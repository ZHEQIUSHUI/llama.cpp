# Axera NPU Backend for llama.cpp

> [!NOTE]
> Early stage. Two modes:
> - **Per op** (default): the matrix products of BF16 GGUF models run on the NPU, everything else on the CPU.
> - **Graph template** (`GGML_AXERA_TEMPLATE=qwen35`): whole Qwen3.5 graphs run as one NPU kernel per layer.
>
> Any GGUF weight type (BF16, Q8_0, Q4_K_M, ...) works with the template; see below.
>
> AX8860, Qwen3.5-0.8B BF16 GGUF, single core: CPU only 473 ms per token, per op 101 ms, template 39.7 ms
> (25 tokens/s), template with MXFP8 weights 30.5 ms (33 tokens/s).

The Axera backend ([`ggml/src/ggml-axera`](../../ggml/src/ggml-axera)) runs GGML ops on the NPU of Axera SoCs (AX8860)
with precompiled kernels, and leaves everything else to the CPU backend.

- **Memory**: tensors of offloaded layers live in CMM (the SoC's physically contiguous memory). CMM is mapped into the
  process, so the CPU backend reads and writes the same buffers without copies; the NPU addresses them physically.
- **Kernels**: AXDSL programs compiled offline into `.axmodel` files. Weights are runtime inputs, so one kernel serves
  every tensor of the same shape and GGUF weights are used as they are. The backend looks kernels up by name in
  `GGML_AXERA_KERNEL_DIR` (default `./axera_kernels`) and only claims an op if its kernel exists. A kernel file is
  `<name>_npu<N>.axmodel`, N the NPU cores it was compiled for (`qwen3_attn_mx8_kvc_c4096_npu1.axmodel`,
  `qwen35_mlp_mc128_npu4.axmodel`); below kernels are named without that suffix.
- **Cores**: `GGML_AXERA_CORES` (default 4) picks `<name>_npu<GGML_AXERA_CORES>` where it exists, else the single-core
  `<name>_npu1`; `GGML_AXERA_CORES=1` runs the single-core builds only. The multi-core builds split their work over the
  cores without synchronizing them (the driver neither synchronizes cores reliably inside a task nor runs tasks on
  different cores concurrently) and need all cores: other NPU users at the same time make their calls time out.

## Supported ops

| Op | Condition | Kernel |
|---|---|---|
| `MUL_MAT` | BF16 weights in CMM, F32 activations | `mul_mat_bf16_n<C>_k<K>`: inputs `x` [1, K] bf16, `w` [C, K] bf16; output `out` [1, C] f32 |

`C` is the weight row count `N` itself, or the largest available `C` that divides `N`: a vocabulary-sized output
projection runs as `N / C` calls. Several activation rows (prompt processing) run row by row and reread the weights
each time, until a batched kernel exists.

## Graph template: Qwen3.5

With `GGML_AXERA_TEMPLATE=qwen35` the backend claims the whole graph and does not run it op by op. Per token, it runs
one AXDSL kernel per layer, either a gated delta net layer or a gated attention layer, and reads the weights straight
from the GGUF tensors in CMM. Two weights are converted once at load: `ssm_a` to `log(-ssm_a)`, and the conv1d taps
are reversed and stored as bf16. Then the CPU applies the final norm and the GEMV kernels compute the output
projection.

Kernels in `GGML_AXERA_KERNEL_DIR`: `qwen35_gdn`, the attention kernels `qwen35_attn_kvc_c<LMAX>`
(one per supported context, see below), plus the `mul_mat` kernel for the output projection. Nothing else is needed:
the backend reads the context `LMAX` and the KV block size from the attention kernel's mask input, and the RoPE
parameters (rotated dims, base, linear scale) from the graph's ROPE nodes.

Optional prefill kernels `qwen35_gdn_prefill<P>` / `qwen35_attn_prefill<P>` (P = 128, 64, 32, 16, 8, 4)
process P consecutive prompt tokens per call. The projections and the MLP read every weight once per chunk. For
P >= 16 the delta rule runs in the chunked form (the triangular solve of `gdn_stable_chunk`, 32 tokens per solve,
8 heads batched), so a chunk is mostly matrix products; for P = 8 / 4 it runs token by token. The backend splits a
batch greedily into the largest chunks that fit and runs the rest through the decode kernels. On AX8860 (BF16) a
1708-token prompt takes 1.0 s (0.58 ms per token, 1712 tokens/s; token by token through the decode kernels it was
26-30 ms per token).

Limits of the template:
- The recurrent state and the KV cache live in the llama.cpp memory, so its checkpoints, prompt cache, sequence edits
  and parallel sequences apply.
- Contexts up to 32768 cells with the fused attention kernels, beyond that split attention (flash decoding).
- Prompt tokens go through the prefill kernels in chunks of 64 / 32 / 16 / 8 / 4, if present, and through the decode
  kernels otherwise.

### Context length

An NPU kernel's IO sizes are fixed at compile time (the runtime rejects a smaller buffer), and the attention kernels
take the whole KV cache of a layer as one IO. Kernels are therefore compiled per context and named by it:
`qwen35_attn[_mx8]_kvc_c<LMAX>` and `qwen35_attn_prefill<P>[_mx8]_kvc_c<LMAX>`, several side by side.
For `-c N` the backend takes the smallest `LMAX >= N` and allocates the llama.cpp KV cache tensors with room for
`LMAX` cells (`get_alloc_size`); the kernels only scan the blocks in use. `-c 6000` runs on the 8192 kernels with a
96 MiB KV buffer instead of 72. The attention kernels' cost does not depend on `LMAX`: decode is 40.6 ms per token with
`-c 4096` and 41.2 ms with `-c 32768`.

The fused kernels end at 32768 cells: the SDMA bounds register of a dynamically indexed tensor (`bytes_max`, 24 bits)
limits one KV IO to 32 MiB, and a Qwen3.5-0.8B cell is 1 KiB (2 KV heads x 256 x f16).

### Long contexts: split attention (flash decoding)

Softmax attention over N cells can be computed per segment and merged exactly: a segment i gives its running max
`m_i`, sum `l_i = sum exp(s - m_i)` and unnormalized output `o_i = sum exp(s - m_i) v`, and
`o = sum_i exp(m_i - m) o_i / sum_i exp(m_i - m) l_i` with `m = max_i m_i`. For a KV cache no fused kernel holds (or
with `GGML_AXERA_FLASH=1`), every attention layer runs as three kernels per batch of R tokens (R = 1, 2, 4, ..., 64
for decode, batched decode and prefill chunks):

- `qwen35_attn_fd_head<R>`: norm, projections, q / k norm, RoPE; outputs q, the gate and the new K / V rows, which
  the host copies to their cells.
- `qwen35_attn_fd_seg<R>_c<S>`: the tokens' attention over one segment of S = 32768 cells, bound by address into the
  llama.cpp KV cache; called once per segment in use, with that segment's slice of the KQ mask. Each block of 256
  cells (512 for R = 1) is loaded and converted once for both KV heads; the block size is read from the kernel's mask
  input.
- `qwen35_attn_fd_tail<R>`: the merged attention output (merged on the CPU) through the gate, `o_proj` and the MLP.

The kernels do not depend on the context, so any `-c` works (the KV cache is allocated in whole segments). On AX8860,
a 39666-token prompt (`-c 65536`, a number stated in its first line, then 30 copies of the GPL) was prefilled in
81 s (2.0 ms per token; the attention grows with the context) and the model answered with the number; decode at that
context took 65 ms per token (with 64-cell blocks, one loop per KV head: 116 s and 106 ms). Below 32768 cells the
fused kernels stay faster (the split costs two more kernel calls per attention layer): forced split attention at
`-c 4096` decodes at 43.2 ms instead of 40.7 ms per token, with PPL 2.4297 / 40.2034 instead of 2.4237 / 40.1693
(other block sizes and the final division on the CPU in f32 change the bf16 rounding).

```bash
GGML_AXERA_TEMPLATE=qwen35 ./build-axera/bin/llama-completion -m qwen3.5-0.8b-bf16.gguf -ngl 99 -c 4096 -p "..." -n 128
```

### llama-server

```bash
GGML_AXERA_TEMPLATE=qwen35 ./build-axera/bin/llama-server -m qwen3.5-0.8b-bf16.gguf -ngl 99 -c 16384 -np 4 \
    --host 0.0.0.0 --port 8080
```

The kernels work on llama.cpp's own memory:
- The delta-net layers update their recurrent state in place in the recurrent cache tensors (`cache_s_l*`,
  `cache_r_l*`).
- The attention layers write and read the llama.cpp KV cache (`cache_k_l*`, `cache_v_l*`) in its layout: one f16 row
  per cell, holding all KV heads. They write at the cells llama.cpp assigns (`attn_inp_k_idxs`) and attend through
  its KQ mask (`attn_inp_kq_mask`).

So every server feature that saves and restores memory works unchanged: context checkpoints, the RAM prompt cache, and
removing tokens from a sequence. On AX8860, a three-turn chat's third turn processed 33 prompt tokens in 0.7 s. After
an unrelated conversation in between, returning to the first one processed 4 prompt tokens instead of 132, with the
same answer as without the detour.

The llama.cpp KV cache is used if it is f16 (the default), V is not transposed (flash attention, chosen
automatically), and a `qwen35_attn*_kvc_c<LMAX>` kernel holds it. Otherwise the template keeps a private KV cache
for a single sequence, and the server then needs `-np 1 --cache-ram 0`.

Parallel sequences (`-np N`, `llama-parallel`, `llama-perplexity` with several chunks per batch) work in both KV
layouts: per-sequence streams (the default for an explicit `-np`) and the unified cache (`-kvu`). Each token of a
batch runs with its sequence's row of the recurrent cache and its stream's KV cells; prompt chunks never cross a
sequence. `llama-perplexity -b 2048` (4 sequences per batch) gives the same PPL as `-b 512` (2.4237 / 40.1693). With
`-np 4`, four chat requests sent at once return the same answers as one at a time (with `-kvu` a sequence's cells
interleave with the others', which changes the bf16 summation order, and a long answer can flip a near-tie token).
### Batched decode

With the unified KV cache (`-kvu`; llama-server's default `-np auto` sets it), a decode batch of several sequences
runs through the batched kernels `qwen35_gdn_batch<B>[_mx8]` / `qwen35_attn_batch<B>[_mx8]_kvc_c<LMAX>`
(B = 4, 2):
one call per layer for one token of each of B sequences, so every weight is read once per B tokens. The delta-net
kernel updates B consecutive rows of the recurrent cache in place, each sequence with its own conv and delta-rule
state; the attention kernel writes each token's K/V at its own cell and attends through its own KQ mask row. The
output projection runs B rows per call through `mul_mat_<bf16|mx8>_b<B>_n<C>_k<K>` (also for the rows of
`llama-perplexity`). Per-sequence KV streams (an explicit `-np` without `-kvu`) keep the token-by-token path.

`llama-batched-bench -npp 128 -ntg 64 -kvu`, decode tokens per second:

| Sequences | BF16 | MXFP8 |
|---|---|---|
| 1 | 24.4 | 31.9 |
| 2 | 46.6 | 59.9 |
| 4 | 84.3 | 106.4 |

With `GGML_AXERA_WEIGHTS=mxfp8` the batched kernels are the MXFP8 ones (bf16 batched kernels, if only those exist, use
the bf16 GGUF tensors). PPL does not depend on the batching: BF16 2.4237 / 40.1693, MXFP8 2.4121 / 40.2833 for
`-b 512` and `-b 2048` alike (the prompt rows go through the bf16 prefill kernels, the logits through the MXFP8 output
projection). The delta-net batch kernels match the decode kernels bit for bit on the simulator; with `-kvu` the cells of the sequences
interleave, so a long greedy answer can still part from the one-at-a-time answer at a near tie (four chat requests at
once: one of the four, from its 100th character on).

A request the template cannot continue from fails with an HTTP 500 error and a log line; the server keeps running.
With BF16 weights, the chat endpoint streams at 43 ms per token.

### Quantized GGUF (Q8_0, Q4_K_M, Q4_0, ...)

The template accepts GGUF files of any weight type that ggml can dequantize. At load it converts each projection to
the kernels' format, BF16 or (with `GGML_AXERA_WEIGHTS=mxfp8`) MXFP8, using ggml's own `to_float`, and keeps the copy
in CMM. The speed is therefore that of BF16 / MXFP8, and the accuracy is that of the GGUF. Qwen3.5-0.8B,
`llama-perplexity` (English / Chinese), against the llama.cpp CPU backend on the same file:

| GGUF | CPU backend | template BF16 | template MXFP8 | decode ms per token (BF16 / MXFP8) |
|---|---|---|---|---|
| Q8_0 | 2.476 / 40.13 | 2.440 / 40.21 | 2.428 / 40.30 | 40.5 / 31.0 |
| Q4_K_M | 3.571 / 42.14 | 3.557 / 41.94 | 3.544 / 42.11 | 40.6 / 30.9 |
| Q4_0 | 5.593 / 46.89 | 5.558 / 46.95 | 5.548 / 47.34 | 40.6 / 31.0 |

The NPU's 4-bit formats do not help here. Re-quantizing a Q4_K_M model to MXFP4 or NVFP4 by rounding loses far more
than Q4_K_M itself; in transformers, 900-token Chinese / English texts give Q4_K_M 38.1 / 3.11, MXFP4 51.9 / 10.7,
and NVFP4 46.1 / 6.76. 4-bit NPU weights would need a calibrated quantization, not a format conversion.

### Loading: the GGUF stays mapped

With a template the backend takes llama.cpp's file mapping (mmap, the default) as weight buffers of the device
(`buffer_from_host_ptr`): the GGUF is not copied into CMM, only the converted weights the kernels read are (MXFP8 /
bf16 projections, small tensors, and for MoE the MXFP4 experts, which the kernels repack in place). The graph stays
whole because the mapped weights count as the device's. CMM peak on a 48 GB AX8860 board (same outputs, bit for bit):

| Model (GGUF) | weights copied into CMM | mapped |
|---|---|---|
| Qwen3.5-9B Q4_K_M | 13.4 GB | 8.6 GB |
| Qwen3.5-35B-A3B MXFP4_MOE | 24.8 GB | 22.0 GB |
| Qwen3.8-27B Q4_K_M | 41.3 GB | 26.3 GB |

The mapping should be on local storage (eMMC): over NFS every load pages the file in again. `--no-mmap` or
`GGML_AXERA_MMAP=0` loads the GGUF into CMM as before; without a template the per-op `MUL_MAT` kernels read bf16 weights
in CMM, so there the default is off (`GGML_AXERA_MMAP=1` forces it).

### MXFP8 weights

With `GGML_AXERA_WEIGHTS=mxfp8` the template quantizes the bf16 GGUF weights to MXFP8 at load: E4M3 elements and one
E8M0 scale per 32 inputs, with the exponent chosen so that nothing saturates. This roughly halves the bytes read per
token. It needs the kernels `qwen35_gdn_mx8`, `qwen35_attn_mx8` and `mul_mat_mx8_n<C>_k<K>`.

Qwen3.5-0.8B, `llama-perplexity`, 2 x 512 tokens:

| Weights | ms per token | English (GPL text) | Chinese (technical prose) |
|---|---|---|---|
| BF16 | 39.7 | 2.431 | 40.18 |
| MXFP8 | 30.5 | 2.598 (+6.9%) | 40.57 (+1.0%) |

(PPL with the prefill kernels; decode only it was 2.427 / 40.24 for BF16 and 2.582 / 40.40 for MXFP8.)

The loss is spread over the MLP and the delta-net projections and is the same for other exponent rules (OCP floor,
best of two), so keeping single tensors in bf16 does not recover it. Use MXFP8 when the speed matters more.

MXFP8 helps decode only, where the weights are read once per token and bandwidth bounds the time. In prefill a chunk
reads each weight once for up to 64 tokens, so for Qwen3.5-0.8B it is compute bound and MXFP8 only adds
dequantization (1708-token prompt: 0.75 ms per token with MXFP8 prefill kernels, 0.58 ms with BF16). Larger models are
bound by reading the weights in prefill too (9B, below). With `GGML_AXERA_WEIGHTS=mxfp8` the prefill kernels therefore
take the MXFP8 weights if the kernel directory has MXFP8 prefill kernels (`qwen35_gdn_prefill<P>_mx8`, build them where
they pay off), else the bf16 GGUF tensors (already in CMM), sharing the recurrent state and KV cache with the MXFP8
decode kernels either way (0.8B: prompts at 0.58 ms per token, decode at 31.5 ms per token).
`GGML_AXERA_PREFILL_WEIGHTS=mxfp8` / `bf16` overrides the choice.

### Long prompts

A prompt chunk of P tokens prepares the KQ mask of every segment once for all attention layers: llama.cpp's mask rows
are converted to bf16 on a host thread pool, and each row's first and last visible cell decide which blocks and which
segment-kernel cores have work. With a fused kernel for the context, batches over masks of at least 4096 cells
(`GGML_AXERA_SPLIT_MIN`, 0: always) take the split attention kernels too, which read only the blocks a token can see.
On AX8860 the 39666-token prompt above takes 42.6 s instead of 81 s (decode 69 ms per token), PPL is unchanged
(2.4237 / 40.1693).

### Images (llama-mtmd-cli, llama-server --mmproj)

Qwen3.5 checkpoints carry a vision encoder (a ViT: 12 layers of 768 for 0.8B, 24 x 1024 for 2B / 4B, 27 x 1152 for
9B and up). Its mmproj GGUF (`convert_hf_to_gguf.py --mmproj`) runs on the NPU when the kernels
`qwen35_vit_{embed,pre,attn,post,merge}<N>_h<hidden>` exist (`q35_vit.py`; N = 256, 1024, 2048, 3072
patches): the backend recognizes the clip graph by its `v.blk.*` weights, cuts the image into patches in the 2 x 2
merge order, interpolates the learned position table and builds the 2-D RoPE tables on the host, and runs the patch
projection, every layer (three kernels: norm / qkv / RoPE, bidirectional attention, output projection / MLP) and the
merger on the NPU, the image padded to the next N with the padding masked. Images of more than 3072 patches need
`--image-max-tokens 768`; `--no-mmproj-offload` runs the encoder on the CPU instead.

The text kernels take M-RoPE positions (an image's tokens share their temporal position and differ in height and
width; interleaved sections as `GGML_ROPE_TYPE_IMROPE`) and prompt chunks need consecutive cells, not consecutive
positions. On AX8860 a 1000 x 667 photo (2604 patches, 651 image tokens) encodes in 471 ms (the CPU: 138.6 s) and
Qwen3.5-0.8B describes it with the same sentence as llama.cpp on x86.

### Qwen3.5-9B

The kernels follow the model's config, so the same scripts build them for 9B (`MODEL=<Qwen3.5-9B dir>`, kernels in
their own `GGML_AXERA_KERNEL_DIR`). Its delta-net layers have 32 value heads for 16 key heads; llama.cpp's converter
reorders the value heads from the checkpoint's grouped order to a tiled one (value head j uses key head j % 16), and
the kernels take that order (`VORDER=tiled`). A 9B cell is 2 KiB, so the fused attention kernels go up to 8192 cells
and longer contexts use the split attention (8192-cell segments). The vision encoder (27 layers of 1152, head dim 72)
has kernels up to 3072 patches (`--image-max-tokens 768`).

On AX8860 with the Q4_K_M GGUF (5.8 GB; the template converts the weights at load), single-core:

| Weights | Load | Decode | PPL (zh, 2 x 512) |
|---|---|---|---|
| CPU llama.cpp (x86), same GGUF | | | 18.9679 |
| BF16 | 19 s | 277 ms per token | 19.0128 |
| MXFP8 (`GGML_AXERA_WEIGHTS=mxfp8`) | 40 s | 164 ms per token | 19.0561 |

Prompts: at 64 tokens per chunk the 9B layers wait on the weights (about 48 GB/s), so the 9B kernel directory holds
prefill kernels of 128 tokens with MXFP8 weights (`q35_prefill.py` with `PCHUNK=128 WFMT=mx8`; the MLP runs in passes
over its intermediate channels to fit the on-chip memory). A 7962-token prompt (a number stated in its first line)
takes 22.7 s instead of 41.0 s with BF16 chunks of 64 (2.9 ms per token), the answer names the number, and PPL is
19.1367. Beyond 8192 cells the split attention's head and tail kernels take the MXFP8 weights too
(`qwen35_attn_fd_{head,tail}<R>_mx8`, `q35_flash.py` with `WFMT=mx8`, sharing the quantized weights of the decode
kernels), with a 128-row segment kernel for the chunks of 128: 15888 tokens in 44.9 s instead of 82.8 s and 39666 tokens
in 134 s instead of 229 s, the number found in both; decode at 40k context 209 ms per token instead of 248.

The text matches llama.cpp on the CPU. The cat photo above (limited to 512 image tokens) encodes in 1.1 s on the NPU
(attention in blocks of 512 query rows; 1.8 s before; 2.0 s with up to 768 image tokens) and is described as "A stylish gray-and-white cat wearing oversized round sunglasses and a shiny pink hooded jacket
with white drawstrings, set against a soft pink background."

### Qwen3.5-35B-A3B (MoE)

The MoE models (`qwen35moe` GGUF) run with their routed experts in MXFP4, as `llama-quantize ... MXFP4_MOE` stores
them (the rest Q8_0, converted at load to BF16 or MXFP8 like any GGUF). Each layer takes two kernel calls:
`qwen35moe_gdn[_mx8]` / `qwen35moe_attn[_mx8]_kvc_c<LMAX>` run the layer up to the MLP and then its dense part (the
shared expert with its sigmoid gate on the residual, and the router logits); the host takes softmax / top-K /
renormalization of the 256 logits and binds the 8 chosen experts' weights, by address, to
`qwen35moe_experts8_gu_mxfp4`, which adds them to the residual. At load the experts are repacked in place (on the
host thread pool) from ggml's MXFP4 blocks (scale byte + 16 bytes of codes) to the kernels' layout (codes
[rows][K / 2], then the scales), with each expert's gate and up rows as one weight (one projection instead of two;
the board pays ~25 us per small projection): llama.cpp allocates `ffn_gate_exps` and `ffn_up_exps` back to back, the
few layers where a buffer boundary falls between get a buffer of their own (a GGUF with `ffn_gate_up_exps` works as
is). `qwen35moe_experts8_mxfp4` (gate and up apart) serves when the gate-up kernel is missing.

Prompts go in chunks of 128 / 64 / 16 / 4 tokens: `qwen35moe_{gdn,attn}_prefill<P>` (BF16 weights) end with the
chunk's base, h2 and router logits rows; up to 2048 tokens of chunks of 128 (`GGML_AXERA_MOE_GROUP` chunks, default
2048 / 128) run layer by layer and the routed experts once per layer for the whole group: the host picks each token's
experts, groups the picks by expert into slots of up to 64 rows (gathering the h2 rows into non-cacheable CMM), and
`qwen35moe_slots32_g64_gu_mxfp4[_bo]` runs 32 slots per call (each slot one expert's weights, bound by address), whose
rows (bf16 with `_bo`) the host adds to the tokens' base. A group holds at most one ubatch, so run MoE models with
`-ub 2048` (llama.cpp's default 512 leaves about 16 rows per expert where a slot takes 64: 46.2 s instead of 33.4 s for
15888 tokens). With MXFP8 weights and the four-core kernels (`q35_mc.py` with the MoE config: `qwen35_gdn_mc128_mx8`,
`qwen35_attn_{head,tail}_mc128_mx8`, and `qwen35_moe_pre_mc128_mx8` in place of the MLP: x = base + the partials, 32
tokens per core, the shared expert, h2 and the router logits) the 128-token chunks run on the multi-core chain, one
layer per call between the experts (the layer input in A / B without partials); the attention segments merge on the
NPU (the tail's slots) instead of the host.
Long contexts take the split attention (`qwen35_attn_fd_{head,seg}*` with `qwen35moe_attn_fd_tail<R>`, which ends with
the MoE's dense part); the 128-row segment kernel takes the query heads in 4 passes per block (`SEG_PARTS`: 1024 score
rows at a time overflow the on-chip memory; 29.3 -> 12.1 ms over 32768 cells). Kernels: `q35_moe.py`,
`q35_moe_prefill.py`, `q35_moe_flash.py` (MXFP4 inputs declared as uint8 for the board runtime, `board_io.py`); the
output projection GEMV is the 2B one (`n31040_k2048`), the vision encoder the 9B one with its own merger (out 2048).

On AX8860 (20 GB GGUF, loaded into CMM in 38 s): decode at 88 ms per token with MXFP8 (gdn layers 36 ms, attention
10 ms, experts 25 ms, output projection 10 ms) and 114 ms with BF16; PPL (zh, 2 x 512) 16.67 (MXFP8 decode) / 16.62
(BF16) against 16.58 for llama.cpp on x86 with the same GGUF (en 1.0128 against 1.0087); the cat photo above at 768
image tokens in about 1.8 s, described as "A stylish cat with fluffy white and gray fur wears a shiny pink leather
hoodie and oversized round sunglasses, posing confidently against a soft pink background." On a board with 48 GB of
CMM (single core / 4 cores): 1717-token prompt 7.3 / 6.9 s, 15888 tokens 59 / 55 s, 66086 tokens 308 / 263 s, 118926
tokens 663 / 539 s, the needle found in each; decode 95 ms per token at 4k, 199 / 172 ms at 119k.

### Multi-core kernels

By default (`GGML_AXERA_CORES=4`) the kernels that have a four-core build (`<name>_npu4`, AXDSL without synchronization
between the cores: each core takes its share of heads, output channels, slots or KV blocks and writes only its own
outputs) replace the single-core ones: for the dense models prefill chunks of 128 / 64 tokens and decode
(`GGML_AXERA_MC_DECODE=0` keeps decode single-core) run as a chain of `qwen35_{gdn,mlp,attn_head,attn_tail}_mc<P>`
kernels that pass the residual as a base plus one fp32 partial per core, the segment kernels split their blocks over
the cores and the tail merges the partial results on the NPU (more segments than its slots: merged on the CPU in
between), and the output projection uses the four-core `mul_mat_bf16_n<C>_k<K>`; for the MoE models the segment kernels, the
slots kernel and the gate-up experts kernel (the cores' partial sums added on the host). Qwen3 models take the same
chain without the gated delta net (`qwen3_{mlp,attn_head,attn_tail}_mc<P>_mx8`, MXFP8 weights sliced per core with
their scales, the four-core `mul_mat_mx8_n<C>_k<K>` output projection); a layer with less than 32 MB of weights (0.6B)
decodes on the single-core fused kernel below 1024 cells of KV (`GGML_AXERA_MC_DECODE_CELLS=<n>` overrides), where the
chain's four calls per layer cost more than the bandwidth gains. The split attention takes the largest segment kernel
within the cache (fewer segments to merge than the tail's slots).

On boards with the October 16-19 images this is stable (0.8B `llama-perplexity` three times bit-identical, PPL zh
46.81 against 46.74 single-core from the order of the partial sums; on an older board it gave NaN and timeouts). The
gain is limited by the DRAM bandwidth those boards reach, about 84 GB/s for all cores together (one core about
53 GB/s; the CPU and the NPU together share the same 84 GB/s): compute-bound kernels scale (35B segment kernel 12.1 ->
3.3 ms, 119k-token prompt 663 -> 539 s), bandwidth-bound ones barely (0.8B decode 42 -> 40 ms). A board with its DRAM
at 7500 MT/s reaches 165 GB/s with four cores (a copy kernel: 256 MB in 1.55 ms), while one core stays at about
53 GB/s on either board: there the single-core kernels gain little and the bandwidth-bound ones need their four-core
builds. `GGML_AXERA_PROF=1` prints per-kernel times at exit.

## Graph template: Qwen3 transformers

The same template runs Qwen3 transformers: attention layers only, no output gate, full-width RoPE, RMSNorm weights as
they are. That covers Qwen3 itself and the models built on its text model: the language model of Qwen3-VL and
Qwen3-ASR (llama.cpp arch `qwen3vl`), the Qwen3-TTS talker (`qwen3tts`), Qwen3-Embedding and Qwen3-Reranker. The
backend tells the two families apart by the graph (gated delta net layers or not) and takes `qwen3_*` kernels for
Qwen3, built by the same scripts from the model's config (`MODEL=<Qwen3 dir>`: a config without `text_config` /
`layer_types` is a Qwen3 one, `q35_step.py` then leaves out the gate and the delta-net layer). `GGML_AXERA_TEMPLATE=qwen3`
is a synonym of `qwen35`.

```sh
# kernels of a Qwen3 model (random weights of the config's shapes); HEAD_C divides the vocabulary (151936 = 8 x 18992)
E="QW_RANDOM=1 KVL=cell BC=256 MODEL=<Qwen3 dir> HEAD_C=18992"
env $E LMAX=4096 WFMT=mx8 python ggml_template_build.py <kernel_dir>              # decode, output projection
env $E NODEC=1 LMAX=4096 PCHUNK=64 WFMT=mx8 python q35_prefill.py <kernel_dir>    # prefill chunks (P = 128 .. 2)
env $E R=1 S=16384 CHECK=0 WFMT=mx8 python q35_flash.py <kernel_dir>             # split attention (R = 1 .. 64)
env $E LMAX=4096 B=2 WFMT=mx8 python q35_batch.py <kernel_dir>                   # batched decode
# four-core chain (P = 1, 16, 64, 128), output projection and segments
env $E CORES=4 P=64 WFMT=mx8 LM_C=18992 python q35_mc.py <kernel_dir>
R=64 S=4096 CORES=4 python q35_flash.py <kernel_dir>                               # and S=16384
```

Models of the same shapes share their kernels: Qwen3-0.6B, Qwen3-ASR-0.6B, Qwen3-Embedding-0.6B and
Qwen3-Reranker-0.6B (1024 wide, 16 / 8 heads of 128, MLP 3072) run on one kernel directory; Qwen3-ASR-1.7B and the
Qwen3-TTS-1.7B talker (2048, MLP 6144) on another, the talker with an output projection of its 3072 codec tokens
(`HEAD_C=3072`).

On AX8860 (MXFP8 weights, the GGUF mapped from eMMC, DRAM at 7500 MT/s; one core / four cores):

| Model (GGUF) | Decode, short | Decode at 1.9k | Prompt (1852 tokens) | PPL (zh, 2 x 512) | CPU PPL, same GGUF |
|---|---|---|---|---|---|
| Qwen3-0.6B (BF16) | 28.0 / 26.4 ms | 63.6 / 36.7 ms | | 43.845 / 43.350 | 42.61 |
| Qwen3-8B (Q4_K_M) | 152.7 / 75.5 ms | 198.3 / 77.3 ms | 6.30 / 2.81 s | 17.886 / 17.818 | 17.754 |

A 15539-token prompt with a number stated in its first line (split attention) takes 76.2 / 20.6 s on Qwen3-8B (decode
243 / 97 ms per token there), and both answer with the number; Qwen3-0.6B took 19.9 s single-core. Qwen3-8B peaks at
10.1 GB of CMM at 16k context. Four-core runs repeat bit-identically (three perplexity runs); they differ from one core
by the order of the cores' partial sums.

### Embeddings and reranking

With pooled embeddings (`llama-embedding`, `llama-server --embedding` / `--rerank`) the graph has no output
projection: the template stops at the final norm (`result_norm`, every row) and the pooling after it (`inp_cls` /
`inp_mean` rows, the rerank head `cls.output`, its softmax) runs on the CPU; the backend does not claim those ops.

Qwen3-Embedding-0.6B (last-token pooling) gives embeddings of cosine 0.996 - 0.997 to llama.cpp's on the CPU and the
same ranking of documents for each query, Qwen3-Embedding-8B (Q8_0 GGUF, on Qwen3-8B's kernels) 0.994 - 0.996 and the
same rankings (six inputs, 114 tokens, in 1.2 s; converting its weights to MXFP8 at the first graph takes 19 s); Qwen3-Reranker-0.6B (`--pooling rank`) gives the same relevance decisions
as on the CPU.

### Speech recognition: Qwen3-ASR

`llama-mtmd-cli -m qwen3-asr.gguf --mmproj mmproj-qwen3-asr.gguf --audio speech.wav -p "Transcribe the audio."`. The
transformers checkpoints (`Qwen/Qwen3-ASR-0.6B-hf`, `-1.7B-hf`) keep the audio config at the top level, the audio
tower as `model.audio_tower.*` and the projector as `model.multi_modal_projector.linear_{1,2}`, and give the text
model plain RoPE; the converter maps them to the Qwen3-Omni names and writes M-RoPE sections that are plain RoPE for
the sequential positions of audio and text.

The audio encoder's 18 (0.6B) / 24 (1.7B) transformer layers run on `q3a_audio.py`'s kernels
(`qwen3a_{pre,attn,post}<N>_h<H>`: q35_vit.py's layer kernels without RoPE, GELU erf, LayerNorm 1e-5; N rows
padded, 13 rows per second of audio), the conv front-end before them and the post norm and projector after them are
the graph's own nodes, run by the CPU backend from inside the backend. Without kernels for the row count the whole
encoder runs on the CPU.

The matrix products of the front-end's two 480-channel convolutions (ggml's im2col rows times the kernel), most of its
CPU time, run on `q3a_conv.py`'s kernels (`qwen3a_conv_o<OC>_k<K>_r<R>`, rows padded to R: 6400 and 1664 cover the
8 one-second chunks of an encoder call); bias, GELU and the rest of the front-end stay on the CPU.

On AX8860 (text model MXFP8, mmproj Q8_0), per encoder call (8 s of audio):

| Model | Encoder, CPU | Encoder, NPU | Transcript of 60 s of Chinese speech |
|---|---|---|---|
| Qwen3-ASR-0.6B | 2.8 s | 0.45 s | as on the CPU but for one interjection |
| Qwen3-ASR-1.7B | 3.3 s | 0.47 s | as on the CPU |

Of the 0.45 s about 0.35 s is the front-end's CPU part; the 18 (0.6B) / 24 (1.7B) layers take 33 / 47 ms.

### Speech synthesis: Qwen3-TTS

`llama-tts -m qwen3-tts.gguf -mm mmproj-qwen3-tts.gguf -ngl 99 -c 4096 -p "<text>" --tts-lang zh --tts-speaker-file
speaker.wav --output out.wav` (`-c 4096`: llama-tts's default context of 32768 cells would take the split attention,
slower per frame; a generation is at most 512 frames). Three models take part per 80 ms codec frame:

- **The talker**, the 28-layer Qwen3 text model with an output projection of 3072 codec tokens, runs on the template.
  llama-tts keeps its hidden states (embeddings without pooling: the final norm is named `result_embd_pooled`, the
  template writes it with the logits), pads the codec logits to the text vocabulary with -inf (the template fills that
  concat) and feeds mixed inputs, token embeddings set into rows of given embeddings (the template runs those input
  nodes on the CPU before its layers).
- **The code predictor** (5 Qwen3 layers of 1024, Qwen3-0.6B's shapes) predicts the frame's other 15 codes, 16
  positions per frame, in mtmd's generation graph.
- **The codec decoder** (an 8-layer transformer, ConvNeXt upsampling and a DAC decoder with SnakeBeta activations)
  turns windows of 72 frames into 24 kHz audio, also an mtmd graph.

mtmd's graphs have no template of their own (`ggml-axera-gen.cpp`): the backend runs them on the CPU backend from
inside the backend and takes over what has an NPU kernel:

- each code predictor layer call as one call of Qwen3-0.6B's fused decode layer kernel (`qwen3tts_cp_attn_kvc_c4096`,
  a copy of its bf16 `qwen3_attn_kvc_c4096`, or the MXFP8 `qwen3tts_cp_attn_mx8_kvc_c4096`, the run's weight format
  first; the 16 positions' KV resident in the kernel). The backend finds the calls in the graph (`ln1` ... `ffn_down` of each layer and position)
  and skips the nodes only they need, the attention and the graph's own KV cache among them. With a stacked kernel
  `qwen3tts_cp_stack5[_mx8]_c<LMAX>` in the directory (`q3tts_cp.py`: the 5 layers in one kernel, the smallest LMAX
  that holds the positions) a position's 5 layers are one call (16 instead of 80 calls per frame) when the layers
  follow each other in the graph and nothing else reads their outputs; `GGML_AXERA_CP_STACK=0` keeps one call per layer;
- the convolutions (ggml's im2col and matrix product) as `conv1d_bf16_ic<IC>_oc<OC>_k<K>_d<D>_l<L>` kernels
  (`conv1d_kernels.py`): K shifted matrix products accumulated in f32 on the input in ggml's [channels][time] layout,
  so the im2col never runs; the kernel's taps repacked once;
- the other matrix products: single rows on the GEMV kernels (one call for the q / k / v or gate / up of one input),
  several rows on `mm_bf16_m<M>_n<N>_k<K>` GEMM kernels of their shapes (`mm_kernels.py`; convolutions without a
  conv1d kernel there too, the backend writing their im2col rows as bf16). A weight the graph transposes on every run
  (a transposed convolution's kernel, `cont(transpose(w))`) is read once through the view's strides and kept as bf16;
  the CONT does not run;
- SnakeBeta (x + sin²(αx) · β⁻¹ per channel) in one pass on the host workers instead of five CPU ops, sin by range
  reduction and a polynomial (vectorized);
- on the host workers too, instead of the CPU backend's element-by-element loops: the concats of the convolutions'
  left context (row copies), the transposes between the [time][channels] and [channels][time] layouts (64 x 64
  blocks), col2im of the transposed convolutions (blocks of output steps) and the ConvNeXt blocks' depthwise
  convolution (computed directly in f32, its f16 im2col not run).

mtmd itself keeps the code generation step's graph: the step builds the same graph for every frame (only its inputs
differ), so `clip_encode` reuses it with the scheduler's allocation while the step's parameters stay, instead of
building and allocating some 6000 nodes again (a codec decoder window or another graph in between drops it).

The rest (the codebook lookups and sampling of the code predictor, the codec decoder's other elementwise ops, its
attention) stays on the CPU backend, slices of small tensors on one thread
(views do not count). The template of a graph is kept for the same graph (telling them apart reads every node). The CPU backend gets threadpools of its own
(without one it starts its threads on every call, here thousands of times per graph).

Precision decides whether a generation ends: the talker samples the end of speech from its codec logits frame by
frame, and with MXFP8 weights Qwen3-TTS-0.6B's talker missed it for some seeds (2 of 4 generations of a Chinese
sentence ran to the 512-frame limit; with the code predictor in bf16 1 of 6, with the code predictor on the CPU 1 of 5
ran twice as long), the CPU never (3 of 3). With bf16 weights (no `GGML_AXERA_WEIGHTS`) all 15 seeds tried ended as on
the CPU. Qwen3-TTS-1.7B's talker in MXFP8 ended every time (13 of 13). The code predictor in MXFP8 does not change
that: 16 generations each with the bf16 and the MXFP8 code predictor (both models, a Chinese and an English sentence,
4 seeds) all ended and read back exactly, and an MXFP8 layer call takes 0.885 ms instead of 1.06 ms. The code
predictor follows `GGML_AXERA_WEIGHTS`: MXFP8 for an MXFP8 run (the 1.7B), bf16 otherwise (the 0.6B).

On AX8860 (`-ngl 99 -c 4096`; 0.6B bf16, 1.7B MXFP8 with its code predictor in MXFP8, the stacked code predictor
kernels, four cores), one Chinese and one English sentence, on a board with its DRAM at 3200 and at 7500 MT/s (the
four-core talker sums in another order, so the 1.7B's samples and lengths differ between the boards):

| Model | Sentence | Audio, time (3200) | Audio, time (7500) |
|---|---|---|---|
| Qwen3-TTS-1.7B | zh | 6.6 s, 15.8 s | 7.0 s, 14.0 s |
| | en | 5.4 s, 12.6 s | 5.2 s, 10.4 s |
| Qwen3-TTS-0.6B | zh | 7.1 s, 15.0 s | 7.1 s, 14.1 s |
| | en | 4.7 s, 10.0 s | 4.7 s, 9.5 s |

Every output, transcribed back by Qwen3-ASR-1.7B, reads exactly as its input text. Per frame (1.7B at 7500 MT/s,
145 ms) the talker takes 35 ms on four cores (41 ms on one; on the 3200 board four cores are slower, 48 ms, the
cores sharing 84 GB/s) and the code predictor's 16 stacked calls 57 ms (80 single-layer calls took 71 ms; 0.6B: bf16
weights, 71 ms); mtmd, sampling and the CPU slices the rest. The code predictor runs on one core, which reads DRAM at
about 53 GB/s on either board. The codec decoder takes about 1.1 s per window of 72 frames (5.8 s of audio). Before
the conv1d kernels, the bf16 weight cache, mtmd's graph reuse and the host workers' concat / transpose / col2im a frame
took 248 ms and a window 4.3 s.

## Source layout

The backend separates what any model uses from the graph templates of one architecture:

| File | Contents |
|---|---|
| `ggml-axera-impl.h` | internal declarations, CMM buffers, kernel registry, host thread pool, profiling timers |
| `ggml-axera.cpp` | backend, device and registry interface; dispatch of a graph to the per-op path or a template |
| `ggml-axera-util.cpp` | runtime init, CMM allocation, settings (`GGML_AXERA_*`), kernel IO helpers, graph slices on the CPU backend |
| `ggml-axera-buffer.cpp` | the CMM buffer type, and the mapped buffers (the GGUF file mapping as device weights) |
| `ggml-axera-weights.cpp` | weight formats (BF16 copies, MXFP8, MXFP4 repacking) and the GEMV calls, per-op `MUL_MAT` |
| `ggml-axera-graph.cpp` | for templates: the graph's tensors by name, the output head (final norm, output projection, MTP's `h_nextn`) |
| `ggml-axera-attn.h/.cpp` | for templates: the llama.cpp KV cache and KQ mask of a batch, split attention (segments, merge), RoPE / M-RoPE tables |
| `ggml-axera-q35.h` | Qwen3.5 / Qwen3 template: its state and per-graph run state |
| `ggml-axera-q35-bind.cpp` | Qwen3.5: kernel loading, binding GGUF tensors and cache tensors to kernel IO |
| `ggml-axera-q35-run.cpp` | Qwen3.5: one graph (recurrent-state bookkeeping, MTP input, chunks, layer calls) |
| `ggml-axera-q35-moe.cpp` | Qwen3.5 MoE: expert routing and the experts / slots kernels |
| `ggml-axera-q35-mc.cpp` | Qwen3.5: the multi-core prefill chain |
| `ggml-axera-vit.cpp` | the Qwen3.5 vision encoder (mtmd) |
| `ggml-axera-audio.cpp` | the Qwen3-ASR audio encoder (mtmd) |
| `ggml-axera-gen.cpp` | graphs without a template (mtmd's generation graphs: the Qwen3-TTS code predictor and codec decoder) on the CPU backend, with their layers and matrix products on the NPU |

A template for another architecture adds its layer kernels (AXDSL), a binding of its GGUF tensor names to their inputs
and its layer loop; the KV cache handling, split attention, RoPE, the output head and the weight formats are shared.

## Build

Cross-compile on an x86-64 host with the Arm GNU toolchain 13.2 (the BSP libraries need glibc 2.38) and the AX8860
BSP (`include/ax_engine_api.h`, `lib/libax_engine.so`, ...):

```bash
T=<arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu>/bin
cmake -S . -B build-axera \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=$T/aarch64-none-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$T/aarch64-none-linux-gnu-g++ \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
    -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
    -DGGML_AXERA=ON -DGGML_AXERA_BSP_DIR=<bsp>/arm64_glibc
cmake --build build-axera -j --target llama-cli llama-completion llama-bench test-backend-ops
```

## Run

On the board, with the BSP runtime libraries on the library path and the kernels for the model's weight shapes:

```bash
export LD_LIBRARY_PATH=/opt/lib:$PWD/build-axera/bin
export GGML_AXERA_KERNEL_DIR=$PWD/axera_kernels
./build-axera/bin/llama-completion -m qwen3.5-0.8b-bf16.gguf -ngl 99 -p "The capital of France is" -n 32
```

`-ngl` places the layers' weights in CMM, which the NPU needs. `GGML_AXERA_DEBUG=1` logs every op shape whose kernel is
missing, i.e. the kernels to compile next.

## Kernels

Kernel files are named `<name>_npu<cores>.axmodel` (see Cores above). The kernels are generated from the GGUF itself:
every BF16 2-D weight shape gets one GEMV kernel (in chunks for
vocabulary-sized shapes), checked on the NPU simulator against PyTorch. The generator needs the Axera `axnn` toolchain and is kept with the AXDSL kernel sources,
outside this repository.
