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
  `GGML_AXERA_KERNEL_DIR` (default `./axera_kernels`) and only claims an op if its kernel exists.
- **Cores**: kernels are single-core. The current NPU driver neither synchronizes cores reliably inside a task nor runs
  tasks on different cores concurrently.

## Supported ops

| Op | Condition | Kernel |
|---|---|---|
| `MUL_MAT` | BF16 weights in CMM, F32 activations | `mul_mat_bf16_n<C>_k<K>.axmodel`: inputs `x` [1, K] bf16, `w` [C, K] bf16; output `out` [1, C] f32 |

`C` is the weight row count `N` itself, or the largest available `C` that divides `N`: a vocabulary-sized output
projection runs as `N / C` calls. Several activation rows (prompt processing) run row by row and reread the weights
each time, until a batched kernel exists.

## Graph template: Qwen3.5

With `GGML_AXERA_TEMPLATE=qwen35` the backend claims the whole graph and does not run it op by op. Per token, it runs
one AXDSL kernel per layer, either a gated delta net layer or a gated attention layer, and reads the weights straight
from the GGUF tensors in CMM. Two weights are converted once at load: `ssm_a` to `log(-ssm_a)`, and the conv1d taps
are reversed and stored as bf16. Then the CPU applies the final norm and the GEMV kernels compute the output
projection.

Kernels in `GGML_AXERA_KERNEL_DIR`: `qwen35_gdn.axmodel`, the attention kernels `qwen35_attn_kvc_c<LMAX>.axmodel`
(one per supported context, see below), plus the `mul_mat` kernel for the output projection. Nothing else is needed:
the backend reads the context `LMAX` and the KV block size from the attention kernel's mask input, and the RoPE
parameters (rotated dims, base, linear scale) from the graph's ROPE nodes.

Optional prefill kernels `qwen35_gdn_prefill<P>.axmodel` / `qwen35_attn_prefill<P>.axmodel` (P = 128, 64, 32, 16, 8, 4)
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
`qwen35_attn[_mx8]_kvc_c<LMAX>.axmodel` and `qwen35_attn_prefill<P>[_mx8]_kvc_c<LMAX>.axmodel`, several side by side.
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
runs through the batched kernels `qwen35_gdn_batch<B>[_mx8].axmodel` / `qwen35_attn_batch<B>[_mx8]_kvc_c<LMAX>.axmodel`
(B = 4, 2):
one call per layer for one token of each of B sequences, so every weight is read once per B tokens. The delta-net
kernel updates B consecutive rows of the recurrent cache in place, each sequence with its own conv and delta-rule
state; the attention kernel writes each token's K/V at its own cell and attends through its own KQ mask row. The
output projection runs B rows per call through `mul_mat_<bf16|mx8>_b<B>_n<C>_k<K>.axmodel` (also for the rows of
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
token. It needs the kernels `qwen35_gdn_mx8.axmodel`, `qwen35_attn_mx8.axmodel` and `mul_mat_mx8_n<C>_k<K>.axmodel`.

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
`qwen35_vit_{embed,pre,attn,post,merge}<N>_h<hidden>.axmodel` exist (`q35_vit.py`; N = 256, 1024, 2048, 3072
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
chunk's base, h2 and router logits rows; up to 8 chunks of 128 (`GGML_AXERA_MOE_GROUP`) run layer by layer and the
routed experts once per layer for the whole group: the host picks each token's experts, groups the picks by expert
into slots of up to 64 rows (gathering the h2 rows into non-cacheable CMM), and `qwen35moe_slots32_g64_gu_mxfp4` runs
32 slots per call (each slot one expert's weights, bound by address), whose rows the host adds to the tokens' base.
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

With `GGML_AXERA_CORES=4` the kernels that have a four-core build (`<name>_x4`, AXDSL without synchronization
between the cores: each core takes its share of heads, output channels, slots or KV blocks and writes only its own
outputs) replace the single-core ones: for the dense models prefill chunks of 128 / 64 tokens and decode
(`GGML_AXERA_MC_DECODE=0` keeps decode single-core) run as a chain of `qwen35_{gdn,mlp,attn_head,attn_tail}_mc<P>_x4`
kernels that pass the residual as a base plus one fp32 partial per core, the segment kernels split their blocks over
the cores and the tail merges the partial results on the NPU (more segments than its slots: merged on the CPU in
between), and the output projection uses `mul_mat_bf16_n<C>_k<K>_x4`; for the MoE models the segment kernels, the
slots kernel and the gate-up experts kernel (the cores' partial sums added on the host).

On boards with the October 16-19 images this is stable (0.8B `llama-perplexity` three times bit-identical, PPL zh
46.81 against 46.74 single-core from the order of the partial sums; on an older board it gave NaN and timeouts). The
gain is limited by the DRAM bandwidth those boards reach, about 84 GB/s for all cores together (one core about
53 GB/s; the CPU and the NPU together share the same 84 GB/s): compute-bound kernels scale (35B segment kernel 12.1 ->
3.3 ms, 119k-token prompt 663 -> 539 s), bandwidth-bound ones barely (0.8B decode 42 -> 40 ms). `GGML_AXERA_PROF=1`
prints per-kernel times at exit.

## Source layout

The backend separates what any model uses from the graph templates of one architecture:

| File | Contents |
|---|---|
| `ggml-axera-impl.h` | internal declarations, CMM buffers, kernel registry, host thread pool, profiling timers |
| `ggml-axera.cpp` | backend, device and registry interface; dispatch of a graph to the per-op path or a template |
| `ggml-axera-util.cpp` | runtime init, CMM allocation, settings (`GGML_AXERA_*`), kernel IO helpers |
| `ggml-axera-buffer.cpp` | the CMM buffer type, and the mapped buffers (the GGUF file mapping as device weights) |
| `ggml-axera-weights.cpp` | weight formats (BF16 copies, MXFP8, MXFP4 repacking) and the GEMV calls, per-op `MUL_MAT` |
| `ggml-axera-graph.cpp` | for templates: the graph's tensors by name, the output head (final norm, output projection, MTP's `h_nextn`) |
| `ggml-axera-attn.h/.cpp` | for templates: the llama.cpp KV cache and KQ mask of a batch, split attention (segments, merge), RoPE / M-RoPE tables |
| `ggml-axera-q35.h` | Qwen3.5 template: its state and per-graph run state |
| `ggml-axera-q35-bind.cpp` | Qwen3.5: kernel loading, binding GGUF tensors and cache tensors to kernel IO |
| `ggml-axera-q35-run.cpp` | Qwen3.5: one graph (recurrent-state bookkeeping, MTP input, chunks, layer calls) |
| `ggml-axera-q35-moe.cpp` | Qwen3.5 MoE: expert routing and the experts / slots kernels |
| `ggml-axera-q35-mc.cpp` | Qwen3.5: the multi-core prefill chain |
| `ggml-axera-vit.cpp` | the Qwen3.5 vision encoder (mtmd) |

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

The kernels are generated from the GGUF itself: every BF16 2-D weight shape gets one GEMV kernel (in chunks for
vocabulary-sized shapes), checked on the NPU simulator against PyTorch. The generator needs the Axera `axnn` toolchain and is kept with the AXDSL kernel sources,
outside this repository.
