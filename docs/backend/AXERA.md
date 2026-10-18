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

Optional prefill kernels `qwen35_gdn_prefill<P>.axmodel` / `qwen35_attn_prefill<P>.axmodel` (P = 64, 32, 16, 8, 4)
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
  llama.cpp KV cache; called once per segment in use, with that segment's slice of the KQ mask.
- `qwen35_attn_fd_tail<R>`: the merged attention output (merged on the CPU) through the gate, `o_proj` and the MLP.

The kernels do not depend on the context, so any `-c` works (the KV cache is allocated in whole segments). On AX8860,
a 39666-token prompt (`-c 65536`, a number stated in its first line, then 30 copies of the GPL) was prefilled in
116 s (2.9 ms per token; the attention grows with the context) and the model answered with the number; decode at that
context took 106 ms per token. Below 32768 cells the fused kernels stay faster (the split costs two more kernel calls
per attention layer): forced split attention at `-c 4096` decodes at 43.4 ms instead of 40.7 ms per token, with PPL
2.4239 / 40.1532 instead of 2.4237 / 40.1693 (the final division happens on the CPU in f32).

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
runs through the batched kernels `qwen35_gdn_batch<B>.axmodel` / `qwen35_attn_batch<B>_kvc_c<LMAX>.axmodel` (B = 4, 2):
one call per layer for one token of each of B sequences, so every weight is read once per B tokens. The delta-net
kernel updates B consecutive rows of the recurrent cache in place, each sequence with its own conv and delta-rule
state; the attention kernel writes each token's K/V at its own cell and attends through its own KQ mask row. The
output projection runs B rows per call through `mul_mat_bf16_b<B>_n<C>_k<K>.axmodel` (also for the rows of
`llama-perplexity`). Per-sequence KV streams (an explicit `-np` without `-kvu`) keep the token-by-token path.

`llama-batched-bench -npp 128 -ntg 64 -kvu`, decode tokens per second:

| Sequences | BF16 | MXFP8 |
|---|---|---|
| 1 | 24.4 | 31.7 |
| 2 | 46.6 | 45.5 |
| 4 | 84.3 | 67.5 |

The batched kernels are bf16; with `GGML_AXERA_WEIGHTS=mxfp8` they use the bf16 GGUF tensors and the output projection
stays the row-by-row MXFP8 GEMV (MXFP8 batched kernels do not exist yet). PPL is unchanged (2.4237 / 40.1693). The
delta-net batch kernel matches the decode kernel bit for bit on the simulator; with `-kvu` the cells of the sequences
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
reads each weight once for up to 64 tokens, so it is compute bound and MXFP8 only adds dequantization (1708-token
prompt: 0.75 ms per token with MXFP8 prefill kernels, 0.58 ms with BF16). With `GGML_AXERA_WEIGHTS=mxfp8` the prefill
kernels therefore use the bf16 GGUF tensors (already in CMM) and share the recurrent state and KV cache with the MXFP8
decode kernels: prompts at 0.58 ms per token, decode at 31.5 ms per token. `GGML_AXERA_PREFILL_WEIGHTS=mxfp8` selects
the MXFP8 prefill kernels instead.

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
