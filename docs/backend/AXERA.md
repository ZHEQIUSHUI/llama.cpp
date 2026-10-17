# Axera NPU Backend for llama.cpp

> [!NOTE]
> Early stage. Two modes:
> - **Per op** (default): the matrix products of BF16 GGUF models run on the NPU, everything else on the CPU.
> - **Graph template** (`GGML_AXERA_TEMPLATE=qwen35`): whole Qwen3.5 graphs run as one NPU kernel per layer.
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

Kernels in `GGML_AXERA_KERNEL_DIR`: `qwen35_gdn.axmodel`, `qwen35_attn.axmodel`, `qwen35.txt` (`LMAX BC ROT THETA`),
plus the `mul_mat` kernel for the output projection.

Optional prefill kernels `qwen35_gdn_prefill<P>.axmodel` / `qwen35_attn_prefill<P>.axmodel` (P = 64, 32, 16, 8, 4)
process P consecutive prompt tokens per call. The projections and the MLP read every weight once per chunk. For
P >= 16 the delta rule runs in the chunked form (the triangular solve of `gdn_stable_chunk`, 32 tokens per solve,
8 heads batched), so a chunk is mostly matrix products; for P = 8 / 4 it runs token by token. The backend splits a
batch greedily into the largest chunks that fit and runs the rest through the decode kernels. On AX8860 (BF16) a
1708-token prompt takes 1.0 s (0.58 ms per token, 1712 tokens/s; token by token through the decode kernels it was
26-30 ms per token).

Limits of the template:
- One sequence. Positions continue, restart at 0, or go back to a context checkpoint; the recurrent state lives in
  the llama.cpp recurrent cache, while the KV cache lives in the backend.
- The context is limited to the kernels' `LMAX` (4096).
- Prompt tokens go through the prefill kernels in chunks of 64 / 32 / 16 / 8 / 4, if present, and through the decode
  kernels otherwise.

```bash
GGML_AXERA_TEMPLATE=qwen35 ./build-axera/bin/llama-completion -m qwen3.5-0.8b-bf16.gguf -ngl 99 -c 4096 -p "..." -n 128
```

### llama-server

```bash
GGML_AXERA_TEMPLATE=qwen35 ./build-axera/bin/llama-server -m qwen3.5-0.8b-bf16.gguf -ngl 99 -c 4096 \
    -np 1 --cache-ram 0 --host 0.0.0.0 --port 8080
```

The delta-net layers keep their recurrent state in place in the llama.cpp recurrent cache tensors (`cache_s_l*`,
`cache_r_l*`, in CMM). So the server's context checkpoints save and restore exactly the kernels' state, and a
request that shares a prefix with the previous one resumes from the last checkpoint. The attention layers then mask
the dropped positions. A conversation's later turns only process the new messages: on AX8860, a third turn took 33
prompt tokens and 1.4 s instead of 162 tokens and 4.5 s, with the same answer.

Still needed:
- `-np 1`: one sequence.
- `--cache-ram 0`: the RAM prompt cache restores KV cache rows, whose layout the attention kernel does not share.

A request the template cannot continue from fails with an HTTP 500 error and a log line naming the flags; the server
keeps running, and a request starting at position 0 works again. With BF16 weights, the chat endpoint streams at
42.5 ms per token and takes 25-28 ms per prompt token.

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
