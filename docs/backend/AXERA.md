# Axera NPU Backend for llama.cpp

The Axera backend ([`ggml/src/ggml-axera`](../../ggml/src/ggml-axera)) runs whole model graphs on the NPU of the AX8860
with precompiled four-core kernels (AXDSL, [ax-dsl-kernels](https://github.com/ZHEQIUSHUI/ax-dsl-kernels)). One kernel
directory holds one model's kernels and a `manifest.json`; the model adapter the manifest names runs every graph of the
process with exactly those kernels. Graph nodes an adapter leaves to the CPU (the token embedding, sampling, pooling,
parts of the TTS codec decoder) run on the CPU backend on the same memory.

| Model | Adapter (`models/`) | Graphs on the NPU | Context |
|---|---|---|---|
| Qwen3.5-35B-A3B (MoE, MXFP4 experts) | `qwen35_moe` + `qwen35_vit` | language model, 4-core ViT | 16k |
| Qwen3.8-27B (MXFP4) | `qwen35_dense` + `qwen35_vit` | language model, 4-core ViT | 16k |
| Qwen3.5-9B (MXFP4) | `qwen35_dense` | language model | 16k |
| Qwen3-Embedding-0.6B | `qwen3` | model up to the final norm (pooling on the CPU) | 16k |
| Qwen3-ASR-0.6B / 1.7B | `qwen3` + `qwen3_asr` | text model, audio encoder | 4k |
| Qwen3-TTS-0.6B / 1.7B | `qwen3` + `qwen3_tts` | talker, code predictor frames, codec decoder GEMMs / convolutions | 4k |

## Design

- **Manifest** (`core/manifest.h`): the only contract between the kernels and this backend. It names the model adapter,
  the context the attention kernels hold, the kernels by role (file, IO table, packed weights' layout) and, per
  parameter kind, which GGUF tensor each kernel parameter comes from and how it is converted (bf16, MXFP8, MXFP4,
  planar MXFP4 experts, ...). Every kernel listed is loaded at start; nothing else in the directory is read. A missing
  file, an IO table that differs from the manifest's, or a role an adapter asks for and the manifest lacks stops the
  process. Manifests are generated in ax-dsl-kernels (`tools/manifest/mkmanifest.py`) from the kernels' IO tables and
  layouts.
- **Adapters** (`models/<model>/`): one per model family, depending on `core/` and `ops/` only, never on each other. An
  adapter has one path: no fallbacks, no switches. A new model is a new adapter (or a manifest for an existing one);
  the others do not change.
- **core/**: the NPU runtime and CMM buffers (`runtime`), the manifest and its kernels (`manifest`), weight conversions
  (`quant`), the parameters and packs filled from the GGUF tensors (`params`).
- **ops/**: what adapters share around their kernels: the graph's tensors by name, the llama.cpp KV cache and KQ mask,
  RoPE / M-RoPE tables, the output head, CPU slices of a graph.
- **Memory**: CMM (physically contiguous, mapped into the process): the CPU backend reads and writes the same buffers,
  the NPU addresses them physically. The GGUF stays mapped (mmap); adapters convert what their kernels read into CMM.
  The KV cache and recurrent state live in llama.cpp's memory, in CMM.
- **One sequence** per context for the language models (`-np 1` with llama-server).

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
cmake --build build-axera -j --target llama-completion llama-server llama-mtmd-cli llama-tts llama-embedding
```

## Run

On the board, with the BSP runtime libraries on the library path and the model's kernel directory:

```bash
export LD_LIBRARY_PATH=/opt/lib:$PWD/build-axera/bin GGML_AXERA_KERNEL_DIR=$PWD/kernels
# language models (27B, 35B-A3B, 9B)
llama-completion -m qwen3.8-27b-mxfp4.gguf -ngl 99 -c 16384 -p "The capital of France is" -n 64
# images (27B, 35B-A3B): at most 3072 patches per image
llama-mtmd-cli -m qwen3.8-27b-mxfp4.gguf --mmproj mmproj-qwen3.8-27b-bf16.gguf -ngl 99 -c 16384 \
    --image bus.jpg --image-max-tokens 768 -p "Describe this image."
# speech recognition
llama-mtmd-cli -m qwen3-asr-1.7b-bf16.gguf --mmproj mmproj-qwen3-asr-1.7b-q8_0.gguf -ngl 99 -c 4096 \
    --audio speech.wav -p "Transcribe the audio."
# speech synthesis
llama-tts -m qwen3-tts-1.7b-bf16.gguf -mm mmproj-qwen3-tts-1.7b-bf16.gguf -ngl 99 -c 4096 -p "今天天气很好。" \
    --tts-lang zh --tts-speaker-file speaker.wav --output out.wav
# embeddings
llama-embedding -m qwen3-emb-0.6b-bf16.gguf -ngl 99 -c 16384 -p "What is the capital of China?" --embd-normalize 2
```

`-ngl 99`: every layer on the device (the adapters run whole graphs). The TTS code predictor samples with top-k on the
NPU (top-k 1 and 50 kernels); top-p < 1 is not supported.

## Settings

| Variable | Effect |
|---|---|
| `GGML_AXERA_KERNEL_DIR` | the kernel directory (default `./axera_kernels`); without a `manifest.json` the backend has no device |
| `GGML_AXERA_PROF=1` | time per kernel role and host step, printed at exit |
| `GGML_AXERA_TRACE=1` | every graph: node count, host gap before it, run time |

None of them changes a result.

## Source layout

| Path | Contents |
|---|---|
| `ggml-axera.cpp` | backend, device and registry; a graph to the directory's model adapter |
| `ggml-axera-buffer.cpp` | the CMM buffer type (KV cache tensors sized to the manifest's context), the mapped GGUF buffers |
| `core/` | runtime, CMM, profiling; manifest and kernels; weight conversions; parameters and packs |
| `ops/` | graph tensors by name, KV cache view and masks, RoPE, output head, CPU slices |
| `models/model.cpp` | the adapter by manifest model name; a composite per graph kind (vision, audio, TTS generation) |
| `models/qwen35_moe/` | Qwen3.5-35B-A3B: 8-layer decode groups, 256-row prefill groups with routed-expert slots |
| `models/qwen35_dense/` | Qwen3.5 dense (9B, 27B): decode groups (9B: whole model with the head), padded prefill chunks |
| `models/qwen3/` | Qwen3 text decoders: Embedding, the ASR text model, the TTS talker |
| `models/qwen35_vit/` | the Qwen3.5 vision encoder (4-core ViT) |
| `models/qwen3_asr/` | the Qwen3-ASR audio encoder |
| `models/qwen3_tts/` | the Qwen3-TTS code predictor (frame kernel), codec decoder and speaker encoder |

## Regression gate

Every change runs, per model, fixed cases whose outputs must stay bit-identical (md5) and whose prefill / decode speed
must stay within 2 %: 27B / 35B-A3B / 9B short, 4k prompt + 1k tokens, 16k prompt, an image (27B / 35B-A3B); Embedding
six inputs; ASR three recordings; TTS a Chinese and an English sentence. No kernel directory may contain single-core
kernels.
