# ggml-axera：爱芯 AX8860 NPU 后端

ggml-axera 是 llama.cpp 的一个 ggml 后端，在爱芯 AX8860 的 NPU（4 核）上运行模型。

- 每个模型对应一组整模型 kernel（`*.axmodel`）和一个 `manifest.json`，一般是 prefill、decode、ViT 或编码器各一个。权重不在 kernel 里，加载时从 GGUF 转换到 NPU 格式（MXFP8 / MXFP4 / BF16）。
- 计算图由各模型的适配器（`models/<model>`）接管，整图交给 NPU；采样、tokenizer、服务端等仍然用 llama.cpp 原有的。
- 可用的程序：`llama-completion`、`llama-server`、`llama-mtmd-cli`、`llama-embedding`、`llama-tts`、`llama-perplexity`、`llama-bench`。

## 模型支持

| 模型 | 适配器 | 功能 | HF 发布仓库 |
|---|---|---|---|
| Qwen3.8-27B | `qwen38_27b`（ViT：`qwen35_vit`） | 文本、图像输入；MTP 投机解码（开发中） | [AXERA-TECH/Qwen3.8-27B-AX8860-llama.cpp](https://huggingface.co/AXERA-TECH/Qwen3.8-27B-AX8860-llama.cpp) |
| Qwen3.5-35B-A3B | `qwen35_moe`（整模型版 `qwen35_35b_a3b` 开发中） | 文本 | 待发布 |
| Qwen3.5-9B | `qwen35_dense` / `qwen35_whole` | 文本、图像输入（`qwen35_whole` 整模型路径） | 待发布 |
| Qwen3.5-2B | `qwen35_whole` | 文本、图像输入 | 待发布 |
| Qwen3-Embedding-0.6B | `qwen3` | 文本向量 | [AXERA-TECH/Qwen3-Embedding-0.6B-AX8860-llama.cpp](https://huggingface.co/AXERA-TECH/Qwen3-Embedding-0.6B-AX8860-llama.cpp) |
| Qwen3-ASR-0.6B / 1.7B | `qwen3` + `qwen3_asr` | 语音识别 | [0.6B](https://huggingface.co/AXERA-TECH/Qwen3-ASR-0.6B-AX8860-llama.cpp) / [1.7B](https://huggingface.co/AXERA-TECH/Qwen3-ASR-1.7B-AX8860-llama.cpp) |
| Qwen3-TTS-12Hz-0.6B / 1.7B-Base | `qwen3` + `qwen3_tts` | 语音合成、声音克隆（x-vector / ICL） | [0.6B](https://huggingface.co/AXERA-TECH/Qwen3-TTS-0.6B-AX8860-llama.cpp) / [1.7B](https://huggingface.co/AXERA-TECH/Qwen3-TTS-1.7B-AX8860-llama.cpp) |

HF 仓库里有 GGUF、静态链接的板端程序、kernel 目录和 `run.sh`，下载到板子上就能运行。

## 简要性能

测试条件：AX8860，SDK V0.5.x，NPU 4 核。

| 模型 | 结果 |
|---|---|
| Qwen3.8-27B | 4k prompt prefill 1231 tokens/s；decode 13.7 tokens/s（4k 上下文）；图像编码 0.28 s（768 图像 token） |
| Qwen3.5-35B-A3B | 4k prompt prefill 约 2400 tokens/s；decode 约 56 tokens/s |
| Qwen3-Embedding-0.6B | 单条 query 22 ms；229 token 36 ms |
| Qwen3-ASR-0.6B / 1.7B | 60 s 录音 1.7 s / 2.5 s（RTF 0.028 / 0.041） |
| Qwen3-TTS-0.6B / 1.7B | RTF 0.46 / 0.53（不计首次声码器权重转换：0.35 / 0.40） |

测试方法和完整数据见 [docs/performance.md](docs/performance.md)。

## 快速开始

```sh
# 交叉编译（x86 主机，aarch64 工具链 + AX8860 BSP）
cmake -S . -B build-axera -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-none-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-none-linux-gnu-g++ \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_AXERA=ON -DGGML_AXERA_BSP_DIR=<bsp>/arm64_glibc
cmake --build build-axera -j 16 --target llama-completion llama-server

# 板上运行
export LD_LIBRARY_PATH=/opt/lib GGML_AXERA_KERNEL_DIR=<kernel 目录>
./llama-completion -m model.gguf -ngl 99 -p "The capital of France is" -n 64
```

完整步骤见 [docs/build.md](docs/build.md)，各模型的运行方式见 [docs/run.md](docs/run.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/build.md](docs/build.md) | 编译：工具链、BSP、CMake 选项、动态/静态构建、部署到板子 |
| [docs/run.md](docs/run.md) | 运行：环境变量、各模型命令行、服务端、性能分析开关 |
| [docs/models.md](docs/models.md) | 各模型的适配器、kernel 角色、GGUF 要求、限制 |
| [docs/performance.md](docs/performance.md) | 性能数据与测试方法 |
| [docs/design.md](docs/design.md) | 后端结构：manifest 约定、core / ops / models 分层、如何加新模型 |

## 源码与 kernel

- 后端源码：本目录（`ggml/src/ggml-axera`）。
- kernel 源码与 manifest 生成：[ZHEQIUSHUI/ax-dsl-kernels](https://github.com/ZHEQIUSHUI/ax-dsl-kernels)（`models/<model>`）。发布仓库的 `kernels/` 目录可以直接用，不需要自己编 kernel。
