# 后端结构

## 总体

llama.cpp 照常构图，由 ggml 调度器把计算图交给 ggml-axera。后端不逐个算子执行，而是识别出整张图属于哪个模型的哪个阶段（prefill、decode、ViT、音频编码器、TTS 生成），再交给该模型的适配器，用整模型 kernel 一次算完，最后把结果（logits、embedding、图像 token 等）写回图的输出张量。

```
llama.cpp / mtmd 计算图
        │
ggml-axera.cpp          后端注册、buffer、graph_compute 入口
        │
models/model.cpp        按 manifest 选适配器；composite 按图的类型分派
        │
models/<model>/         适配器：状态管理、分页、调 kernel、写回输出
        │
ops/                    共用的算子：输出头（lm_head）、采样尾巴（PAD / TOP_K）等
core/                   manifest、kernel 运行时（AX_ENGINE）、CMM、GGUF 权重转换
```

## 目录

| 路径 | 内容 |
|---|---|
| `ggml-axera.cpp` | 后端设备、graph_compute、`GGML_AXERA_TRACE` |
| `ggml-axera-buffer.cpp` | 后端 buffer（CMM） |
| `core/manifest.*` | 读取 `manifest.json`，启动时加载所有 kernel，核对 IO 表 |
| `core/runtime.*` | AX_ENGINE 句柄、IO 绑定、CMM 分配、host 线程池、`GGML_AXERA_PROF` |
| `core/params.*` | 把 GGUF 张量转成 kernel 参数：MXFP8 / MXFP4 / BF16、pack / bank 布局、value head 重排 |
| `core/quant.*`、`core/mx.h` | MX 格式量化 |
| `ops/` | 多个模型共用的 kernel 调用：输出头、多行 lm_head、host 端采样尾巴 |
| `models/model.*` | 适配器接口、按 manifest 创建适配器、composite 分派 |
| `models/qwen38_27b` | Qwen3.8-27B |
| `models/qwen35_moe`、`qwen35_35b_a3b` | Qwen3.5-35B-A3B（发布版 / 整模型版） |
| `models/qwen35_dense` | Qwen3.5-9B（发布版路径） |
| `models/qwen35_whole` | Qwen3.5-9B / 4B / 2B / 0.8B 整模型（prefill bank + decode group） |
| `models/qwen3` | Qwen3 文本模型：Embedding、ASR 文本部分、TTS talker |
| `models/qwen3_asr` | Qwen3-ASR 音频编码器 |
| `models/qwen3_tts` | Qwen3-TTS 的 code predictor 和声码器（`codec.*`） |
| `models/qwen35_vit` | Qwen3.5 / 3.8 视觉编码器 |

## manifest：kernel 和后端之间的约定

kernel 目录里只有 `*.axmodel` 和 `manifest.json` 两类文件。manifest 由 ax-dsl-kernels 生成，是后端和 kernel 之间唯一的约定，主要字段：

| 字段 | 含义 |
|---|---|
| `format` | 格式版本，目前是 1 |
| `model` | 运行这个目录的适配器名 |
| `context` | kernel 支持的 KV 长度 |
| `params` | 每个层参数对应的 GGUF 张量（`blk.<i>.` 之后的部分）、转换方式（`mxfp8`、`mxfp4`、`tensor` 等）、kernel 里的形状，以及可选的 value head 重排（`vperm`） |
| `kernels` | 按角色列出 kernel：文件名、输入输出表、pack 及其成员（每层参数在 pack 里的偏移） |

启动时加载 manifest 里列出的所有 kernel，目录里其他文件一概不读。下面任何一种情况都会让进程直接报错退出：

- kernel 文件不存在。
- kernel 的 IO 表和 manifest 记录的不一致。
- 适配器需要的角色在 manifest 里没有。

这样，不配套的 kernel 和程序在启动时就会暴露，而不会在运行中算错。

## 适配器的分派

`models/model.cpp` 按 manifest 的 `model` 字段创建语言模型适配器。如果 kernel 目录里还有下面这些角色，会再挂上对应的适配器：

- `vit*`：`qwen35_vit`
- `audio.*`：`qwen3_asr`
- `cp.*`：`qwen3_tts`

每次 graph_compute 时，composite 根据图的内容决定交给谁：

| 判断依据 | 交给 |
|---|---|
| 图里读到 `v.blk.0.attn_qkv.weight` | 视觉编码器 |
| 图属于 Qwen3-ASR 音频编码器 | 音频编码器 |
| 有 TTS 生成适配器，且图里没有 `attn_norm-0` / `result_output` / `result_norm` | TTS 生成 |
| 其他 | 语言模型 |

## 权重

- kernel 不带权重。适配器在第一次构图时，从图里找到 GGUF 张量，按 manifest 的 `params` 转换后放进 CMM。
- 转换方式：
  - 稠密层转成 MXFP8。
  - MXFP4 GGUF（27B、35B 的路由专家）直接重排，不损失精度。
  - 视觉编码器、音频编码器、声码器转成 BF16。
- 转换后，pack 或 bank 里已经有一份的原始张量会被释放，节省 CMM。

## 新增一个模型

1. 在 ax-dsl-kernels 的 `models/<model>` 下写 kernel，生成 kernel 目录和 manifest。
2. 在本目录新建 `models/<model>/<model>.cpp`，实现 `model` 接口（`compute(ggml_cgraph *)`）。适配器只通过 manifest 的角色名取 kernel。
3. 在 `models/model.cpp` 的 `make_model_of` 里注册 manifest 里的模型名，并把源文件加进 `CMakeLists.txt`。
4. 验收：
   - 和 CPU 后端比较 PPL、补全输出。
   - 新 kernel 要在模拟器上做数值对比，再在板上单独运行，和参考结果对比。
   - 已有模型的输出 md5 和速度不能回退。

每个模型的适配器和 kernel 各自独立。新增或修改一个模型，不要改动其他模型的代码路径。
