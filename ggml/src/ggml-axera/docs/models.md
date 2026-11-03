# 模型

每个 kernel 目录由一个适配器（`models/<model>`）运行，适配器由 `manifest.json` 里的 `model` 字段决定。kernel 目录里如果还有 ViT、音频编码器或 TTS 生成相关的 kernel，会分别交给 `qwen35_vit`、`qwen3_asr`、`qwen3_tts` 处理，具体结构见 [design.md](design.md)。

所有模型都按下面的约定运行：

- 权重只从 GGUF 读取，kernel 里不带权重。加载时转换成 kernel 要求的格式。
- 只支持单序列，服务端需要 `-np 1`。
- 运行时占用全部 4 个 NPU 核。

## Qwen3.8-27B（`qwen38_27b`）

| 项 | 内容 |
|---|---|
| GGUF | `qwen3.8-27b-mxfp4.gguf`（MXFP4），`mmproj-qwen3.8-27b-bf16.gguf`（视觉，BF16） |
| kernel | `prefill`：整模型一次处理一页 256 token，16 个 block 在 kernel 内循环<br>`decode.group`：每次 16 层，每个 token 调 4 次<br>`lm_head`：输出头<br>`vit`：整个 ViT（3072 patch） |
| 上下文 | 16384 |
| 状态 | KV cache 分页存放（256 token 一页，BF16），GDN 递归状态和 conv 状态由适配器在 CMM 里自己分配 |
| 多轮对话 | prefill 只能从页边界开始。适配器在每页开头保存递归状态，并保留本页已处理的输入；新一轮从页中间开始时，恢复该页开头的状态，把本页旧输入和新输入拼起来重算这一页 |
| MTP（开发中） | 用模型自带的单层 MTP 头连续出 3 个 draft，再一次验证 4 行（kernel 角色 `decode.verify`、`mtp.t1`、`mtp.prompt`，`lm_head` 一次可算多行）。验证后保留每一行的状态快照，用于回退 |
| 加载 | 约 70 s，大部分时间是 GGUF 转 NPU 格式 |

## Qwen3.5-35B-A3B（`qwen35_moe`）

| 项 | 内容 |
|---|---|
| GGUF | `qwen3.5-35b-a3b-mxfp4_moe.gguf`：路由专家 MXFP4，其余 Q8_0（`llama-quantize ... MXFP4_MOE`） |
| kernel | `decode.group`：每次若干个完整 MoE 层（含 router、top-K 和专家）<br>`prefill256.*` 等：分块 prefill，每层先跑层 kernel，再跑路由专家<br>`moe.slots*` / `moe.combine_t*`：把同一专家的 token 收进槽位一起计算，再加回各 token<br>`lm_head`：输出头 |
| 上下文 | 8192（发布版 kernel） |
| 状态 | KV cache 和递归状态放在 llama.cpp 的 memory 里（CMM） |

整模型 kernel 版（`qwen35_35b_a3b`，每层 kernel 内含注意力 / GDN、路由和共享专家，专家按层集中计算）还在开发中，manifest 里有对应角色时才会启用。

## Qwen3.5-9B（`qwen35_dense`）

| 项 | 内容 |
|---|---|
| kernel | `decode.group`（整模型一组，带输出头），`prefill128.*` / 64 / 16 / 4 分块 prefill |
| 状态 | 和 35B 一样放在 llama.cpp 的 memory 里 |

暂未发布。

## Qwen3.5-9B / 2B 整模型（`qwen35_whole`）

| 项 | 内容 |
|---|---|
| GGUF | 9B：`qwen3.5-9b-Q4_K_M.gguf`（MXFP4）+ `mmproj-qwen3.5-9b-bf16.gguf`；2B：`Qwen3.5-2B-Q4_K_M.gguf`（MXFP4）+ `mmproj-Qwen3.5-2B-BF16.gguf` |
| kernel | `prefill`：整个模型一个 kernel，运行时按 block 循环读取 `b0..b3` 权重 bank<br>`decode.group`：每次 16 层（9B）/ 12 层（2B），每 token 2 次<br>`lm_head`：输出头<br>`vit`：整个 ViT（最多 3072 patch） |
| 上下文 | 16384 |
| 权重 | decode group 的投影权重通过 `k_<bank>` 直接读 prefill 的 bank；CMM 里只有一份主干权重 |
| 状态 | KV cache 分页存放（256 token 一页），GDN 递归 / conv 状态由适配器管理 |
| 限制 | 单序列（`-np 1`），整模型加载（`-ngl 99`） |

适配器目录是 `models/qwen35_whole`，由 manifest 的 `"model": "qwen35_whole"` 选择。9B 和 2B 共用适配器，层数和
decode group 大小从 manifest / kernel 的层数得到。

## Qwen3-Embedding-0.6B（`qwen3`）

| 项 | 内容 |
|---|---|
| GGUF | `qwen3-emb-0.6b-bf16.gguf`（BF16，加载时转成 MXFP8） |
| kernel | `prefill`：整个模型一个 kernel，28 层在 kernel 内循环 |
| 上下文 | 4096 |
| 输出 | last-token pooling（`--pooling last`），建议归一化（`--embd-normalize 2`） |

## Qwen3-ASR-0.6B / 1.7B（`qwen3` + `qwen3_asr`）

| 项 | 内容 |
|---|---|
| GGUF | `qwen3-asr-*-bf16.gguf`（语言模型，转成 MXFP8），`mmproj-qwen3-asr-*-bf16.gguf`（音频编码器，转成 BF16） |
| kernel | `audio.enc`：音频编码器（Whisper 结构，运行时 key mask）<br>`prefill`：文本 prefill<br>`decode`：文本 decode（含输出头） |
| 上下文 | 4096 |
| 输入 | wav / mp3 |

## Qwen3-TTS-12Hz-0.6B / 1.7B-Base（`qwen3` + `qwen3_tts`）

| 项 | 内容 |
|---|---|
| GGUF | `qwen3-tts-*-bf16.gguf`（talker），`mmproj-qwen3-tts-*-bf16.gguf`（code predictor、声码器、说话人编码器、语音编码器） |
| kernel | `prefill` / `decode`：talker；`decode` 里包含 codec 输出头<br>`cp.frame`：code predictor，每次出一帧，top-k 在运行时指定（最大 50）<br>`gen.codec`：声码器，每次处理 72 帧 |
| 上下文 | 4096；一次合成最多 2000 帧（约 160 s） |
| 克隆 | 只给录音：说话人向量（x-vector）<br>录音加文字（`--tts-ref-text`）：官方 ICL 方式，录音先由语音编码器转成 codec token，再放进 prompt |
| 在 CPU 上的部分 | 说话人编码器、语音编码器（参考录音只编码一次）、采样 |

CustomVoice / VoiceDesign 两个变体的结构和 Base 相同，kernel 可以复用，正在做 GGUF 转换和 llama-tts 适配。

## 视觉编码器（`qwen35_vit`）

27B 和 35B-A3B 共用这个适配器。整个 ViT 是一个 kernel（角色 `vit`），图像最多 3072 个 patch，合并后是 768 个图像 token（`--image-max-tokens 768`）。旧版的分段 ViT kernel（`vit.embed` / `vit.attn` / `vit.merge` 等）仍然支持。
