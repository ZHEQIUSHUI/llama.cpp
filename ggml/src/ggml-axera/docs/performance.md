# 性能

## 测试条件

- AX8860，NPU 4 核。
- SDK V0.5.1；35B 用的是 V0.5.0。
- 使用 HF 发布仓库里的程序和 kernel。
- 数值是多次运行的范围或典型值。

NPU 上的累加顺序不固定，所以同一输入多次运行，输出偶尔有极小差异。这不影响输出质量。

## Qwen3.8-27B（MXFP4）

| 项 | 结果 |
|---|---|
| 4k prompt（3908 token）prefill | 3.18 s（1231 tokens/s） |
| 16k prompt prefill | 1168 tokens/s |
| decode，4k 上下文 | 72.8 ms/token（13.7 tokens/s） |
| decode，16k 上下文 | 81.0 ms/token（12.3 tokens/s） |
| 图像编码（3072 patch = 768 图像 token） | 0.28 s |
| 加载 | 约 70 s |
| 中文 PPL（内部测试集，`-c 512`） | 10.85 |

MTP 投机解码（开发中，`--spec-draft-n-max 3`）：

- 短 prompt 生成速度 18–19 tokens/s，长文 16.5 tokens/s。
- draft 接受率约 42–50%。同样的设置在 CPU 上跑参考实现，接受率也是这个水平，说明这是模型本身的特性，不是 NPU 精度造成的。

## Qwen3.5-35B-A3B（路由专家 MXFP4）

| 项 | 结果 |
|---|---|
| 4k prompt（3908 token）prefill | 1.63–1.66 s（2357–2397 tokens/s） |
| decode | 17.7–17.9 ms/token（约 56 tokens/s） |
| 文档问答（3920 token prompt） | prefill 1.78 s，回答正确 |

## Qwen3-Embedding-0.6B

服务模式下的单次请求耗时：

| 输入 | 耗时 |
|---|---|
| 1 条 query（9 token） | 22 ms |
| 1 条 229 token 文本 | 36 ms |
| 2 条文本，共 884 token | 69 ms |

## Qwen3-ASR

服务模式，端到端计时，即从发出请求到收到结果。

| 录音长度 | 0.6B 耗时 | 0.6B RTF | 1.7B 耗时 | 1.7B RTF |
|---|---|---|---|---|
| 0.6 s | 0.07 s | 0.12 | 0.10 s | 0.17 |
| 7.3 s | 0.38–0.47 s | 0.05–0.06 | 0.53–0.64 s | 0.07–0.09 |
| 60 s | 1.7 s | 0.028 | 2.5 s | 0.041 |

每次启动后的第一个请求会多出约 0.4–0.5 s，用于权重转换。

## Qwen3-TTS

测试内容：一段中文，生成 27.2 s 语音。

| 项 | 0.6B | 1.7B |
|---|---|---|
| 合成耗时（RTF） | 12.96 s（0.46） | 14.41 s（0.53） |
| 不计首次声码器权重转换 | RTF 0.35，每帧 27 ms | RTF 0.40，每帧 32 ms |
| NPU 每帧耗时：code predictor / talker decode | 7.3 ms / 3.7 ms | 7.8 ms / 8.1 ms |

一帧对应 80 ms 音频。每个进程还有以下一次性开销：

- 加载和权重转换：0.6B 约 14 s，1.7B 约 25 s。
- 第一次合成时声码器权重转换约 3 s，已经算在上表的合成耗时里。
- 参考录音在 CPU 上编码，耗时随录音长度增加，22 s 的录音约 10 s。

## 复现方法

```sh
# prefill / decode：llama-completion 退出时打印的 prompt eval / eval time
GGML_AXERA_PROF=1 llama-completion -m <gguf> -ngl 99 -c 8192 -b 2048 -ub 2048 -f prompt4k.txt -n 128 --temp 0 -no-cnv
# PPL
llama-perplexity -m <gguf> -ngl 99 -c 512 -f <测试文本>
```

`GGML_AXERA_PROF=1` 会按阶段打印耗时：prefill、decode、lm_head、host 端开销、权重转换。可以用来判断时间花在 NPU 上还是 host 上。
