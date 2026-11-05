# 运行

下面的命令都在 AX8860 板子上执行。程序、GGUF 和 kernel 目录的准备见 [build.md](build.md)。如果用的是 HF 发布仓库，直接运行仓库里的 `run.sh` 就行，它封装了本页的命令。

## 环境变量

| 变量 | 说明 |
|---|---|
| `LD_LIBRARY_PATH` | 必须包含 `/opt/lib`（板子的 SDK 库）；动态构建还要包含程序所在目录 |
| `GGML_AXERA_KERNEL_DIR` | 必填，kernel 目录（`*.axmodel` + `manifest.json`） |
| `GGML_AXERA_PROF=1` | 进程退出时打印各阶段耗时（prefill、decode、lm_head、权重转换等） |
| `GGML_AXERA_GRAPH_FAST=1` | 按图指针缓存名字映射（decode 图复用时省掉每次约 0.3 ms 的 3847 节点扫描；实验开关） |
| `GGML_AXERA_TRACE=1` | 每个计算图打印节点数、两图之间的 host 间隔、NPU 运行时间 |
| `GGML_AXERA_CODEC_DUMP=<目录>` | 仅调试用：把 TTS 声码器转换后的权重写成文件 |

## 通用参数

- `-ngl 99`：所有层都交给后端。必须加，不加就在 CPU 上跑。
- `-c`：上下文长度，不能超过 kernel 的 `context`（manifest 里的值，见 [models.md](models.md)）。
- 服务端 `llama-server` 必须加 `-np 1`，适配器只支持单序列。
- 每个进程独占 NPU 的 4 个核，不要同时运行两个 NPU 程序。

## Qwen3.8-27B

```sh
M="-m qwen3.8-27b-mxfp4.gguf -ngl 99"
V="--mmproj mmproj-qwen3.8-27b-bf16.gguf"

# 文本补全
llama-completion $M -c 16384 -ub 4096 -b 4096 -p "The capital of France is" -n 64 --temp 0 -no-cnv
# 图像
llama-mtmd-cli $M $V --image bus.jpg --image-max-tokens 768 -p "Describe this image." -c 8192 -n 256 --temp 0
# OpenAI 兼容服务（文本 + 图像）
llama-server $M $V -c 16384 -np 1 --host 0.0.0.0 --port 8080
```

MTP 投机解码目前还在开发中。它需要一个含 MTP kernel 的 kernel 目录（尚未发布），服务端再加上：

```sh
--spec-type draft-mtp --spec-draft-n-max 3
```

## Qwen3.5-35B-A3B

```sh
llama-completion -m qwen3.5-35b-a3b-mxfp4_moe.gguf -ngl 99 -c 8192 -b 2048 -ub 2048 -f prompt.txt -n 1024 --temp 0 -no-cnv
```

路由专家必须是 MXFP4（`llama-quantize ... MXFP4_MOE`），其余权重用 Q8_0。用其他量化格式加载会报错退出。

## Qwen3-Embedding-0.6B

```sh
M="-m qwen3-emb-0.6b-bf16.gguf -ngl 99 -c 4096"

llama-embedding $M -ub 4096 -b 4096 -p "今天天气很好" --embd-normalize 2 --embd-output-format json
llama-server $M --embedding --pooling last -np 1 --host 0.0.0.0 --port 8080
curl http://<board>:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"今天天气很好"}'
```

做检索时，query 要按模型卡加上指令前缀：`Instruct: <task>\nQuery: <query>`。

## Qwen3-ASR

```sh
M="-m qwen3-asr-1.7b-bf16.gguf --mmproj mmproj-qwen3-asr-1.7b-bf16.gguf -ngl 99 -c 4096"

llama-mtmd-cli $M --audio speech.wav -p "Transcribe the audio." -n 1024 --temp 0
llama-server $M -np 1 --host 0.0.0.0 --port 8080
```

服务端请求发到 `/v1/chat/completions`，在 content 里放：

```json
{"type": "input_audio", "input_audio": {"format": "wav", "data": "<base64>"}}
```

## Qwen3-TTS

```sh
M="-m qwen3-tts-1.7b-bf16.gguf -mm mmproj-qwen3-tts-1.7b-bf16.gguf -ngl 99 -c 4096"

# 只给参考录音：用说话人向量（x-vector）克隆声音
llama-tts $M -p "今天天气很好。" --tts-speaker-file speaker.wav --output out.wav --tts-lang zh --temp 0.9 --top-k 50 -n 2000
# 参考录音加上它的文字：官方 ICL 方式，克隆出的声音更像
llama-tts $M -p "今天天气很好。" --tts-speaker-file speaker.wav --tts-ref-text "参考录音里说的话" \
  --output out.wav --tts-lang zh --temp 0.9 --top-k 50 -n 2000
```

- 语言：`zh en de it pt es ja ko fr ru`。
- `top-k` 最大 50，这是 kernel 的上限。
- 不建议用贪心解码（`--temp 0`），合成出的语音偏长、停顿多。
- 参考录音建议几秒到十几秒，过短（不到 1 s）克隆效果差。

## 性能分析

```sh
GGML_AXERA_PROF=1 llama-completion ...   # 退出时打印各阶段累计耗时 / 调用次数 / 平均耗时
GGML_AXERA_TRACE=1 llama-server ...      # 每个计算图一行：nodes、gap（host 间隔）、run（NPU 时间）
```

`init: ...` 开头的行是加载时的权重转换耗时，只在启动时发生一次。
