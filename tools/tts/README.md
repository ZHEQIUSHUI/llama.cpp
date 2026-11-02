# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

The model type is read from the talker GGUF metadata. Available prompt options
depend on the variant:

- Base: `--tts-speaker-file` points to a speaker reference audio file (wav,
  mp3). `--tts-ref-text` is its transcript; when given, it enables full ICL
  voice cloning (reference text and codec codes are placed in the talker
  prompt).
- CustomVoice: `--tts-speaker <name>` selects one of the built-in speakers
  (`aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`,
  `vivian`). `--tts-instruct "<style>"` is optional, except that the 0.6B
  CustomVoice model ignores it, as in the official API.
- VoiceDesign: `--tts-instruct "<voice description>"` is required.

Common params:
- `--tts-lang` can be `auto`, `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`,
  `fr`, `ru` (default: `auto` for CustomVoice/VoiceDesign, `en` for Base).
  `auto` uses the official `nothink` prompt; `dylan` and `eric` select the
  Beijing and Sichuan dialect language ids for Chinese/auto.
- `--temp 0 --top-k 1` requests greedy decoding for both the talker and the
  code predictor.

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

For ICL cloning, pass the reference transcript as well:

```sh
llama-tts -m qwen3-tts.gguf -mm mmproj-qwen3-tts.gguf \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.wav \
    --tts-ref-text "This is the reference transcript." \
    --output out.wav
```

CustomVoice example:

```sh
llama-tts -m qwen3-tts-customvoice.gguf -mm mmproj-qwen3-tts-customvoice.gguf \
    -p "Hello world" \
    --tts-speaker vivian \
    --tts-instruct "Speak in a warm, cheerful tone." \
    --output out.wav
```

VoiceDesign example:

```sh
llama-tts -m qwen3-tts-voicedesign.gguf -mm mmproj-qwen3-tts-voicedesign.gguf \
    -p "Hello world" \
    --tts-instruct "A deep, warm male narrator voice, calm and measured." \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```
