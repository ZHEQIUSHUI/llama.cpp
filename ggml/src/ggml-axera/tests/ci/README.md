# ggml-axera CI

A change is gated by the host tests and by the board smoke cases of the adapters it touches, so a typical change runs
a handful of small models instead of every published package.

```sh
# a pull request: host tests + the pr cases the change since origin/axera-dev needs
sh tests/ci/ci.sh origin/axera-dev <bin dir> <out dir>
# nightly / release: every case
sh tests/ci/ci.sh - <bin dir> <out dir> full
```

`<bin dir>` holds the programs to test (the static `llama-*` build). The board run needs:

| variable | |
|---|---|
| `AXERA_PKG_ROOT` | directory holding the published packages by repo name (`Qwen3.5-0.8B-AX8860-llama.cpp`, ...) |
| `AXERA_CI_DATA` | the case inputs: `bus.jpg`, `smoke_en.wav`, `smoke_zh.wav` |
| `AXERA_CI_WRAP` | optional prefix for every run, e.g. a one-NPU-job-at-a-time lock |
| `AXERA_CI_HEALTH` | optional command before and after each case (e.g. an NPU state check); non-zero fails it |

## Files

| file | |
|---|---|
| `cases.txt` | one or two cases per adapter (`models/<dir>`), the smallest package that runs it; tier `pr` / `full`; `det` |
| `select.sh` | the cases a change needs: a file of `models/<dir>/` selects the cases covering `<dir>` (its `full` cases when it has no `pr` one); `core/`, `ops/`, `models/model.*`, the backend files and `CMakeLists.txt` select every case; docs and tests none |
| `board.sh` | runs cases in a shadow copy of each package whose `bin/` is the build under test |
| `golden.txt` | the output md5 of every `det=1` case |

A `det=1` case must reproduce its golden md5. A `det=0` case only has to exit 0 with an output: the board's NVFP4 MoE
kernels and the TTS samplers are not run-to-run exact, so the same program gives different text on a rerun.

After an intended output change (a new kernel set in a package, a fix that changes numerics), refresh the goldens with
the reviewed build: `AXERA_CI_RECORD=1 sh tests/ci/board.sh <bin dir> <out dir>`.

## Adding a model

Add its adapter's case(s) to `cases.txt`: a `pr` case when a small package runs the adapter, else `full` (it then
still runs on every change of that adapter). Record its golden.
