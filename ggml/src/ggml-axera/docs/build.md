# 编译

在 x86_64 Linux 主机上交叉编译，产物拷到 AX8860 板子上运行。

## 准备

| 项 | 说明 |
|---|---|
| 交叉工具链 | Arm GNU Toolchain 13.2.Rel1（`aarch64-none-linux-gnu`）。板子 glibc 较新，旧版 gcc 编出的程序可能缺符号，建议就用 13.2 |
| AX8860 BSP | 目录下需要 `include/ax_engine_api.h`，以及 `lib/` 里的 `libax_engine`、`libax_interpreter`、`libax_sys`。下文记作 `<bsp>`，即 BSP 的 `arm64_glibc` 目录 |
| CMake | 3.14 或更新 |

BSP 的库还依赖板子 `/opt/lib` 下的其他系统库。这些依赖在链接时不解析，到运行时才加载（`-Wl,--allow-shlib-undefined`），所以主机上只需要上面三个库。

## 动态链接构建（开发用）

```sh
T=<toolchain>/bin
cmake -S . -B build-axera \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$T/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$T/aarch64-none-linux-gnu-g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_AXERA=ON -DGGML_AXERA_BSP_DIR=<bsp>
cmake --build build-axera -j 16 --target \
  llama-completion llama-server llama-mtmd-cli llama-embedding llama-tts llama-perplexity llama-bench
```

产物在 `build-axera/bin/`，包括程序本身和 `libggml*.so*`、`libllama.so*`、`libmtmd.so*`。部署时把它们放进同一个目录，并把这个目录加进 `LD_LIBRARY_PATH`。

## 静态链接构建（发布用）

在上面的配置基础上再加两个选项：

```sh
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF
```

编出来的程序只依赖板子的 `/opt/lib` 和 libc，可以单独拷走。发布前用 `aarch64-none-linux-gnu-strip` 去掉符号。HF 发布仓库里的 `bin/` 就是这样构建的。

## CMake 选项

| 选项 | 说明 |
|---|---|
| `GGML_AXERA=ON` | 编译 ggml-axera 后端 |
| `GGML_AXERA_BSP_DIR` | 必填，BSP 目录（含 `include/`、`lib/`） |
| `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` | CPU 后端指令集。仍在 CPU 上算的部分（采样、少量算子）会用到 |
| `GGML_OPENMP=OFF` | 不用 OpenMP |
| `LLAMA_CURL=OFF`、`LLAMA_OPENSSL=OFF` | 板上不需要下载模型 |

## 部署到板子

板子需要 SDK V0.5.0 或更新，可以用 `cat /proc/ax_proc/version` 查看。V0.4.0 在 CMM 用量较大时会静默算错，不要用。

需要拷到板子上的东西：

1. 程序：动态构建是 `bin/` 目录加上各个 `.so`；静态构建只要程序本身。
2. GGUF：模型权重，多模态模型还要 mmproj。
3. kernel 目录：`*.axmodel` 加 `manifest.json`。可以直接用 HF 发布仓库的 `kernels/`，也可以用 ax-dsl-kernels 生成。

然后在板上运行：

```sh
export LD_LIBRARY_PATH=/opt/lib:<程序目录>
export GGML_AXERA_KERNEL_DIR=<kernel 目录>
<程序目录>/llama-completion -m <gguf> -ngl 99 -p "The capital of France is" -n 32
```

日志里出现 `axera: <kernel 目录>: model <适配器>, context <N>, <K> kernels loaded in <t> s`，说明 NPU 路径已经生效。各模型的完整命令见 [run.md](run.md)。

## 常见问题

| 现象 | 原因 |
|---|---|
| CMake 报 `GGML_AXERA requires GGML_AXERA_BSP_DIR` | 没有指定 BSP 目录 |
| 运行时找不到 `libax_*.so` | `LD_LIBRARY_PATH` 里没有 `/opt/lib` |
| `axera: .../manifest.json: no adapter for model ...` | kernel 目录和程序版本不匹配：manifest 里的模型名没有对应的适配器 |
| `axera: kernel ...: its IO is not the manifest's` | kernel 文件和 manifest 不是同一次生成的，要成套使用 |
| `failed to allocate ... bytes of CMM` |  CMM 不够：板子 CMM 分区太小，或者别的进程占着 |
| 多个 NPU 程序同时运行时出错或卡住 | 每个进程都独占 NPU 的 4 个核，同一时间只能跑一个 |
