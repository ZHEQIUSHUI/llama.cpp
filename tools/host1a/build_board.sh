#!/bin/sh
# Cross-compile the host1a profiling build on the 4090.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
TOOL=${TOOL:-/data/junda/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu}
BSP=${BSP:-/data/junda/work/ax8860_bsp/arm64_glibc}
BUILD=${BUILD:-$ROOT/build-host1a}

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="$TOOL/bin/aarch64-none-linux-gnu-gcc" \
    -DCMAKE_CXX_COMPILER="$TOOL/bin/aarch64-none-linux-gnu-g++" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_AXERA=ON -DGGML_AXERA_BSP_DIR="$BSP"

cmake --build "$BUILD" -j"${JOBS:-16}" --target llama-completion llama-server llama-mtmd-cli

"$TOOL/bin/aarch64-none-linux-gnu-g++" -O2 -std=c++17 -Wall -Wextra \
    -I"$BSP/include" \
    "$ROOT/tools/host1a/cmm_bw.cpp" -o "$BUILD/bin/cmm_bw" \
    -L"$BSP/lib" \
    -lax_sys -lax_dma -Wl,--allow-shlib-undefined

"$TOOL/bin/aarch64-none-linux-gnu-g++" -O2 -std=c++17 -Wall -Wextra \
    "$ROOT/tools/host1a/disk_bw.cpp" -o "$BUILD/bin/disk_bw" -pthread

"$TOOL/bin/aarch64-none-linux-gnu-g++" -O2 -std=c++17 -Wall -Wextra \
    -I"$BSP/include" \
    "$ROOT/tools/host1a/disk_cmm.cpp" -o "$BUILD/bin/disk_cmm" \
    -L"$BSP/lib" -lax_sys -Wl,--allow-shlib-undefined

echo "built: $BUILD/bin/llama-completion $BUILD/bin/llama-server $BUILD/bin/cmm_bw $BUILD/bin/disk_bw $BUILD/bin/disk_cmm"
