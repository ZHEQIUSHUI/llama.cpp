#!/bin/sh
# One bounded .209 batch: CMM bandwidth, 2B host timings, and 35B decode timings.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== CMM bandwidth =="
"$BIN_DIR/cmm_bw" 2>&1 | tee "$OUT_DIR/cmm_bw_209.csv"
echo "== snapshot 248 MiB =="
"$BIN_DIR/cmm_bw" snapshot 248 5 2>&1 | tee "$OUT_DIR/snapshot_248_209.csv"

echo "== 2B token timing =="
sh "$HERE/board_token_bench.sh" /data/qiushui/fp8_2b qwen3.5-2b-q8_0.gguf 2b_prof 2 128 || true
echo "== 2B load cold/hot =="
sh "$HERE/board_load_bench.sh" /data/qiushui/fp8_2b qwen3.5-2b-q8_0.gguf 2b_load 1 1 || true

echo "== 35B token timing =="
sh "$HERE/board_token_bench.sh" /data/qiushui/rel1109/pkg/35b qwen3.5-35b-a3b-nvfp4.gguf 35b_prof 2 128 || true

echo "== done =="
cmm_line
