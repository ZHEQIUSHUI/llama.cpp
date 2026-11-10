#!/bin/sh
# One bounded .184 batch: snapshot copy cost and prefix-hit tail recompute for 27B, 9B control.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== snapshot 590 MiB =="
"$BIN_DIR/cmm_bw" snapshot 590 5 2>&1 | tee "$OUT_DIR/snapshot_590_184.csv"
echo "== snapshot 248 MiB =="
"$BIN_DIR/cmm_bw" snapshot 248 5 2>&1 | tee "$OUT_DIR/snapshot_248_184.csv"

echo "== 27B tail =="
CTX=4096 CTXCP=4 sh "$HERE/board_tail_bench.sh" /data/qiushui/rel1109/27b qwen3.8-27b-mxfp4.gguf 27b_tail 18152 768 || true
echo "== 9B tail control =="
CTX=4096 CTXCP=4 sh "$HERE/board_tail_bench.sh" /data/qiushui/rel1109/9b qwen3.5-9b-mxfp4.gguf 9b_tail 18153 768 || true

echo "== done =="
cmm_line
