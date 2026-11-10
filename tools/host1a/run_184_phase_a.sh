#!/bin/sh
# One bounded .184 batch: snapshot copy costs and role-tagged decode timing for 9B/27B.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== snapshot 590 MiB =="
"$BIN_DIR/cmm_bw" snapshot 590 5 2>&1 | tee "$OUT_DIR/snapshot_590_184.csv"
echo "== snapshot 248 MiB =="
"$BIN_DIR/cmm_bw" snapshot 248 5 2>&1 | tee "$OUT_DIR/snapshot_248_184.csv"

echo "== 9B role-tagged decode =="
sh "$HERE/board_token_bench.sh" /data/qiushui/rel1109/9b qwen3.5-9b-mxfp4.gguf 9b_prof 2 128 || true
echo "== 27B role-tagged decode =="
sh "$HERE/board_token_bench.sh" /data/qiushui/rel1109/27b qwen3.8-27b-mxfp4.gguf 27b_prof 2 128 || true

echo "== done =="
cmm_line
