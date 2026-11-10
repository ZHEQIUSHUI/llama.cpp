#!/bin/sh
# Release-package 2B timing on .184 (the .209 board only has the older fp8_2b package).
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

DIR=/data/qiushui/rel1109/2b
GGUF=qwen3.5-2b-q8_0.gguf

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== 2B release token timing =="
sh "$HERE/board_token_bench.sh" "$DIR" "$GGUF" 2b_rel 2 128 || true
echo "== 2B release load cold/hot =="
sh "$HERE/board_load_bench.sh" "$DIR" "$GGUF" 2b_rel_load 1 1 || true
echo "== 2B release resident CMM, 4K =="
sh "$HERE/board_cmm_hold.sh" "$DIR" "$GGUF" 2b_rel_ctx4k 4096 18161 || true
echo "== 2B release resident CMM, 256K =="
sh "$HERE/board_cmm_hold.sh" "$DIR" "$GGUF" 2b_rel_ctx256k 262144 18162 || true
echo "== 2B release raw disk read, first 1 GiB =="
echo "KIND,SCOPE,FILE,BYTES,BLOCK,THREADS,DIRECT,MEDIAN_MS,MIN_MS,MAX_MS,GBPS,SAMPLES"
for bs in 1048576 4194304; do
    for direct in 0 1; do
        for nt in 1 4; do
            sync
            echo 3 > /proc/sys/vm/drop_caches
            "$BIN_DIR/disk_bw" "$DIR/$GGUF" "$bs" "$nt" "$direct" 1073741824 1 2>/dev/null || true
        done
    done
done

echo "== done =="
cmm_line
