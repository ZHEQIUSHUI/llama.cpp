#!/bin/sh
# One bounded .209 batch: 35B load, resident CMM, and raw-disk read speed.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

DIR=/data/qiushui/rel1109/pkg/35b
GGUF=qwen3.5-35b-a3b-nvfp4.gguf

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== 35B load cold/hot =="
sh "$HERE/board_load_bench.sh" "$DIR" "$GGUF" 35b_load 1 1 || true
echo "== 35B resident CMM, 4K =="
sh "$HERE/board_cmm_hold.sh" "$DIR" "$GGUF" 35b_ctx4k 4096 18158 || true

echo "== raw disk read, first 1 GiB of the 35B GGUF =="
echo "KIND,SCOPE,FILE,BYTES,BLOCK,THREADS,DIRECT,MEDIAN_MS,MIN_MS,MAX_MS,GBPS,SAMPLES"
for bs in 4096 1048576 4194304; do
    for direct in 0 1; do
        [ "$direct" = 1 ] && [ "$bs" = 4096 ] && continue
        for nt in 1 4; do
            sync
            echo 3 > /proc/sys/vm/drop_caches
            "$BIN_DIR/disk_bw" "$DIR/$GGUF" "$bs" "$nt" "$direct" 1073741824 1 2>/dev/null || true
        done
    done
done

echo "== done =="
cmm_line
