#!/bin/sh
# 2B raw disk and resident CMM. Run under lockrun.sh host1a.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

DIR=/data/qiushui/fp8_2b
GGUF=qwen3.5-2b-q8_0.gguf
echo "== 2B disk =="
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
echo "== 2B resident CMM, 4K =="
sh "$HERE/board_cmm_hold.sh" "$DIR" "$GGUF" 2b_ctx4k 4096 18157 || true
echo "== 2B resident CMM, 256K =="
sh "$HERE/board_cmm_hold.sh" "$DIR" "$GGUF" 2b_ctx256k 262144 18158 || true
echo "== 2B prompt cache state size, 4K =="
CTX=4096 CTXCP=4 CACHE_DIR=/data/qiushui/host1a/cache/2b_pc \
    sh "$HERE/board_promptcache_bench.sh" "$DIR" "$GGUF" 2b_pc 18159 768 || true
