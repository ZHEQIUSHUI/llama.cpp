#!/bin/sh
# Raw sequential reads of the 9B and 27B GGUF files. Run under lockrun.sh host1a.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "KIND,SCOPE,FILE,BYTES,BLOCK,THREADS,DIRECT,MEDIAN_MS,MIN_MS,MAX_MS,GBPS,SAMPLES"
for spec in "/data/qiushui/rel1109/9b/qwen3.5-9b-mxfp4.gguf 9b_gguf" "/data/qiushui/rel1109/27b/qwen3.8-27b-mxfp4.gguf 27b_gguf"; do
    set -- $spec
    f=$1
    tag=$2
    echo "SIZE,$tag,$(wc -c < "$f")"
    for bs in 1048576 4194304; do
        for direct in 0 1; do
            for nt in 1 4; do
                sync
                echo 3 > /proc/sys/vm/drop_caches
                "$BIN_DIR/disk_bw" "$f" "$bs" "$nt" "$direct" 1073741824 1 2>/dev/null || true
            done
        done
    done
done
