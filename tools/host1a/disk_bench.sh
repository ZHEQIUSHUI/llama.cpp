#!/bin/sh
# Cold-cache disk read matrix.  Run the whole script under lockrun.sh: it drops caches before every
# sample.  Results are CSV lines beginning with RESULT.
set -eu
. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/board_common.sh"
host1a_lock_assert
F=${1:?file}
BLOCKS=${2:-"4096 65536 1048576 4194304 16777216"}
THREADS=${3:-"1 2 4 8"}
BYTES=${4:-1073741824}
REPS=${5:-2}
DIR=${6:-/data/qiushui/host1a/bin}

echo "KIND,SCOPE,FILE,BYTES,BLOCK,THREADS,DIRECT,MEDIAN_MS,MIN_MS,MAX_MS,GBPS,SAMPLES"
for bs in $BLOCKS; do
    for nt in $THREADS; do
        for direct in 0 1; do
            if [ "$direct" = 1 ] && [ $((bs % 4096)) -ne 0 ]; then
                continue
            fi
            for r in $(seq 1 "$REPS"); do
                sync
                echo 3 > /proc/sys/vm/drop_caches
                "$DIR/disk_bw" "$F" "$bs" "$nt" "$direct" "$BYTES" 1 2>/dev/null || true
            done
        done
    done
done
