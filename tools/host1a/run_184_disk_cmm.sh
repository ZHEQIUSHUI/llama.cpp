#!/bin/sh
# Cold-cache prompt-cache reads into host and CMM. Run under lockrun.sh host1a.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

PC=$(find /data/qiushui/host1a/cache/27b_pc2 -name 'state-*.bin' -type f 2>/dev/null | head -1)
if [ -z "$PC" ]; then
    echo "no prompt-cache file"
    exit 3
fi
BYTES=$(wc -c < "$PC")
echo "PC=$PC BYTES=$BYTES"
echo "KIND,MODE,BYTES,BLOCK,DIRECT,READ_MS,COPY_MS,FLUSH_MS,TOTAL_MS,GBPS,SAMPLES"
for mode in host cmm cmm_buffered host_to_cmm host_to_cmm_chunk; do
    for run in 1 2; do
        sync
        echo 3 > /proc/sys/vm/drop_caches
        "$BIN_DIR/disk_cmm" "$PC" "$mode" 4194304 "$BYTES" 1
    done
done
