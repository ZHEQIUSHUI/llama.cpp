#!/bin/sh
# Cold-cache read matrix for the 27B prompt-cache state file. Run under lockrun.sh host1a.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
F=$(find /data/qiushui/host1a/cache/27b_pc2/prompt-cache -name 'state-*.bin' -type f 2>/dev/null | head -1)
[ -n "$F" ] || { echo "no prompt-cache file"; exit 3; }
BYTES=$(wc -c < "$F")
echo "FILE=$F BYTES=$BYTES"
sh "$HERE/disk_bench.sh" "$F" "4096 1048576 4194304" "1 4" "$BYTES" 2
