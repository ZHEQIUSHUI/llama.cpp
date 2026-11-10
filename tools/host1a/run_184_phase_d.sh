#!/bin/sh
# One bounded .184 batch: cold/hot model-load stages and the prompt-cache file's disk read matrix.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert
export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== 9B load cold/hot =="
sh "$HERE/board_load_bench.sh" /data/qiushui/rel1109/9b qwen3.5-9b-mxfp4.gguf 9b_load 1 1 || true
echo "== 27B load cold/hot =="
sh "$HERE/board_load_bench.sh" /data/qiushui/rel1109/27b qwen3.8-27b-mxfp4.gguf 27b_load 1 1 || true

PC=/data/qiushui/host1a/cache/27b_pc2/prompt-cache/state-334073047512130-0.bin
if [ -f "$PC" ]; then
    echo "== prompt-cache file disk matrix =="
    sh "$HERE/disk_bench.sh" "$PC" "4096 1048576 4194304" "1 4" 1073741824 2
fi

echo "== done =="
cmm_line
