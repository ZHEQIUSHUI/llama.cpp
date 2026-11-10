#!/bin/sh
# Host-pool scaling for the weight-conversion phase.  Run under lockrun.sh host1a.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== 9B hot load, pool threads 1/4/8/15 =="
for t in 1 4 8 15; do
    sh "$HERE/board_load_bench.sh" /data/qiushui/rel1109/9b qwen3.5-9b-mxfp4.gguf 9b_pool 0 1 "$t" || true
done

echo "== 27B hot load, pool threads 1/4/8/15 =="
for t in 1 4 8 15; do
    sh "$HERE/board_load_bench.sh" /data/qiushui/rel1109/27b qwen3.8-27b-mxfp4.gguf 27b_pool 0 1 "$t" || true
done

echo "== done =="
cmm_line
