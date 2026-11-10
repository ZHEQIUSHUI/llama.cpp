#!/bin/sh
# Resident CMM remaining after loading 9B at 4K and 256K. Run under lockrun.sh host1a.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

echo "== lock =="
cat /data/qiushui/.npu_lock/owner
cmm_line

echo "== 9B resident CMM, 4K =="
sh "$HERE/board_cmm_hold.sh" /data/qiushui/rel1109/9b qwen3.5-9b-mxfp4.gguf 9b_ctx4k 4096 18159 || true
echo "== 9B resident CMM, 256K =="
sh "$HERE/board_cmm_hold.sh" /data/qiushui/rel1109/9b qwen3.5-9b-mxfp4.gguf 9b_ctx256k 262144 18160 || true

echo "== done =="
cmm_line
