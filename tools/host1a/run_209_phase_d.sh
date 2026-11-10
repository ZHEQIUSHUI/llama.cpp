#!/bin/sh
# Follow-up .209 batch: 35B resident CMM at the full 256K context. Run under lockrun.sh host1a.
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

echo "== 35B resident CMM, 256K =="
sh "$HERE/board_cmm_hold.sh" "$DIR" "$GGUF" 35b_ctx256k 262144 18160 || true

echo "== done =="
cmm_line
