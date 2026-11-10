#!/bin/sh
# Load a model in llama-server and report the maximum CMM in use while it is resident.
#   sh board_cmm_hold.sh DIR GGUF TAG [CTX] [PORT]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

DIR=${1:?model directory}
GGUF=${2:?gguf filename}
TAG=${3:?tag}
CTX=${4:-4096}
PORT=${5:-18155}
LOG="$OUT_DIR/cmm_${TAG}.server.log"

export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_AXERA_KERNEL_DIR="$DIR/kernels"

SPID=
cleanup() {
    if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
        kill -TERM "$SPID" 2>/dev/null || true
        wait "$SPID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$BIN_DIR/llama-server" -m "$DIR/$GGUF" -ngl 99 -c "$CTX" -b 4096 -ub 4096 -np 1 \
    --host 127.0.0.1 --port "$PORT" -lv 3 > "$LOG" 2>&1 &
SPID=$!

max_kb=0
i=0
ready=0
while [ "$i" -lt 900 ]; do
    line=$(cmm_line)
    kb=$(printf '%s' "$line" | sed -n 's/.*used=\([0-9][0-9]*\)KB.*/\1/p')
    [ -n "$kb" ] && [ "$kb" -gt "$max_kb" ] && max_kb=$kb
    if [ "$ready" -eq 0 ] && wget -q -O - "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
        ready=1
        sleep 2
        break
    fi
    kill -0 "$SPID" 2>/dev/null || break
    i=$((i + 1)); sleep 1
done
line=$(cmm_line)
echo "RESULT,cmm_hold,$TAG,ready=$ready,ctx=$CTX,max_used_kb=$max_kb,$line"
cleanup
SPID=
