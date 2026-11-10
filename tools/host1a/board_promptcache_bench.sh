#!/bin/sh
# Disk prompt-cache write/read timing for one package.  Run under lockrun.sh host1a.
#   sh board_promptcache_bench.sh DIR GGUF TAG [PORT] [PREFIX_WORDS]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

DIR=${1:?model directory}
GGUF=${2:?gguf filename}
TAG=${3:?tag}
PORT=${4:-18153}
PREFIX_WORDS=${5:-768}
CTX=${CTX:-4096}
CTXCP=${CTXCP:-4}
CACHE_DIR=${CACHE_DIR:-/data/qiushui/host1a/cache/$TAG}
TMP=${TMPDIR:-/tmp}/host1a_pc_$$
mkdir -p "$CACHE_DIR" "$TMP"

export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_AXERA_KERNEL_DIR="$DIR/kernels"
export LLAMA_AXERA_HOST_PROF=1

SPID=
cleanup() {
    if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
        kill -TERM "$SPID" 2>/dev/null || true
        wait "$SPID" 2>/dev/null || true
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

make_req() { # file prefix_words n_predict
    awk -v p="$2" -v n="$3" 'BEGIN {
        printf "{\"prompt\":\"";
        for (i = 0; i < p; ++i) printf (i ? " a" : "a");
        printf "\",\"n_predict\":%d,\"temperature\":0,\"ignore_eos\":true,\"cache_prompt\":true,\"id_slot\":0,\"stream\":false}", n;
    }' > "$1"
}

start_server() { # log
    "$BIN_DIR/llama-server" -m "$DIR/$GGUF" -ngl 99 -c "$CTX" -b 4096 -ub 4096 -np 1 \
        -ctxcp "$CTXCP" --cache-ram 0 --slot-save-path "$CACHE_DIR/" --cache-disk 32768 \
        --host 127.0.0.1 --port "$PORT" -lv 4 > "$1" 2>&1 &
    SPID=$!
    i=0
    while [ "$i" -lt 600 ]; do
        if wget -q -O - "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then return 0; fi
        kill -0 "$SPID" 2>/dev/null || return 1
        i=$((i + 1)); sleep 1
    done
    return 1
}

stop_server() {
    kill -TERM "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=
}

make_req "$TMP/req.json" "$PREFIX_WORDS" 1

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"
start_server "$OUT_DIR/pc_${TAG}_write.server.log" || { echo SERVER_WRITE_FAIL; exit 3; }
wget -q -O "$TMP/write.json" --header='Content-Type: application/json' --post-file="$TMP/req.json" \
    "http://127.0.0.1:$PORT/completion" || true
echo "RESULT,pc_write,$TAG,prompt_n=$(json_num "$TMP/write.json" prompt_n),cache_n=$(json_num "$TMP/write.json" cache_n),prompt_ms=$(json_num "$TMP/write.json" prompt_ms),predicted_ms=$(json_num "$TMP/write.json" predicted_ms)"

i=0
last=0
stable=0
while [ "$i" -lt 300 ]; do
    f=$(ls -1 "$CACHE_DIR"/prompt-cache/state-*.bin 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        cur=$(stat -c %s "$f" 2>/dev/null || echo 0)
        if [ "$cur" -gt 0 ] && [ "$cur" = "$last" ]; then
            stable=$((stable + 1))
            [ "$stable" -ge 5 ] && break
        else
            stable=0
        fi
        last=$cur
    fi
    i=$((i + 1)); sleep 1
done
echo "RESULT,pc_file,$TAG,path=$f,bytes=$last"
stop_server

sync
echo 3 > /proc/sys/vm/drop_caches
start_server "$OUT_DIR/pc_${TAG}_read.server.log" || { echo SERVER_READ_FAIL; exit 3; }
sync
echo 3 > /proc/sys/vm/drop_caches
wget -q -O "$TMP/read.json" --header='Content-Type: application/json' --post-file="$TMP/req.json" \
    "http://127.0.0.1:$PORT/completion" || true
echo "RESULT,pc_read,$TAG,prompt_n=$(json_num "$TMP/read.json" prompt_n),cache_n=$(json_num "$TMP/read.json" cache_n),prompt_ms=$(json_num "$TMP/read.json" prompt_ms),predicted_ms=$(json_num "$TMP/read.json" predicted_ms)"
stop_server

grep -aE '^host prof: cache: disk|reading prompt cache|read prompt cache|cache: disk' "$OUT_DIR/pc_${TAG}_read.server.log" | tail -30
if [ -n "$f" ] && [ "$last" -gt 0 ]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
    "$BIN_DIR/disk_bw" "$f" 4194304 1 1 "$last" 1
    sync
    echo 3 > /proc/sys/vm/drop_caches
    "$BIN_DIR/disk_bw" "$f" 4194304 1 0 "$last" 1
fi
