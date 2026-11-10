#!/bin/sh
# Prefix-hit / tail-recompute timing on a local llama-server.  Run under lockrun.sh host1a.
#   sh board_tail_bench.sh DIR GGUF TAG [PORT] [PREFIX_WORDS]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

DIR=${1:?model directory}
GGUF=${2:?gguf filename}
TAG=${3:?tag}
PORT=${4:-18152}
PREFIX_WORDS=${5:-768}
CTX=${CTX:-4096}
CTXCP=${CTXCP:-4}
CACHE_RAM=${CACHE_RAM:-4096}
LOG="$OUT_DIR/tail_${TAG}.server.log"
TMP=${TMPDIR:-/tmp}/host1a_tail_$$
mkdir -p "$TMP"

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

make_req() { # file prefix_words suffix_words n_predict cache
    awk -v p="$2" -v s="$3" -v n="$4" -v c="$5" 'BEGIN {
        printf "{\"prompt\":\"";
        for (i = 0; i < p; ++i) printf (i ? " a" : "a");
        for (i = 0; i < s; ++i) printf " b";
        printf "\",\"n_predict\":%d,\"temperature\":0,\"ignore_eos\":true,\"cache_prompt\":%s,\"id_slot\":0,\"stream\":false}", n, c;
    }' > "$1"
}

post() { # name file
    name=$1
    req=$2
    out="$TMP/$name.json"
    wget -q -O "$out" --header='Content-Type: application/json' --post-file="$req" \
        "http://127.0.0.1:$PORT/completion" || { echo "RESULT,tail,$TAG,$name,HTTP_FAIL"; return; }
    pn=$(json_num "$out" prompt_n)
    cn=$(json_num "$out" cache_n)
    pm=$(json_num "$out" prompt_ms)
    dm=$(json_num "$out" predicted_ms)
    dn=$(json_num "$out" predicted_n)
    echo "RESULT,tail,$TAG,$name,$pn,$cn,$pm,$dm,$dn"
}

make_req "$TMP/prefix.json" "$PREFIX_WORDS" 0 1 true
make_req "$TMP/s1.json"     "$PREFIX_WORDS" 1 1 true
make_req "$TMP/s4.json"     "$PREFIX_WORDS" 4 1 true
make_req "$TMP/s8.json"     "$PREFIX_WORDS" 8 1 true
make_req "$TMP/s16.json"    "$PREFIX_WORDS" 16 1 true
make_req "$TMP/s23.json"    "$PREFIX_WORDS" 23 1 true
make_req "$TMP/reset.json"  "$PREFIX_WORDS" 0 1 false

"$BIN_DIR/llama-server" -m "$DIR/$GGUF" -ngl 99 -c "$CTX" -b 4096 -ub 4096 -np 1 \
    -ctxcp "$CTXCP" --cache-ram "$CACHE_RAM" --host 127.0.0.1 --port "$PORT" -lv 4 \
    > "$LOG" 2>&1 &
SPID=$!

i=0
while [ "$i" -lt 600 ]; do
    if wget -q -O - "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then break; fi
    kill -0 "$SPID" 2>/dev/null || { echo "SERVER_FAIL"; exit 3; }
    i=$((i + 1)); sleep 1
done
[ "$i" -lt 600 ] || { echo "SERVER_TIMEOUT"; exit 3; }

echo "HEADER,kind,model,case,prompt_n,cache_n,prompt_ms,predicted_ms,predicted_n"
post cold_prefix "$TMP/prefix.json"
if [ "${TAIL_SKIP_SAME:-0}" != 1 ]; then
    post same_prefix "$TMP/prefix.json"
fi
post same_s1     "$TMP/s1.json"
post same_s4     "$TMP/s4.json"
post same_s8     "$TMP/s8.json"
post same_s16    "$TMP/s16.json"
post same_s23    "$TMP/s23.json"
post reset       "$TMP/reset.json"
post restore_s1  "$TMP/s1.json"
post restore_s4  "$TMP/s4.json"
post restore_s8  "$TMP/s8.json"
post restore_s16 "$TMP/s16.json"
post restore_s23 "$TMP/s23.json"

kill -TERM "$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
SPID=

grep -aE 'restored context checkpoint|saving prompt with length|created context checkpoint' "$LOG" | tail -80
