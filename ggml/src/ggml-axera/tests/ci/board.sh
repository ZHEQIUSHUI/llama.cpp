#!/bin/sh
# board.sh <bin dir> <out dir> [tag...]: run board smoke cases (tests/ci/cases.txt) with the programs of <bin dir> on
# the board and gate them: a det=1 case's output md5 must equal its golden (tests/ci/golden.txt), any case must exit 0
# with a non-empty output. No tags: every case. AXERA_CI_RECORD=1 writes the det=1 md5s into golden.txt instead.
#
# Each case runs its package's own run.sh in a shadow copy whose bin/ is <bin dir> (the package's kernels, GGUF and
# scripts as published).
#   AXERA_PKG_ROOT  directory holding the packages by repo name (required)
#   AXERA_CI_DATA   directory holding the inputs the cases name (bus.jpg, smoke_en.wav, smoke_zh.wav)
#   AXERA_CI_WRAP   optional command prefix for every run (e.g. a one-NPU-job-at-a-time lock)
#   AXERA_CI_HEALTH optional command run before and after each case; non-zero fails the case (e.g. an NPU state check)
set -u
d=$(cd "$(dirname "$0")" && pwd)
bin=$(cd "${1:?usage: board.sh <bin dir> <out dir> [tag...]}" && pwd)
out=${2:?usage: board.sh <bin dir> <out dir> [tag...]}
shift 2
: "${AXERA_PKG_ROOT:?set AXERA_PKG_ROOT}"
export AXERA_CI_DATA=${AXERA_CI_DATA:-}
mkdir -p "$out"
out=$(cd "$out" && pwd)
golden=$d/golden.txt
[ -f "$golden" ] || : > "$golden"
fail=0
pass=0

trim() { echo "$1" | sed 's/^[ \t]*//; s/[ \t]*$//'; }
health() {
    [ -z "${AXERA_CI_HEALTH:-}" ] && return 0
    sh -c "$AXERA_CI_HEALTH" > "$out/health.$1" 2>&1
}

while IFS='|' read -r tag tier cov pkg args det; do
    tag=$(trim "$tag")
    case "$tag" in ''|\#*) continue ;; esac
    if [ $# -gt 0 ]; then
        hit=0
        for t in "$@"; do [ "$t" = "$tag" ] && hit=1; done
        [ $hit = 1 ] || continue
    fi
    pkg=$(trim "$pkg"); det=$(trim "$det"); args=$(trim "$args")
    P=$AXERA_PKG_ROOT/$pkg
    if [ ! -f "$P/run.sh" ]; then
        echo "== $tag SKIP (no package $P)"
        continue
    fi
    S=$out/shadow/$tag
    rm -rf "$S"; mkdir -p "$S"
    for e in "$P"/* "$P"/.[!.]*; do
        [ -e "$e" ] || continue
        b=$(basename "$e")
        case "$b" in bin|run.sh) ;; *) ln -s "$e" "$S/$b" ;; esac
    done
    cp "$P/run.sh" "$S/run.sh"
    ln -s "$bin" "$S/bin"
    OUT=$out/$tag.wav
    rm -f "$OUT"
    health "$tag.pre" || { echo "== $tag FAIL (health check before the run)"; fail=$((fail + 1)); continue; }
    t0=$(date +%s)
    ( cd "$out" && eval "OUT=\"$OUT\" ${AXERA_CI_WRAP:-} sh \"$S/run.sh\" $args" ) < /dev/null > "$out/$tag.out" 2> "$out/$tag.err"
    rc=$?
    t=$(( $(date +%s) - t0 ))
    health "$tag.post" || { echo "== $tag FAIL (health check after the run)"; fail=$((fail + 1)); continue; }
    md=$(grep -a -v -E 'npu-normal|^[[:space:]]*$' "$out/$tag.out" | md5sum | cut -c1-12)
    nonempty=$( { grep -a -v -E 'npu-normal|^[[:space:]]*$' "$out/$tag.out"; [ -s "$OUT" ] && echo wav; } | head -1)
    want=$(awk -v t="$tag" '$1 == t { print $2 }' "$golden")
    verdict=PASS
    if [ $rc != 0 ] || [ -z "$nonempty" ]; then
        verdict="FAIL (rc $rc, output $( [ -n "$nonempty" ] && echo present || echo empty))"
    elif [ "$det" = 1 ] && [ "${AXERA_CI_RECORD:-0}" = 1 ]; then
        awk -v t="$tag" '$1 != t' "$golden" > "$golden.tmp"; echo "$tag $md" >> "$golden.tmp"
        sort "$golden.tmp" > "$golden"; rm -f "$golden.tmp"
        verdict="RECORDED"
    elif [ "$det" = 1 ] && [ -z "$want" ]; then
        verdict="FAIL (no golden md5: run once with AXERA_CI_RECORD=1)"
    elif [ "$det" = 1 ] && [ "$md" != "$want" ]; then
        verdict="FAIL (md5 $md, golden $want)"
    fi
    case "$verdict" in FAIL*) fail=$((fail + 1)) ;; *) pass=$((pass + 1)) ;; esac
    ev=$(grep -a 'eval time' "$out/$tag.err" | grep -v prompt | grep -oE '[0-9.]+ ms per token' | head -1)
    echo "== $tag $verdict ${t}s md5=$md ${ev:+[$ev]} | $(grep -a -v npu-normal "$out/$tag.out" | tr '\n' ' ' | cut -c1-100)"
    [ "${verdict#FAIL}" != "$verdict" ] && grep -a -iE 'error|abort|fail' "$out/$tag.err" | head -3
done < "$d/cases.txt"

echo "board smoke: $pass passed, $fail failed"
[ $fail = 0 ]
