#!/bin/sh
# select.sh <base ref> [tier]: the board smoke cases (tests/ci/cases.txt tags) a change since <base ref> needs.
#   a file of models/<dir>/      -> the cases that cover <dir>
#   core/, ops/, models/model.*, the backend files, CMakeLists.txt -> every case (shared code)
#   docs, tests, README          -> none (the host tests still run)
# tier pr (default): the pr cases only; full: every case (a nightly / release run passes full and no ref: "-")
set -e
d=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$d/../.." && pwd)  # ggml/src/ggml-axera
base=${1:?usage: select.sh <base ref | -> [pr|full]}
tier=${2:-pr}

if [ "$base" = "-" ]; then
    dirs="*"
else
    top=$(git -C "$root" rev-parse --show-toplevel)
    pre=${root#"$top"}; pre=${pre#/}; [ -n "$pre" ] && pre="$pre/"
    dirs=$(git -C "$root" diff --name-only "$base" -- . | sed -n "s#^$pre##p" | while read -r f; do
        case "$f" in
            models/model.*)               echo "*" ;;
            models/*/*)                   echo "$f" | cut -d/ -f2 ;;
            core/*|ops/*|ggml-axera*|CMakeLists.txt) echo "*" ;;
            docs/*|tests/*|*.md)          ;;
            *)                            echo "*" ;;
        esac
    done | sort -u | tr '\n' ' ')
fi
[ -z "$dirs" ] && exit 0

awk -F'|' -v dirs="$dirs" -v tier="$tier" '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    /^[ \t]*#/ || NF < 6 { next }
    { n++; tag[n] = trim($1); tr[n] = trim($2); cov[n] = " " trim($3) " "; gsub(/[ \t]+/, " ", cov[n])
      if (tr[n] == "pr") { m = split(trim($3), c, /[ \t]+/); for (i = 1; i <= m; i++) has_pr[c[i]] = 1 } }
    END {
        all = dirs ~ /\*/
        nd = split(dirs, dd, / +/)
        for (k = 1; k <= n; k++) {
            want = 0
            for (i = 1; i <= nd && !want; i++) {
                if (dd[i] == "" || dd[i] == "*") continue
                # a changed adapter without a pr case gets its full ones
                if (index(cov[k], " " dd[i] " ") && (tier == "full" || tr[k] == "pr" || !has_pr[dd[i]])) want = 1
            }
            if (all && (tier == "full" || tr[k] == "pr")) want = 1
            if (want) print tag[k]
        }
    }' "$d/cases.txt"
