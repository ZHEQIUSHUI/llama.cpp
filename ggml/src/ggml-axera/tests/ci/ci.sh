#!/bin/sh
# ci.sh <base ref | -> <bin dir> <out dir> [pr|full]: the ggml-axera gate of a change. The host tests (no board), then
# the board smoke cases the change needs (select.sh) with the programs of <bin dir> (board.sh: see its environment).
set -e
d=$(cd "$(dirname "$0")" && pwd)
base=${1:?usage: ci.sh <base ref | -> <bin dir> <out dir> [pr|full]}
bin=${2:?}; out=${3:?}; tier=${4:-pr}
sh "$d/../run-host-tests.sh"
tags=$(sh "$d/select.sh" "$base" "$tier" | tr '\n' ' ')
if [ -z "$tags" ]; then
    echo "ci: no board case covers this change"
    exit 0
fi
echo "ci: board cases: $tags"
sh "$d/board.sh" "$bin" "$out" $tags
