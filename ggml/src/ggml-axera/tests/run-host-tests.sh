#!/bin/sh
# the ggml-axera tests that need no board and no SDK (x86 or the board alike)
set -e
d=$(cd "$(dirname "$0")" && pwd)
o=${TMPDIR:-/tmp}/ggml-axera-host-tests.$$
mkdir -p "$o"
${CXX:-g++} -std=c++17 -O2 -Wall -Wextra -pthread "$d/../core/seqstate.cpp" "$d/test-seqstate.cpp" -o "$o/test-seqstate"
"$o/test-seqstate" "$@"
${CXX:-g++} -std=c++17 -O2 -Wall -Wextra -pthread "$d/../core/seqstate.cpp" "$d/../core/kv_state.cpp" "$d/test-kv-state.cpp" -o "$o/test-kv-state"
"$o/test-kv-state"
${CXX:-g++} -std=c++17 -O2 -Wall -Wextra -pthread "$d/../core/patch_embed.cpp" "$d/test-patch-embed.cpp" -o "$o/test-patch-embed"
"$o/test-patch-embed"
${CXX:-g++} -std=c++17 -O2 -Wall -Wextra "$d/../core/affinity.cpp" "$d/test-affinity.cpp" -o "$o/test-affinity"
"$o/test-affinity"
rm -rf "$o"
