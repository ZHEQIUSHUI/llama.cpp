#!/bin/sh
# the ggml-axera tests that need no board and no SDK (x86 or the board alike)
set -e
d=$(cd "$(dirname "$0")" && pwd)
o=${TMPDIR:-/tmp}/ggml-axera-host-tests.$$
mkdir -p "$o"
${CXX:-g++} -std=c++17 -O2 -Wall -Wextra "$d/../core/seqstate.cpp" "$d/test-seqstate.cpp" -o "$o/test-seqstate"
"$o/test-seqstate" "$@"
rm -rf "$o"
