#!/bin/sh
# Shared helpers for the host1a board runs.  Source this only from a command launched by
# /data/qiushui/lockrun.sh host1a ...; the lock check below makes an accidental direct run fail.

BIN_DIR=${BIN_DIR:-/data/qiushui/host1a/bin}
OUT_DIR=${OUT_DIR:-/data/qiushui/host1a/logs}
LOCK=/data/qiushui/.npu_lock

mkdir -p "$OUT_DIR"

host1a_lock_assert() {
    owner=$(cat "$LOCK/owner" 2>/dev/null || true)
    case "$owner" in
        host1a\ *) ;;
        *) echo "host1a: refusing to run outside lockrun.sh host1a (owner='$owner')" >&2; exit 2 ;;
    esac
}

cmm_line() {
    grep 'total size=' /proc/ax_proc/mem_cmm_info 2>/dev/null | tail -1
}

prof_field() { # file label
    awk -v label="$2" 'index($0, "axera prof: " label) == 1 {
        for (i = 1; i <= NF; ++i) if ($i == "ms") { print $(i-1); exit }
    }' "$1"
}

host_prof_field() { # file label
    awk -v label="$2" 'index($0, "host prof: " label) == 1 {
        for (i = 1; i <= NF; ++i) if ($i == "ms") { print $(i-1); exit }
    }' "$1"
}

ms_field() { # file label
    awk -v label="$2" 'index($0, label) {
        for (i = 1; i <= NF; ++i) if ($i == "ms") { print $(i-1); exit }
    }' "$1" | tail -1
}

json_num() { # file key
    sed -n "s/.*\"$2\":\([-0-9][0-9.eE+-]*\).*/\1/p" "$1" | head -1
}
