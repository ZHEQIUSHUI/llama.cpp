#!/bin/sh
# Cold/hot model-load stages for one release package.  Run as:
#   /data/qiushui/lockrun.sh host1a sh board_load_bench.sh DIR GGUF TAG [COLD_RUNS] [HOT_RUNS] [POOL_THREADS]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

DIR=${1:?model directory}
GGUF=${2:?gguf filename}
TAG=${3:?tag}
COLD_RUNS=${4:-2}
HOT_RUNS=${5:-2}
POOL_THREADS=${6:-}

export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_AXERA_KERNEL_DIR="$DIR/kernels"
export LLAMA_AXERA_HOST_PROF=1
if [ -n "$POOL_THREADS" ]; then
    export AXERA_HOST_POOL_THREADS=$POOL_THREADS
fi

run_one() {
    cache=$1
    r=$2
    log="$OUT_DIR/load_${TAG}_${cache}_${r}${POOL_THREADS:+_t$POOL_THREADS}.log"
    if [ "$cache" = cold ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
    fi
    cmm_before=$(cmm_line)
    start=$(date +%s)
    "$BIN_DIR/llama-completion" -m "$DIR/$GGUF" -ngl 99 -c 4096 -b 4096 -ub 4096 \
        -p Hello -n 1 --temp 0 -no-cnv > "$log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))
    cmm_after=$(cmm_line)
    load_ms=$(ms_field "$log" 'load time')
    total_ms=$(host_prof_field "$log" 'load: total')
    loader_ms=$(host_prof_field "$log" 'load: loader/mmap')
    vocab_ms=$(host_prof_field "$log" 'load: vocab')
    tensors_ms=$(host_prof_field "$log" 'load: tensors total')
    tensor_data_ms=$(host_prof_field "$log" 'load: tensor data')
    ctx_ms=$(host_prof_field "$log" 'load: context init')
    banks_ms=$(prof_field "$log" 'init: prefill banks')
    fill_alloc_ms=$(prof_field "$log" 'fill: alloc')
    fill_layers_ms=$(prof_field "$log" 'fill: layers')
    fill_flush_ms=$(prof_field "$log" 'fill: flush')
    quant4_ms=$(prof_field "$log" 'load: quantize mxfp4')
    quant8_ms=$(prof_field "$log" 'load: quantize mxfp8')
    echo "RESULT,load,$TAG,$cache,$r,$rc,$elapsed,$load_ms,$total_ms,$loader_ms,$vocab_ms,$tensors_ms,$tensor_data_ms,$ctx_ms,$banks_ms,$fill_alloc_ms,$fill_layers_ms,$fill_flush_ms,$quant4_ms,$quant8_ms,$POOL_THREADS"
    echo "CMM_BEFORE,$cmm_before" >> "$OUT_DIR/load_${TAG}_${cache}_${r}.cmm"
    echo "CMM_AFTER,$cmm_after" >> "$OUT_DIR/load_${TAG}_${cache}_${r}.cmm"
}

echo "HEADER,kind,model,cache,run,rc,elapsed_s,load_ms,host_total_ms,loader_mmap_ms,vocab_ms,tensors_ms,tensor_data_ms,context_init_ms,prefill_banks_ms,fill_alloc_ms,fill_layers_ms,fill_flush_ms,quant_mxfp4_ms,quant_mxfp8_ms,pool_threads"
for r in $(seq 1 "$COLD_RUNS"); do run_one cold "$r"; done
for r in $(seq 1 "$HOT_RUNS"); do run_one hot "$r"; done
