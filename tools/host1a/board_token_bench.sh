#!/bin/sh
# Greedy and recommended-sampling decode timing for one package.  Run under lockrun.sh host1a.
#   sh board_token_bench.sh DIR GGUF TAG [RUNS] [N_PREDICT] [POOL_THREADS]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/board_common.sh"
host1a_lock_assert

DIR=${1:?model directory}
GGUF=${2:?gguf filename}
TAG=${3:?tag}
RUNS=${4:-3}
NPRED=${5:-128}
POOL_THREADS=${6:-}

export LD_LIBRARY_PATH=/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_AXERA_KERNEL_DIR="$DIR/kernels"
export LLAMA_AXERA_HOST_PROF=1
if [ -n "$POOL_THREADS" ]; then
    export AXERA_HOST_POOL_THREADS=$POOL_THREADS
fi

run_one() {
    mode=$1
    r=$2
    case "$mode" in
        greedy) S="--temp 0" ;;
        rec)    S="--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0" ;;
    esac
    log="$OUT_DIR/token_${TAG}_${mode}_${r}${POOL_THREADS:+_t$POOL_THREADS}.log"
    "$BIN_DIR/llama-completion" -m "$DIR/$GGUF" -ngl 99 -c 4096 -b 4096 -ub 4096 \
        -p Hello -n "$NPRED" --seed 1234 -no-cnv $S > "$log" 2>&1
    rc=$?
    eval_ms=$(sed -n 's/.*eval time = *\([0-9][0-9.]*\) ms \/ *\([0-9][0-9]*\) runs.*/\1/p' "$log" | tail -1)
    eval_n=$(sed -n 's/.*eval time = *[0-9][0-9.]* ms \/ *\([0-9][0-9]*\) runs.*/\1/p' "$log" | tail -1)
    tps=$(sed -n 's/.*, *\([0-9][0-9.]*\) tokens per second.*/\1/p' "$log" | tail -1)
    graph_ms=$(prof_field "$log" 'graph (decode)')
    layers_ms=$(prof_field "$log" 'layers (decode)')
    sync_ms=$(prof_field "$log" 'kernel: RunSync')
    head_ms=$(prof_field "$log" 'lm_head (combined)')
    topk_ms=$(prof_field "$log" 'lm_head.topk')
    sample_ms=$(host_prof_field "$log" 'sampling: common_sampler_sample (after sync)')
    echo "RESULT,token,$TAG,$mode,$r,$rc,$eval_ms,$eval_n,$tps,$graph_ms,$layers_ms,$sync_ms,$head_ms,$topk_ms,$sample_ms,$POOL_THREADS"
}

echo "HEADER,kind,model,mode,run,rc,eval_ms,eval_n,tok_s,graph_decode_ms,layers_ms,runsync_ms,lm_head_combined_ms,lm_head_topk_ms,sampling_ms,pool_threads"
for mode in greedy rec; do
    for r in $(seq 1 "$RUNS"); do run_one "$mode" "$r"; done
done
