set -x

GPUS=(0)
BATCHES=(32 64)
GRAD_ACUMS=(4 8)
LR=5e-6
MODEL="Qwen/Qwen2.5-1.5B"

run_job(){
    GPU=$1
    BS=$2
    GA=$3
    CUDA_VISIBLE_DEVICES=$GPU uv run train_sft_with_compute_acc_batched.py \
    --model_name $MODEL \
    --batch_size "$BS" \
    --lr "$LR" \
    --grad_accum "$GA" \
    --epochs 3
}

idx=0
for GPU in "${GPUS[@]}"; do
    BS=${BATCHES[$idx]}
    GA=${GRAD_ACUMS[$idx]}
    run_job "$GPU" "$BS" "$GA" &
    idx=$((idx+1))
done 

while [ $idx -lt ${#BATCHES[@]} ]; do
    wait -n
    for GPU in "${GPUS[@]}"; do
        BS=${BATCHES[$idx]}
        GA=${GRAD_ACUMS[$idx]}
        run_job "$GPU" "$BS" "$GA" &
        idx=$((idx+1))
        [ $idx -ge ${#BATCHES[@]} ] && break
    done 
done
wait
