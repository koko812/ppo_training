set -x

GPUS=(1)
BATCHES=16
LRS=(5e-5 5e-6 5e-7)
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

run_job(){
    GPU=$1
    LR=$2
    CUDA_VISIBLE_DEVICES=$GPU uv run train_sft_with_compute_acc_batched.py \
    --model_name $MODEL \
    --batch_size $BATCHES \
    --lr "$LR" \
    --run_name "instruct" \
    --epochs 3
}

idx=0
for GPU in "${GPUS[@]}"; do
    LR=${LRS[$idx]}
    run_job "$GPU" "$LR" &
    idx=$((idx+1))
done 

while [ $idx -lt ${#LRS[@]} ]; do
    wait -n
    for GPU in "${GPUS[@]}"; do
        LR=${LRS[$idx]}
        run_job "$GPU" "$LR" &
        idx=$((idx+1))
        [ $idx -ge ${#LRS[@]} ] && break
    done 
done
wait
