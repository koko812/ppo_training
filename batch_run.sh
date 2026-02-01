set -x

GPUS=(0 1 2 3)
BATCHES=(1 2 4 8)
MODEL="Qwen/Qwen2.5-0.5B"

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    BS=${BATCHES[$i]}
    CUDA_VISIBLE_DEVICES=$GPU uv run train_sft_with_compute_acc_batched.py \
    --model_name $MODEL \
    --batch_size "$BS" \
    --epochs 5 &
done 
wait
