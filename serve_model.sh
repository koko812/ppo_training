PORT=${1:-8000}
uv run trl vllm-serve \
  --model Qwen/Qwen2.5-0.5B \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu_memory_utilization 0.90 \
  --max_model_len 1024
