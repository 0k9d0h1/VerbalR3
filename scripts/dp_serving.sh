conda activate vllm_serve
export CUDA_VISIBLE_DEVICES=0,1

echo "SERVING SCRIPT (REPLICATION): Starting on GPUs: $CUDA_VISIBLE_DEVICES"

VLLM_PIDS=()
cleanup() {
    echo "SERVING SCRIPT: Caught exit signal. Shutting down all vLLM servers..."
    if [ ${#VLLM_PIDS[@]} -gt 0 ]; then
        kill "${VLLM_PIDS[@]}" || true
    fi
}
trap cleanup EXIT

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
VLLM_URLS=()
BASE_PORT=8001

for i in "${!gpus[@]}"; do
    gpu_id="${gpus[$i]}"
    port=$((BASE_PORT + i))
    url="http://localhost:$port/v1"
    
    echo "SERVING SCRIPT: Starting vLLM server on GPU $gpu_id (Port $port)..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id vllm serve 0k9d0h1/reranker3b-sft \
        --port $port \
        --max-model-len 4096 \
        --disable-uvicorn-access-log \
        --gpu-memory-utilization 0.95 \
    
    VLLM_PIDS+=("$!")
    VLLM_URLS+=("$url")
done

echo "SERVING SCRIPT: Waiting for all ${#gpus[@]} vLLM servers..."
for i in "${!gpus[@]}"; do
    port=$((BASE_PORT + i))
    echo "SERVING SCRIPT: Waiting for server on port $port..."
    while ! curl -s http://localhost:$port/health; do
        sleep 2
    done
    echo "SERVING SCRIPT: Server on port $port is ready!"
done
echo "SERVING SCRIPT: All vLLM Servers are ready!"

vllm_url_string=$(IFS=,; echo "${VLLM_URLS[*]}")
echo "SERVING SCRIPT: Starting Reranker server on port 8005..."
echo "SERVING SCRIPT: Connecting to vLLM URLs: $vllm_url_string"

python -m utils.dp_reranker_server \
    --model-name-or-path "0k9d0h1/reranker3b-sft" \
    --max-output-tokens 1024 \
    --max-score 5 \
    --concurrency 1024 \
    --retriever-url "http://localhost:7999/retrieve" \
    --vllm-url "$vllm_url_string" \
    --temperature 0.6 \
    --top-p 0.95