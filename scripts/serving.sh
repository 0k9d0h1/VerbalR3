echo "--> Starting vLLM server on port 8001..."
srun vllm serve 0k9d0h1/reranker1.5b-sft --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --disable-uvicorn-access-log --port 8001 &

echo "--> Waiting for vLLM server to become available..."
while ! curl -s http://localhost:8001/health; do
    echo "  - vLLM server not ready yet, waiting 5 seconds..."
    sleep 5
done
echo "--> vLLM Server is ready!"

echo "--> Starting Reranker server..."
python -m utils.reranker_server \
    --model-name-or-path "0k9d0h1/reranker1.5b-sft" \
    --max-output-tokens 1024 \
    --max-score 5 \
    --concurrency 1024 \
    --retriever-url "http://localhost:7999/retrieve" \
    --vllm-url "http://localhost:8001/v1" \
    --temperature 0.6 \
    --top-p 0.95 
