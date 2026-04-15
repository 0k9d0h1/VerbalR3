# Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning
---

Official implementation for [Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning](http://arxiv.org/abs/).

## Links
#### Models

## Installation
---
### Reranker Training Environment
We recommend using `uv` for fast and reliable management.
```bash
conda create -n reranker python=3.12 -y
conda activate reranker
pip install uv
uv pip install ms-swift==3.7.3
uv pip install deepspeed==0.16.9
uv pip install vllm==0.10.1.1
uv pip install huggingface_hub[cli] hf_transfer
uv pip install wandb
uv pip install scikit-learn
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Reranker Training
Verbal Reranker is trained using the distilled trajectories from GPT-OSS-120B.
After opening the retriever server, you can get the NQ retrieved data from it.
```bash
python src/retrieve_nq.py --search_url {SEARCH_URL}
```

Batches for llm input with the retrieved data is formend as follows.
```bash
python src/forma_batch.py
```

Open the vllm server with GPT-OSS-120B (This can be altered with the desired model).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-120b \
    --port 8017 \
    --tensor_parallel_size 4 \
    --data_parallel_size 1 \
    --served_model_name rerank
```

Use the server for inferencing. this automatically supports resuming feature.
```
python src/inference.py \
  --model_name rerank\
  --base_url "http://localhost:8017/v1/" \
  --api_key EMPTY \
  --max_tokens 16384 \
  --max_workers 256 \
  --input_path batch.jsonl \
  --output_path vllm_output.jsonl \
  --messages_key "messages"
```

Reformat the data for SFT distillation. Filtering is applied here, while the splitting must be done manually.
```
python src/build_sft_data.py \
  --input vllm_output.jsonl
  --output sft_data.jsonl
```

Using the formated data, perform sft to get the reranker
```
WANDB_PROJECT="RERANK" NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train_type full \
    --dataset /home/tkdrnjs0621/work/rerank/data/sft_data.jsonl\
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --save_strategy epoch \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir saves/reranker_sft_3b \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --report_to wandb \
    --run_name reranker_sft_3b \
    --use_hf true \
    --deepspeed zero3 \
    --attn_impl flash_attn 
```
Trained reranker can be used in a following python code.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "your-username/your-model-name"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

SYSTEM_PROMPT = """You are an evaluator that judges how informative a document is for answering a given question. You will receive a Question and a Document.

Carefully assess relevance and usefulness with a brief, evidence-based comment that supports downstream filtering (mention concrete entities/claims/dates/metrics when helpful), and then a score of relevance.

Scoring rubric (1-5):
1 — Unrelated: The document has nothing to do with the question.
2 — Loosely related: Contains information that might potentially help, but is unlikely to.
3 — Partially informative: Contains information that can potentially help answer the question.
4 — Substantively informative: Related and includes relevant information.
5 — Direct answer: Clearly related and includes key information to directly answer the question.

Output format (exactly):
Comment: <concise assessment citing specific evidence>
Score: <1-5>"""


def evaluate(question: str, document: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nDocument: {document}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs, max_new_tokens=256, temperature=1.0, top_p=0.95
    )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response


# --- Example usage ---
question = "When was the Eiffel Tower built?"
document = "The Eiffel Tower is a wrought-iron lattice tower in Paris, constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."

result = evaluate(question, document)
print(result)
```
