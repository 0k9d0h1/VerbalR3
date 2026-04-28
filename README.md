# Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning

Official implementation for **Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning**.

## Links

- 📄 Paper: [arXiv](http://arxiv.org/abs/XXXX.XXXXX)
- 🤗 Models: [Hugging Face](https://huggingface.co/0k9d0h1/reranker1.5b-sft, https://huggingface.co/0k9d0h1/reranker3b-sft, https://huggingface.co/0k9d0h1/3b-planner-1.5b-reranker-nq-hotpotqa-filtered-tp-reranker, https://huggingface.co/0k9d0h1/7b-planner-1.5b-reranker-nq-hotpotqa-filtered-tp-reranker)

---

## Overview

This repository contains code for:

- **Verbal Reranker training and inference**
- **Planner RL training**
- **Planner generation / evaluation**
- **Retriever indexing and serving**

The overall pipeline consists of three components:

1. **Retriever**: serves document retrieval over the indexed corpus
2. **Verbal Reranker**: scores and annotates retrieved documents
3. **Planner**: performs retrieval-augmented reasoning and generation

---

## Environment Setup

### 1. Planner Environment

```bash
conda create -n planner python=3.10 -y
conda activate planner

conda install -c conda-forge cuda-toolkit=12.4

cd verl
scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

pip install grpcio==1.74.0
pip install ray==2.49.0
```

### 2. Retriever Environment

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

# We recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

# Install the GPU version of faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# API server dependencies
pip install uvicorn fastapi
```

### 3. Reranker Environment

We recommend using `uv` for fast and reliable package management.

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

---

## Retriever Data Preparation

### 1. Download Retriever Data

```bash
save_path=/the/path/to/save
python utils/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### 2. Data Preprocessing

Run the following preprocessing scripts:

```bash
python utils/data_preprocess/planner_test_merge.py
python utils/data_preprocess/planner_train_merge.py
python utils/make_filtering_indices.py
python utils/filter_dataset.py
```

---

## Reranker Training

The Verbal Reranker is trained using distilled trajectories from GPT-OSS-120B. The training pipeline consists of the following steps.

### 1. Retrieve Data

After starting the retriever server, retrieve NQ data:

```bash
python src/retrieve_nq.py --search_url {SEARCH_URL}
```

### 2. Format Batches

Form batches for LLM input from the retrieved data:

```bash
python src/format_batch.py
```

### 3. Run Inference with Teacher Model

Launch a vLLM server with GPT-OSS-120B (this can be replaced with another teacher model if desired):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-120b \
    --port 8017 \
    --tensor_parallel_size 4 \
    --data_parallel_size 1 \
    --served_model_name rerank
```

Run inference against the server (supports automatic resume):

```bash
python src/inference.py \
    --model_name rerank \
    --base_url "http://localhost:8017/v1/" \
    --api_key EMPTY \
    --max_tokens 16384 \
    --max_workers 256 \
    --input_path batch.jsonl \
    --output_path vllm_output.jsonl \
    --messages_key "messages"
```

### 4. Build SFT Data

Reformat the output for SFT distillation. Filtering is applied automatically, but train/val splitting must be done manually.

```bash
python src/build_sft_data.py \
    --input vllm_output.jsonl \
    --output sft_data.jsonl
```

### 5. Fine-tune the Reranker

```bash
WANDB_PROJECT="RERANK" NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train_type full \
    --dataset /path/to/sft_data.jsonl \
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

---

## Serving

### 1. Retriever Server

Launch the retriever server with:

```bash
scripts/retriever_launch.sh
```

### 2. Reranker Server

We provide two serving modes:

```bash
scripts/dp_serving.sh
scripts/serving.sh
```

- `dp_serving.sh` uses **data parallelism** and is generally faster.
- However, we observed that it can be **unstable during planner RL training**.
- If you encounter instability during planner training, we recommend using `serving.sh` instead.

---

## Planner Training and Evaluation

The planner relies on the retriever and reranker servers during both training and evaluation.

- For **planner RL training**, we use the **1.5B reranker**.
- For **planner evaluation**, we use the **3B reranker**.

### 1. Planner RL Training

```bash
scripts/training.sh
```

### 2. Planner Generation

```bash
scripts/generation.sh
```

### 3. Evaluation

```bash
python utils/evaluation_from_jsonl.py
```

Before evaluation, make sure to modify the `input_file` path accordingly.

---

## Reranker Usage

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

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:], skip_special_tokens=True
    )
    return response


# Example
question = "When was the Eiffel Tower built?"
document = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, "
    "constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."
)

result = evaluate(question, document)
print(result)
```

---

## TODO

- Add test-time scaling implementation and usage instructions.

---

## Acknowledgement

This codebase is built upon:

- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)

We thank the authors for open-sourcing their code.

---

## Citation

```bibtex
@article{verbal-r3,
  title={Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
