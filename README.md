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
