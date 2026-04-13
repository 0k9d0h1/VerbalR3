import pandas as pd
import ray
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pickle
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def build_prompt(question: str) -> str:
    chat = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": question},
    ]
    # keep <|im_end|> etc. exactly as Qwen expects
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def batch_generate(start, queries, answers):
    model_answerable_idx = []
    prompts = [build_prompt(q) for q in queries]

    # 3) vLLM does the batching & GPU sharding internally
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        answer = answers[i]
        query = queries[i]
        skip = False
        for ans in answer:
            if ans.strip().lower() in ["yes", "no"]:
                skip = True
                break
            if normalize_answer(ans) in normalize_answer(query):
                skip = True
                break
        if skip:
            continue

        model_answerable = False
        for j, completion in enumerate(output.outputs):
            generated_text = completion.text
            for ans in answer:
                if normalize_answer(ans) in normalize_answer(generated_text):
                    model_answerable = True
                    break
            if model_answerable:
                break
        if model_answerable:
            model_answerable_idx.append(start + i)
    return model_answerable_idx


ray.init(
    num_gpus=1,
    include_dashboard=False,
)

train_path = "./data/planner/train.parquet"
model_name = "Qwen/Qwen2.5-3B"
batch_size = 512
num_return_sequences = 10
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="bfloat16",
    trust_remote_code=True,
)
sampling_params = SamplingParams(
    temperature=1.0, top_p=0.95, max_tokens=512, n=num_return_sequences
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, padding_side="left"
)

train_df = pd.read_parquet(train_path)
model_answerable_idx = []

for start in tqdm(range(0, len(train_df), batch_size)):
    end = start + batch_size
    batch_df = train_df.iloc[start:end]
    answers = batch_df["golden_answers"].tolist()
    queries = batch_df["question"].tolist()
    payload = {"queries": queries, "topk": 3, "return_scores": True}
    if type(answers[0]) is str:
        print("answers are not a list, convert them to a list")
        answers = [[a] for a in answers]
    model_answerable_idx.extend(batch_generate(start, queries, answers))


print(f"Model answerable indices: {len(model_answerable_idx)}")
print("Saving direct model answerable indices as pickle")
with open("./data/planner/answerable_indices.pkl", "wb") as f:
    pickle.dump(
        {
            "model_answerable": model_answerable_idx,
        },
        f,
    )
