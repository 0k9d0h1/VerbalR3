# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import json
import os
import re
import string
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.planner_utils.planner_generation import PlannerGenerationConfig, PlannerGenerationManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


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


def _calculate_f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    """Helper function to calculate F1 score between two lists of tokens."""
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    prediction_counter = Counter(prediction_tokens)
    ground_truth_counter = Counter(ground_truth_tokens)

    common_tokens = prediction_counter & ground_truth_counter
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_max_f1(prediction: str, golden_answers: list[str]) -> float:
    """
    Calculates the maximum F1 score between a prediction string and a list of golden answers.
    """
    if prediction is None:
        return 0.0

    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()

    max_f1 = 0.0
    for golden_answer in golden_answers:
        normalized_golden = normalize_answer(golden_answer)
        golden_tokens = normalized_golden.split()

        current_f1 = _calculate_f1(prediction_tokens, golden_tokens)

        if current_f1 > max_f1:
            max_f1 = current_f1

    return max_f1


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def extract_answer(text: str) -> str:
    """Return the substring inside <answer> ... </answer>; fall back to full text."""
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    if config.data.shuffle:
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]
    gt_lst = []
    for data in dataset["reward_model"]:
        gt_lst.append(data["ground_truth"]["target"])
    data_source_lst = dataset["data_source"].tolist()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]

    correct_score = {}
    retrieval_correct_score = {}
    f1_score = {}
    data_source_len = {}
    for data_source in set(data_source_lst):
        f1_score[data_source] = 0
        correct_score[data_source] = 0
        retrieval_correct_score[data_source] = 0
        data_source_len[data_source] = 0

    gen_config = PlannerGenerationConfig(
        max_turns=config.max_turns,
        max_start_length=config.data.max_start_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
        max_obs_length=config.data.max_obs_length,
        num_gpus=config.trainer.n_gpus_per_node * config.trainer.nnodes,
        no_think_rl=config.algorithm.no_think_rl,
        search_url=config.reranker.url,
        topk=config.reranker.topk,
        retriever_initial_topk=config.reranker.retriever_initial_topk,
        return_full_documents=config.reranker.get("return_full_documents", False),
    )

    # Agent config preparation
    generation_manager = PlannerGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=wg,
        config=gen_config,
        is_validation=True,
    )

    json_file_path = f"generation_results_{config.reranker.size}_top{config.reranker.retriever_initial_topk}_{config.model.path.split('/')[-1].replace('-', '_')}.jsonl"

    if not config.do_search:
        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

            data = DataProto.from_dict(batch_dict)
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
            for n_sample in range(config.data.n_samples):
                output_padded = wg.generate_sequences(data_padded)
                output = unpad_dataproto(output_padded, pad_size=pad_size)

                output_texts = []
                for i in range(len(output)):
                    data_item = output[i]
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    output_texts.append(response_str)

                output_lst[n_sample].extend(output_texts)

    else:
        for batch_idx in tqdm(range(num_batch)):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            for i in range(config.data.n_samples):
                batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
                batch_gt_lst = gt_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
                batch_data_source_lst = data_source_lst[
                    batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size
                ]
                inputs = tokenizer.apply_chat_template(
                    batch_chat_lst,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=config.rollout.prompt_length,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                position_ids = compute_position_id_with_mask(attention_mask)
                batch_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)

                test_gen_batch = test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                # Set validate False to use rollout config
                test_gen_batch.meta_info = {
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": False,
                }
                first_input_ids = test_gen_batch.batch["input_ids"][:, -gen_config.max_start_length :].clone()
                final_gen_batch_output = generation_manager.run_llm_loop(
                    gen_batch=test_gen_batch,
                    initial_input_ids=first_input_ids,
                )

                test_batch = test_batch.union(final_gen_batch_output)
                for key in test_batch.batch.keys():
                    test_batch.batch[key] = test_batch.batch[key].long()

                prompts = tokenizer.batch_decode(first_input_ids, skip_special_tokens=False)
                responses = tokenizer.batch_decode(test_batch.batch["responses"], skip_special_tokens=True)

                for idx, (prompt, response, gt, data_source) in enumerate(
                    zip(prompts, responses, batch_gt_lst, batch_data_source_lst)
                ):
                    answer = extract_answer(response)
                    correct = False
                    for golden_answer in gt:
                        if answer is not None and normalize_answer(golden_answer) == normalize_answer(answer):
                            correct = True
                            break
                    retrieval_correct = is_retrieval_correct(response, gt)

                    f1 = calculate_max_f1(answer, gt)

                    if config.output_format == "json":
                        out_path = os.path.join(
                            getattr(config, "output_dir", "."),  # default "./"
                            json_file_path,
                        )
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "a", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "idx": idx + batch_idx * config_batch_size,
                                    "data_source": data_source,
                                    "prompt": prompt.replace(tokenizer.eos_token, ""),
                                    "ground_truth": gt.tolist(),
                                    "response": response,
                                    "answer": answer,
                                    "correct": correct,
                                    "f1": f1,
                                    "retrieval_correct": retrieval_correct,
                                },
                                f,
                                ensure_ascii=False,
                            )
                            f.write("\n")
                            f.flush()


if __name__ == "__main__":
    os.unsetenv("ROCR_VISIBLE_DEVICES")
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    print("ROCR_VISIBLE_DEVICES:", os.environ.get("ROCR_VISIBLE_DEVICES", None))
    print(torch.cuda.device_count())
    main()
