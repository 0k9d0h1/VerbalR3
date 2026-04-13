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
Preprocess the QA dataset to parquet format
"""

import math
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp["question"]

    # NOTE: also need to change reward_score/countdown.py
    if template_type == "base":
        """This works for any base model"""
        prefix = f"""You are an expert in retrieval who specializes in thinking, searching, and answering.

Before performing any search or providing an answer, you must first reason about your next action, including before your initial search.

After reasoning, if your knowledge is not enough to answer the question, you must call the reranker function by placing your query between <search> and </search>. The reranker function will return the relevant documents with their relevance scores(1~5 scale, 5 for most relevant and 1 for least relevant) between <information> and </information>.

If the retrieved results do not contain enough information for answering the question, perform additional searches to gather more context.
Continue searching iteratively until you have gathered sufficient information to respond accurately.

Once you determine that no further external information is required, provide your answer directly within <answer> and </answer>, without detailed explanations. For example:
<answer> Beijing </answer>.

Question: {question}"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/planner")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")
    parser.add_argument(
        "--data_sources",
        default="nq,hotpotqa,triviaqa,popqa,2wikimultihopqa,bamboogle,musique",
    )

    args = parser.parse_args()

    data_sources = args.data_sources.split(",")
    all_dataset = []
    least_num_samples = 2000

    for data_source in data_sources:
        if data_source != "strategyqa":
            dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
        else:
            dataset = datasets.load_dataset(
                "json",
                data_files="/home/peterjin/mnt/data/strategyqa/test_correct.jsonl",
            )

        if "test" in dataset:
            print(f"Using the {data_source} test dataset...")
            test_dataset = dataset["test"]
        elif "dev" in dataset:
            print(f"Using the {data_source} dev dataset...")
            test_dataset = dataset["dev"]
        else:
            print(f"Using the {data_source} train dataset...")
            test_dataset = dataset["train"]

        dataset_len = len(test_dataset)
        num_samples = math.ceil(least_num_samples / dataset_len)
        print(
            f"Dataset {data_source} has {dataset_len} samples, repeating {num_samples} times to reach at least {least_num_samples} samples."
        )
        test_dataset = datasets.concatenate_datasets([test_dataset] * num_samples)
        print(
            f"After repeating, dataset {data_source} has {len(test_dataset)} samples."
        )

        # add a row to each data item that represents a unique id
        def make_map_fn(split):
            def process_fn(example, idx):
                example["question"] = example["question"].strip()
                if example["question"][-1] != "?":
                    example["question"] += "?"
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example["golden_answers"],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "fact-reasoning",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                    },
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, "test_2000.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
