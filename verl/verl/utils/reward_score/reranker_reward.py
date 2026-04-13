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

import re
import string
import random
import math


def extract_score(solution_str):
    """
    Extract the string inside the <answer> and </answer> tags,
    as well as the strings before and after the tags.
    """

    if "Score:" not in solution_str:
        return None
    else:
        score_str = solution_str.split("Score:")[-1].strip()
        if score_str.isdigit():
            return int(score_str)
        else:
            inside = re.findall(r"Score:\s*([-+]?\d*\.\d+|\d+)", score_str)
            if len(inside) == 0:
                return None
            else:
                return int(inside[0])


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    score = extract_score(solution_str)
    if score is None:
        return 0.0
    do_print = random.random() < 0.02

    max_score = int(extra_info["max_score"])

    max_error = max_score - 1
    reward = 1.0 - (abs(score - ground_truth) / max_error)

    if do_print:
        print("--------------------------------")
        print(f"Ground truth answer: {ground_truth}")
        print(f"Final score: {reward}")
        print(f"Solution string: {solution_str}")
    return reward
