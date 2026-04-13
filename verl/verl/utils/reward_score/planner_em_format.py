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


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    content = text

    # Consider both verbal signal and original document format for retrieval check
    if "<information>[Doc 1]" not in content and "<information>Doc 1" not in content:
        return False, "Missing retrieval"

    # Check for balanced tags
    tags_to_check = ["search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return (
                False,
                f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags",
            )

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> search -> think -> information -> ... -> think -> answer -> end

    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<search>" and state == "think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state == "start" or state == "information":
                state = "think"
            elif state in ["think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["after_search"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return (
                        False,
                        f"Unexpected content '{part.strip()}' between tags (state: {state})",
                    )
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """
    Extract the string inside the <answer> and </answer> tags,
    as well as the strings before and after the tags.
    """

    answer_pattern = r"<answer>(.*?)</answer>"
    match_iter = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match_iter)

    # If there are 0 or exactly 1 matches, return None for all parts
    if len(matches) < 1:
        return None
    # If there are 2 or more matches, use the last one
    last_match = matches[-1]
    if len(matches) > 1:
        second_last_match = matches[-2]
        before = solution_str[second_last_match.end(0) : last_match.start(0)]
    else:
        before = solution_str[: last_match.start(0)]

    if (
        "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between "
        in before
    ):
        return None
    inside = last_match.group(1).strip()
    return inside


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def find_max_relevance_score(text: str) -> int | None:
    """
    Finds all relevance scores in a string and returns the maximum.

    Args:
        text: The input string containing relevance scores.

    Returns:
        The maximum integer score, or None if no scores are found.
    """
    # 1. Define the regular expression pattern
    #    \(Relevance score: - Matches the literal text, escaping the parenthesis
    #    \s* - Matches any (or no) whitespace
    #    (\d+)               - Captures one or more digits (the score)
    #    \)                  - Matches the closing parenthesis
    pattern = r"\(Relevance score:\s*(\d+)\)"

    # 2. Use re.findall() to get a list of all captured scores
    #    This will return a list of strings, e.g., ['5', '4', '1']
    matches = re.findall(pattern, text)

    # 3. Handle the case where no scores are found
    if not matches:
        return None

    # 4. Convert the list of score strings to integers
    scores = [int(score) for score in matches]

    # 5. Return the maximum score
    return max(scores)


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score(data_source, solution_str, solution_ids, prompt_str, ground_truth, extra_info):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    structure_format_score = extra_info.get("structure_format_score", 0.2)
    final_format_score = extra_info.get("final_format_score", 0.1)
    retrieval_score = extra_info.get("retrieval_score", 0.0)
    score = extra_info.get("score", 1.0)

    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth)
    answer = extract_solution(solution_str=solution_str)
    do_print = True
    # do_print = random.randint(1, 64) == 1

    final_score = None

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                final_score = structure_format_score + retrieval_score  # 0.3
            else:
                final_score = structure_format_score  # 0.2
        else:
            final_score = 0
    else:
        if em_check(answer, ground_truth):
            if is_valid_format:
                final_score = score  # 1
            else:
                final_score = score - structure_format_score  # 0.8
        elif is_valid_format:
            if retrieval_correct:
                final_score = structure_format_score + retrieval_score  # 0.3
            else:
                final_score = structure_format_score  # 0.2
        else:
            final_score = final_format_score  # 0.1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Final score: {final_score}")
        print(f"Solution string: {prompt_str + ' ' + solution_str}")

    return {
        "score": final_score,
        "phrase_token_lengths": 0,
        "advantage_penalty": 0.0,
    }
