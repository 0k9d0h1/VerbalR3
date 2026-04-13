import json
import collections
import re
import string
import sys
import numpy as np
from typing import List


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


def _calculate_f1(
    prediction_tokens: List[str], ground_truth_tokens: List[str]
) -> float:
    """Helper function to calculate F1 score between two lists of tokens."""
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    prediction_counter = collections.Counter(prediction_tokens)
    ground_truth_counter = collections.Counter(ground_truth_tokens)

    common_tokens = prediction_counter & ground_truth_counter
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_max_f1(prediction: str, golden_answers: List[str]) -> float:
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


def evaluate(file_path):
    data_by_idx = collections.defaultdict(list)

    # Read the file
    print(f"Reading from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data_by_idx[item["idx"]].append(item)

    # Initialize global stats
    total_em = 0
    total_f1 = 0
    total_reranker_call = 0
    count = 0

    # Initialize per-source stats
    # Structure: {'source_name': {'em': 0, 'f1': 0, 'reranker': 0, 'count': 0}}
    source_stats = collections.defaultdict(
        lambda: {"em": 0, "f1": 0, "reranker": 0, "count": 0}
    )

    print(f"Found {len(data_by_idx)} unique indices.")

    for idx, items in data_by_idx.items():
        # Identify Data Source (assuming all items with same idx have same source)
        data_source = items[0].get("data_source", "unknown")

        # Get answers
        answers = [item.get("answer", "") for item in items]
        # Filter None answers if any
        answers = [str(a) if a is not None else "" for a in answers]

        # Majority vote based on normalized answer
        normalized_answers = [normalize_answer(a) for a in answers]
        counter = collections.Counter(normalized_answers)
        majority_normalized_answer = counter.most_common(1)[0][0]

        # Pick original answer string corresponding to majority normalized form
        majority_answer_original = ""
        for ans in answers:
            if normalize_answer(ans) == majority_normalized_answer:
                majority_answer_original = ans
                break

        ground_truth = items[0]["ground_truth"]

        # Calculate EM
        is_em = False
        for gt in ground_truth:
            if normalize_answer(gt) == majority_normalized_answer:
                is_em = True
                break

        # Calculate F1
        f1 = calculate_max_f1(majority_answer_original, ground_truth)

        # Calculate Reranker Calls
        current_reranker_calls = 0
        if "reranker_calls" in items[0] and items[0]["reranker_calls"] is not None:
            current_reranker_calls += items[0]["reranker_calls"]
        else:
            for item in items:
                # Check if 'reranker_calls' column exists
                # Fallback to regex counting
                resp = item.get("response", "")
                if resp:
                    matches = re.findall(
                        r"<information>.*?</information>", resp, flags=re.DOTALL
                    )
                    if matches:
                        current_reranker_calls += len(matches)

        # Update Global Stats
        total_em += 1 if is_em else 0
        total_f1 += f1
        total_reranker_call += current_reranker_calls
        count += 1

        # Update Per-Source Stats
        source_stats[data_source]["em"] += 1 if is_em else 0
        source_stats[data_source]["f1"] += f1
        source_stats[data_source]["reranker"] += current_reranker_calls
        source_stats[data_source]["count"] += 1

    if count == 0:
        print("No data found.")
        return

    # Print Global Results
    avg_em = total_em / count
    avg_f1 = total_f1 / count
    avg_reranker_call = total_reranker_call / count

    print("=" * 30)
    print("OVERALL RESULTS")
    print("=" * 30)
    print(f"Average EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average reranker call: {avg_reranker_call:.4f}")
    print("\n")

    # Print Per-Source Results
    print("=" * 30)
    print("RESULTS BY DATA SOURCE")
    print("=" * 30)
    # Sort by source name for consistent output
    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        s_count = stats["count"]

        if s_count > 0:
            s_avg_em = stats["em"] / s_count
            s_avg_f1 = stats["f1"] / s_count
            s_avg_reranker = stats["reranker"] / s_count

            print(f"Source: {source} (n={s_count})")
            print(f"  - EM: {s_avg_em:.4f}")
            print(f"  - F1: {s_avg_f1:.4f}")
            print(f"  - Reranker Calls: {s_avg_reranker:.4f}")
        else:
            print(f"Source: {source} (n=0)")
        print("-" * 20)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default path if not provided
        file_path = "Path to the generation result for test-time scaling"
    else:
        file_path = sys.argv[1]

    evaluate(file_path)
