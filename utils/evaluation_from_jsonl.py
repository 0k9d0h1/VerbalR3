import json
import re
from collections import defaultdict
from tqdm import tqdm


def analyze_scores(file_path: str):
    """
    Reads a JSONL file and calculates the average scores for each data source.

    Args:
        file_path (str): The path to the input .jsonl file.
    """
    # Use a nested defaultdict to easily store scores for each source and metric.
    # Structure: { 'source_name': { 'metric_name': [score1, score2, ...] } }
    results = defaultdict(lambda: defaultdict(list))

    print(f"Reading and processing data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Analyzing samples"):
                try:
                    data = json.loads(line)
                    source = data.get("data_source")

                    if source:
                        # Append the scores to the appropriate list for the given source
                        results[source]["correct"].append(data.get("correct", 0))
                        results[source]["f1"].append(data.get("f1", 0.0))
                        results[source]["retrieval_correct"].append(
                            data.get("retrieval_correct", 0)
                        )
                        if data.get("reranker_calls"):
                            results[source]["reranker_calls"].append(
                                data.get("reranker_calls", 0)
                            )
                        else:
                            matches = re.findall(
                                r"<information>.*?</information>",
                                data.get("response", ""),
                                flags=re.DOTALL,
                            )
                            results[source]["reranker_calls"].append(len(matches))

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping malformed line or missing key: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # --- Calculate and Print the Final Summary Table ---
    print("\n--- Average Score Summary by Data Source ---")
    print(
        f"{'Data Source':<20} | {'Avg Correct':<15} | {'Avg F1':<15} | {'Avg Retrieval':<20} | {'Avg Reranker Calls':<20}"
    )
    print("-" * 100)

    for source, metrics in sorted(results.items()):
        # Calculate the mean for each list of scores.
        # Use np.mean for robustness, defaulting to 0 if the list is empty.
        import numpy as np

        avg_correct = np.mean(metrics["correct"]) if metrics["correct"] else 0.0
        avg_f1 = np.mean(metrics["f1"]) if metrics["f1"] else 0.0
        avg_retrieval = (
            np.mean(metrics["retrieval_correct"])
            if metrics["retrieval_correct"]
            else 0.0
        )
        avg_reranker_call = (
            np.mean(metrics["reranker_calls"]) if metrics["reranker_calls"] else 0.0
        )

        # Print a formatted row in the table
        print(
            f"{source:<20} | {avg_correct:<15.4f} | {avg_f1:<15.4f} | {avg_retrieval:<20.4f} | {avg_reranker_call:<20.4f}"
        )

    print("-" * 100)


if __name__ == "__main__":
    input_file = "Path to the generation result"
    analyze_scores(input_file)
