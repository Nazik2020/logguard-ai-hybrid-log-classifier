import sys
from pathlib import Path
import pandas as pd

# Allow importing sibling modules whether run from project root or training/
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from processor_regex import classify_with_regex
from processor_BERT import classify_with_bert
from processor_LLM import classify_with_llm


def classify_log_with_tier(source, log_message):
    """Classify one log and return (predicted_label, tier_used)."""
    # Tier 3: Rare source with limited training data -> LLM
    if source == "LegacyCRM":
        return classify_with_llm(log_message), "LLM"

    # Tier 1: Predictable templates -> fast Regex rules
    label = classify_with_regex(log_message)
    if label is not None:
        return label, "Regex"

    # Tier 2: Unstructured logs -> BERT embeddings + Logistic Regression
    return classify_with_bert(log_message), "BERT"


def classify_log(source, log_message):
    """Classify a single log message using the 3-tier hybrid approach."""
    label, _ = classify_log_with_tier(source, log_message)
    return label


def classify(logs):
    """Classify a list of (source, log_message) pairs."""
    labels = []
    for source, log_message in logs:
        label = classify_log(source, log_message)
        labels.append(label)
    return labels


def classify_csv(input_file, output_file=None):
    """Classify all logs in a CSV file and save results with target_label."""
    df = pd.read_csv(input_file)
    if "source" not in df.columns or "log_message" not in df.columns:
        raise ValueError("CSV must contain 'source' and 'log_message' columns.")

    results = [
        classify_log_with_tier(source, log_message)
        for source, log_message in zip(df["source"], df["log_message"])
    ]
    df["target_label"] = [label for label, _ in results]
    df["classification_tier"] = [tier for _, tier in results]

    if output_file is None:
        output_file = CURRENT_DIR / "resources" / "output.csv"
    df.to_csv(output_file, index=False)
    return str(output_file)


if __name__ == "__main__":
    test_csv_path = CURRENT_DIR / "resources" / "test.csv"
    output_path = classify_csv(test_csv_path)
    print(f"Classified CSV saved to: {output_path}")
    print(pd.read_csv(output_path)[["source", "target_label", "classification_tier"]])
