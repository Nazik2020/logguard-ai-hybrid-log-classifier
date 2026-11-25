from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

# Path to the trained Logistic Regression model (works from any working directory)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "log_classifier.joblib"

# Load models once when this file is imported (fast for repeated predictions)
transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
classifier_model = joblib.load(MODEL_PATH)


def classify_with_bert(log_message):
    """Classify a log message using Sentence-BERT embeddings and Logistic Regression."""
    if not isinstance(log_message, str) or not log_message.strip():
        return "Unclassified"

    # 1. Convert log text into a numeric BERT embedding
    message_embedding = transformer_model.encode([log_message])

    # 2. Check confidence score; if too low, mark as Unclassified
    probabilities = classifier_model.predict_proba(message_embedding)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"

    # 3. Return the predicted category
    predicted_label = classifier_model.predict(message_embedding)[0]
    return predicted_label


if __name__ == "__main__":
    sample_logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer",
    ]

    for log in sample_logs:
        label = classify_with_bert(log)
        print(f"{label}  |  {log}")
