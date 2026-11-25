from pathlib import Path
import re
from dotenv import load_dotenv
from groq import Groq

# Load GROQ_API_KEY from training/.env (or project root .env)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

groq = Groq()


def classify_with_llm(log_message):
    """Classify rare or complex log messages (e.g. LegacyCRM) using Groq LLM."""
    if not isinstance(log_message, str) or not log_message.strip():
        return "Unclassified"

    prompt = f"""Classify the log message into one of these categories:
(1) Workflow Error
(2) Deprecation Warning
If you can't figure out a category, use "Unclassified".
Put the category inside <category> </category> tags.
Log message: {log_message}"""

    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-120b",
        temperature=0.2,
    )

    content = chat_completion.choices[0].message.content or ""
    match = re.search(r"<category>(.*?)</category>", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Unclassified"


if __name__ == "__main__":
    sample_logs = [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' feature is no longer supported. Use 'ExportToCSV' instead.",
        "Hello how are you?",
    ]

    for log in sample_logs:
        label = classify_with_llm(log)
        print(f"{label}  |  {log}")
