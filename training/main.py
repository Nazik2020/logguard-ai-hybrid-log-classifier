from datetime import datetime
import sys
import time
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Allow importing sibling modules whether run from project root or training/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DASHBOARD_HTML = PROJECT_ROOT / "dashboard" / "index.html"
OUTPUT_CSV = CURRENT_DIR / "resources" / "output.csv"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from classify import classify_log_with_tier

app = FastAPI(
    title="Hybrid AI Log Classification API",
    description="Classifies real-world IT logs using Regex rules, BERT + Logistic Regression, and Groq LLM.",
    version="1.0.0",
)

# In-memory session store so the dashboard starts empty with zero static/mock data
SESSION_LOGS: list[dict] = []

MESSAGE_ALIASES = ("log_message", "message", "log", "text", "msg", "content", "event")
SOURCE_ALIASES = ("source", "service", "app", "application", "host", "component", "system")
TIME_ALIASES = ("timestamp", "time", "datetime", "date", "created_at", "@timestamp")


class LogRequest(BaseModel):
    source: str
    log_message: str
    add_to_session: bool = False


def _find_matching_column(columns, aliases):
    lower_map = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _normalize_dataframe(df: pd.DataFrame, default_source: str = "Application") -> pd.DataFrame:
    msg_col = _find_matching_column(df.columns, MESSAGE_ALIASES)
    if msg_col is None:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a log message column (e.g. 'log_message', 'message', 'log', or 'text').",
        )

    src_col = _find_matching_column(df.columns, SOURCE_ALIASES)
    ts_col = _find_matching_column(df.columns, TIME_ALIASES)

    normalized = pd.DataFrame()
    normalized["timestamp"] = df[ts_col].fillna("").astype(str) if ts_col else ""
    normalized["source"] = (
        df[src_col].fillna(default_source).astype(str) if src_col else default_source
    )
    normalized["log_message"] = df[msg_col].fillna("").astype(str)
    normalized = normalized[normalized["log_message"].str.strip() != ""].copy()

    if normalized.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no non-empty log messages.")
    return normalized


def _classify_dataframe(df: pd.DataFrame, default_source: str = "Application") -> pd.DataFrame:
    clean_df = _normalize_dataframe(df, default_source=default_source)
    results = [
        classify_log_with_tier(source, log_message)
        for source, log_message in zip(clean_df["source"], clean_df["log_message"])
    ]
    clean_df["target_label"] = [label for label, _ in results]
    clean_df["classification_tier"] = [tier for _, tier in results]
    return clean_df


def _parse_uploaded_file(file: UploadFile, default_source: str) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file.file)
    if filename.endswith((".log", ".txt")):
        raw_text = file.file.read().decode("utf-8", errors="replace")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return pd.DataFrame({"source": default_source, "log_message": lines})
    raise HTTPException(status_code=400, detail="Unsupported file format. Upload a .csv, .log, or .txt file.")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the responsive LogGuard AI web dashboard."""
    if DASHBOARD_HTML.exists():
        return FileResponse(DASHBOARD_HTML, media_type="text/html")
    return {"status": "ok", "message": "Hybrid AI Log Classification API is running"}


@app.get("/health")
def health_check():
    """Verify the API service is running."""
    return {"status": "ok", "message": "Hybrid AI Log Classification API is running"}


@app.get("/api/logs")
def get_classified_logs():
    """Return real logs classified during the current session (starts empty)."""
    return {"logs": SESSION_LOGS}


@app.post("/api/clear")
def clear_session_logs():
    """Clear all classified logs in the current session."""
    SESSION_LOGS.clear()
    return {"logs": []}


@app.post("/api/load-sample")
def load_sample_dataset():
    """Quick-load the 18 sample real-world logs for instant demoing and analysis."""
    sample_path = CURRENT_DIR / "resources" / "sample_upload.csv"
    if not sample_path.exists():
        sample_path = CURRENT_DIR / "resources" / "test.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found.")

    df = pd.read_csv(sample_path)
    df_classified = _classify_dataframe(df, default_source="ProductionServer")
    records = df_classified.fillna("").to_dict(orient="records")
    SESSION_LOGS.clear()
    SESSION_LOGS.extend(records)
    return {"logs": SESSION_LOGS, "count": len(SESSION_LOGS)}



@app.post("/api/classify-csv")
async def classify_file_json(file: UploadFile, default_source: str = Form("Application")):
    """Upload a real .csv, .log, or .txt file and return classified rows as JSON."""
    try:
        df_raw = _parse_uploaded_file(file, default_source.strip() or "Application")
        df_classified = _classify_dataframe(df_raw, default_source=default_source.strip() or "Application")

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_classified.to_csv(OUTPUT_CSV, index=False)

        records = df_classified.fillna("").to_dict(orient="records")
        SESSION_LOGS.clear()
        SESSION_LOGS.extend(records)
        return {"logs": SESSION_LOGS}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()


@app.post("/classify-log/")
def classify_single_log(request: LogRequest):
    """Classify a single log message and return its label, routing tier, and latency."""
    source = request.source.strip() or "Application"
    start = time.perf_counter()
    label, tier = classify_log_with_tier(source, request.log_message)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "log_message": request.log_message,
        "target_label": label,
        "classification_tier": tier,
    }
    if request.add_to_session:
        SESSION_LOGS.insert(0, record)

    return {
        **record,
        "latency_ms": latency_ms,
        "logs": SESSION_LOGS if request.add_to_session else None,
    }


@app.post("/classify/")
async def classify_logs_csv(file: UploadFile):
    """Upload a CSV file and download classified output.csv."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv file.")

    try:
        df = pd.read_csv(file.file)
        df_classified = _classify_dataframe(df)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_classified.to_csv(OUTPUT_CSV, index=False)

        return FileResponse(
            path=OUTPUT_CSV,
            media_type="text/csv",
            filename="output.csv",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()
