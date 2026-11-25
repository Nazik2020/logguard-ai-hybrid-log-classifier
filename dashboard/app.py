from io import BytesIO
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from classify import classify_log_with_tier

st.set_page_config(
    page_title="LogGuard AI — Log Classification & Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
      }
      [data-testid="stMetric"] {
        background: rgba(27, 35, 54, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 14px 16px;
        font-variant-numeric: tabular-nums;
      }
      .stButton > button, .stDownloadButton > button {
        min-height: 44px;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 160ms cubic-bezier(0.23, 1, 0.32, 1);
      }
      .stButton > button:active, .stDownloadButton > button:active {
        transform: scale(0.97);
      }
      @media (max-width: 768px) {
        .block-container {
          padding-left: 1rem !important;
          padding-right: 1rem !important;
          padding-top: 1.5rem !important;
        }
      }
      @media (prefers-reduced-motion: reduce) {
        .stButton > button, .stDownloadButton > button {
          transition: none !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LogGuard AI — Real-World Log Classification")
st.caption("Tier 1: Regex Rules  ·  Tier 2: BERT + Logistic Regression  ·  Tier 3: Groq LLM")


@st.cache_data(show_spinner="Classifying logs through hybrid pipeline…")
def classify_dataframe(df_bytes: bytes, file_name: str, default_src: str) -> pd.DataFrame:
    if file_name.lower().endswith((".log", ".txt")):
        lines = [
            line.strip()
            for line in df_bytes.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        df = pd.DataFrame({"source": default_src, "log_message": lines})
    else:
        df = pd.read_csv(BytesIO(df_bytes))
        if "source" not in df.columns:
            df["source"] = default_src

    results = [
        classify_log_with_tier(source, msg)
        for source, msg in zip(df["source"], df["log_message"])
    ]
    df["target_label"] = [label for label, _ in results]
    df["classification_tier"] = [tier for _, tier in results]
    return df


with st.sidebar:
    st.header("Upload Real Logs")
    default_source = st.text_input("Default Source (if file has no source column)", value="ProductionServer")
    uploaded_file = st.file_uploader("Upload .csv, .log, or .txt file", type=["csv", "log", "txt"])

    st.divider()
    st.header("Live Single-Log Classifier")
    test_source = st.text_input("Application Source", placeholder="e.g. Nginx, AuthService, LegacyCRM")
    test_message = st.text_area("Log Message", placeholder="Paste any real log line…", height=96)
    if st.button("Classify Single Log", type="primary", use_container_width=True):
        if test_message.strip():
            label, tier = classify_log_with_tier(test_source.strip() or "Application", test_message)
            st.success(f"**Category:** {label}\n\n**Handled by:** {tier}")
        else:
            st.warning("Enter a log message first.")


if uploaded_file is None:
    st.info("Open the sidebar to upload a real `.csv`, `.log`, or `.txt` file—or test a single log line.")
    st.stop()

raw_bytes = uploaded_file.getvalue()
if uploaded_file.name.lower().endswith(".csv"):
    temp_df = pd.read_csv(BytesIO(raw_bytes))
    if "log_message" not in temp_df.columns:
        st.error("Uploaded CSV must contain a `log_message` column.")
        st.stop()

df_classified = classify_dataframe(raw_bytes, uploaded_file.name, default_source.strip() or "ProductionServer")

with st.expander("Search & Filter Classified Logs", expanded=False):
    search_query = st.text_input("Search log text", placeholder="Filter by IP, user ID, error…")
    f1, f2, f3 = st.columns(3)
    selected_sources = f1.multiselect(
        "Sources",
        options=sorted(df_classified["source"].dropna().unique()),
        default=sorted(df_classified["source"].dropna().unique()),
    )
    selected_labels = f2.multiselect(
        "Categories",
        options=sorted(df_classified["target_label"].dropna().unique()),
        default=sorted(df_classified["target_label"].dropna().unique()),
    )
    selected_tiers = f3.multiselect(
        "Classification Tiers",
        options=sorted(df_classified["classification_tier"].dropna().unique()),
        default=sorted(df_classified["classification_tier"].dropna().unique()),
    )

filtered_df = df_classified[
    df_classified["source"].isin(selected_sources)
    & df_classified["target_label"].isin(selected_labels)
    & df_classified["classification_tier"].isin(selected_tiers)
]
if search_query.strip():
    filtered_df = filtered_df[
        filtered_df["log_message"].astype(str).str.contains(search_query.strip(), case=False, na=False)
    ]

total_logs = len(filtered_df)
security_count = int((filtered_df["target_label"] == "Security Alert").sum())
error_labels = {"Error", "Critical Error", "Workflow Error"}
error_count = int(filtered_df["target_label"].isin(error_labels).sum())
local_count = int(filtered_df["classification_tier"].isin(["Regex", "BERT"]).sum())
local_pct = (local_count / total_logs * 100) if total_logs > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Logs", total_logs)
k2.metric("Security Alerts", security_count)
k3.metric("System & Workflow Errors", error_count)
k4.metric("Handled Locally (No LLM Cost)", f"{local_pct:.1f}%")

st.divider()

c1, c2 = st.columns(2)
with c1:
    label_counts = filtered_df["target_label"].value_counts().reset_index()
    label_counts.columns = ["Category", "Count"]
    fig_labels = px.bar(
        label_counts,
        x="Count",
        y="Category",
        orientation="h",
        color="Category",
        title="Log Distribution by Predicted Category",
        template="plotly_dark",
    )
    fig_labels.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_labels, use_container_width=True)

with c2:
    tier_counts = filtered_df["classification_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig_tiers = px.pie(
        tier_counts,
        names="Tier",
        values="Count",
        hole=0.48,
        title="Hybrid Architecture Routing Share",
        template="plotly_dark",
        color="Tier",
        color_discrete_map={"Regex": "#22C55E", "BERT": "#38BDF8", "LLM": "#A78BFA"},
    )
    fig_tiers.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_tiers, use_container_width=True)

tab_alerts, tab_all, tab_sources = st.tabs(["Security & Errors", "All Classified Logs", "Source Breakdown"])

with tab_alerts:
    priority_labels = {"Security Alert", "Critical Error", "Error", "Workflow Error"}
    priority_df = filtered_df[filtered_df["target_label"].isin(priority_labels)]
    if priority_df.empty:
        st.success("No high-severity alerts or errors found in the current selection.")
    else:
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

with tab_all:
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    csv_export = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Classified CSV",
        data=csv_export,
        file_name="classified_logs.csv",
        mime="text/csv",
    )

with tab_sources:
    source_cat = filtered_df.groupby(["source", "target_label"]).size().reset_index(name="Count")
    fig_source = px.bar(
        source_cat,
        x="Count",
        y="source",
        orientation="h",
        color="target_label",
        title="Categories by Application Source",
        barmode="stack",
        template="plotly_dark",
    )
    fig_source.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_source, use_container_width=True)
