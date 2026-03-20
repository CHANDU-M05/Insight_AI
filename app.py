"""
InsightAI — AI-Powered Data Analysis
Upload data → auto profile → ask questions → get code + charts → export report
"""

import os
import io
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from core.data_loader import load_file, load_sample, profile_dataframe
from core.analyst import ask
from core.reporter import generate_report

load_dotenv()

st.set_page_config(
    page_title="InsightAI — Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## InsightAI")
    st.caption("AI-powered data analysis — local, private, free")

    provider = st.selectbox("AI Provider", ["gemini", "openai"])

    if provider == "gemini":
        api_key = st.text_input("Gemini API Key", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""))
    else:
        api_key = st.text_input("OpenAI API Key", type="password",
                                value=os.getenv("OPENAI_API_KEY", ""))

    st.markdown("---")
    if st.button("Clear session"):
        for key in ["df", "profile", "history", "figures"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.caption("Your data never leaves your machine.")
    st.caption("Built by Chandu | github.com/CHANDU-M05")

# ── Session state ──
for key, default in [("df", None), ("profile", None), ("history", []), ("figures", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["Data", "Analyze", "Report"])


# ════════════════════════════════════════
# TAB 1 — DATA
# ════════════════════════════════════════
with tab1:
    st.markdown("### Load your dataset")

    load_mode = st.radio("Source", ["Upload file", "Use sample dataset"], horizontal=True)

    if load_mode == "Upload file":
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded:
            df, error = load_file(uploaded.read(), uploaded.name)
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.session_state.profile = profile_dataframe(df, uploaded.name)
                st.session_state.history = []
                st.session_state.figures = []
                st.success(f"Loaded {uploaded.name} — {df.shape[0]:,} rows x {df.shape[1]} columns")

    else:
        sample = st.selectbox("Sample dataset", ["sales", "titanic"])
        if st.button("Load sample"):
            df, error = load_sample(sample)
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.session_state.profile = profile_dataframe(df, f"{sample}.csv")
                st.session_state.history = []
                st.session_state.figures = []
                st.success(f"Loaded {sample} dataset — {df.shape[0]:,} rows x {df.shape[1]} columns")

    if st.session_state.df is not None:
        df = st.session_state.df
        profile = st.session_state.profile

        st.markdown("---")
        st.markdown("#### Auto profile")

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{profile.row_count:,}")
        c2.metric("Columns", profile.col_count)
        c3.metric("Missing values", f"{profile.null_pct}%")
        c4.metric("Numeric columns", len(profile.numeric_cols))

        # Warnings
        if profile.warnings:
            for w in profile.warnings:
                st.warning(w)

        # Column breakdown
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Numeric columns**")
            if profile.numeric_cols:
                st.success(" · ".join(profile.numeric_cols))
            else:
                st.caption("None")

            st.markdown("**Categorical columns**")
            if profile.categorical_cols:
                st.info(" · ".join(profile.categorical_cols))
            else:
                st.caption("None")

        with col_b:
            st.markdown("**Top correlations**")
            if profile.top_correlations:
                for c in profile.top_correlations[:3]:
                    bar = "█" * int(abs(c["correlation"]) * 10)
                    direction = "+" if c["correlation"] > 0 else "-"
                    st.caption(f"{c['col1']} ↔ {c['col2']}: {direction}{abs(c['correlation'])} {bar}")
            else:
                st.caption("Not enough numeric columns")

        # Column detail table
        with st.expander("Column details"):
            col_data = [{
                "Column": c.name,
                "Type": c.dtype,
                "Nulls": f"{c.null_count} ({c.null_pct}%)",
                "Unique": c.unique_count,
                "Min": c.min_val or "—",
                "Max": c.max_val or "—",
                "Mean": c.mean_val or "—",
                "Sample": ", ".join(c.sample_values[:2]),
            } for c in profile.columns]
            st.dataframe(col_data, use_container_width=True)

        # Raw data preview
        with st.expander("Raw data preview"):
            st.dataframe(df.head(20), use_container_width=True)

        # Auto distribution charts
        if profile.numeric_cols:
            st.markdown("#### Distributions")
            cols_to_plot = profile.numeric_cols[:4]
            chart_cols = st.columns(len(cols_to_plot))
            for i, col in enumerate(cols_to_plot):
                with chart_cols[i]:
                    fig = px.histogram(df, x=col, title=col, height=200)
                    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════
# TAB 2 — ANALYZE
# ════════════════════════════════════════
with tab2:
    st.markdown("### Ask questions about your data")

    if st.session_state.df is None:
        st.warning("Load a dataset in the Data tab first.")
        st.stop()

    if not api_key:
        st.error("Add your API key in the sidebar.")
        st.stop()

    df = st.session_state.df

    # Suggested questions
    profile = st.session_state.profile
    suggestions = []
    if profile.numeric_cols:
        suggestions.append(f"What is the distribution of {profile.numeric_cols[0]}?")
        suggestions.append(f"Show me the top 10 rows by {profile.numeric_cols[0]}")
    if profile.categorical_cols:
        suggestions.append(f"How many records per {profile.categorical_cols[0]}?")
    if len(profile.numeric_cols) >= 2:
        suggestions.append(f"Is there a correlation between {profile.numeric_cols[0]} and {profile.numeric_cols[1]}?")
    suggestions.append("What are the key insights from this dataset?")

    st.markdown("**Suggested questions:**")
    cols = st.columns(len(suggestions[:3]))
    for i, sug in enumerate(suggestions[:3]):
        with cols[i]:
            if st.button(sug, key=f"sug_{i}"):
                st.session_state["prefill"] = sug

    # Chat history display
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            if entry.get("explanation"):
                st.markdown(entry["explanation"])
            if entry.get("code"):
                with st.expander("Generated code", expanded=False):
                    st.code(entry["code"], language="python")
            if entry.get("fig") is not None:
                st.plotly_chart(entry["fig"], use_container_width=True)
            elif entry.get("result") is not None:
                if isinstance(entry["result"], pd.DataFrame):
                    st.dataframe(entry["result"], use_container_width=True)
                else:
                    st.markdown(str(entry["result"]))
            if entry.get("error"):
                st.error(f"Execution error: {entry['error']}")

    # Input
    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask anything about your data...")

    if question or prefill:
        q = question or prefill

        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = ask(
                    question=q,
                    df=df,
                    history=st.session_state.history,
                    api_key=api_key,
                    provider=provider,
                )

            if result["explanation"]:
                st.markdown(result["explanation"])

            if result["code"]:
                with st.expander("Generated code", expanded=True):
                    st.code(result["code"], language="python")

            if result["fig"] is not None:
                st.plotly_chart(result["fig"], use_container_width=True)
            elif result["result"] is not None:
                if isinstance(result["result"], pd.DataFrame):
                    st.dataframe(result["result"], use_container_width=True)
                else:
                    st.markdown(str(result["result"]))

            if result["error"]:
                st.error(f"Execution error: {result['error']}")
                st.caption("Try rephrasing your question.")

        # Save to history
        st.session_state.history.append({
            "question": q,
            "code": result["code"],
            "explanation": result["explanation"],
            "result": result["result"],
            "fig": result["fig"],
            "error": result["error"],
            "answer_text": result["explanation"] or str(result["result"])[:200],
        })
        st.session_state.figures.append(result["fig"])


# ════════════════════════════════════════
# TAB 3 — REPORT
# ════════════════════════════════════════
with tab3:
    st.markdown("### Export analysis report")
    st.caption("Download your full session — profile + questions + answers + charts — as a PDF.")

    if st.session_state.df is None:
        st.warning("Load a dataset first.")
        st.stop()

    if not st.session_state.history:
        st.warning("Ask at least one question in the Analyze tab first.")
        st.stop()

    st.markdown(f"**Session summary:** {len(st.session_state.history)} questions analyzed.")

    for entry in st.session_state.history:
        st.markdown(f"- {entry['question']}")

    if st.button("Generate PDF report", type="primary"):
        with st.spinner("Building report..."):
            try:
                pdf_bytes = generate_report(
                    filename=st.session_state.profile.filename,
                    profile=st.session_state.profile,
                    history=st.session_state.history,
                    figures=st.session_state.figures,
                )
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name=f"insightai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
                st.success("Report ready.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
