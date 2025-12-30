"""
dashboard.py

Streamlit-based web dashboard for visualizing benchmark results.
Allows users to upload CSV files from benchmark runs and view:
- Raw results table
- Aggregated metrics by backend and scenario
- Latency comparison charts
- Accuracy comparison by scenario

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd

st.title("Letta vs Mem0 Benchmark Dashboard")
upload_csv = st.file_uploader("Upload CSV results", type=["csv"])
if upload_csv:
    df = pd.read_csv(upload_csv)
    st.write("Raw Results", df)
    if not df.empty:
        grp = df.groupby(["backend","scenario"]).agg(mean_latency_ms=("latency_ms","mean"), accuracy=("accuracy","mean")).reset_index()
        st.subheader("Aggregates")
        st.dataframe(grp)
        st.subheader("Latency by Scenario")
        for sc in grp["scenario"].unique():
            sub = df[df["scenario"]==sc]
            st.bar_chart(sub, x="backend", y="latency_ms")
        st.subheader("Accuracy by Scenario")
        acc = grp.pivot(index="scenario", columns="backend", values="accuracy").fillna(0.0)
        st.dataframe(acc)
else:
    st.info("Upload a CSV from src.benchmark to view summaries.")
