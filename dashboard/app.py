"""
Interactive experiment dashboard (Streamlit).

    streamlit run dashboard/app.py

Reads the SQLite warehouse and renders the same readout an analyst would share
with a product team: headline metrics, segment breakdowns, and the retention
curve. Mirrors what would be a Looker / LookML dashboard in production.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DB_PATH = os.path.join(ROOT, "data", "warehouse.db")

st.set_page_config(page_title="Smart Savings Vault — A/B Readout",
                   page_icon="📊", layout="wide")


@st.cache_data
def load():
    if not os.path.exists(DB_PATH):
        return None
    from ab_test import load_user_metrics, run_full_readout
    df = load_user_metrics(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    users = pd.read_sql_query("SELECT * FROM users", con)
    con.close()
    df = df.merge(users[["user_id", "country", "plan", "acquisition_channel"]],
                  on="user_id", how="left")
    return df, run_full_readout(DB_PATH)


data = load()
st.title("📊 Smart Savings Vault — A/B Test Readout")
st.caption("Synthetic neobank data · SQL + Python · frequentist & Bayesian analysis")

if data is None:
    st.error("Warehouse not found. Run:\n\n"
             "`python data/generate_data.py && python src/etl.py`")
    st.stop()

df, readout = data
p, s, g = (readout["primary_metric"], readout["secondary_metric"],
           readout["guardrail_metric"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("D30 retention", f"{p['treatment']}%",
          f"{p['abs_lift']:+}pp vs control")
c2.metric("Weekly transactions", f"{s['treatment']}",
          f"{s['rel_lift_pct']:+}%")
c3.metric("Avg txn value (guardrail)", f"{g['treatment']}",
          f"{g['rel_lift_pct']:+}%")
c4.metric("Decision", readout["decision"],
          f"P(better)={readout['bayesian']['prob_treatment_better']:.0%}")

st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("Headline metrics")
    fig_path = os.path.join(ROOT, "reports", "figures", "metric_comparison.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    st.subheader("Significance")
    st.dataframe(pd.DataFrame([
        {"metric": "D30 retention", "test": p["test"], "p_value": p["p_value"],
         "significant": p["significant"]},
        {"metric": "Weekly txns", "test": s["test"], "p_value": s["p_value"],
         "significant": s["significant"]},
        {"metric": "Avg txn value", "test": g["test"], "p_value": g["p_value"],
         "significant": g["significant"]},
    ]), hide_index=True, use_container_width=True)

with right:
    st.subheader("Retention curve")
    rc = os.path.join(ROOT, "reports", "figures", "retention_curve.png")
    if os.path.exists(rc):
        st.image(rc, use_container_width=True)
    st.subheader("Bayesian posterior")
    bp = os.path.join(ROOT, "reports", "figures", "bayesian_posterior.png")
    if os.path.exists(bp):
        st.image(bp, use_container_width=True)

st.divider()
st.subheader("Segment explorer")
seg = st.selectbox("Break down retention by", ["plan", "country",
                                               "acquisition_channel"])
pivot = (df.groupby([seg, "experiment_group"])["retained_d30"]
           .mean().mul(100).round(1).reset_index()
           .pivot(index=seg, columns="experiment_group", values="retained_d30"))
pivot["lift_pp"] = (pivot["treatment"] - pivot["control"]).round(1)
st.dataframe(pivot, use_container_width=True)
st.bar_chart(pivot[["control", "treatment"]])
