# streamlit_app.py
import streamlit as st
from taskos_sim import run_sim
import pandas as pd

st.set_page_config(page_title="TaskOS Simulation", layout="wide")
st.title("🏙️ TaskOS™ Simulation Dashboard")
st.caption("AI-driven gig orchestration for real estate portfolios")

sim_time = st.slider("Simulation Duration (hours)", 1, 48, 12)
if st.button("Run Simulation"):
    df = run_sim(sim_minutes=sim_time*60)
    st.success(f"Simulation complete with {len(df)} events")
    st.metric("Completed Tasks", df['event'].eq('task_completed').sum())
    st.metric("Completion Rate", f"{df['event'].eq('task_completed').mean()*100:.1f}%")
    st.metric("Total Payout ($)", round(df['payout'].sum(),2))
    st.line_chart(df.groupby('time')['event'].count())
    st.dataframe(df.head(30))
