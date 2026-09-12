"""
TCX3901 Industrial Practice — Report 1 EDA Dashboard
Delivery Performance: Lead Time and Late-Delivery Risk

Student: Yeo Hui Hui, A0310673L, Group 5

Note: 
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Olist Delivery Performance — EDA Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

DATA_FILE = "order_level.csv"


@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=["order_purchase_timestamp"])
    return df


st.title("Olist Delivery Performance — Exploratory Dashboard")
st.caption("TCX3901 Industrial Practice, Report 1 · Yeo Hui Hui, A0310673L, Group 5")

try:
    full = load_data(DATA_FILE)
except FileNotFoundError:
    st.warning(
        f"Could not find `{DATA_FILE}` in the app folder. "
        "Upload it below (it's the joined order-level table from the analysis notebook)."
    )
    uploaded = st.file_uploader("Upload order_level.csv", type="csv")
    if uploaded is None:
        st.stop()
    full = load_data(uploaded)

required_cols = {
    "order_purchase_timestamp", "lead_time_days", "promised_days", "late",
    "customer_state", "is_cross_state",
}
missing_cols = required_cols - set(full.columns)
if missing_cols:
    st.error(f"The uploaded file is missing expected columns: {sorted(missing_cols)}")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------

st.sidebar.header("Filters")

states = sorted(full.customer_state.dropna().unique())
selected_states = st.sidebar.multiselect(
    "Customer state (leave empty for all)", states, default=[]
)

min_date = full.order_purchase_timestamp.min().date()
max_date = full.order_purchase_timestamp.max().date()
date_range = st.sidebar.slider(
    "Purchase date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

shipment_filter = st.sidebar.radio(
    "Shipment type", ["All", "Same-state only", "Cross-state only"], index=0
)

filtered = full.copy()
if selected_states:
    filtered = filtered[filtered.customer_state.isin(selected_states)]
filtered = filtered[
    (filtered.order_purchase_timestamp.dt.date >= date_range[0])
    & (filtered.order_purchase_timestamp.dt.date <= date_range[1])
]
if shipment_filter == "Same-state only":
    filtered = filtered[filtered.is_cross_state == 0]
elif shipment_filter == "Cross-state only":
    filtered = filtered[filtered.is_cross_state == 1]

st.sidebar.markdown(f"**{len(filtered):,}** of {len(full):,} orders match the current filters.")

if len(filtered) == 0:
    st.warning("No orders match the current filter combination. Widen a filter in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI header row
# ----------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Orders", f"{len(filtered):,}")
col2.metric("Late rate", f"{filtered.late.mean():.1%}")
col3.metric("Mean actual lead time", f"{filtered.lead_time_days.mean():.1f} days")
col4.metric("Mean promised window", f"{filtered.promised_days.mean():.1f} days")

st.divider()

# ----------------------------------------------------------------------------
# View 1 — Lead time distribution: actual vs. promised
# ----------------------------------------------------------------------------

st.subheader("1. Actual lead time vs. the platform's promised window")
fig1 = go.Figure()
fig1.add_trace(go.Histogram(
    x=filtered.lead_time_days, name="Actual lead time",
    opacity=0.75, marker_color="#2E5A87",
    xbins=dict(start=0, end=60, size=1),
))
fig1.add_trace(go.Histogram(
    x=filtered.promised_days, name="Promised window",
    opacity=0.5, marker_color="#C0392B",
    xbins=dict(start=0, end=60, size=1),
))
fig1.add_vline(x=filtered.lead_time_days.median(), line_dash="dash", line_color="#2E5A87",
               annotation_text=f"Median actual: {filtered.lead_time_days.median():.1f}d")
fig1.add_vline(x=filtered.promised_days.median(), line_dash="dash", line_color="#C0392B",
               annotation_text=f"Median promised: {filtered.promised_days.median():.1f}d")
fig1.update_layout(
    barmode="overlay", xaxis_title="Days from purchase", yaxis_title="Number of orders",
    height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------------------------------------------------------
# View 2 — Monthly order volume and late-delivery rate
# ----------------------------------------------------------------------------

st.subheader("2. Order volume and late-delivery rate by month")
monthly = (
    filtered.groupby(filtered.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp())
    .agg(orders=("order_id", "count"), late_rate=("late", "mean"))
    .reset_index()
    .rename(columns={"order_purchase_timestamp": "month"})
)
monthly = monthly[monthly.orders >= 5]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=monthly.month, y=monthly.orders, name="Orders placed", marker_color="#B7C9DC", yaxis="y1",
))
fig2.add_trace(go.Scatter(
    x=monthly.month, y=monthly.late_rate * 100, name="Late rate (%)",
    marker_color="#C0392B", mode="lines+markers", yaxis="y2",
))
fig2.update_layout(
    xaxis_title="Purchase month",
    yaxis=dict(title="Orders placed"),
    yaxis2=dict(title="Late deliveries (%)", overlaying="y", side="right"),
    height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# View 3 — Late-delivery rate by customer state, with 95% CI
# ----------------------------------------------------------------------------

st.subheader("3. Late-delivery rate by customer state")
top_states = filtered.customer_state.value_counts().head(10).index
s = (
    filtered[filtered.customer_state.isin(top_states)]
    .groupby("customer_state")
    .agg(late_rate=("late", "mean"), n=("order_id", "count"))
    .reset_index()
)
s["se"] = np.sqrt(s.late_rate * (1 - s.late_rate) / s.n)
s["ci"] = 1.96 * s.se
s = s.sort_values("late_rate", ascending=False)
s["label"] = s.apply(lambda r: f"n={r.n:,}", axis=1)

fig3 = px.bar(
    s, x="late_rate", y="customer_state", orientation="h",
    error_x="ci", text="label", color_discrete_sequence=["#2E5A87"],
)
fig3.update_traces(textposition="outside")
fig3.update_layout(
    xaxis_title="Late-delivery rate", yaxis_title="Customer state",
    yaxis=dict(autorange="reversed"), height=440,
)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# View 4 — Baseline comparison (fixed reference numbers from Report 1, Section 5)
# ----------------------------------------------------------------------------

st.subheader("4. Baseline comparison (measured on the official chronological hold-out)")
st.caption(
    "These five bars are the fixed baseline numbers from Report 1, Section 5, measured once "
    "on the 80/20 chronological train/test split described there. They do not change with the "
    "sidebar filters above — recomputing a 'baseline' on an arbitrary filtered subset would not "
    "be the same experiment, and would defeat the purpose of a single fixed number to beat."
)
baseline_df = pd.DataFrame({
    "Baseline": [
        "Platform estimate", "Global mean", "Same/cross-state (2-bucket)",
        "Customer-state mean", "Route mean (chosen)",
    ],
    "Test MAE (days)": [13.14, 6.43, 5.86, 5.74, 5.63],
})
fig4 = px.bar(
    baseline_df, x="Baseline", y="Test MAE (days)", text="Test MAE (days)",
    color="Baseline",
    color_discrete_sequence=["#C0392B", "#B7C9DC", "#8FA8C4", "#5A7A9C", "#2E5A87"],
)
fig4.update_traces(textposition="outside")
fig4.update_layout(showlegend=False, height=420, yaxis_title="Test MAE (days)")
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption(
    "Data: Olist Brazilian E-Commerce Public Dataset (Olist & Sionek, 2018), joined and "
    "aggregated to order level as described in Report 1, Section 3."
)
