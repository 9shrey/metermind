from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_model import score_meters
from src.explanations import attach_explanations
from src.feature_engineering import build_meter_features
from src.generate_data import save_sample_data


APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "data" / "smart_meter_sample.csv"


st.set_page_config(
    page_title="MeterMind",
    page_icon="MM",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --bg: #f5f7fb;
        --panel: #ffffff;
        --text: #172033;
        --muted: #667085;
        --border: #d9e0ea;
        --accent: #1f7a8c;
        --danger: #cf3f3f;
        --warning: #d98a1d;
        --success: #228b64;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] { background: #102033; }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    h1, h2, h3 { letter-spacing: 0; color: var(--text); }
    .hero {
        padding: 30px 34px;
        border: 1px solid var(--border);
        background:
            radial-gradient(circle at 92% 12%, rgba(31, 122, 140, 0.18), transparent 28%),
            linear-gradient(135deg, #ffffff 0%, #edf7f8 58%, #e9f0f7 100%);
        border-radius: 8px;
        margin-bottom: 18px;
        box-shadow: 0 18px 45px rgba(33, 52, 74, 0.08);
    }
    .hero h1 { margin: 0 0 6px; font-size: 42px; line-height: 1.05; }
    .hero h2 { margin: 0 0 12px; font-size: 22px; line-height: 1.25; font-weight: 750; color: #223149; }
    .hero p { margin: 0; color: var(--muted); font-size: 16px; max-width: 760px; }
    .explain-box {
        padding: 15px 18px;
        border: 1px solid #b9d9e1;
        background: #edf8fa;
        border-radius: 8px;
        color: #21384f;
        font-size: 15px;
        margin-bottom: 18px;
    }
    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
        border: 1px solid #d3dde9;
        border-radius: 8px;
        padding: 18px 17px 15px;
        box-shadow: 0 15px 32px rgba(34, 50, 75, 0.08);
        min-height: 122px;
    }
    .metric-label { color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }
    .metric-value { color: var(--text); font-size: 32px; font-weight: 850; margin-top: 8px; }
    .metric-note { color: var(--muted); font-size: 12px; margin-top: 5px; }
    .metric-accent {
        width: 34px;
        height: 3px;
        border-radius: 999px;
        background: var(--accent);
        margin-bottom: 10px;
    }
    .section-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 12px 28px rgba(34, 50, 75, 0.05);
        margin-bottom: 18px;
    }
    .risk-score {
        border-radius: 8px;
        padding: 22px;
        background: #102033;
        color: #ffffff;
        min-height: 160px;
    }
    .risk-score .label { color: #a9bacd; font-size: 13px; font-weight: 700; text-transform: uppercase; }
    .risk-score .value { font-size: 54px; font-weight: 850; line-height: 1; margin: 10px 0; }
    .risk-score .level { font-size: 17px; font-weight: 700; }
    .flag {
        background: #f7fafc;
        border-left: 4px solid var(--accent);
        padding: 11px 12px;
        border-radius: 6px;
        margin-bottom: 9px;
        color: #26364a;
        font-size: 14px;
    }
    .score-guide {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }
    .guide-item {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: #ffffff;
        padding: 14px 15px;
    }
    .guide-label { font-size: 14px; font-weight: 800; margin-bottom: 5px; }
    .guide-copy { font-size: 13px; color: var(--muted); }
    .panel-kicker {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .sidebar-disclaimer {
        margin-top: 24px;
        padding: 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.22);
        font-size: 12px;
        line-height: 1.45;
    }
    .footer-disclaimer {
        color: #526276;
        font-size: 13px;
        padding: 8px 2px 20px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""


@st.cache_data(show_spinner=False)
def load_readings() -> pd.DataFrame:
    if not DATA_PATH.exists():
        save_sample_data(DATA_PATH)
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return data


@st.cache_data(show_spinner=False)
def prepare_scores(data: pd.DataFrame) -> pd.DataFrame:
    return attach_explanations(score_meters(build_meter_features(data)))


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-accent"></div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def risk_color(level: str) -> str:
    return {"High": "#cf3f3f", "Medium": "#d98a1d", "Low": "#228b64"}.get(level, "#1f7a8c")


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
readings = load_readings()
scored = prepare_scores(readings)

st.sidebar.title("MeterMind")
st.sidebar.caption("Behavioral fingerprinting filters")
areas = st.sidebar.multiselect("Area", sorted(scored["area"].unique()), default=sorted(scored["area"].unique()))
consumer_types = st.sidebar.multiselect(
    "Consumer type",
    sorted(scored["consumer_type"].unique()),
    default=sorted(scored["consumer_type"].unique()),
)
risk_levels = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
st.sidebar.markdown(
    """
    <div class="sidebar-disclaimer">
        Prototype uses synthetic data. Risk score indicates inspection priority, not proof of tampering.
    </div>
    """,
    unsafe_allow_html=True,
)

filtered = scored[
    scored["area"].isin(areas)
    & scored["consumer_type"].isin(consumer_types)
    & scored["risk_level"].isin(risk_levels)
].copy()

st.markdown(
    """
    <div class="hero">
        <h1>MeterMind</h1>
        <h2>Behavioral Fingerprinting for Non-Technical Loss Detection in Smart Meters</h2>
        <p>AI-assisted risk ranking for smart meter inspection prioritization.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="explain-box">
        This prototype uses synthetic BESCOM-style smart meter data to demonstrate how behavioral anomaly detection can help prioritize non-technical loss investigations.
    </div>
    """,
    unsafe_allow_html=True,
)

total_meters = len(filtered)
high_risk = int((filtered["risk_level"] == "High").sum())
medium_risk = int((filtered["risk_level"] == "Medium").sum())
avg_risk = filtered["risk_score"].mean() if total_meters else 0
estimated_suspicious_units = (
    (filtered["previous_30_day_avg"] - filtered["recent_7_day_avg"]).clip(lower=0) * 7
).sum()

kpi_cols = st.columns(5)
kpi_cols[0].markdown(metric_card("Total Meters", f"{total_meters:,}", "after active filters"), unsafe_allow_html=True)
kpi_cols[1].markdown(metric_card("High Risk Meters", f"{high_risk:,}", "prioritize inspection"), unsafe_allow_html=True)
kpi_cols[2].markdown(metric_card("Medium Risk Meters", f"{medium_risk:,}", "needs monitoring"), unsafe_allow_html=True)
kpi_cols[3].markdown(metric_card("Average Risk Score", f"{avg_risk:.1f}", "0 to 100 scale"), unsafe_allow_html=True)
kpi_cols[4].markdown(
    metric_card("Estimated Suspicious Units", f"{estimated_suspicious_units:,.0f}", "recent shortfall estimate"),
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-panel">
        <div class="panel-kicker">How to interpret risk score</div>
        <div class="score-guide">
            <div class="guide-item"><div class="guide-label" style="color:#228b64;">Low</div><div class="guide-copy">Normal behavior</div></div>
            <div class="guide-item"><div class="guide-label" style="color:#d98a1d;">Medium</div><div class="guide-copy">Needs monitoring</div></div>
            <div class="guide-item"><div class="guide-label" style="color:#cf3f3f;">High</div><div class="guide-copy">Prioritize for inspection</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns([1.05, 1])
with chart_left:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.subheader("Risk Distribution")
    risk_counts = filtered["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    risk_counts.columns = ["risk_level", "meters"]
    fig = px.bar(
        risk_counts,
        x="risk_level",
        y="meters",
        color="risk_level",
        color_discrete_map={"High": "#cf3f3f", "Medium": "#d98a1d", "Low": "#228b64"},
        text="meters",
    )
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    fig.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with chart_right:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.subheader("Area-Wise High-Risk Meters")
    area_high = (
        filtered[filtered["risk_level"] == "High"]
        .groupby("area")
        .size()
        .sort_values(ascending=True)
        .reset_index(name="high_risk_meters")
    )
    fig = px.bar(area_high, x="high_risk_meters", y="area", orientation="h", color_discrete_sequence=["#1f7a8c"])
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-panel">', unsafe_allow_html=True)
st.subheader("Top 10 Suspicious Meters")
ranking = filtered[
    [
        "meter_id",
        "risk_score",
        "risk_level",
        "consumer_type",
        "area",
        "feeder_id",
        "percentage_consumption_drop",
        "zero_consumption_days",
        "peer_underconsumption_pct",
        "voltage_anomaly_count",
        "power_factor_anomaly_count",
    ]
].head(10).rename(
    columns={
        "meter_id": "Meter ID",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "consumer_type": "Consumer Type",
        "area": "Area",
        "feeder_id": "Feeder ID",
        "percentage_consumption_drop": "Consumption Drop %",
        "zero_consumption_days": "Zero Consumption Days",
        "peer_underconsumption_pct": "Peer Underconsumption %",
        "voltage_anomaly_count": "Voltage Anomalies",
        "power_factor_anomaly_count": "Power Factor Anomalies",
    }
)
st.dataframe(
    ranking,
    width="stretch",
    hide_index=True,
    column_config={
        "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
        "Consumption Drop %": st.column_config.NumberColumn("Consumption Drop %", format="%.1f"),
        "Peer Underconsumption %": st.column_config.NumberColumn("Peer Underconsumption %", format="%.1f"),
    },
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-panel">', unsafe_allow_html=True)
st.subheader("Investigation Panel")
default_meter = filtered.iloc[0]["meter_id"] if not filtered.empty else scored.iloc[0]["meter_id"]
selected_meter = st.selectbox("Select meter", filtered["meter_id"].tolist() or scored["meter_id"].tolist(), index=0)
meter_row = scored[scored["meter_id"] == selected_meter].iloc[0]
meter_readings = readings[readings["meter_id"] == selected_meter].sort_values("date")

profile_cols = st.columns([0.78, 1.35, 1.1])
with profile_cols[0]:
    st.markdown('<div class="panel-kicker">Behavioral Fingerprint</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-score">
            <div class="label">Risk score</div>
            <div class="value" style="color:{risk_color(meter_row['risk_level'])};">{meter_row['risk_score']:.1f}</div>
            <div class="level">{meter_row['risk_level']} risk</div>
            <div style="color:#a9bacd; margin-top:12px;">{meter_row['consumer_type']} | {meter_row['area']} | {meter_row['feeder_id']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.metric("Sanctioned load", f"{meter_row['sanctioned_load_kw']:.1f} kW")
    st.metric("Recent 7-day avg", f"{meter_row['recent_7_day_avg']:.1f} units")
    st.metric("Previous 30-day avg", f"{meter_row['previous_30_day_avg']:.1f} units")

with profile_cols[1]:
    st.markdown('<div class="panel-kicker">Consumption Trend</div>', unsafe_allow_html=True)
    trend = px.line(meter_readings, x="date", y="units_consumed", markers=False)
    trend.update_traces(line_color="#1f7a8c", line_width=3)
    trend.update_layout(height=365, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_title="Units")
    st.plotly_chart(trend, width="stretch")

with profile_cols[2]:
    st.markdown('<div class="panel-kicker">Explanation Flags</div>', unsafe_allow_html=True)
    for explanation in meter_row["explanations"]:
        st.markdown(f'<div class="flag">{explanation}</div>', unsafe_allow_html=True)

compare_cols = st.columns(2)
with compare_cols[0]:
    st.markdown('<div class="panel-kicker">Baseline vs Recent</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame(
        {
            "window": ["Previous 30-day baseline", "Recent 7-day average"],
            "units": [meter_row["previous_30_day_avg"], meter_row["recent_7_day_avg"]],
        }
    )
    fig = px.bar(comparison, x="window", y="units", color="window", color_discrete_sequence=["#667085", "#1f7a8c"])
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

with compare_cols[1]:
    st.markdown('<div class="panel-kicker">Peer Comparison</div>', unsafe_allow_html=True)
    peer_median = scored[
        (scored["consumer_type"] == meter_row["consumer_type"]) & (scored["area"] == meter_row["area"])
    ]["mean_consumption"].median()
    peer_df = pd.DataFrame(
        {"metric": ["Selected meter", "Peer group median"], "units": [meter_row["mean_consumption"], peer_median]}
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=peer_df["metric"],
                y=peer_df["units"],
                marker_color=["#cf3f3f" if meter_row["risk_level"] == "High" else "#1f7a8c", "#667085"],
            )
        ]
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", yaxis_title="Mean daily units")
    st.plotly_chart(fig, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-disclaimer">
        Prototype uses synthetic data. Risk score indicates inspection priority, not proof of tampering.
    </div>
    """,
    unsafe_allow_html=True,
)
