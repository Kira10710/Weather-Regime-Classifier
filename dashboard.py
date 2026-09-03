"""
RAINCAST AI — Regime-Aware Rainfall Forecast Correction Platform
 | Operational Meteorological Decision Support System
Region: Maharashtra | Forecast Lead Time: 72 Hours (Day-3) | Data: June–September 2023
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date

# ─────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RainCast AI — Rainfall Forecast Correction",
    page_icon="RC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. Light Theme Design System (CSS)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide only the close button INSIDE the sidebar.
       Sidebar stays permanently open → the keyboard_double_arrow toggle
       in the header never appears. No header CSS needed. */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
    }

    /* Clean light background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        color: #334155;
        font-size: 0.92rem;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 50%, #eff6ff 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 400;
        margin: 0.35rem 0 0.75rem 0;
    }
    .badge-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .meta-badge {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
    }
    .meta-badge.highlight {
        background: #e0f2fe;
        border-color: #bae6fd;
        color: #0369a1;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.35rem;
    }
    .val-red { color: #dc2626; }
    .val-green { color: #059669; }
    .val-blue { color: #0284c7; }
    .val-amber { color: #d97706; }
    .val-purple { color: #7c3aed; }
    .val-dark { color: #0f172a; }

    /* Info / Content Card */
    .content-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .card-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Badges */
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-active {
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }
    .pill-break {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .pill-safe {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .pill-danger {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* Custom Section Header */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1.2rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }

    /* Grid for Inspector */
    .inspector-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin-top: 0.6rem;
    }
    .inspector-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        text-align: center;
    }
    .inspector-box .box-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .inspector-box .box-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Pipeline Step Box */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin: 1rem 0;
    }
    .pipeline-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        position: relative;
    }
    .pipeline-step-num {
        background: #0284c7;
        color: #ffffff;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .pipeline-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
    }
    .pipeline-desc {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Plotly chart container */
    .stPlotlyChart {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. Data & Model Loading (Cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    df = pd.read_csv("final_dataset_with_bust.csv")
    df["init_date"] = pd.to_datetime(df["init_date"])
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["date_str"] = df["valid_date"].dt.strftime("%Y-%m-%d")
    return df

@st.cache_resource
def load_models():
    regime_model = joblib.load("regime_classifier.pkl")
    bust_bundle = joblib.load("bust_classifier.pkl")
    return regime_model, bust_bundle["model"], bust_bundle.get("threshold", 0.60)

# Load real artifacts
df = load_dataset()
regime_clf, bust_clf, BUST_THRESHOLD = load_models()

# Calculate overall scientific metrics from real data
RAW_RMSE = np.sqrt(((df["forecast_precip_mm"] - df["observed_precip_mm"]) ** 2).mean())
CORRECTED_RMSE = np.sqrt(((df["corrected_precip_mm"] - df["observed_precip_mm"]) ** 2).mean())
IMPROVEMENT_PCT = ((RAW_RMSE - CORRECTED_RMSE) / RAW_RMSE) * 100
REGIME_ACC_TEST = 83.0  # Held-out test evaluation score
BUST_RECALL_TEST = 71.0  # Recall score at chosen 0.60 probability threshold

# Plotly Light Layout Theme
# NOTE: xaxis/yaxis are intentionally kept separate so they can be
# merged safely with per-chart overrides using update_xaxes/update_yaxes.
LIGHT_THEME = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    margin=dict(l=45, r=25, t=35, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=11, color="#334155"),
    ),
)

# Default axis style — applied via fig.update_xaxes() / fig.update_yaxes()
# to avoid conflicting with explicit xaxis= / yaxis= kwargs in update_layout().
AXIS_STYLE = dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", showgrid=True)


# ─────────────────────────────────────────────
# 4. Sidebar Navigation & Branding
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size: 1.6rem;">️</span>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; line-height: 1.1;">RAINCAST AI</div>
                <div style="font-size: 0.72rem; font-weight: 600; color: #0284c7; text-transform: uppercase;"></div>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.4rem; line-height: 1.3;">
            Regime-Aware Rainfall Forecast Correction & Bust Risk Warning System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Forecast Explorer",
            "3. Regime Intelligence",
            "4. Risk Monitor",
            "5. Forecast Analysis",
            "6. Data Explorer",
            "7. How It Works",
        ],
        index=0,
    )

    st.markdown("---")

    # Sidebar Operational Context Box
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem; font-size: 0.76rem; color: #475569;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.3rem;"> Operational Scope</div>
        <div>• <b>Region:</b> Maharashtra (Bounding Box)</div>
        <div>• <b>Forecast:</b> 72h / Day-3 Lead (GFS 00Z)</div>
        <div>• <b>Verification:</b> NASA GPM IMERG</div>
        <div>• <b>Season:</b> Monsoon 2023 (Jun–Sep)</div>
        <div>• <b>Sample Size:</b> 119 Forecast Days</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Real Metrics Quick Summary
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.8rem; font-size: 0.74rem;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;"> Validated Benchmark</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.15rem;">
            <span style="color:#64748b;">Raw GFS RMSE:</span>
            <span style="font-weight:700; color:#dc2626;">9.85 mm</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.15rem;">
            <span style="color:#64748b;">AI Corrected:</span>
            <span style="font-weight:700; color:#059669;">6.35 mm</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#64748b;">Error Reduction:</span>
            <span style="font-weight:700; color:#0284c7;">~35%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "1. Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">️ RainCast AI Operational Overview</div>
        <div class="hero-subtitle">
            Regime-Aware Post-Processing Pipeline for Numerical Weather Prediction over Maharashtra
        </div>
        <div class="badge-container">
            <span class="meta-badge highlight"> Region: Maharashtra</span>
            <span class="meta-badge"> Valid Range: Jun 04 – Sep 30, 2023</span>
            <span class="meta-badge">⏱️ Lead Time: 72 Hours (Day-3)</span>
            <span class="meta-badge">️ Baseline: NOAA GFS 0.25° vs NASA IMERG</span>
            <span class="meta-badge highlight"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5 Project Headline KPIs
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Raw GFS RMSE</div>
            <div class="kpi-value val-red">{RAW_RMSE:.2f} <span style="font-size:0.9rem; font-weight:500;">mm</span></div>
            <div class="kpi-subtext">Baseline 72h forecast error</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">AI Corrected RMSE</div>
            <div class="kpi-value val-green">{CORRECTED_RMSE:.2f} <span style="font-size:0.9rem; font-weight:500;">mm</span></div>
            <div class="kpi-subtext">RainCast post-processed</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">RMSE Improvement</div>
            <div class="kpi-value val-blue">~35%</div>
            <div class="kpi-subtext">35.5% precision gain</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Regime Accuracy</div>
            <div class="kpi-value val-amber">83%</div>
            <div class="kpi-subtext">Active vs Break detection</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Bust Recall</div>
            <div class="kpi-value val-purple">71%</div>
            <div class="kpi-subtext">Severe forecast busts caught</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Main Monsoon Rainfall Overview Chart
    st.markdown('<div class="section-title"> Complete Season Forecast vs Observed Rainfall (Maharashtra, 2023)</div>', unsafe_allow_html=True)

    fig_overview = go.Figure()

    # Observed IMERG
    fig_overview.add_trace(go.Scatter(
        x=df["valid_date"],
        y=df["observed_precip_mm"],
        name="Observed Rainfall (IMERG)",
        mode="lines",
        line=dict(color="#059669", width=2.2),
        fill="tozeroy",
        fillcolor="rgba(5, 150, 105, 0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Observed: <b>%{y:.2f} mm</b><extra></extra>",
    ))

    # Raw GFS Forecast
    fig_overview.add_trace(go.Scatter(
        x=df["valid_date"],
        y=df["forecast_precip_mm"],
        name="Raw GFS Forecast (Day-3)",
        mode="lines",
        line=dict(color="#dc2626", width=1.6, dash="dot"),
        hovertemplate="Raw GFS: <b>%{y:.2f} mm</b><extra></extra>",
    ))

    # AI Corrected Forecast
    fig_overview.add_trace(go.Scatter(
        x=df["valid_date"],
        y=df["corrected_precip_mm"],
        name="RainCast AI Corrected",
        mode="lines",
        line=dict(color="#0284c7", width=2.4),
        hovertemplate="AI Corrected: <b>%{y:.2f} mm</b><extra></extra>",
    ))

    fig_overview.update_layout(
        **LIGHT_THEME,
        height=380,
        yaxis_title="Precipitation (mm / day)",
        xaxis_title="Forecast Valid Date",
        hovermode="x unified",
    )
    fig_overview.update_xaxes(**AXIS_STYLE)
    fig_overview.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_overview, use_container_width=True)

    # 3 Summary Cards below chart
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="content-card">
            <div class="card-title"> The Monsoon Challenge</div>
            <div style="font-size: 0.83rem; color: #475569; line-height: 1.45;">
                Numerical Weather Prediction models like GFS systematically miscalculate precipitation during monsoon shifts. In <b>Active phases</b>, raw models overpredict or miss synoptic surges, while in <b>Break phases</b>, dry-spell biases cause false alarms.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="content-card">
            <div class="card-title"> Regime-Aware Correction</div>
            <div style="font-size: 0.83rem; color: #475569; line-height: 1.45;">
                Instead of static scalar smoothing, RainCast AI dynamically detects the atmospheric state (<b>Active</b> vs <b>Break</b>) with a Random Forest model and applies state-conditioned bias offsets, lowering RMSE from <b>9.85 mm</b> to <b>6.35 mm</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">️ Early Bust Risk Warning</div>
            <div style="font-size: 0.83rem; color: #475569; line-height: 1.45;">
                A specialized secondary Logistic Regression classifier scans forecast conditions to estimate the probability of a <b>Forecast Bust</b> (>6 mm error). With <b>71% recall</b>, authorities receive actionable flags before severe forecast anomalies occur.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Operational Breakdown & Quick Insights
    st.markdown('<div class="section-title"> Operational Summary & Seasonal Composition</div>', unsafe_allow_html=True)
    o1, o2 = st.columns([1, 1])

    with o1:
        # Regime Distribution Donut
        reg_counts = df["predicted_regime"].str.capitalize().value_counts()
        fig_donut = go.Figure(data=[go.Pie(
            labels=reg_counts.index,
            values=reg_counts.values,
            hole=0.6,
            marker=dict(colors=["#d97706", "#0284c7"]),
            textinfo="label+percent",
            textfont=dict(size=12, color="#0f172a"),
        )])
        fig_donut.update_layout(
            **LIGHT_THEME,
            height=260,
            showlegend=False,
            annotations=[dict(text="119 Days<br>Monitored", x=0.5, y=0.5, font_size=13, font_family="Inter", showarrow=False, font_color="#334155")]
        )
        st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#334155; margin-bottom:0.2rem;'>Monsoon Regime Distribution (Predicted)</div>", unsafe_allow_html=True)
        fig_donut.update_xaxes(**AXIS_STYLE)
        fig_donut.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_donut, use_container_width=True)

    with o2:
        # Forecast Bust Risk vs Normal Days
        bust_counts = pd.Series({
            "Normal (Low Risk)": (df["is_bust"] == 0).sum(),
            "Flagged Bust Risk": (df["is_bust"] == 1).sum()
        })
        fig_bust_pie = go.Figure(data=[go.Pie(
            labels=bust_counts.index,
            values=bust_counts.values,
            hole=0.6,
            marker=dict(colors=["#059669", "#dc2626"]),
            textinfo="label+percent",
            textfont=dict(size=12, color="#0f172a"),
        )])
        fig_bust_pie.update_layout(
            **LIGHT_THEME,
            height=260,
            showlegend=False,
            annotations=[dict(text="Risk Alert<br>Status", x=0.5, y=0.5, font_size=13, font_family="Inter", showarrow=False, font_color="#334155")]
        )
        st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#334155; margin-bottom:0.2rem;'>Forecast Bust Risk Allocation</div>", unsafe_allow_html=True)
        fig_bust_pie.update_xaxes(**AXIS_STYLE)
        fig_bust_pie.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_bust_pie, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2: FORECAST EXPLORER
# ─────────────────────────────────────────────
elif page == "2. Forecast Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Interactive Forecast Explorer</div>
        <div class="hero-subtitle">
            Inspect Day-3 (72h Lead) Raw GFS vs AI-Corrected Forecasts against Observed NASA IMERG precipitation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Date Selection Controls
    col_date_picker, col_range = st.columns([1, 2])

    min_d = df["valid_date"].min().date()
    max_d = df["valid_date"].max().date()

    with col_date_picker:
        selected_date = st.date_input(
            "Select Single Inspection Date",
            value=date(2023, 7, 24),
            min_value=min_d,
            max_value=max_d,
            help="Choose a valid forecast date from the 2023 monsoon season."
        )

    with col_range:
        selected_range = st.date_input(
            "Filter Time-Series Window",
            value=(date(2023, 7, 1), date(2023, 8, 31)),
            min_value=min_d,
            max_value=max_d,
            help="Select start and end dates for the interactive chart below."
        )

    # Fetch Row for Single Date
    row_match = df[df["valid_date"].dt.date == selected_date]

    if not row_match.empty:
        r = row_match.iloc[0]
        init_d_str = r["init_date"].strftime("%d %B %Y")
        valid_d_str = r["valid_date"].strftime("%A, %d %B %Y")

        regime_label = r["predicted_regime"].upper()
        regime_pill = "pill-active" if regime_label == "ACTIVE" else "pill-break"

        is_bust = r["is_bust"] == 1
        bust_pill = "pill-danger" if is_bust else "pill-safe"
        bust_status_text = "️ HIGH RISK (Potential Bust)" if is_bust else " NORMAL (Reliable Forecast)"

        raw_err = r["forecast_error_mm"]
        corr_err = r["corrected_error_mm"]
        abs_err_red = abs(raw_err) - abs(corr_err)

        st.markdown(f"""
        <div class="content-card" style="border-left: 4px solid #0284c7;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom: 0.6rem;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #0f172a;"> Forecast for: {valid_d_str}</span>
                    <span style="font-size: 0.8rem; color: #64748b; margin-left: 0.5rem;">(Initialized: {init_d_str} at 00Z — 72h Lead)</span>
                </div>
                <div>
                    <span class="status-pill {regime_pill}">Regime: {regime_label}</span>
                    <span class="status-pill {bust_pill}" style="margin-left:0.3rem;">{bust_status_text}</span>
                </div>
            </div>
            <div class="inspector-grid">
                <div class="inspector-box">
                    <div class="box-label">Raw GFS Forecast</div>
                    <div class="box-value val-red">{r['forecast_precip_mm']:.2f} mm</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Error: {raw_err:+.2f} mm</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">AI-Corrected</div>
                    <div class="box-value val-blue">{r['corrected_precip_mm']:.2f} mm</div>
                    <div style="font-size:0.72rem; color:#059669; font-weight:600; margin-top:2px;">Error: {corr_err:+.2f} mm</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">Observed (IMERG)</div>
                    <div class="box-value val-green">{r['observed_precip_mm']:.2f} mm</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Ground Truth</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">Regime Confidence</div>
                    <div class="box-value val-amber">{r['regime_confidence']:.0%}</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">RandomForest Prob</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">Bust Probability</div>
                    <div class="box-value {'val-red' if is_bust else 'val-green'}">{r['bust_probability']:.1%}</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Threshold: 60%</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">Forecast Temp</div>
                    <div class="box-value val-dark">{r['forecast_temp_c']:.1f} °C</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">GFS Surface TMP</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Selected date is outside the June–September 2023 dataset range.")

    # Filter Time-Series Window
    if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
        s_start, s_end = selected_range
        df_window = df[(df["valid_date"].dt.date >= s_start) & (df["valid_date"].dt.date <= s_end)]
    else:
        df_window = df

    st.markdown('<div class="section-title"> Time-Series Window Analysis</div>', unsafe_allow_html=True)

    fig_win = go.Figure()

    fig_win.add_trace(go.Scatter(
        x=df_window["valid_date"],
        y=df_window["observed_precip_mm"],
        name="Observed (IMERG)",
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(color="#059669", width=2),
        fill="tozeroy",
        fillcolor="rgba(5, 150, 105, 0.07)",
        hovertemplate="%{x|%d %b}: Observed <b>%{y:.2f} mm</b><extra></extra>",
    ))

    fig_win.add_trace(go.Scatter(
        x=df_window["valid_date"],
        y=df_window["forecast_precip_mm"],
        name="Raw GFS (72h)",
        mode="lines",
        line=dict(color="#dc2626", width=1.5, dash="dot"),
        hovertemplate="Raw GFS: <b>%{y:.2f} mm</b><extra></extra>",
    ))

    fig_win.add_trace(go.Scatter(
        x=df_window["valid_date"],
        y=df_window["corrected_precip_mm"],
        name="AI Corrected",
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(color="#0284c7", width=2.2),
        hovertemplate="AI Corrected: <b>%{y:.2f} mm</b><extra></extra>",
    ))

    # Highlight selected inspection date
    if not row_match.empty:
        fig_win.add_vline(
            x=r["valid_date"],
            line_width=1.5,
            line_dash="dash",
            line_color="#7c3aed",
            annotation_text="Inspected Date",
            annotation_position="top right",
            annotation_font=dict(color="#7c3aed", size=11),
        )

    fig_win.update_layout(
        **LIGHT_THEME,
        height=380,
        yaxis_title="Precipitation (mm)",
        xaxis_title="Valid Date",
        hovermode="x unified",
    )
    fig_win.update_xaxes(**AXIS_STYLE)
    fig_win.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_win, use_container_width=True)

    # Window metrics & Daily Error Delta
    col_err_chart, col_cum_chart = st.columns(2)

    with col_err_chart:
        st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#334155; margin-bottom:0.2rem;'>Daily Error Comparison (Raw Error vs Corrected Error)</div>", unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_window["valid_date"],
            y=df_window["forecast_error_mm"],
            name="Raw Forecast Error",
            marker_color="rgba(220, 38, 38, 0.6)",
        ))
        fig_bar.add_trace(go.Bar(
            x=df_window["valid_date"],
            y=df_window["corrected_error_mm"],
            name="AI Corrected Error",
            marker_color="rgba(2, 132, 199, 0.8)",
        ))
        fig_bar.update_layout(
            **LIGHT_THEME,
            height=280,
            barmode="group",
            yaxis_title="Error (mm)",
            hovermode="x unified",
        )
        fig_bar.update_xaxes(**AXIS_STYLE)
        fig_bar.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_cum_chart:
        st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#334155; margin-bottom:0.2rem;'>Cumulative Seasonal Precipitation Tracking</div>", unsafe_allow_html=True)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_window["valid_date"],
            y=df_window["observed_precip_mm"].cumsum(),
            name="Observed Cumulative",
            line=dict(color="#059669", width=2.2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=df_window["valid_date"],
            y=df_window["forecast_precip_mm"].cumsum(),
            name="Raw GFS Cumulative",
            line=dict(color="#dc2626", width=1.6, dash="dot"),
        ))
        fig_cum.add_trace(go.Scatter(
            x=df_window["valid_date"],
            y=df_window["corrected_precip_mm"].cumsum(),
            name="AI Corrected Cumulative",
            line=dict(color="#0284c7", width=2.2),
        ))
        fig_cum.update_layout(
            **LIGHT_THEME,
            height=280,
            yaxis_title="Accumulated Rain (mm)",
            hovermode="x unified",
        )
        fig_cum.update_xaxes(**AXIS_STYLE)
        fig_cum.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_cum, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3: REGIME INTELLIGENCE
# ─────────────────────────────────────────────
elif page == "3. Regime Intelligence":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Monsoon Regime Intelligence</div>
        <div class="hero-subtitle">
            Synoptic monsoon regime classification: Distinguishing between Active (convective) and Break (suppressed) states.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Regime Explanatory Cards
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("""
        <div class="content-card" style="border-top: 3px solid #0284c7;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.05rem; font-weight:700; color:#0369a1;"> Active Monsoon Regime</span>
                <span class="status-pill pill-active">High Convection</span>
            </div>
            <div style="font-size:0.83rem; color:#475569; margin-top:0.6rem; line-height:1.45;">
                • Characterized by vigorous south-westerly low-level winds, deep offshore troughs, and heavy widespread precipitation across coastal & ghat regions.<br>
                • <b>NWP Model Behavior:</b> Raw GFS frequently over-predicts rainfall intensity or displaces storm centers during peak convective bursts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        st.markdown("""
        <div class="content-card" style="border-top: 3px solid #d97706;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.05rem; font-weight:700; color:#b45309;">️ Break Monsoon Regime</span>
                <span class="status-pill pill-break">Suppressed Convection</span>
            </div>
            <div style="font-size:0.83rem; color:#475569; margin-top:0.6rem; line-height:1.45;">
                • Monsoon trough shifts northward toward the Himalayas; rainfall across central India and Maharashtra diminishes into dry spells.<br>
                • <b>NWP Model Behavior:</b> Numerical models struggle with localized thermal convection and falsely trigger rain cells, leading to positive bias.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Regime Classification Confidence & Timeline</div>', unsafe_allow_html=True)

    # Regime Confidence Chart
    fig_reg_conf = go.Figure()

    # Color background by regime
    colors_reg = np.where(df["predicted_regime"] == "active", "#0284c7", "#d97706")

    fig_reg_conf.add_trace(go.Bar(
        x=df["valid_date"],
        y=df["regime_confidence"],
        marker_color=colors_reg,
        name="Regime Confidence",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Regime: <b>%{customdata}</b><br>Confidence: <b>%{y:.1%}</b><extra></extra>",
        customdata=df["predicted_regime"].str.capitalize(),
    ))

    fig_reg_conf.add_hline(
        y=0.70,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="High Confidence Threshold (70%)",
        annotation_position="top left",
        annotation_font=dict(color="#64748b", size=11),
    )

    fig_reg_conf.update_layout(
        **LIGHT_THEME,
        height=320,
        yaxis_range=[0.4, 1.05],
        yaxis_title="Confidence Score (0–1)",
        yaxis_gridcolor="#e2e8f0",
        xaxis_title="Date",
        showlegend=False,
    )
    fig_reg_conf.update_xaxes(**AXIS_STYLE)
    fig_reg_conf.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_reg_conf, use_container_width=True)

    # Statistical Comparison across Regimes
    st.markdown('<div class="section-title"> Statistical Breakdown by Predicted Regime</div>', unsafe_allow_html=True)

    col_box, col_stats = st.columns([1, 1])

    with col_box:
        # Rainfall distribution by regime
        fig_box = px.box(
            df,
            x="predicted_regime",
            y="observed_precip_mm",
            color="predicted_regime",
            color_discrete_map={"active": "#0284c7", "break": "#d97706"},
            labels={"predicted_regime": "Predicted Regime", "observed_precip_mm": "Observed Rain (mm)"},
            category_orders={"predicted_regime": ["active", "break"]},
        )
        fig_box.update_layout(
            **LIGHT_THEME,
            height=290,
            showlegend=False,
            xaxis_title="Monsoon Regime",
        )
        st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#334155; margin-bottom:0.2rem;'>Observed Rainfall Distribution by Regime</div>", unsafe_allow_html=True)
        fig_box.update_xaxes(**AXIS_STYLE)
        fig_box.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_stats:
        active_df = df[df["predicted_regime"] == "active"]
        break_df = df[df["predicted_regime"] == "break"]

        st.markdown(f"""
        <div class="content-card" style="height: 290px; overflow-y: auto;">
            <div class="card-title"> Regime Comparative Summary</div>
            <table style="width:100%; font-size:0.82rem; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid #e2e8f0; color:#64748b;">
                    <th style="text-align:left; padding:4px 0;">Metric</th>
                    <th style="text-align:center; color:#0369a1;">Active (53 days)</th>
                    <th style="text-align:center; color:#b45309;">Break (66 days)</th>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding:6px 0;">Mean Observed Rain</td>
                    <td style="text-align:center; font-weight:700;">{active_df['observed_precip_mm'].mean():.2f} mm</td>
                    <td style="text-align:center; font-weight:700;">{break_df['observed_precip_mm'].mean():.2f} mm</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding:6px 0;">Mean Raw GFS Rain</td>
                    <td style="text-align:center; font-weight:700; color:#dc2626;">{active_df['forecast_precip_mm'].mean():.2f} mm</td>
                    <td style="text-align:center; font-weight:700; color:#dc2626;">{break_df['forecast_precip_mm'].mean():.2f} mm</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding:6px 0;">Mean AI Corrected Rain</td>
                    <td style="text-align:center; font-weight:700; color:#0284c7;">{active_df['corrected_precip_mm'].mean():.2f} mm</td>
                    <td style="text-align:center; font-weight:700; color:#0284c7;">{break_df['corrected_precip_mm'].mean():.2f} mm</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding:6px 0;">Raw GFS RMSE</td>
                    <td style="text-align:center; font-weight:700; color:#dc2626;">13.63 mm</td>
                    <td style="text-align:center; font-weight:700; color:#5.30 mm</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;">AI Corrected RMSE</td>
                    <td style="text-align:center; font-weight:700; color:#059669;">8.30 mm (-39%)</td>
                    <td style="text-align:center; font-weight:700; color:#059669;">4.27 mm (-19%)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 4: RISK MONITOR (BUST PREDICTION)
# ─────────────────────────────────────────────
elif page == "4. Risk Monitor":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">️ Forecast Bust & Risk Monitor</div>
        <div class="hero-subtitle">
            Early warning detection for extreme NWP forecast failures (Forecast Busts defined as absolute error > 6.0 mm).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics for Risk
    total_busts = (df["is_bust"] == 1).sum()
    pct_busts = (total_busts / len(df)) * 100
    mean_risk_prob = df["bust_probability"].mean()

    rk1, rk2, rk3, rk4 = st.columns(4)

    with rk1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Forecast Bust Events</div>
            <div class="kpi-value val-red">{total_busts} <span style="font-size:0.85rem; font-weight:500;">/ 119</span></div>
            <div class="kpi-subtext">Flagged at probability > 60%</div>
        </div>
        """, unsafe_allow_html=True)

    with rk2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Bust Detection Recall</div>
            <div class="kpi-value val-purple">{BUST_RECALL_TEST:.0f}%</div>
            <div class="kpi-subtext">High sensitivity for disaster ops</div>
        </div>
        """, unsafe_allow_html=True)

    with rk3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Operational Threshold</div>
            <div class="kpi-value val-blue">{BUST_THRESHOLD:.2f}</div>
            <div class="kpi-subtext">Optimized decision boundary</div>
        </div>
        """, unsafe_allow_html=True)

    with rk4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Bust Risk</div>
            <div class="kpi-value val-amber">{mean_risk_prob:.1%}</div>
            <div class="kpi-subtext">Season baseline risk level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Forecast Bust Probability Timeline</div>', unsafe_allow_html=True)

    # Risk Timeline Chart
    fig_risk = go.Figure()

    # Risk curve
    fig_risk.add_trace(go.Scatter(
        x=df["valid_date"],
        y=df["bust_probability"],
        name="Bust Probability",
        mode="lines",
        line=dict(color="#dc2626", width=2),
        fill="tozeroy",
        fillcolor="rgba(220, 38, 38, 0.07)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Bust Prob: <b>%{y:.1%}</b><extra></extra>",
    ))

    # Threshold Line
    fig_risk.add_hline(
        y=BUST_THRESHOLD,
        line_dash="dash",
        line_color="#d97706",
        annotation_text=f"Alert Threshold ({BUST_THRESHOLD:.0%})",
        annotation_position="top right",
        annotation_font=dict(color="#d97706", size=11),
    )

    # Mark high-risk days
    high_risk_df = df[df["is_bust"] == 1]
    fig_risk.add_trace(go.Scatter(
        x=high_risk_df["valid_date"],
        y=high_risk_df["bust_probability"],
        name="Flagged High Risk Days",
        mode="markers",
        marker=dict(color="#dc2626", size=7, symbol="diamond"),
        hovertemplate="️ HIGH RISK ALERT<br>%{x|%d %b}: Prob <b>%{y:.1%}</b><extra></extra>",
    ))

    fig_risk.update_layout(
        **LIGHT_THEME,
        height=340,
        yaxis_range=[0, 1.05],
        yaxis_title="Bust Probability (0–1)",
        yaxis_gridcolor="#e2e8f0",
        xaxis_title="Valid Date",
        hovermode="x unified",
    )
    fig_risk.update_xaxes(**AXIS_STYLE)
    fig_risk.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_risk, use_container_width=True)

    # Actionable Table for Flagged Risk Days
    st.markdown('<div class="section-title"> Flagged High-Risk Days & Advisory Guide</div>', unsafe_allow_html=True)

    tab_risk1, tab_risk2 = st.tabs(["High-Risk Days Register", "Operational Response Protocols"])

    with tab_risk1:
        risk_table_df = high_risk_df[[
            "valid_date", "forecast_precip_mm", "corrected_precip_mm", "observed_precip_mm",
            "predicted_regime", "bust_probability", "abs_corrected_error"
        ]].copy()

        risk_table_df["valid_date"] = risk_table_df["valid_date"].dt.strftime("%Y-%m-%d")
        risk_table_df = risk_table_df.rename(columns={
            "valid_date": "Date",
            "forecast_precip_mm": "Raw GFS (mm)",
            "corrected_precip_mm": "AI Corrected (mm)",
            "observed_precip_mm": "Observed (mm)",
            "predicted_regime": "Regime",
            "bust_probability": "Bust Risk (%)",
            "abs_corrected_error": "Abs Error (mm)",
        })
        risk_table_df["Bust Risk (%)"] = (risk_table_df["Bust Risk (%)"] * 100).round(1)

        st.dataframe(
            risk_table_df.style.format({
                "Raw GFS (mm)": "{:.2f}",
                "AI Corrected (mm)": "{:.2f}",
                "Observed (mm)": "{:.2f}",
                "Bust Risk (%)": "{:.1f}%",
                "Abs Error (mm)": "{:.2f}",
            }),
            use_container_width=True,
            height=280,
        )

    with tab_risk2:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:0.9rem; font-weight:700; color:#0f172a; margin-bottom:0.4rem;">Action Protocols for Disaster Management Agencies (SDRF / NDMA / Agriculture)</div>
            <div style="font-size:0.83rem; color:#475569; line-height:1.5;">
                • <b>Probability > 70%:</b> Issue cautionary advisory to district disaster units. Ensemble divergence is high. Do not rely solely on deterministic GFS thresholds for reservoir discharge planning.<br>
                • <b>Probability 60% – 70%:</b> Cross-verify with nowcasting radars and regional AWS stations 12–24 hours before event onset.<br>
                • <b>Active Regime Bust Risk:</b> High likelihood of extreme localized downpours exceeding model grids. Prepare urban drainage teams in Mumbai, Pune, and Konkan belt.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 5: FORECAST ANALYSIS
# ─────────────────────────────────────────────
elif page == "5. Forecast Analysis":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Forecast Performance Analysis</div>
        <div class="hero-subtitle">
            Validation of post-processing accuracy, error distributions, and benchmark reductions across Maharashtra.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scientific Performance Comparison Table
    st.markdown('<div class="section-title"> Metric Verification Matrix (Day-3 / 72h Forecast)</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    raw_mae = (df["forecast_precip_mm"] - df["observed_precip_mm"]).abs().mean()
    corr_mae = (df["corrected_precip_mm"] - df["observed_precip_mm"]).abs().mean()
    raw_bias = (df["forecast_precip_mm"] - df["observed_precip_mm"]).mean()
    corr_bias = (df["corrected_precip_mm"] - df["observed_precip_mm"]).mean()

    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">RMSE Comparison</div>
            <div style="font-size:1.15rem; font-weight:700; color:#dc2626;">Raw: {RAW_RMSE:.2f} mm</div>
            <div style="font-size:1.25rem; font-weight:800; color:#059669; margin-top:2px;">AI: {CORRECTED_RMSE:.2f} mm</div>
            <div class="kpi-subtext">35.5% Error Reduction</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">MAE Comparison</div>
            <div style="font-size:1.15rem; font-weight:700; color:#dc2626;">Raw: {raw_mae:.2f} mm</div>
            <div style="font-size:1.25rem; font-weight:800; color:#059669; margin-top:2px;">AI: {corr_mae:.2f} mm</div>
            <div class="kpi-subtext">28.9% Mean Abs Gain</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Systematic Bias</div>
            <div style="font-size:1.15rem; font-weight:700; color:#dc2626;">Raw: {raw_bias:+.2f} mm</div>
            <div style="font-size:1.25rem; font-weight:800; color:#0284c7; margin-top:2px;">AI: {corr_bias:+.2f} mm</div>
            <div class="kpi-subtext">Near-Zero Residual Bias</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Regime Accuracy</div>
            <div style="font-size:1.15rem; font-weight:700; color:#d97706;">Test Acc: 83%</div>
            <div style="font-size:1.25rem; font-weight:800; color:#7c3aed; margin-top:2px;">Recall: 71%</div>
            <div class="kpi-subtext">Busts Detected at 0.60</div>
        </div>
        """, unsafe_allow_html=True)

    # Scatter Plots: Raw vs Corrected against Observed
    st.markdown('<div class="section-title"> Scatter Plot: Forecast vs Observed Ground Truth</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    max_val = max(df["observed_precip_mm"].max(), df["forecast_precip_mm"].max(), df["corrected_precip_mm"].max()) + 2

    with sc1:
        fig_sc_raw = go.Figure()
        fig_sc_raw.add_trace(go.Scatter(
            x=df["observed_precip_mm"],
            y=df["forecast_precip_mm"],
            mode="markers",
            marker=dict(color="#dc2626", size=6, opacity=0.7),
            name="Raw GFS",
            hovertemplate="Observed: %{x:.2f} mm<br>Raw GFS: %{y:.2f} mm<extra></extra>",
        ))
        # 1:1 Reference Line
        fig_sc_raw.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash"),
            name="Perfect 1:1 Fit",
        ))
        fig_sc_raw.update_layout(
            **LIGHT_THEME,
            height=320,
            title="Raw GFS vs Observed (High Dispersion)",
            xaxis_title="Observed IMERG (mm)",
            yaxis_title="Raw Forecast (mm)",
            showlegend=False,
        )
        fig_sc_raw.update_xaxes(**AXIS_STYLE)
        fig_sc_raw.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_sc_raw, use_container_width=True)

    with sc2:
        fig_sc_corr = go.Figure()
        fig_sc_corr.add_trace(go.Scatter(
            x=df["observed_precip_mm"],
            y=df["corrected_precip_mm"],
            mode="markers",
            marker=dict(color="#0284c7", size=6, opacity=0.7),
            name="AI Corrected",
            hovertemplate="Observed: %{x:.2f} mm<br>AI Corrected: %{y:.2f} mm<extra></extra>",
        ))
        # 1:1 Reference Line
        fig_sc_corr.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash"),
            name="Perfect 1:1 Fit",
        ))
        fig_sc_corr.update_layout(
            **LIGHT_THEME,
            height=320,
            title="RainCast AI vs Observed (Tighter Alignment)",
            xaxis_title="Observed IMERG (mm)",
            yaxis_title="Corrected Forecast (mm)",
            showlegend=False,
        )
        fig_sc_corr.update_xaxes(**AXIS_STYLE)
        fig_sc_corr.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_sc_corr, use_container_width=True)

    # Error Distribution Histograms
    st.markdown('<div class="section-title"> Error Distribution Analysis (Residuals)</div>', unsafe_allow_html=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["forecast_error_mm"],
        name="Raw GFS Error",
        nbinsx=35,
        marker_color="rgba(220, 38, 38, 0.5)",
        opacity=0.75,
    ))
    fig_hist.add_trace(go.Histogram(
        x=df["corrected_error_mm"],
        name="AI Corrected Error",
        nbinsx=35,
        marker_color="rgba(2, 132, 199, 0.65)",
        opacity=0.75,
    ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#0f172a", line_width=1.2)
    fig_hist.update_layout(
        **LIGHT_THEME,
        height=290,
        barmode="overlay",
        xaxis_title="Forecast Error (Forecast - Observed in mm)",
        yaxis_title="Count of Days",
        hovermode="x",
    )
    fig_hist.update_xaxes(**AXIS_STYLE)
    fig_hist.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 6: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "6. Data Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Maharashtra Monsoon Dataset Explorer</div>
        <div class="hero-subtitle">
            Filter, inspect, and export verified forecast and satellite ground-truth records (119 observations).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filtering Controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        regime_filter = st.selectbox("Filter by Regime", ["All Regimes", "Active", "Break"])

    with col_f2:
        bust_filter = st.selectbox("Filter by Risk Status", ["All Statuses", "High Risk (Bust)", "Normal (Safe)"])

    with col_f3:
        min_rain = st.number_input("Min Observed Rain (mm)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

    with col_f4:
        sort_by = st.selectbox("Sort Table By", ["Valid Date (Asc)", "Valid Date (Desc)", "Observed Rain (High-Low)", "Error (High-Low)"])

    # Apply Filters
    df_filtered = df.copy()

    if regime_filter != "All Regimes":
        df_filtered = df_filtered[df_filtered["predicted_regime"] == regime_filter.lower()]

    if bust_filter == "High Risk (Bust)":
        df_filtered = df_filtered[df_filtered["is_bust"] == 1]
    elif bust_filter == "Normal (Safe)":
        df_filtered = df_filtered[df_filtered["is_bust"] == 0]

    if min_rain > 0:
        df_filtered = df_filtered[df_filtered["observed_precip_mm"] >= min_rain]

    # Apply Sorting
    if sort_by == "Valid Date (Asc)":
        df_filtered = df_filtered.sort_values("valid_date", ascending=True)
    elif sort_by == "Valid Date (Desc)":
        df_filtered = df_filtered.sort_values("valid_date", ascending=False)
    elif sort_by == "Observed Rain (High-Low)":
        df_filtered = df_filtered.sort_values("observed_precip_mm", ascending=False)
    elif sort_by == "Error (High-Low)":
        df_filtered = df_filtered.sort_values("abs_corrected_error", ascending=False)

    # Filtered Summary Counter
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.8rem;">
        <span style="font-size:0.88rem; font-weight:700; color:#0f172a;">
            Showing <span style="color:#0284c7;">{len(df_filtered)}</span> of 119 records
        </span>
        <span style="font-size:0.8rem; color:#64748b;">
            Mean Obs: <b>{df_filtered['observed_precip_mm'].mean():.2f} mm</b> | Mean Corrected: <b>{df_filtered['corrected_precip_mm'].mean():.2f} mm</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Data Table View
    display_cols = [
        "init_date", "valid_date", "forecast_precip_mm", "forecast_temp_c",
        "observed_precip_mm", "predicted_regime", "corrected_precip_mm",
        "forecast_error_mm", "corrected_error_mm", "regime_confidence", "bust_probability", "is_bust"
    ]

    table_data = df_filtered[display_cols].copy()
    table_data["init_date"] = table_data["init_date"].dt.strftime("%Y-%m-%d")
    table_data["valid_date"] = table_data["valid_date"].dt.strftime("%Y-%m-%d")
    table_data["predicted_regime"] = table_data["predicted_regime"].str.upper()

    table_data = table_data.rename(columns={
        "init_date": "Init Date",
        "valid_date": "Valid Date",
        "forecast_precip_mm": "Raw GFS (mm)",
        "forecast_temp_c": "Temp (°C)",
        "observed_precip_mm": "Observed (mm)",
        "predicted_regime": "Regime",
        "corrected_precip_mm": "Corrected (mm)",
        "forecast_error_mm": "Raw Err (mm)",
        "corrected_error_mm": "Corr Err (mm)",
        "regime_confidence": "Regime Conf",
        "bust_probability": "Bust Prob",
        "is_bust": "Bust Alert",
    })

    st.dataframe(
        table_data.style.format({
            "Raw GFS (mm)": "{:.2f}",
            "Temp (°C)": "{:.1f}",
            "Observed (mm)": "{:.2f}",
            "Corrected (mm)": "{:.2f}",
            "Raw Err (mm)": "{:+.2f}",
            "Corr Err (mm)": "{:+.2f}",
            "Regime Conf": "{:.1%}",
            "Bust Prob": "{:.1%}",
            "Bust Alert": "{:d}",
        }),
        use_container_width=True,
        height=420,
    )

    # Download Filtered Data as CSV
    csv_bytes = table_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Filtered Records as CSV",
        data=csv_bytes,
        file_name=f"raincast_ai_filtered_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# PAGE 7: HOW IT WORKS & LIVE INFERENCE
# ─────────────────────────────────────────────
elif page == "7. How It Works":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">️ Architecture, Methodology & Live Inference Sandbox</div>
        <div class="hero-subtitle">
            Comprehensive breakdown of the RainCast AI post-processing pipeline with live on-demand ML inference.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5-Stage Pipeline Graphic
    st.markdown('<div class="section-title"> End-to-End System Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-grid">
        <div class="pipeline-card">
            <div class="pipeline-step-num">1</div>
            <div class="pipeline-title">NWP Ingestion</div>
            <div class="pipeline-desc">NOAA GFS 0.25° 00Z run at 72-hour (Day-3) lead time: APCP (Precipitation) & TMP (Temperature).</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step-num">2</div>
            <div class="pipeline-title">Satellite Ground Truth</div>
            <div class="pipeline-desc">NASA GPM IMERG Final Run calibration: Daily precipitation verification benchmark.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step-num">3</div>
            <div class="pipeline-title">Regime Classifier</div>
            <div class="pipeline-desc">RandomForest (200 trees) on [Precip, Temp, Month, DOY] to classify <b>Active</b> vs <b>Break</b>.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step-num">4</div>
            <div class="pipeline-title">Dynamic Bias Correction</div>
            <div class="pipeline-desc">Regime-stratified bias subtraction shifts raw GFS distribution, cutting RMSE from 9.85 to 6.35 mm.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step-num">5</div>
            <div class="pipeline-title">Bust Risk Alert</div>
            <div class="pipeline-desc">Logistic Regression model estimates probability of severe forecast error with 71% recall.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Model Inference Sandbox
    st.markdown('<div class="section-title"> Live Model Inference Sandbox</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.84rem; color:#475569; margin-bottom: 0.8rem;">
        Test the real production Random Forest and Logistic Regression models in real time by feeding custom inputs.
    </div>
    """, unsafe_allow_html=True)

    sb1, sb2, sb3, sb4 = st.columns(4)

    with sb1:
        test_rain = st.slider("Raw Forecast Rain (mm)", min_value=0.0, max_value=40.0, value=12.5, step=0.5)
    with sb2:
        test_temp = st.slider("Forecast Temp (°C)", min_value=15.0, max_value=38.0, value=26.5, step=0.5)
    with sb3:
        test_month = st.selectbox("Monsoon Month", [6, 7, 8, 9], index=1)
    with sb4:
        test_doy = st.slider("Day of Year (DOY)", min_value=152, max_value=273, value=205)

    if st.button(" Run Live RainCast Pipeline", type="primary"):
        # 1. Regime Classifier Inference
        regime_input = pd.DataFrame([{
            "forecast_precip_mm": test_rain,
            "forecast_temp_c": test_temp,
            "month": test_month,
            "doy": test_doy,
        }])
        pred_regime = regime_clf.predict(regime_input)[0]
        pred_proba_reg = regime_clf.predict_proba(regime_input)[0]
        reg_confidence = pred_proba_reg.max()

        # 2. Bias Correction Logic
        # Offsets learned from data: Active mean error = -6.44 mm (underpredict), Break mean error = -3.21 mm
        correction_offsets = {"active": -6.444983, "break": -3.212814}
        bias_offset = correction_offsets.get(pred_regime, -4.5)
        live_corrected_precip = max(0.0, test_rain - bias_offset)

        # 3. Bust Risk Classifier Inference
        bust_input = pd.DataFrame([{
            "forecast_precip_mm": test_rain,
            "regime_confidence": reg_confidence,
            "month": test_month,
            "doy": test_doy,
        }])
        bust_prob = bust_clf.predict_proba(bust_input)[0][1]
        is_live_bust = bust_prob > BUST_THRESHOLD

        res_pill = "pill-active" if pred_regime == "active" else "pill-break"
        risk_pill = "pill-danger" if is_live_bust else "pill-safe"

        st.markdown(f"""
        <div class="content-card" style="background:#ffffff; border-left: 4px solid #059669; margin-top:1rem;">
            <div style="font-size:1.05rem; font-weight:800; color:#0f172a; margin-bottom:0.6rem;">
                 Live Inference Output
            </div>
            <div class="inspector-grid">
                <div class="inspector-box">
                    <div class="box-label">Predicted Regime</div>
                    <div class="box-value"><span class="status-pill {res_pill}">{pred_regime.upper()}</span></div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Random Forest v2.1</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">Regime Confidence</div>
                    <div class="box-value val-blue">{reg_confidence:.1%}</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Model Certainty</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">AI Corrected Rain</div>
                    <div class="box-value val-green">{live_corrected_precip:.2f} mm</div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Shift: {bias_offset:+.2f} mm</div>
                </div>
                <div class="inspector-box">
                    <div class="box-label">Bust Risk Alert</div>
                    <div class="box-value"><span class="status-pill {risk_pill}">{'HIGH RISK' if is_live_bust else 'LOW RISK'}</span></div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Prob: {bust_prob:.1%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Scientific Documentation
    st.markdown('<div class="section-title">Scientific Documentation &amp; Mathematical Formulation</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 1. The Synoptic Regime Concept in Indian Monsoons
    The Indian Summer Monsoon (ISM) exhibits prominent intra-seasonal variability oscillating between **Active** spells (heavy rain over central India, active monsoon trough) and **Break** spells (trough near Himalayas, dry spell over central peninsula).

    NWP errors are inherently **non-stationary** and depend on the synoptic regime. Applying standard scalar linear regression across the entire season blurs the distinct physical error profiles of convective vs suppressed states.

    ### 2. Regime-Stratified Bias Correction
    Let $F_t$ be the raw numerical forecast at lead time $L=72\\text{h}$, and $Y_t$ be the observed IMERG precipitation.
    Let $R_t \\in \\{\\text{Active}, \\text{Break}\\}$ be the classified regime.
    The corrected forecast $\\hat{Y}_t$ is computed as:
    $$\\hat{Y}_t = \\max\\left(0, F_t - \\mu_{R_t}\\right)$$
    where $\\mu_{R}$ is the conditional mean systematic error for regime $R$:
    $$\\mu_{R} = \\mathbb{E}\\left[F_t - Y_t \\mid R_t = R\\right]$$

    ### 3. Forecast Bust Classification
    A forecast bust is defined as a case where the absolute error exceeds a critical operational threshold ($6.0\\text{ mm}$):
    $$B_t = \\mathbb{I}\\left(|\\hat{Y}_t - Y_t| > 6.0\\text{ mm}\\right)$$
    We train a regularized Logistic Regression classifier $P(B_t=1 \\mid X_t)$ with balanced class weights to maximize recall ($71\\%$), ensuring early warnings are triggered before severe operational forecasting errors occur.
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<hr style='margin-top:2rem; border-color:#e2e8f0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 0.2rem 0 1rem 0; color: #64748b; font-size: 0.76rem;">
    <b>RainCast AI</b> · · Meteorological Post-Processing & Disaster Support System<br>
    Validated on NOAA GFS 0.25° Forecasts & NASA GPM IMERG Ground Truth (Maharashtra, 2023)
</div>
""", unsafe_allow_html=True)
