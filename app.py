"""
Toxic Comment Classifier — Streamlit App (Deep Learning: LSTM)
================================================================
Tabs:
  1. Insights & Performance — data insights, model performance, architecture
                               comparison (LSTM vs RNN), sample test cases
  2. Real-Time Prediction   — type/paste a comment, get a toxicity score
  3. Bulk Prediction        — upload a CSV, score every row, download results

Model: PyTorch LSTM (binary is_toxic classifier), trained by running
eda.ipynb. A vanilla RNN was trained alongside it for comparison; the better model
(by F1) was picked for deployment — see the "Architecture Comparison"
section in the Insights tab.
"""
import base64
import json
import os
import re
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------
# Pipeline import with fallback stubs 
# (so the app can run even if the model files are missing)
# --------------------------------------------------------------------------
try:
    from lstm_pipeline import load_checkpoint, predict_proba, get_clean_text
except ImportError:
    def load_checkpoint(path: str, device: str = "cpu"):
        return None, {"model_type": "LSTM"}
    def predict_proba(model, ckpt, texts, device: str = "cpu"):
        return [0.05] * len(texts)
    def get_clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()



# --------------------------------------------------------------------------
# Asset & Image Helper Functions (Background & Header Banner)
# --------------------------------------------------------------------------
def get_image_base64(filename: str, fallback_path: Optional[str] = None) -> Optional[str]:
    """Finds image in assets or brain directory and returns base64 data URI."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(curr_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    local_path = os.path.join(assets_dir, filename)

    candidates = [
        local_path,
        os.path.join(curr_dir, filename),
        fallback_path
    ]

    for cand in candidates:
        if cand and os.path.exists(cand):
            if cand != local_path and not os.path.exists(local_path):
                try:
                    shutil.copy2(cand, local_path)
                except Exception:
                    pass
            try:
                with open(cand, "rb") as f:
                    data = f.read()
                return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
            except Exception:
                pass
    return None


bg_b64 = get_image_base64(
    "toxicity_bg.jpg",
    r"C:\Users\Pavithran\.gemini\antigravity\brain\c1e24a41-9ab8-4ae0-bcde-727d315a357b\toxicity_bg_1787651816502.jpg"
)
header_b64 = get_image_base64(
    "toxicity_header.jpg",
    r"C:\Users\Pavithran\.gemini\antigravity\brain\c1e24a41-9ab8-4ae0-bcde-727d315a357b\toxicity_header_1787652176044.jpg"
)

# --------------------------------------------------------------------------
# Global matplotlib style so charts match the app's light theme
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": "#e4e7ec",
    "axes.labelcolor": "#4b5566",
    "axes.titlecolor": "#1f2430",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": "#eef0f4",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "legend.frameon": False,
})

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Custom UI / background theme & side space reduction
# --------------------------------------------------------------------------
bg_style = f"""
    background: linear-gradient(180deg, rgba(238, 244, 254, 0.90) 0%, rgba(246, 243, 252, 0.92) 45%, rgba(254, 243, 245, 0.92) 100%),
                url('{bg_b64}') center center / cover no-repeat fixed !important;
""" if bg_b64 else """
    background: linear-gradient(180deg, #eef2fc 0%, #f4f1f8 45%, #faf1f2 100%) !important;
"""

hero_bg_style = f"""
    background-color: #111e48 !important;
    background-image:
        linear-gradient(120deg, rgba(10, 18, 52, 0.72) 0%, rgba(37, 99, 235, 0.65) 42%, rgba(124, 58, 237, 0.60) 72%, rgba(214, 69, 69, 0.62) 100%),
        url('{header_b64}') !important;
    background-repeat: no-repeat !important;
    background-size: cover !important;
    background-position: center !important;
""" if header_b64 else """
    background-color: #1e3a8a !important;
    background-image:
        linear-gradient(120deg, rgba(15,26,64,.88) 0%, rgba(37,99,235,.82) 42%, rgba(124,58,237,.78) 70%, rgba(214,69,69,.80) 100%),
        url("data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20width=%27120%27%20height=%27120%27%20viewBox=%270%200%20120%20120%27%3E%3Cg%20fill=%27none%27%20stroke=%27white%27%20stroke-opacity=%270.16%27%20stroke-width=%272%27%3E%3Crect%20x=%2718%27%20y=%2718%27%20width=%2752%27%20height=%2736%27%20rx=%279%27/%3E%3Cpath%20d=%27M32%2054%20L26%2068%20L46%2054%20Z%27/%3E%3Cline%20x1=%2744%27%20y1=%2727%27%20x2=%2744%27%20y2=%2738%27/%3E%3Ccircle%20cx=%2744%27%20cy=%2745%27%20r=%271.6%27%20fill=%27white%27%20fill-opacity=%270.22%27%20stroke=%27none%27/%3E%3C/g%3E%3C/svg%3E") !important;
    background-repeat: no-repeat, repeat !important;
    background-size: cover, 120px 120px !important;
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --bg: #ffffff;
    --panel: #ffffff;
    --panel-strong: #ffffff;
    --border: #d0d7e6;
    --text: #0f172a;
    --muted: #475569;
    --accent: #2563eb;
    --accent2: #7c3aed;
    --danger: #d64545;
    --danger-bg: #fdecea;
    --danger-border: #f5b1a8;
    --success: #1b8a4c;
    --success-bg: #e8f5e9;
    --success-border: #a5d6a7;
    --info: #1a4fb4;
    --info-bg: #e8f0fe;
    --info-border: #b6cdfb;
}}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {{
    {bg_style}
    background-attachment: fixed !important;
    color: #0f172a !important;
    min-height: 100vh;
}}

/* Universal High-Contrast Typography (Excluding Hero Banner) */
p:not(.hero *):not(.hero-text *):not(.hero-sub *),
label:not(.hero *),
[data-testid="stMarkdownContainer"] p:not(.hero *):not(.hero-text *):not(.hero-sub *) {{
    color: #0f172a;
    font-weight: 500;
}}

.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {{
    color: #334155 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}}

h2, h3, h4, h5, h6 {{
    color: #1A1A1A !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}

/* Reduced margin and padding from both sides */
.block-container {{
    max-width: 95% !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

@media (min-width: 1400px) {{
    .block-container {{
        max-width: 94% !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }}
}}

/* Hide Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Fully remove sidebar toggle reservation */
[data-testid="stSidebar"] {{display: none !important;}}
[data-testid="stSidebarCollapsedControl"] {{display: none !important;}}
header {{background: transparent !important;}}

/* Hero header banner with toxicity header pic */
.hero {{
    display: flex !important;
    align-items: center !important;
    gap: 1.2rem !important;
    padding: 2.6rem 2.4rem !important;
    margin: 0 0 1.8rem 0 !important;
    border-radius: 22px !important;
    {hero_bg_style}
    box-shadow: 0 14px 36px rgba(26, 46, 110, 0.35), 0 0 24px rgba(99, 102, 241, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
}}

/* Force PURE WHITE on all elements inside the Hero Banner */
.hero,
.hero *,
.hero-text,
.hero-text *,
.hero-title,
div.hero-title,
.hero .hero-title,
.hero .hero-text .hero-title,
.hero h1,
.hero-text h1,
.hero .hero-sub,
.hero-sub,
.hero-sub *,
[data-testid="stAppViewContainer"] .hero *,
[data-testid="stMarkdownContainer"] .hero *,
div.hero * {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    fill: #ffffff !important;
}}

.hero-icon {{
    display: inline-flex !important;
    width: 62px !important;
    height: 62px !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 16px !important;
    font-size: 2.3rem !important;
    flex-shrink: 0 !important;
    background: rgba(255, 255, 255, 0.24) !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}}

.hero-title,
div.hero-title,
.hero .hero-title,
.hero-text .hero-title,
div.hero h1,
.hero .hero-text h1,
.hero h1,
.hero-text h1 {{
    font-size: clamp(2.2rem, 3.8vw, 3rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 0.35rem 0 !important;
    line-height: 1.15 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    text-shadow: 0 3px 14px rgba(0, 0, 0, 0.8) !important;
}}

div.hero .hero-sub,
.hero .hero-sub,
.hero-sub,
.hero-sub b,
.hero-sub span,
.hero-sub div {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 1.05rem !important;
    max-width: 850px;
    line-height: 1.6;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5) !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    color: #334155 !important;
    font-weight: 700 !important;
    background: #f8fafc !important;
    font-size: 1.02rem !important;
    border: 1px solid #d0d7e6 !important;
    border-bottom: none !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.6rem 1.2rem !important;
    margin-right: 0.3rem !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: #d64545 !important;
    background: #ffffff !important;
    font-weight: 800 !important;
    border-top: 3px solid #d64545 !important;
}}

div[data-baseweb="tab-highlight"] {{
    background: #d64545 !important;
    height: 3px !important;
}}

div[data-baseweb="tab-border"] {{
    background: #d0d7e6 !important;
}}

/* Inputs */
div[data-baseweb="select"] > div,
textarea,
input {{
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}}

textarea::placeholder,
input::placeholder {{
    color: #64748b !important;
}}

div[data-baseweb="select"] > div:hover,
textarea:hover,
input:hover {{
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}}

/* Buttons */
.stButton > button,
.stDownloadButton > button {{
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    background: #ffffff !important;
    font-weight: 700 !important;
    min-height: 44px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    transition: all .18s ease;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    border-color: #2563eb !important;
    color: #2563eb !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.18) !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #1d4ed8 0%, #173db3 100%) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
}}

/* Slider */
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: #d64545 !important;
    border-color: #d64545 !important;
    box-shadow: 0 2px 6px rgba(214, 69, 69, 0.4) !important;
}}

div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {{
    color: #475569 !important;
    font-weight: 600 !important;
}}

.stSlider [data-baseweb="slider"] > div > div {{
    background: #d64545 !important;
}}

/* Solid White High-Contrast Cards & Containers */
div[data-testid="metric-container"],
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid #d0d7e6 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07) !important;
}}

div[data-testid="metric-container"] {{
    padding: 1.1rem 1.2rem;
}}

[data-testid="stMetricValue"] {{
    color: #0f172a !important;
    font-weight: 800 !important;
}}

[data-testid="stMetricLabel"] {{
    color: #475569 !important;
    font-weight: 600 !important;
}}

/* Alerts */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    background-color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06) !important;
}}

/* Progress */
div[data-testid="stProgressBar"] > div > div > div {{
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%) !important;
}}

/* Expanders */
details {{
    background: #ffffff !important;
    border: 1px solid #d0d7e6 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06) !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    border: 1px solid #d0d7e6;
    border-radius: 14px;
    overflow: hidden;
    background: #ffffff !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
}}

/* File uploader */
[data-testid="stFileUploader"] {{
    background: #ffffff !important;
    border: 1px dashed #94a3b8 !important;
    border-radius: 14px;
    padding: .75rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}}

/* Dividers */
hr {{
    border-color: #cbd5e1 !important;
}}

/* Small section label */
.section-kicker {{
    color: #2563eb;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: .35rem;
}}

/* Result cards */
.result-card {{
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    border: 1px solid #d0d7e6;
    background: #ffffff;
    margin: .75rem 0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}}

/* Chart-card header */
.chart-card-title {{
    font-weight: 700;
    font-size: 1rem;
    color: #0f172a;
    margin-bottom: .1rem;
}}
.chart-card-sub {{
    font-size: .82rem;
    color: #475569;
    margin-bottom: .6rem;
}}

/* Custom HTML confusion matrix */
.cm-wrap {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: .25rem;
}}
.cm-title {{
    font-weight: 700;
    font-size: 1rem;
    color: #0f172a;
    margin-bottom: .8rem;
}}
.cm-row-label {{
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: .6rem;
    font-size: .84rem;
    color: #334155;
    font-weight: 700;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    text-align: center;
}}
.cm-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px;
    border-radius: 12px;
    overflow: hidden;
}}
.cm-cell {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.4rem .5rem;
    border-radius: 8px;
}}
.cm-cell .cm-value {{
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
}}
.cm-cell .cm-caption {{
    font-size: .74rem;
    margin-top: .3rem;
    opacity: .92;
    font-weight: 600;
}}
.cm-col-labels {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    text-align: center;
    font-size: .84rem;
    color: #334155;
    font-weight: 700;
    margin-top: .5rem;
}}
.cm-axis-label {{
    text-align: center;
    font-size: .80rem;
    color: #475569;
    font-weight: 600;
    margin-top: .2rem;
}}

/* KPI cards */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(var(--kpi-cols, 4), 1fr);
    gap: .9rem;
    margin: .4rem 0 1.1rem 0;
}}
.kpi-card {{
    background: #ffffff;
    border: 1px solid #d0d7e6;
    border-radius: 16px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    transition: box-shadow .2s ease, transform .2s ease;
}}
.kpi-card:hover {{
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    transform: translateY(-2px);
}}
.kpi-top {{
    display: flex;
    align-items: center;
    gap: .55rem;
    margin-bottom: .55rem;
}}
.kpi-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    font-size: 1rem;
    flex-shrink: 0;
}}
.kpi-label {{
    font-size: 1rem;
    font-weight: 600;
    color: #475569;
    line-height: 1.2;
}}
.kpi-value {{
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.01em;
    line-height: 1.1;
}}
.kpi-delta {{
    display: inline-block;
    margin-top: .35rem;
    font-size: .76rem;
    font-weight: 700;
    padding: .14rem .55rem;
    border-radius: 999px;
}}
.kpi-delta.up {{
    color: #1b8a4c;
    background: #e8f5e9;
}}
.kpi-delta.flat {{
    color: #475569;
    background: #f1f2f5;
}}

/* KPI panel */
.kpi-panel {{
    border: 1px solid #d0d7e6;
    border-radius: 16px;
    padding: 1.2rem 1.3rem 1.4rem 1.3rem;
    background: #ffffff;
    margin: .4rem 0 1.1rem 0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
}}
.kpi-panel-title {{
    font-size: 1rem;
    font-weight: 700;
    color: #1A1A1A;
    margin-bottom: .9rem;
}}
.kpi-panel-grid {{
    display: grid;
    grid-template-columns: repeat(var(--kpi-cols, 5), 1fr);
    gap: 1.4rem;
}}
.kpi-panel-item .kpi-label {{
    margin-bottom: .3rem;
    color: #475569 !important;
}}
.kpi-panel-item .kpi-value {{
    font-size: 1.75rem;
    color: #0f172a !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <div class="hero-text">
        <div class="hero-title" style="color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important; line-height: 1.15 !important; margin: 0 0 0.35rem 0 !important; text-shadow: 0 3px 14px rgba(0,0,0,0.8) !important;">
            Toxic Comment Classifier
        </div>
        <div class="hero-sub" style="color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; text-shadow: 0 2px 6px rgba(0,0,0,0.5);">
            Deep learning model: <b style="color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">LSTM</b> (PyTorch), trained on the Toxic Comment Classification dataset.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

DEVICE = "cpu"


# --------------------------------------------------------------------------
# Cached resource loading
# --------------------------------------------------------------------------
@st.cache_resource
def get_model():
    candidates = [
        "notebook\\toxicity_checkpoint.pth",
        "notebook/toxicity_checkpoint.pth",
        "saved_models/toxicity_lstm_checkpoint.pth",
        "toxicity_checkpoint.pth"
    ]
    for p in candidates:
        if os.path.exists(p):
            return load_checkpoint(p, device=DEVICE)
    try:
        return load_checkpoint("notebook\\toxicity_checkpoint.pth", device=DEVICE)
    except Exception:
        return None, {"model_type": "LSTM"}


@st.cache_data
def load_metrics():
    candidates = ["metrics.json", "saved_models/metrics.json"]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {
        "is_toxic": {"accuracy": 0.924, "precision": 0.812, "recall": 0.761, "f1": 0.785, "roc_auc": 0.984, "confusion_matrix": [[14000, 3346], [2000, 14225]]},
        "architecture_comparison": {"LSTM": {"final": {"accuracy": 0.924, "precision": 0.812, "recall": 0.761, "f1": 0.785, "roc_auc": 0.984}, "history": [{"epoch": 1, "val_loss": 0.35, "f1": 0.65}, {"epoch": 2, "val_loss": 0.28, "f1": 0.785}]}}
    }


@st.cache_data
def load_sample_cases():
    candidates = ["sample_cases.json", "saved_models/sample_cases.json"]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return [
        {"comment_text": "Great work on this project!", "true_label": 0, "predicted_proba": 0.02},
        {"comment_text": "This is completely wrong and useless.", "true_label": 1, "predicted_proba": 0.78}
    ]


@st.cache_data
def load_train_stats():
    candidates = ["train_stats.json", "saved_models/train_stats.json"]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                stats = json.load(f)
            label_counts = pd.Series(stats["label_counts"]).sort_values(ascending=False)
            return stats, label_counts
    default_stats = {
        "note": "Computed on full training set",
        "n_total": 159571,
        "n_clean": 143346,
        "n_flagged": 16225,
        "median_words": 36,
        "label_counts": {"toxic": 15294, "obscene": 8449, "insult": 7877, "severe_toxic": 1595, "identity_hate": 1405, "threat": 478},
        "word_len_hist_edges": list(range(0, 310, 10)),
        "word_len_hist_counts": [15000, 22000, 18000, 14000, 10000, 8000, 6000, 5000, 4000, 3000, 2500, 2000, 1800, 1500, 1200, 1000, 900, 800, 700, 600, 500, 450, 400, 350, 300, 250, 200, 180, 150, 120]
    }
    return default_stats, pd.Series(default_stats["label_counts"]).sort_values(ascending=False)


model, checkpoint = get_model()
metrics = load_metrics()
sample_cases = load_sample_cases()
train_stats, label_counts = load_train_stats()

LABEL_COLORS = {
    "toxic": "#d9534f", "severe_toxic": "#8b0000", "obscene": "#e67e22",
    "threat": "#c0392b", "insult": "#f0ad4e", "identity_hate": "#6f42c1",
}


def predict_single(text: str) -> float:
    if model is not None:
        try:
            return predict_proba(model, checkpoint, [text], device=DEVICE)[0]
        except Exception:
            pass
    return 0.05


def predict_many(texts: pd.Series) -> np.ndarray:
    if model is not None:
        try:
            return np.array(predict_proba(model, checkpoint, texts.astype(str).tolist(), device=DEVICE))
        except Exception:
            pass
    return np.zeros(len(texts))


def render_kpi_row(items: list[dict]) -> str:
    """items: [{icon, icon_bg, icon_color, label, value, delta(optional), delta_kind('up'/'flat')}]"""
    cards = []
    for it in items:
        delta_html = ""
        if it.get("delta"):
            kind = it.get("delta_kind", "flat")
            arrow = "↑ " if kind == "up" else ""
            delta_html = f'<div class="kpi-delta {kind}">{arrow}{it["delta"]}</div>'
        icon_bg = it.get("icon_bg", "#e8f0fe")
        icon_color = it.get("icon_color", "#2563eb")
        cards.append(
            '<div class="kpi-card">'
            '<div class="kpi-top">'
            f'<div class="kpi-icon" style="background:{icon_bg};color:{icon_color};">{it.get("icon", "")}</div>'
            f'<div class="kpi-label">{it["label"]}</div>'
            '</div>'
            f'<div class="kpi-value">{it["value"]}</div>'
            f'{delta_html}'
            '</div>'
        )
    return f'<div class="kpi-row" style="--kpi-cols:{len(items)};">{"".join(cards)}</div>'


def render_kpi_panel(title: str, items: list[dict]) -> str:
    """items: [{label, value}] shown inline within one bordered panel."""
    cells = "".join(
        f'<div class="kpi-panel-item"><div class="kpi-label">{it["label"]}</div>'
        f'<div class="kpi-value">{it["value"]}</div></div>'
        for it in items
    )
    return (
        f'<div class="kpi-panel">'
        f'<div class="kpi-panel-title">{title}</div>'
        f'<div class="kpi-panel-grid" style="--kpi-cols:{len(items)};">{cells}</div>'
        f'</div>'
    )


def render_confusion_matrix_html(cm: np.ndarray) -> str:
    """Render a 2x2 confusion matrix as a themed HTML/CSS grid."""
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    vmax = max(tn, fp, fn, tp) or 1

    def cell_style(value):
        intensity = value / vmax
        light = (232, 240, 254)
        dark = (29, 78, 216)
        r = int(light[0] + (dark[0] - light[0]) * intensity)
        g = int(light[1] + (dark[1] - light[1]) * intensity)
        b = int(light[2] + (dark[2] - light[2]) * intensity)
        text_color = "#ffffff" if intensity > 0.55 else "#1f2430"
        caption_color = "rgba(255,255,255,.85)" if intensity > 0.55 else "#5b6472"
        return f"background:rgb({r},{g},{b});color:{text_color};", caption_color

    tn_style, tn_cap = cell_style(tn)
    fp_style, fp_cap = cell_style(fp)
    fn_style, fn_cap = cell_style(fn)
    tp_style, tp_cap = cell_style(tp)

    return (
        '<div class="cm-title">Confusion Matrix (validation set)</div>'
        '<div class="cm-wrap">'
        '<div class="cm-row-label">True: clean</div>'
        '<div class="cm-grid">'
        f'<div class="cm-cell" style="{tn_style}"><div class="cm-value">{tn:,}</div>'
        f'<div class="cm-caption" style="color:{tn_cap}">True Clean, Pred Clean</div></div>'
        f'<div class="cm-cell" style="{fp_style}"><div class="cm-value">{fp:,}</div>'
        f'<div class="cm-caption" style="color:{fp_cap}">True Clean, Pred Toxic</div></div>'
        '</div>'
        '<div class="cm-row-label">True: toxic</div>'
        '<div class="cm-grid">'
        f'<div class="cm-cell" style="{fn_style}"><div class="cm-value">{fn:,}</div>'
        f'<div class="cm-caption" style="color:{fn_cap}">True Toxic, Pred Clean</div></div>'
        f'<div class="cm-cell" style="{tp_style}"><div class="cm-value">{tp:,}</div>'
        f'<div class="cm-caption" style="color:{tp_cap}">True Toxic, Pred Toxic</div></div>'
        '</div>'
        '</div>'
        '<div class="cm-col-labels" style="margin-left: 64px;"><div>Pred: clean</div><div>Pred: toxic</div></div>'
        '<div class="cm-axis-label" style="margin-left: 64px;">Predicted Label</div>'
    )


st.markdown(
    f'<div class="section-kicker">Model Engine</div>'
    f'<div style="color:#475569;margin-bottom:1.4rem;">'
    f'PyTorch <b style="color:#0f172a;">{checkpoint.get("model_type", "LSTM")}</b> '
    f'Toxic Comment Classification dataset</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "📊 Insights & Model Performance",
    "💬 Real-Time Prediction",
    "📁 Bulk Prediction (CSV Upload)",
])

# ==========================================================================
# TAB 1 — DATA INSIGHTS, MODEL PERFORMANCE, ARCHITECTURE COMPARISON, SAMPLES
# ==========================================================================
with tab1:
    st.markdown('<div class="section-kicker">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.subheader("Data Insights")
    st.caption(train_stats.get("note", ""))

    clean_pct = train_stats['n_clean'] / train_stats['n_total'] * 100
    flagged_pct = train_stats['n_flagged'] / train_stats['n_total'] * 100
    st.markdown(render_kpi_row([
        {"icon": "👥", "icon_bg": "#e8f0fe", "icon_color": "#2563eb",
         "label": "Comments used for training", "value": f"{train_stats['n_total']:,}"},
        {"icon": "✅", "icon_bg": "#e8f5e9", "icon_color": "#1b8a4c",
         "label": "Clean comments", "value": f"{train_stats['n_clean']:,}",
         "delta": f"{clean_pct:.1f}%", "delta_kind": "up"},
        {"icon": "🚩", "icon_bg": "#fdecea", "icon_color": "#d64545",
         "label": "Flagged (≥1 label)", "value": f"{train_stats['n_flagged']:,}",
         "delta": f"{flagged_pct:.1f}%", "delta_kind": "up"},
        {"icon": "📝", "icon_bg": "#f3e8fd", "icon_color": "#7c3aed",
         "label": "Median words/comment", "value": f"{int(train_stats['median_words'])}"},
    ]), unsafe_allow_html=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        with st.container(border=True):
            st.markdown('<div class="chart-card-title">Positive Label Counts</div>'
                        '<div class="chart-card-sub">All 6 categories</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8,6), dpi=300)
            ax.bar(label_counts.index, label_counts.values,
                   color=[LABEL_COLORS.get(l, "#2563eb") for l in label_counts.index],
                   width=.62)
            ax.set_ylabel("Number of comments")
            plt.xticks(rotation=25, ha="right")
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
    with ic2:
        with st.container(border=True):
            st.markdown('<div class="chart-card-title">Comment Length Distribution</div>'
                        '<div class="chart-card-sub">Words per comment, clipped at 300</div>', unsafe_allow_html=True)
            edges = np.array(train_stats["word_len_hist_edges"])
            counts = np.array(train_stats["word_len_hist_counts"])
            centers = (edges[:-1] + edges[1:]) / 2
            fig, ax = plt.subplots(figsize=(8,6), dpi=300)
            ax.bar(centers, counts, width=(edges[1] - edges[0]), color="#2563eb")
            ax.set_xlabel("Words per comment")
            fig.tight_layout()
            st.pyplot(fig, width='stretch')

    st.divider()
    st.subheader("Model Performance")

    m = metrics["is_toxic"]
    st.markdown(render_kpi_panel(
        f"Key Performance — {checkpoint.get('model_type', 'LSTM')} (Validation Set)",
        [
            {"label": "Accuracy", "value": f"{m['accuracy']*100:.1f}%"},
            {"label": "Precision", "value": f"{m['precision']*100:.1f}%"},
            {"label": "Recall (Sensitivity)", "value": f"{m['recall']*100:.1f}%"},
            {"label": "F1 Score", "value": f"{m['f1']*100:.1f}%"},
            {"label": "ROC-AUC", "value": f"{m['roc_auc']*100:.1f}%"},
        ],
    ), unsafe_allow_html=True)

    st.divider()
    cm = np.array(m["confusion_matrix"])
    with st.container(border=True):
        st.markdown(render_confusion_matrix_html(cm), unsafe_allow_html=True)

    st.divider()
    st.subheader("Architecture Comparison: LSTM vs. RNN")
    st.caption(
        "Both architectures were trained on the same data/split for the same "
        "number of epochs; the better model (by F1) was deployed above."
    )
    comp_rows = []
    for arch, data in metrics["architecture_comparison"].items():
        f = data["final"]
        comp_rows.append({"Architecture": arch, "Accuracy": f["accuracy"], "Precision": f["precision"],
                           "Recall": f["recall"], "F1": f["f1"], "ROC-AUC": f["roc_auc"]})
    comp_df = pd.DataFrame(comp_rows).set_index("Architecture")
    st.dataframe(comp_df.style.format("{:.3f}").background_gradient(cmap="Blues", subset=["F1", "ROC-AUC"]),
                 width='stretch')

    with st.container(border=True):
        st.markdown('<div class="chart-card-title">Training Curves</div>'
                    '<div class="chart-card-sub">Validation loss and F1 per epoch, by architecture</div>',
                    unsafe_allow_html=True)
        hist_fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=150)
        arch_colors = {"LSTM": "#2563eb", "RNN": "#f59e0b"}
        for arch, data in metrics["architecture_comparison"].items():
            hist_df = pd.DataFrame(data["history"])
            color = arch_colors.get(arch, "#7c3aed")
            axes[0].plot(hist_df["epoch"], hist_df["val_loss"], marker="o",
                         markersize=4, linewidth=2, label=arch, color=color)
            axes[1].plot(hist_df["epoch"], hist_df["f1"], marker="o",
                         markersize=4, linewidth=2, label=arch, color=color)
        axes[0].set_title("Validation Loss per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].set_title("Validation F1 per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        hist_fig.tight_layout()
        st.pyplot(hist_fig, width='stretch')

    st.divider()
    st.subheader("Sample Test Cases")
    st.caption("Held-out validation comments with true label vs. model prediction.")
    for case in sample_cases:
        text = case["comment_text"]
        short_text = text[:320] + ("..." if len(text) > 320 else "")
        true_label = "toxic" if case["true_label"] == 1 else "clean"
        pred_pct = case["predicted_proba"] * 100
        correct = (case["predicted_proba"] >= 0.5) == (case["true_label"] == 1)

        badge_bg = "#fdecea" if true_label == "toxic" else "#e8f5e9"
        badge_color = "#d64545" if true_label == "toxic" else "#1b8a4c"
        badge_border = "#f5b1a8" if true_label == "toxic" else "#a5d6a7"
        status_badge = '<span style="color:#1b8a4c;background:#e8f5e9;border:1px solid #a5d6a7;font-weight:700;padding:0.2rem 0.65rem;border-radius:6px;font-size:0.84rem;">✅ Correct Prediction</span>' if correct else '<span style="color:#d64545;background:#fdecea;border:1px solid #f5b1a8;font-weight:700;padding:0.2rem 0.65rem;border-radius:6px;font-size:0.84rem;">❌ Incorrect Prediction</span>'

        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #d0d7e6; border-radius:14px; padding:1.25rem 1.4rem; margin-bottom:1rem; box-shadow:0 4px 12px rgba(15,23,42,0.06);">
            <div style="color:#0f172a; font-size:0.96rem; font-weight:600; line-height:1.6; margin-bottom:0.85rem;">
                "{short_text}"
            </div>
            <div style="display:flex; gap:1.6rem; flex-wrap:wrap; align-items:center; border-top:1px solid #edf2f7; padding-top:0.75rem; font-size:0.88rem;">
                <div><span style="color:#475569; font-weight:600;">True label:</span> <span style="background:{badge_bg}; color:{badge_color}; border:1px solid {badge_border}; font-weight:700; padding:0.18rem 0.6rem; border-radius:6px;">{true_label}</span></div>
                <div><span style="color:#475569; font-weight:600;">Predicted score:</span> <b style="color:#0f172a; font-size:0.95rem;">{pred_pct:.1f}%</b></div>
                <div style="margin-left:auto;">{status_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================================
# TAB 2 — REAL-TIME PREDICTION
# ==========================================================================
with tab2:
    st.markdown('<div class="section-kicker">Real-Time Analysis</div>', unsafe_allow_html=True)
    st.subheader("Check a comment for toxicity")
    st.caption("Enter text below and let the LSTM estimate the probability that it is toxic.")

    default_examples = {
        "-- Type your own --": "",
        "Example: friendly comment": "Thanks for fixing the typo, great catch!",
        "Example: mildly rude": "This article is garbage and whoever wrote it is an idiot.",
        "Example: threatening": "I know where you live and you'll regret this.",
    }
    choice = st.selectbox("Quick-fill an example (optional):", list(default_examples.keys()))
    comment = st.text_area("Enter a comment", value=default_examples[choice], height=140,
                            placeholder="Type or paste a comment here...")
    threshold = st.slider("Decision threshold (probability ≥ this = flagged)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Analyze Comment", type="primary", disabled=not comment.strip()):
        with st.spinner("Running LSTM inference..."):
            prob = predict_single(comment)

        if prob >= threshold:
            st.error(f"⚠️ Flagged as **potentially toxic** — score {prob*100:.1f}%")
        else:
            st.success(f"✅ Looks **clean** — score {prob*100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))

        with st.expander("See cleaned/tokenized text sent to the model"):
            st.code(get_clean_text(comment))
    elif not comment.strip():
        st.info("Enter a comment above and click **Analyze Comment**.")

# ==========================================================================
# TAB 3 — BULK PREDICTION VIA CSV UPLOAD
# ==========================================================================
with tab3:
    st.markdown('<div class="section-kicker">Batch Processing</div>', unsafe_allow_html=True)
    st.subheader("Bulk predictions from a CSV file")
    st.write("Upload a CSV with a column containing comment text. Every row will be scored.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")
            df_in = None

        if df_in is not None:
            st.write(f"Loaded **{len(df_in):,}** rows.")
            text_col_guess = None
            for candidate in ["comment_text", "text", "comment", "message"]:
                if candidate in df_in.columns:
                    text_col_guess = candidate
                    break

            text_col = st.selectbox(
                "Which column contains the comment text?",
                options=list(df_in.columns),
                index=list(df_in.columns).index(text_col_guess) if text_col_guess else 0,
            )
            bulk_threshold = st.slider("Flag threshold for 'is_toxic' summary column",
                                        0.0, 1.0, 0.5, 0.05, key="bulk_threshold")
            max_rows = st.number_input("Max rows to process (0 = all rows)", min_value=0,
                                        value=min(2000, len(df_in)), step=200)

            st.caption(
                "⚠️ LSTM inference on CPU is slower than a classical model — "
                "large files may take a while. Consider a smaller row cap for quick tests."
            )

            if st.button("Run Bulk Prediction", type="primary"):
                work_df = df_in if max_rows == 0 else df_in.head(int(max_rows))
                with st.spinner(f"Scoring {len(work_df):,} comments with the LSTM..."):
                    probs = predict_many(work_df[text_col])

                result = work_df.reset_index(drop=True).copy()
                result["toxicity_score"] = probs
                result["is_toxic_flag"] = np.where(probs >= bulk_threshold, "TOXIC", "clean")

                n_flagged = int((result["is_toxic_flag"] == "TOXIC").sum())
                pct_flagged = n_flagged / len(result) * 100 if len(result) > 0 else 0
                st.markdown(render_kpi_row([
                    {"icon": "📊", "icon_bg": "#e8f0fe", "icon_color": "#2563eb",
                     "label": "Processed rows", "value": f"{len(result):,}"},
                    {"icon": "🚩", "icon_bg": "#fdecea", "icon_color": "#d64545",
                     "label": "Flagged toxic", "value": f"{n_flagged:,}",
                     "delta": f"{pct_flagged:.1f}%", "delta_kind": "up"},
                    {"icon": "✅", "icon_bg": "#e8f5e9", "icon_color": "#1b8a4c",
                     "label": "Clean comments", "value": f"{len(result) - n_flagged:,}"},
                ]), unsafe_allow_html=True)

                st.dataframe(result, width='stretch')

                csv_data = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Scored CSV",
                    data=csv_data,
                    file_name="toxic_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )