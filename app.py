from pickle import load
from venv import create
from pandas import read_csv
import streamlit as st
import numpy as np
from api import predict
import json

with open("stats.json", "r") as f:
    data = json.load(f)
    features = data['features']
    init_features = data['init_features']
    importances = data['importances']

st.set_page_config(
    page_title="Drilling ROP Predictor",
    page_icon="🛢️",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 2px; }

    .title-block {
        background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
        border-left: 4px solid #f97316;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.5rem;
        border-radius: 0 8px 8px 0;
    }
    .title-block h1 { color: #f97316; font-size: 2rem; margin: 0; text-transform: uppercase; }
    .title-block p {
        color: #8b949e; font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem; margin: 0.3rem 0 0; letter-spacing: 1px;
    }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-label {
        color: #8b949e; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 1.5px; font-family: 'Share Tech Mono', monospace;
    }
    .metric-value { color: #f97316; font-size: 1.8rem; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
    .metric-value.green  { color: #3fb950; }
    .metric-value.yellow { color: #d29922; }
    .metric-value.red    { color: #f85149; }

    .result-box {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 2rem; text-align: center; margin-top: 1rem;
    }
    .result-box .label { color: #8b949e; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; font-family: 'Share Tech Mono', monospace; }
    .result-box .value { color: #f97316; font-size: 3.5rem; font-weight: 700; font-family: 'Share Tech Mono', monospace; line-height: 1.1; }
    .result-box .unit  { color: #8b949e; font-size: 1rem; font-family: 'Share Tech Mono', monospace; }

    .stDataFrame { border-radius: 8px; overflow: hidden; }
    div[data-testid="stDataFrame"] table { font-family: 'Share Tech Mono', monospace !important; font-size: 0.82rem !important; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white; border: none; border-radius: 6px; padding: 0.6rem 2rem;
        font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1rem;
        letter-spacing: 2px; text-transform: uppercase; cursor: pointer;
        transition: opacity 0.2s; width: 100%;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.85; }

    .section-label {
        color: #8b949e; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 2px; font-family: 'Share Tech Mono', monospace; margin-bottom: 0.5rem;
    }
    div[data-testid="stNumberInput"] label { font-family: 'Share Tech Mono', monospace !important; font-size: 0.78rem !important; color: #8b949e !important; }

    /* Feature importance bar styles */
    .fi-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }
    .fi-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        gap: 0.8rem;
    }
    .fi-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
        width: 140px;
        flex-shrink: 0;
        text-align: right;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .fi-bar-wrap {
        flex: 1;
        background: #0d1117;
        border-radius: 3px;
        height: 18px;
        overflow: hidden;
    }
    .fi-bar {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .fi-pct {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: #e6edf3;
        width: 48px;
        flex-shrink: 0;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🛢️ Drilling ROP Predictor</h1>
    <p>RATE OF PENETRATION · ANOMALY-AWARE MODEL · LIVE SAMPLING</p>
</div>
""", unsafe_allow_html=True)

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load(open('model.pkl', 'rb'))
def load_kmeans():
    return load(open('kmeans.pkl', 'rb'))
from util import create_features
@st.cache_data
def load_data():
    return create_features(read_csv('cleaned_drilling.csv', usecols=['rpm', 'wob', 'flow_in','rop' , 'depth_tmd' , 'depth_tvd' ]))

try:
    model = load_model()
    kmeans = load_kmeans()
    df    = load_data()
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊  Batch Sample", "🎯  Manual Prediction"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Sample
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.markdown("### ⚙️ SAMPLING CONFIG")
        n_samples   = st.slider("Number of samples", 5, 50, 10)
        st.markdown("---")
        st.markdown(
            "<div style='color:#8b949e;font-size:0.75rem;font-family:Share Tech Mono,monospace'>"
            f"Dataset: {len(df):,} rows<br>Features: {df.shape[1]-1}</div>",
            unsafe_allow_html=True
        )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        rerun = st.button("🔄  RERUN SAMPLE")

    if rerun:
        if "seed_offset" not in st.session_state:
            st.session_state.seed_offset = 0
        st.session_state.seed_offset += 1

    sample_df = df.sample(n=n_samples)
    X_sample  = sample_df.drop(columns=['rop'])
    X_sample['cluster'] = kmeans.predict(X_sample)
    X_sample = create_features(X_sample)[features]
    y_sample  = sample_df['rop'].reset_index(drop=True)
    y_pred    = model.predict(X_sample)
    abs_error = np.abs(y_sample.values - y_pred)
    mae  = abs_error.mean()
    mape = (abs_error / np.where(y_sample != 0, y_sample, np.nan)).mean() * 100
    r2   = 1 - np.sum((y_sample - y_pred)**2) / np.sum((y_sample - y_sample.mean())**2)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Samples</div><div class="metric-value">{n_samples}</div></div>', unsafe_allow_html=True)
    with m2:
        c = 'green' if mae < 1 else 'yellow' if mae < 3 else 'red'
        st.markdown(f'<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value {c}">{mae:.3f}</div></div>', unsafe_allow_html=True)
    with m3:
        c = 'green' if mape < 5 else 'yellow' if mape < 15 else 'red'
        st.markdown(f'<div class="metric-card"><div class="metric-label">MAPE</div><div class="metric-value {c}">{mape:.1f}%</div></div>', unsafe_allow_html=True)
    with m4:
        c = 'green' if r2 > 0.9 else 'yellow' if r2 > 0.7 else 'red'
        st.markdown(f'<div class="metric-card"><div class="metric-label">R²</div><div class="metric-value {c}">{r2:.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    DISPLAY_COLS = ["rpm", "wob", "flow_in"]
    results = X_sample.reset_index(drop=True)[DISPLAY_COLS].copy()
    results.insert(0, "sample_idx", sample_df.index.tolist())
    results["rop_actual"] = y_sample.values
    results["rop_pred"]   = np.round(y_pred, 4)
    results["abs_error"]  = np.round(abs_error, 4)

    def highlight_error(val):
        if val < 1:   return "background-color:#0d2818; color:#3fb950"
        elif val < 3: return "background-color:#2d1f00; color:#d29922"
        else:         return "background-color:#2d0f0f; color:#f85149"

    styled = (
        results.style
        .map(highlight_error, subset=["abs_error"])
        .format(precision=4)
        .set_properties(**{"font-family": "Share Tech Mono, monospace", "font-size": "0.82rem"})
    )

    st.markdown('<div class="section-label">Prediction Results Table</div>', unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, height=420)

    # ── Feature Importance ────────────────────────────────────────────────────
    import streamlit.components.v1 as components

    st.markdown("<br>", unsafe_allow_html=True)

    fi_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:6:]
    max_imp  = max(imp for _, imp in fi_pairs) if fi_pairs else 1
    colours  = ["#f97316", "#fb923c", "#fdba74", "#94a3b8"]

    rows_html = ""
    for i, (feat, imp) in enumerate(fi_pairs):
        pct      = imp / max_imp * 100
        pct_disp = imp * 100
        colour   = colours[min(i, len(colours) - 1)]
        rows_html += f"""
        <div style="display:flex; align-items:center; margin-bottom:0.9rem; gap:0.8rem;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:#8b949e;
                        width:140px; flex-shrink:0; text-align:right; text-transform:uppercase; letter-spacing:1px;">
                {feat}
            </div>
            <div style="flex:1; background:#0d1117; border-radius:3px; height:20px; overflow:hidden;">
                <div style="width:{pct:.1f}%; height:100%; background:{colour}; border-radius:3px;"></div>
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:#e6edf3;
                        width:48px; flex-shrink:0; text-align:right;">
                {pct_disp:.1f}%
            </div>
        </div>
        """

    fi_html = f"""
    <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1.2rem 1.5rem;">
        <div style="color:#8b949e; font-size:0.72rem; text-transform:uppercase; letter-spacing:2px;
                    font-family:'Share Tech Mono',monospace; margin-bottom:1rem;">
            Model Feature Importance
        </div>
        {rows_html}
    </div>
    """

    components.html(fi_html, height=60 + len(fi_pairs) * 48)


with tab2:
    st.markdown('<div class="section-label" style="margin-bottom:1rem">Enter drilling parameters — derived features are computed automatically by the API</div>', unsafe_allow_html=True)

    def num_input(label, col, fmt="%.4f"):
        return st.number_input(label, value=None, placeholder="Enter value...", format=fmt, key=f"manual_{col}")

    c1, c2 = st.columns(2)
    with c1:
        rpm           = num_input("rpm", "rpm")
        wob           = num_input("wob", "wob")
        flow_in        = num_input("flow_in", "flow_in")
    with c2:
        
        depth_tmd = num_input("depth_tmd", "depth_tmd")
        depth_tvd = num_input("depth_tvd", "depth_tvd")
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  PREDICT ROP", key="predict_manual")

    if predict_btn:
        input_data = {
            "rpm":            [rpm],
            "flow_in":        [flow_in],
            "wob":            [wob],
            "depth_tmd":            [depth_tmd],
            "depth_tvd":            [depth_tvd],
        }

        prediction = predict(input_data)

        st.markdown(f"""
        <div class="result-box">
            <div class="label">Predicted Rate of Penetration</div>
            <div class="value">{prediction:.4f}</div>
            <div class="unit">m/hr</div>
        </div>
        """, unsafe_allow_html=True)