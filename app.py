"""
============================================================
  Analisis Cluster E-Wallet — Full Version
  Algoritma K-Means Clustering dengan CRISP-DM Framework
  Dapat digunakan untuk presentasi kepada dosen
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Analisis Cluster E-Wallet",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Geist', sans-serif;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.015em;
}
.main {
    background-color: #07101F;
    background-image:
        radial-gradient(ellipse at 75% 0%,  rgba(37, 99, 235, 0.10) 0%, transparent 55%),
        radial-gradient(ellipse at 10% 85%, rgba(16, 185, 129, 0.07) 0%, transparent 50%);
    min-height: 100vh;
}
.block-container { padding: 2.5rem 3rem; max-width: 1240px; }
h1,h2,h3,h4,p,span,div { color: inherit; }
.stMarkdown p { color: #94A3B8; }

/* ── App Header ── */
.app-header {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #0F2155 0%, #1A3A8F 45%, #0E2847 100%);
    border-radius: 24px; padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 24px 64px rgba(0,0,0,0.55);
}
.app-header::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:320px; height:320px; border-radius:50%;
    background:rgba(255,255,255,0.03); filter:blur(40px);
}
.header-eyebrow {
    display:flex; align-items:center; gap:10px;
    font-family:'JetBrains Mono',monospace; font-size:0.68rem;
    font-weight:500; letter-spacing:0.14em; text-transform:uppercase;
    color:rgba(147,197,253,0.55); margin-bottom:1rem;
}
.app-header h1 {
    font-family:'Fraunces', serif;
    font-size:2.6rem; font-weight:400; color:#FFFFFF !important;
    letter-spacing:-0.02em; line-height:1.08; margin:0 0 0.6rem;
    font-style: italic;
}
.app-header p {
    font-size:0.875rem; color:rgba(186,215,250,0.6) !important;
    font-weight:400; line-height:1.75; max-width:520px; margin:0;
    letter-spacing: -0.005em;
}
.header-badge {
    position:absolute; top:2.5rem; right:3rem;
    background:rgba(255,255,255,0.07); backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,0.09); border-radius:16px;
    padding:0.8rem 1.2rem; display:flex; align-items:center; gap:12px; z-index:1;
}
.badge-icon { width:36px; height:36px; background:rgba(255,255,255,0.07); border-radius:10px;
              display:flex; align-items:center; justify-content:center; font-size:1rem; }
.badge-title { font-size:0.78rem; font-weight:700; color:white !important; display:block; }
.badge-sub { display:flex; align-items:center; gap:5px; margin-top:2px; }
.badge-dot { width:6px; height:6px; background:#34D399; border-radius:50%;
             display:inline-block; animation:pulse-dot 2s infinite; }
@keyframes pulse-dot {
    0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)}
}
.badge-status {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    letter-spacing:0.1em; color:rgba(186,215,250,0.45) !important; text-transform:uppercase;
}

/* ── Step Bar ── */
.step-bar {
    display:flex; gap:4px; margin-bottom:2rem;
    background:rgba(10,18,35,0.8); backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,0.05); border-radius:16px; padding:6px;
}
.step {
    flex:1; padding:0.85rem 0.5rem; text-align:center;
    font-size:0.8rem; font-weight:500; color:#3D5070;
    border-radius:10px; transition:all 0.2s; letter-spacing:0.01em;
    display:flex; align-items:center; justify-content:center; gap:6px;
}
.step-num { font-family:'JetBrains Mono',monospace; font-size:0.62rem; opacity:0.5; }
.step.done { color:#3B82F6; background:rgba(59,130,246,0.07); }
.step.active {
    background:#1D4ED8; color:white !important; font-weight:600;
    box-shadow:0 4px 18px rgba(29,78,216,0.4);
}

/* ── Section Title ── */
.section-title {
    font-size:1.05rem; font-weight:700; color:#F1F5F9 !important;
    margin-bottom:1.2rem; display:flex; align-items:center; gap:10px;
}
.section-title-icon {
    width:30px; height:30px; border-radius:8px; display:flex;
    align-items:center; justify-content:center; font-size:0.85rem;
    background:rgba(59,130,246,0.13);
}

/* ── Stat Cards ── */
.stat-row { display:flex; gap:1rem; margin-bottom:2rem; }
.stat-card {
    background:rgba(10,18,35,0.8); border:1px solid rgba(255,255,255,0.05);
    border-radius:16px; padding:1.4rem 1.6rem; flex:1;
    position:relative; overflow:hidden;
}
.stat-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#1D4ED8,#6366F1); border-radius:16px 16px 0 0;
}
.stat-number { font-size:2rem; font-weight:700; color:#F1F5F9 !important;
               line-height:1; margin-bottom:5px; letter-spacing:-0.04em; }
.stat-label { font-family:'JetBrains Mono',monospace; font-size:0.65rem;
              color:#3D5070 !important; font-weight:500; text-transform:uppercase; letter-spacing:0.08em; }

/* ── Upload Info Box ── */
.upload-info {
    background:rgba(29,78,216,0.06); border:1px solid rgba(59,130,246,0.18);
    border-radius:16px; padding:1.3rem 1.5rem; box-sizing:border-box;
    display:flex; flex-direction:column;
}
.upload-info-header {
    display:flex; align-items:center; gap:6px;
    font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase; color:#60A5FA !important; margin-bottom:0.7rem;
}
.upload-info p { font-size:0.84rem; color:#4E6380 !important; line-height:1.6; margin:0 0 0.9rem; }
.col-list { display:flex; flex-direction:column; gap:6px; flex:1; }
.col-item {
    display:flex; align-items:center; gap:8px; padding:0.5rem 0.8rem;
    background:rgba(7,16,31,0.7); border:1px solid rgba(255,255,255,0.04);
    border-radius:8px;
}
.col-dot { width:5px; height:5px; background:#3B82F6; border-radius:50%; flex-shrink:0; }
.col-name { font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#94A3B8 !important; }

/* ── Algorithm Step Box ── */
.algo-phase {
    background:rgba(10,18,35,0.8); border:1px solid rgba(255,255,255,0.05);
    border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1rem;
    position:relative; overflow:hidden;
}
.algo-phase-header {
    display:flex; align-items:center; gap:10px; margin-bottom:0.8rem;
}
.algo-phase-num {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:500;
    color:#1D4ED8; background:rgba(29,78,216,0.12); padding:3px 8px;
    border-radius:5px; letter-spacing:0.1em; flex-shrink:0;
}
.algo-phase-title { font-size:0.9rem; font-weight:700; color:#CBD5E1 !important; }
.algo-phase-body { font-size:0.82rem; color:#4E6380 !important; line-height:1.7; }
.algo-phase-body code {
    background:rgba(255,255,255,0.05); color:#93C5FD !important;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    padding:2px 6px; border-radius:4px;
}
.algo-phase-accent {
    position:absolute; left:0; top:0; bottom:0; width:3px;
    border-radius:16px 0 0 16px;
}
.algo-step-log {
    background:rgba(7,16,31,0.9); border:1px solid rgba(255,255,255,0.04);
    border-radius:12px; padding:1rem 1.2rem; margin-top:0.6rem;
    font-family:'JetBrains Mono',monospace; font-size:0.75rem; line-height:1.9;
}
.log-line { display:flex; gap:10px; align-items:flex-start; }
.log-tag {
    flex-shrink:0; padding:1px 7px; border-radius:4px;
    font-size:0.65rem; font-weight:600; letter-spacing:0.06em;
}
.log-tag.ok  { background:rgba(52,211,153,0.12); color:#34D399; }
.log-tag.run { background:rgba(251,191,36,0.12);  color:#FBBF24; }
.log-tag.inf { background:rgba(99,102,241,0.12);  color:#818CF8; }
.log-val { color:#E2E8F0 !important; }
.log-dim { color:#3D5070 !important; }

/* ── Formula Box ── */
.formula-box {
    background:rgba(7,16,31,0.9); border:1px solid rgba(59,130,246,0.15);
    border-radius:12px; padding:1rem 1.4rem; margin:0.6rem 0;
    font-family:'JetBrains Mono',monospace; font-size:0.82rem;
    color:#93C5FD !important; text-align:center; letter-spacing:0.02em;
}
.formula-label {
    font-family:'Geist',sans-serif; font-size:0.72rem;
    color:#3D5070 !important; text-align:center; margin-top:4px;
}

/* ── Pill Row ── */
.pill-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:1.5rem; align-items:center; }
.pill {
    background:rgba(59,130,246,0.09); color:#60A5FA !important;
    border:1px solid rgba(59,130,246,0.18); border-radius:7px;
    padding:4px 12px; font-size:0.76rem; font-weight:500;
}
.pill-label { font-size:0.76rem; color:#3D5070 !important; font-weight:500; }

/* ── Persona Cards ── */
.persona-card {
    background:rgba(10,18,35,0.85); border:1px solid rgba(255,255,255,0.05);
    border-radius:18px; overflow:hidden; margin-bottom:1rem;
    transition:border-color 0.25s, transform 0.2s, box-shadow 0.2s;
}
.persona-card:hover {
    border-color:rgba(59,130,246,0.28); transform:translateY(-2px);
    box-shadow:0 12px 32px rgba(0,0,0,0.35);
}
.persona-accent { height:3px; }
.persona-body   { padding:1.3rem 1.4rem; }
.persona-badge {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:0.65rem; font-weight:600; letter-spacing:0.08em;
    text-transform:uppercase; padding:3px 10px; border-radius:5px; margin-bottom:0.7rem;
}
.persona-name { font-size:0.95rem; font-weight:700; color:#E2E8F0 !important;
                margin-bottom:0.9rem; line-height:1.3; }
.persona-chips { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:0.8rem; }
.persona-chip {
    background:rgba(255,255,255,0.04); color:#64748B !important;
    border:1px solid rgba(255,255,255,0.05); border-radius:6px;
    padding:4px 10px; font-size:0.73rem; font-weight:500;
}
.persona-desc  { font-size:0.81rem; color:#4E6380 !important; line-height:1.6; }
.persona-footer {
    margin-top:0.9rem; padding-top:0.9rem;
    border-top:1px solid rgba(255,255,255,0.04);
    font-size:0.76rem; color:#3D5070 !important;
    display:flex; gap:0.9rem; flex-wrap:wrap;
}

/* ── Reco Cards ── */
.reco-card {
    background:rgba(10,18,35,0.85); border:1px solid rgba(255,255,255,0.05);
    border-radius:16px; padding:1.3rem 1.4rem; height:100%;
}
.reco-header {
    font-weight:700; font-size:0.88rem; color:#E2E8F0 !important;
    margin-bottom:1rem; display:flex; align-items:center; gap:8px;
}
.reco-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.reco-item {
    font-size:0.8rem; color:#4E6380 !important;
    margin:0.55rem 0; display:flex; gap:8px;
    align-items:flex-start; line-height:1.4;
}
.reco-item strong { color:#64748B !important; font-weight:600;
                    min-width:88px; flex-shrink:0; }

/* ── Divider ── */
.section-divider {
    height:1px; background:rgba(255,255,255,0.04);
    border:none; margin:2rem 0;
}

/* ── Streamlit Overrides ── */
.stButton > button {
    background:rgba(255,255,255,0.04) !important; color:#94A3B8 !important;
    border:1px solid rgba(255,255,255,0.08) !important; border-radius:12px !important;
    font-weight:600 !important; font-size:0.875rem !important;
    padding:0.85rem 1.4rem !important; box-shadow:none !important;
    transition:all 0.2s !important; margin-top:0.75rem !important;
}
.stButton > button:hover {
    background:rgba(255,255,255,0.07) !important;
    border-color:rgba(255,255,255,0.14) !important;
}
.stButton > button[kind="primary"] {
    background:#1D4ED8 !important; color:white !important;
    border:none !important; box-shadow:0 4px 14px rgba(29,78,216,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background:#1E40AF !important; box-shadow:0 6px 20px rgba(29,78,216,0.45) !important;
    transform:translateY(-1px) !important;
}
.stDataFrame { border-radius:12px !important; overflow:hidden;
               border:1px solid rgba(255,255,255,0.05) !important; }

/* ── Styled File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(10,18,35,0.7);
    border: 2px dashed rgba(59,130,246,0.35);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.6);
}
[data-testid="stFileUploader"] label {
    color: #94A3B8 !important;
    font-size: 0.88rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #64748B !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 0.85rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(59,130,246,0.12) !important;
    border: 1px solid rgba(59,130,246,0.28) !important;
    color: #60A5FA !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    margin-top: 0.5rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(59,130,246,0.22) !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill: #3B82F6 !important;
    opacity: 0.6;
}

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.01); }
::-webkit-scrollbar-thumb { background:rgba(59,130,246,0.28); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(59,130,246,0.48); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_COLORS      = ["#3B82F6", "#10B981", "#F59E0B", "#A78BFA"]
CLUSTER_COLORS_LIGHT= ["rgba(59,130,246,0.11)","rgba(16,185,129,0.11)",
                        "rgba(245,158,11,0.11)","rgba(167,139,250,0.11)"]
CLUSTER_COLORS_TEXT = ["#60A5FA","#34D399","#FCD34D","#C4B5FD"]
PHASE_COLORS        = ["#3B82F6","#10B981","#F59E0B","#A78BFA","#F43F5E","#0EA5E9"]

FITUR_LABELS = [
    "transaction_hour","product_category_enc","product_amount",
    "cashback","loyalty_points","payment_method_enc","device_type_enc"
]

DARK_LAYOUT = dict(
    plot_bgcolor="#08111E", paper_bgcolor="#08111E",
    font=dict(family="Geist, sans-serif", color="#4E6380"),
    title_font=dict(size=13, color="#94A3B8", family="Geist"),
    margin=dict(t=50,b=40,l=45,r=20),
)
DARK_GRID = dict(
    showgrid=True, gridcolor="rgba(255,255,255,0.03)",
    showline=True, linecolor="rgba(255,255,255,0.05)"
)

# ══════════════════════════════════════════════════════════════════════════════
#  MODUL ALGORITMA — CRISP-DM + KMeans
# ══════════════════════════════════════════════════════════════════════════════

def validate_columns(df: pd.DataFrame) -> bool:
    """Validasi 7 kolom wajib menggunakan set.issubset() sesuai metodologi."""
    required = {
        "transaction_date", "product_category", "product_amount",
        "cashback", "loyalty_points", "payment_method", "device_type"
    }
    return required.issubset(set(df.columns))


def fase2_data_understanding(df: pd.DataFrame) -> dict:
    """
    FASE 2 — Data Collection & Understanding
    EDA awal: total baris, kolom, missing values, distribusi variabel kategorikal.
    """
    stats = {
        "n_rows"         : len(df),
        "n_cols"         : len(df.columns),
        "missing_total"  : int(df.isnull().sum().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "dtypes"         : df.dtypes.astype(str).to_dict(),
        "numeric_desc"   : df.describe().to_dict(),
        # distribusi variabel kategorikal
        "unique_payment" : df["payment_method"].nunique() if "payment_method" in df.columns else 0,
        "unique_device"  : df["device_type"].nunique()    if "device_type"    in df.columns else 0,
        "unique_category": df["product_category"].nunique() if "product_category" in df.columns else 0,
        # distribusi payment method (%)
        "payment_dist"   : (df["payment_method"].value_counts(normalize=True)*100
                            ).round(1).to_dict() if "payment_method" in df.columns else {},
        # distribusi device type (%)
        "device_dist"    : (df["device_type"].value_counts(normalize=True)*100
                            ).round(1).to_dict() if "device_type" in df.columns else {},
    }
    return stats


def fase3_data_preparation(df: pd.DataFrame) -> tuple:
    """
    FASE 3 — Data Preparation (sesuai metodologi artikel):
      3a. Parsing datetime & ekstrak transaction_hour (0-23)
      3b. Label Encoding untuk payment_method, device_type, product_category
      3c. Seleksi 7 fitur → matrix X
      3d. Normalisasi Z-Score dengan StandardScaler
    """
    df = df.copy()
    prep_log = {}

    # 3a. Parsing tanggal & ekstrak jam — format DD/MM/YYYY HH:MM
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True, errors="coerce")
    df["transaction_hour"] = df["transaction_date"].dt.hour
    prep_log["3a"] = "pd.to_datetime(dayfirst=True) → .dt.hour → transaction_hour ∈ [0, 23]"

    # 3b. Label Encoding — mengubah teks ke integer ordinal
    le = LabelEncoder()
    le_info = {}
    for col, new_col in [
        ("payment_method",   "payment_method_enc"),
        ("device_type",      "device_type_enc"),
        ("product_category", "product_category_enc"),
    ]:
        df[new_col] = le.fit_transform(df[col].astype(str))
        le_info[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        prep_log[f"3b_{col}"] = f"LabelEncoder(): {dict(list(le_info[col].items())[:4])}"

    # 3c. Seleksi 7 fitur → array features
    features = [
        "transaction_hour",
        "product_category_enc",
        "product_amount",
        "cashback",
        "loyalty_points",
        "payment_method_enc",
        "device_type_enc",
    ]
    prep_log["3c"] = f"features = {features}"

    X_raw    = df[features].dropna()
    # 3d. Normalisasi Z-Score: x' = (x − μ) / σ
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    prep_log["3d"] = (
        f"StandardScaler: μ={scaler.mean_.round(3)}, σ={scaler.scale_.round(3)}"
    )

    return df, X_scaled, features, le_info, scaler, prep_log


def fase4a_elbow_method(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple:
    """
    FASE 4a — Elbow Method
    Loop k=2..10: fit KMeans lalu simpan .inertia_ (WCSS).
    Kurva WCSS diplot, titik siku = k kandidat optimal.
    """
    k_range   = list(range(k_min, k_max + 1))
    wcss      = []
    elbow_log = []

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=42
        )
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        elbow_log.append({
            "k"           : k,
            "wcss"        : round(km.inertia_, 2),
            "n_iterations": km.n_iter_,
        })

    return k_range, wcss, elbow_log


def fase4b_silhouette_analysis(X_scaled: np.ndarray, k_range: list) -> tuple:
    """
    FASE 4b — Silhouette Score
    Hitung silhouette_score(X_scaled, km.labels_) per k.
    k dengan np.argmax(sil_scores) ditetapkan sebagai optimal_k.
    Kriteria keberhasilan: score >= 0.4 (sesuai Business Understanding).
    """
    sil_scores = []
    sil_log    = []

    for k in k_range:
        km     = KMeans(n_clusters=k, init="k-means++",
                        max_iter=300, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        sil_log.append({
            "k"    : k,
            "score": round(score, 4),
            "best" : False,
        })

    best_idx = int(np.argmax(sil_scores))
    sil_log[best_idx]["best"] = True

    return sil_scores, sil_log


def fase4c_kmeans_final(X_scaled: np.ndarray, k: int) -> tuple:
    """
    FASE 4c — Training Model Final (run_kmeans)
    KMeans.fit_predict() → array label integer per baris transaksi.
    Menggunakan k-means++ init untuk inisialisasi centroid yang lebih baik.
    """
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    )
    labels = km.fit_predict(X_scaled)

    km_log = {
        "k"              : k,
        "n_iterations"   : km.n_iter_,
        "final_inertia"  : round(km.inertia_, 4),
        "centroids_shape": km.cluster_centers_.shape,
        "convergence"    : km.n_iter_ < km.max_iter,
        "cluster_sizes"  : {int(i): int(np.sum(labels == i)) for i in range(k)},
    }

    return labels, km, km_log


def fase4d_pca_visualization(X_scaled: np.ndarray, labels: np.ndarray) -> tuple:
    """
    FASE 4d — Reduksi PCA 2D (hanya untuk visualisasi)
    PCA(n_components=2).fit_transform(X_scaled) → koordinat 2D per transaksi.
    explained_variance_ratio_ ditampilkan di judul scatter plot.
    Clustering sesungguhnya tetap di ruang 7D.
    """
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pca_log = {
        "explained_var_ratio" : pca.explained_variance_ratio_.round(4).tolist(),
        "total_var_explained" : round(sum(pca.explained_variance_ratio_) * 100, 2),
        "n_components"        : 2,
        "note"                : "PCA hanya untuk visualisasi — label cluster dari K-Means 7D"
    }

    return coords, pca.explained_variance_ratio_, pca_log


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 5 — EVALUATION: Profiling & Persona
# ══════════════════════════════════════════════════════════════════════════════

def fase5_profiling(df_proc: pd.DataFrame) -> pd.DataFrame:
    """
    Profiling cluster menggunakan df.groupby("cluster").agg():
    - mean untuk product_amount, cashback, loyalty_points
    - mode()[0] untuk payment_method, device_type, transaction_hour, product_category
    """
    profile = df_proc.groupby("cluster").agg(
        jumlah   = ("cluster",          "count"),
        nominal  = ("product_amount",    "mean"),
        cashback = ("cashback",          "mean"),
        poin     = ("loyalty_points",    "mean"),
        jam      = ("transaction_hour",  lambda x: x.mode()[0]),
        metode   = ("payment_method",    lambda x: x.mode()[0]),
        device   = ("device_type",       lambda x: x.mode()[0]),
        kategori = ("product_category",  lambda x: x.mode()[0]),
    ).reset_index()
    return profile


def get_persona_name(row) -> str:
    """
    Fungsi get_persona_name(row):
    Klasifikasi sesi hari dari jam puncak → level nominal.
    Malam ≥20 atau <5 / Pagi <12 / Siang <17 / Sore.
    Premium >6.000 / Menengah >3.000 / Hemat.
    """
    sesi  = "Malam"   if row["jam"] >= 20 or row["jam"] < 5 else \
            "Pagi"    if row["jam"] < 12 else \
            "Siang"   if row["jam"] < 17 else "Sore"
    level = "Premium" if row["nominal"] > 6000 else \
            "Menengah" if row["nominal"] > 3000 else "Hemat"
    return f"Pengguna {level} · {row['device']} · {sesi}"


def get_persona_desc(row) -> str:
    """
    Fungsi get_persona_desc(row):
    2 kalimat deskriptif otomatis: jam dominan, device, metode, kategori.
    """
    return (f"Transaksi dominan pukul {row['jam']:02d}:00 melalui {row['device']}. "
            f"Metode favorit <strong style='color:#93C5FD'>{row['metode']}</strong> "
            f"dengan fokus kategori "
            f"<strong style='color:#CBD5E1'>{row['kategori']}</strong>.")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def gs(key, default=None):
    return st.session_state.get(key, default)

def render_algo_phase(num: str, title: str, body: str, color: str, log_html: str = ""):
    st.markdown(f"""
    <div class="algo-phase">
        <div class="algo-phase-accent" style="background:{color}"></div>
        <div class="algo-phase-header">
            <span class="algo-phase-num">{num}</span>
            <span class="algo-phase-title">{title}</span>
        </div>
        <div class="algo-phase-body">{body}</div>
        {log_html}
    </div>
    """, unsafe_allow_html=True)

def make_log_html(lines: list) -> str:
    inner = ""
    for tag, text in lines:
        inner += (f'<div class="log-line">'
                  f'<span class="log-tag {tag}">{tag.upper()}</span>'
                  f'<span class="log-val">{text}</span></div>')
    return f'<div class="algo-step-log">{inner}</div>'


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-eyebrow">Fintech Analytics · CRISP-DM Framework</div>
    <h1>Analisis Cluster E-Wallet</h1>
    <p>Segmentasi pola perilaku transaksi digital menggunakan K-Means Clustering
       dengan pendekatan 5 fase CRISP-DM.</p>
    <div class="header-badge">
        <div class="badge-icon">🧩</div>
        <div>
            <span class="badge-title">Behavioral Intelligence</span>
            <div class="badge-sub">
                <span class="badge-dot"></span>
                <span class="badge-status">System Ready</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# STEP BAR
step_now    = gs("step", "upload")
steps       = ["upload", "preview", "analisis", "hasil"]
step_icons  = ["↑", "⌕", "◎", "⊞"]
step_labels = ["Unggah Data", "Preview & Validasi", "Proses Algoritma", "Hasil & Insight"]
step_nums   = ["01", "02", "03", "04"]

bar = '<div class="step-bar">'
for s, icon, label, num in zip(steps, step_icons, step_labels, step_nums):
    idx_now = steps.index(step_now)
    idx_s   = steps.index(s)
    cls = "active" if s == step_now else ("done" if idx_s < idx_now else "step")
    bar += (f'<div class="step {cls}">'
            f'<span class="step-num">({num})</span> {icon} {label}</div>')
bar += "</div>"
st.markdown(bar, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if step_now == "upload":
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">🗄</span> Unggah File CSV Transaksi
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="upload-info">
            <div class="upload-info-header">ⓘ &nbsp;Persyaratan Data</div>
            <p>File CSV harus memiliki tepat <strong style="color:#60A5FA">7 kolom</strong>
               berikut. Separator kolom menggunakan titik koma
               <code style="color:#93C5FD;background:rgba(255,255,255,0.05);
               padding:1px 5px;border-radius:3px">;</code></p>
            <div class="col-list">
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">transaction_date</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Datetime · DD/MM/YYYY HH:MM</span>
                </div>
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">product_category</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Kategorikal · 20 kategori</span>
                </div>
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">product_amount</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Numerik · 10–9.997</span>
                </div>
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">cashback</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Numerik · 0–100</span>
                </div>
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">loyalty_points</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Numerik · 0–999</span>
                </div>
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">payment_method</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Kategorikal · 5 metode</span>
                </div>
                <div class="col-item"><span class="col-dot"></span>
                    <span class="col-name">device_type</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Kategorikal · 3 platform</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "☁️  Seret & lepas file CSV, atau klik **Browse files**",
            type=["csv"],
            key="file_uploader",
            help="Format CSV dengan separator titik koma (;) · Maks. 200MB"
        )

        # ── Setelah file dipilih: tampilkan preview dulu, tunggu konfirmasi user
        if uploaded is not None:
            try:
                df_preview = pd.read_csv(uploaded, sep=";")
                if not validate_columns(df_preview):
                    st.error("❌ Kolom tidak sesuai. Pastikan CSV memiliki 7 kolom wajib.")
                else:
                    # Simpan sementara ke session state tapi BELUM pindah step
                    st.session_state["df_pending"] = df_preview

                    n_rows = len(df_preview)
                    n_cols = len(df_preview.columns)
                    missing = int(df_preview.isnull().sum().sum())

                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);
                                border-radius:12px;padding:1rem 1.2rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                    font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                                    color:#34D399;margin-bottom:0.6rem;">✓ File Terdeteksi</div>
                        <div style="font-size:0.82rem;color:#64748B;line-height:1.8;">
                            📄 <b style="color:#94A3B8">{uploaded.name}</b><br>
                            📊 {n_rows:,} baris &nbsp;·&nbsp; {n_cols} kolom &nbsp;·&nbsp;
                            {'<span style="color:#F87171">'+str(missing)+' missing</span>' if missing > 0
                             else '<span style="color:#34D399">0 missing ✓</span>'}
                        </div>
                        <div style="margin-top:0.7rem;font-size:0.75rem;color:#3D5070;">
                            Preview 3 baris pertama:
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(df_preview.head(3), use_container_width=True, height=140)

                    if st.button("✅  Gunakan File Ini — Lanjut ke Preview & Validasi",
                                 type="primary", use_container_width=True):
                        st.session_state["df_raw"]    = df_preview
                        st.session_state["df_pending"] = None
                        st.session_state["step"]       = "preview"
                        st.rerun()

            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

        else:
            # Tidak ada file yang dipilih — tampilkan tombol data contoh
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button(
            "🗄  Gunakan Data Contoh (project_Data_ewallet.csv)",
            use_container_width=True
        )

        if use_sample:
            try:
                df = pd.read_csv("/mnt/user-data/uploads/project_Data_ewallet.csv", sep=";")
                st.session_state["df_raw"] = df
                st.session_state["step"]   = "preview"
                st.rerun()
            except Exception as e:
                st.error(f"❌ File contoh tidak ditemukan: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PREVIEW & VALIDASI (FASE 2: Data Collection & Understanding)
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "preview":
    df = gs("df_raw")
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">⌕</span> Preview & Validasi — FASE 2: Data Collection & Understanding
    </div>
    """, unsafe_allow_html=True)

    du      = fase2_data_understanding(df)
    missing = du["missing_total"]

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-number">{du['n_rows']:,}</div>
            <div class="stat-label">Total Transaksi</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{du['n_cols']}</div>
            <div class="stat-label">Kolom Terdeteksi</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{missing}</div>
            <div class="stat-label">Missing Values</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{du['unique_payment']}</div>
            <div class="stat-label">Metode Pembayaran</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{du['unique_device']}</div>
            <div class="stat-label">Jenis Perangkat</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{du['unique_category']}</div>
            <div class="stat-label">Kategori Produk</div>
        </div>
    </div>
    <div class="pill-row">
        <span class="pill-label">7 Fitur yang akan dianalisis :</span>
        {''.join(f'<span class="pill">{f}</span>' for f in FITUR_LABELS)}
    </div>
    """, unsafe_allow_html=True)

    # Distribusi variabel kategorikal
    pay_dist_str = "  |  ".join(
        [f"{k}: {v:.1f}%" for k, v in list(du["payment_dist"].items())[:5]]
    )
    dev_dist_str = "  |  ".join(
        [f"{k}: {v:.1f}%" for k, v in list(du["device_dist"].items())[:3]]
    )

    render_algo_phase(
        "FASE 02", "DATA COLLECTION & UNDERSTANDING",
        f"""Eksplorasi awal sebelum pemrosesan. Dataset berisi
        <code>{du['n_rows']:,}</code> record transaksi dengan <code>{du['n_cols']}</code> kolom.
        <br><br>
        <b>Distribusi payment_method:</b> {pay_dist_str}<br>
        <b>Distribusi device_type:</b> {dev_dist_str}<br><br>
        Validasi kolom dilakukan dengan <code>set.issubset()</code> — memastikan 7 kolom wajib ada.
        Missing values dideteksi dengan <code>df.isnull().sum()</code>.
        Missing values <code>{missing}</code> {'→ akan di-drop saat preprocessing.' if missing > 0 else '✓ — data bersih, tidak ada missing values.'}""",
        PHASE_COLORS[1],
        make_log_html([
            ("ok",  f"validate_columns() → 7 kolom wajib terverifikasi via set.issubset()"),
            ("ok",  f"n_rows = {du['n_rows']:,}  |  n_cols = {du['n_cols']}"),
            ("inf", f"df.isnull().sum() → missing_total = {missing}"),
            ("inf", f"payment_method: {du['unique_payment']} metode unik"),
            ("inf", f"device_type: {du['unique_device']} platform unik"),
            ("inf", f"product_category: {du['unique_category']} kategori unik"),
            ("ok" if missing == 0 else "run",
             f"Status missing values: {'Bersih ✓' if missing == 0 else f'{missing} baris → akan di-drop pada fase preparasi'}"),
        ])
    )

    if missing > 0:
        st.warning(f"⚠️ {missing} nilai kosong akan dihapus saat preprocessing.")

    st.markdown('<p style="font-size:0.85rem;color:#3D5070;margin:1rem 0 0.5rem;font-weight:600;">5 Baris Pertama Data</p>',
                unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True, height=220)

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("← Ganti File"):
            st.session_state["step"] = "upload"; st.rerun()
    with c2:
        if st.button("Lanjut ke Proses Algoritma →", type="primary", use_container_width=True):
            st.session_state["step"] = "analisis"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — ANALISIS (FASE 1 Business Understanding + FASE 3-4 Modeling)
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "analisis":
    df = gs("df_raw")
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">⚙</span> Pipeline K-Means Clustering — CRISP-DM Fase 1, 3 & 4
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Memulai pipeline CRISP-DM…")
    status_text  = st.empty()

    # ── FASE 1: Business Understanding
    render_algo_phase(
        "FASE 01", "BUSINESS UNDERSTANDING",
        """Tujuan analisis: mengidentifikasi pola perilaku pengguna e-wallet
        menggunakan <b>7 variabel gabungan</b> — waktu transaksi, jenis produk,
        nilai transaksi, insentif (cashback & poin), metode pembayaran, dan jenis perangkat.
        <br><br>
        <b>Kriteria keberhasilan</b> (ditetapkan via <code>sklearn.metrics</code>):
        Silhouette Score <code>≥ 0.4</code> → cluster ideal dengan karakteristik unik
        yang dapat diinterpretasi secara bisnis.
        <br><br>
        <b>Output yang diharapkan:</b> Pengelompokan deskriptif yang memetakan
        gaya hidup digital pengguna + rekomendasi strategi pemasaran personal.""",
        PHASE_COLORS[0],
        make_log_html([
            ("inf", "Tujuan: segmentasi perilaku transaksi digital pengguna e-wallet"),
            ("inf", "Variabel: 7 fitur gabungan (waktu + insentif + teknologi akses)"),
            ("inf", "Kriteria sukses: Silhouette Score ≥ 0.4"),
            ("inf", "Library evaluasi: sklearn.metrics.silhouette_score"),
        ])
    )
    progress_bar.progress(10, text="Fase 1 selesai…")

    # ── FASE 3: Data Preparation
    status_text.markdown("⚙️ **FASE 3** — Data Preparation…")
    time.sleep(0.3)
    progress_bar.progress(20, text="Fase 3: Preprocessing data…")

    df_proc, X_scaled, features, le_info, scaler, prep_log = fase3_data_preparation(df)

    render_algo_phase(
        "FASE 03", "DATA PREPARATION",
        """<b>3a. Parsing Datetime & Ekstraksi Fitur Waktu</b><br>
        <code>pd.to_datetime(dayfirst=True, errors="coerce")</code> pada kolom
        <code>transaction_date</code>, lalu <code>.dt.hour</code> → kolom baru
        <code>transaction_hour</code> ∈ [0, 23] sebagai fitur ritme harian.
        <br><br>
        <b>3b. Label Encoding</b><br>
        <code>LabelEncoder()</code> dipakai pada <code>payment_method</code>,
        <code>device_type</code>, dan <code>product_category</code> — mengubah
        variabel teks menjadi integer ordinal agar dapat diproses K-Means.
        <br><br>
        <b>3c. Seleksi 7 Fitur</b><br>
        Array <code>features</code> diisi: <code>transaction_hour</code>,
        <code>product_category_enc</code>, <code>product_amount</code>,
        <code>cashback</code>, <code>loyalty_points</code>,
        <code>payment_method_enc</code>, <code>device_type_enc</code> → matrix X.
        <br><br>
        <b>3d. Normalisasi Z-Score</b><br>
        <code>StandardScaler().fit_transform(X)</code> → mean=0, std=1.
        Wajib dilakukan agar perbedaan skala tidak mendominasi jarak Euclidean K-Means
        (misal <code>product_amount</code> dalam ribuan vs <code>device_type_enc</code> 0-2).""",
        PHASE_COLORS[2],
        make_log_html([
            ("ok",  f"3a. transaction_hour diekstrak: {df_proc['transaction_hour'].min():.0f}–{df_proc['transaction_hour'].max():.0f}"),
            ("ok",  f"3b. LabelEncoding payment_method → {dict(list(le_info.get('payment_method',{}).items())[:3])}"),
            ("ok",  f"3b. LabelEncoding device_type    → {dict(list(le_info.get('device_type',{}).items())[:3])}"),
            ("ok",  f"3b. LabelEncoding product_category → {dict(list(le_info.get('product_category',{}).items())[:3])}"),
            ("ok",  f"3c. Matrix X shape = {X_scaled.shape}  (baris × 7 fitur)"),
            ("ok",  f"3d. μ (mean) = {scaler.mean_.round(2)}"),
            ("ok",  f"3d. σ (std)  = {scaler.scale_.round(2)}"),
        ])
    )
    st.markdown("""
    <div class="formula-box">x' = (x − μ) / σ</div>
    <div class="formula-label">Rumus Standardisasi Z-Score — menyamakan skala semua 7 fitur</div>
    """, unsafe_allow_html=True)

    # ── FASE 4a: Elbow Method
    status_text.markdown("📈 **FASE 4a** — Elbow Method: k=2..10…")
    progress_bar.progress(35, text="Fase 4a: Elbow Method…")

    k_range, wcss, elbow_log = fase4a_elbow_method(X_scaled)

    render_algo_phase(
        "FASE 04a", "MODELING — Elbow Method (k=2–10)",
        """Loop <code>k</code> dari 2 hingga 10. Tiap iterasi jalankan
        <code>KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)</code>
        lalu simpan <code>.inertia_</code> (WCSS = Within-Cluster Sum of Squares).
        <br><br>
        WCSS diplot sebagai kurva Elbow dengan Plotly <code>go.Scatter</code> dan
        garis vertikal putus-putus <code>fig.add_vline</code> di posisi k optimal.
        Titik "siku" kurva = kandidat k optimal.""",
        PHASE_COLORS[0],
        make_log_html([
            ("run", f"Loop k = {k_range}"),
        ] + [
            ("ok", f"k={e['k']}: WCSS={e['wcss']:,.1f}  |  n_iter={e['n_iterations']}")
            for e in elbow_log
        ])
    )
    st.markdown("""
    <div class="formula-box">WCSS = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²</div>
    <div class="formula-label">Within-Cluster Sum of Squares — semakin kecil semakin padat cluster</div>
    """, unsafe_allow_html=True)

    # ── FASE 4b: Silhouette
    status_text.markdown("📊 **FASE 4b** — Silhouette Score…")
    progress_bar.progress(55, text="Fase 4b: Silhouette Analysis…")

    sil_scores, sil_log = fase4b_silhouette_analysis(X_scaled, k_range)
    optimal_k = k_range[int(np.argmax(sil_scores))]
    best_sil  = max(sil_scores)
    threshold_met = "✓ Memenuhi kriteria" if best_sil >= 0.4 else "⚠ Di bawah threshold"

    render_algo_phase(
        "FASE 04b", "MODELING — Silhouette Score (Validasi k Optimal)",
        f"""Dalam loop yang sama, hitung <code>silhouette_score(X_scaled, km.labels_)</code> per k.
        k dengan <code>np.argmax(sil_scores)</code> ditetapkan sebagai <code>optimal_k</code>.
        Bar chart Plotly dengan bar tertinggi disorot warna solid.
        <br><br>
        <b>Formula per titik data:</b><br>
        • <b>a(i)</b> = rata-rata jarak ke anggota cluster sendiri (cohesion)<br>
        • <b>b(i)</b> = rata-rata jarak ke cluster terdekat lainnya (separation)<br><br>
        Score mendekati +1 = cluster sangat padat & terpisah baik.
        Hasil: <code>optimal_k = {optimal_k}</code>, score = <code>{best_sil:.4f}</code>
        &nbsp;→ <b>{threshold_met} (≥ 0.4)</b>""",
        PHASE_COLORS[3],
        make_log_html([
            ("run", "silhouette_score(X_scaled, km.labels_) per k"),
        ] + [
            ("ok" if e["best"] else "inf",
             f"k={e['k']}: score={e['score']}  {'← OPTIMAL ✓ | np.argmax(sil_scores)' if e['best'] else ''}")
            for e in sil_log
        ] + [
            ("ok" if best_sil >= 0.4 else "run",
             f"Kriteria ≥ 0.4: {threshold_met} → optimal_k = {optimal_k}"),
        ])
    )
    st.markdown("""
    <div class="formula-box">s(i) = (b(i) − a(i)) / max(a(i), b(i))</div>
    <div class="formula-label">Silhouette Score per titik — rata-rata seluruh titik = score keseluruhan</div>
    """, unsafe_allow_html=True)

    # ── FASE 4c: K-Means Final
    status_text.markdown(f"🎯 **FASE 4c** — Training K-Means Final k={optimal_k}…")
    progress_bar.progress(72, text=f"Fase 4c: K-Means k={optimal_k}…")

    labels, km_model, km_log = fase4c_kmeans_final(X_scaled, optimal_k)

    render_algo_phase(
        "FASE 04c", f"MODELING — Training Model Final K-Means (k={optimal_k})",
        f"""Fungsi <code>run_kmeans(X_scaled, optimal_k)</code> →
        <code>KMeans.fit_predict()</code> menghasilkan array label integer per baris transaksi.
        Label disimpan di <code>st.session_state["labels"]</code> dan
        ditambahkan ke DataFrame sebagai kolom <code>cluster</code>.
        <br><br>
        <b>Assignment Step (E-Step):</b> setiap titik di-assign ke centroid terdekat
        berdasarkan jarak Euclidean.<br>
        <b>Update Step (M-Step):</b> centroid diperbarui ke rata-rata anggota cluster.
        <br><br>
        Inisialisasi <b>k-means++</b>: memilih centroid awal yang tersebar jauh satu sama lain,
        menghindari konvergensi ke local minimum.
        Konvergen setelah <code>{km_log['n_iterations']} iterasi</code>,
        WCSS final = <code>{km_log['final_inertia']:,.2f}</code>.""",
        PHASE_COLORS[4],
        make_log_html([
            ("run", f"KMeans(n_clusters={optimal_k}, init='k-means++', max_iter=300, n_init=10, random_state=42)"),
            ("run", ".fit_predict(X_scaled) → array label integer per transaksi"),
            ("ok",  f"Konvergen dalam {km_log['n_iterations']} iterasi"),
            ("ok",  f"Final WCSS (inertia) = {km_log['final_inertia']:,.2f}"),
            ("ok",  f"Distribusi cluster: { {f'C{k}':v for k,v in km_log['cluster_sizes'].items()} }"),
            ("inf", "Label disimpan → st.session_state['labels'] + df['cluster']"),
        ])
    )
    st.markdown("""
    <div class="formula-box">c(x) = argminᵢ ‖x − μᵢ‖²  &nbsp;→&nbsp;  μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x</div>
    <div class="formula-label">Assignment Step (kiri) dan Update Step (kanan) — diulang hingga label tidak berubah</div>
    """, unsafe_allow_html=True)

    # ── FASE 4d: PCA Visualization
    status_text.markdown("🗺 **FASE 4d** — PCA 2D untuk visualisasi…")
    progress_bar.progress(88, text="Fase 4d: PCA Dimensionality Reduction…")

    coords, var_ratio, pca_log = fase4d_pca_visualization(X_scaled, labels)

    render_algo_phase(
        "FASE 04d", "MODELING — Reduksi PCA 2D (Visualisasi)",
        f"""<code>PCA(n_components=2, random_state=42).fit_transform(X_scaled)</code>
        menghasilkan koordinat 2D tiap transaksi.
        <code>explained_variance_ratio_</code> ditampilkan di judul scatter plot
        sebagai persentase variansi yang dijelaskan.
        <br><br>
        <b>Catatan penting:</b> PCA digunakan <i>hanya untuk visualisasi</i>.
        Clustering sesungguhnya terjadi di ruang 7 dimensi — label cluster berasal
        dari K-Means 7D, bukan dari PCA.
        <br><br>
        PC1 menjelaskan <code>{var_ratio[0]*100:.1f}%</code> variansi,
        PC2 menjelaskan <code>{var_ratio[1]*100:.1f}%</code>.
        Total informasi dipertahankan: <code>{pca_log['total_var_explained']:.1f}%</code>.""",
        PHASE_COLORS[5],
        make_log_html([
            ("run", "PCA(n_components=2, random_state=42).fit_transform(X_scaled)"),
            ("ok",  f"PC1 explained_variance_ratio_ = {var_ratio[0]*100:.2f}%"),
            ("ok",  f"PC2 explained_variance_ratio_ = {var_ratio[1]*100:.2f}%"),
            ("inf", f"Total variansi dipertahankan  = {pca_log['total_var_explained']:.1f}%"),
            ("inf", "Label cluster = K-Means 7D (bukan dari PCA)"),
        ])
    )

    # Simpan state
    st.session_state.update({
        "df_proc"  : df_proc,
        "X_scaled" : X_scaled,
        "features" : features,
        "le_info"  : le_info,
        "scaler"   : scaler,
        "k_range"  : k_range,
        "wcss"     : wcss,
        "sil_scores": sil_scores,
        "sil_log"  : sil_log,
        "optimal_k": optimal_k,
        "labels"   : labels,
        "km_model" : km_model,
        "km_log"   : km_log,
        "coords"   : coords,
        "var_ratio": var_ratio,
    })

    progress_bar.progress(100, text="✅ Semua fase selesai!")
    time.sleep(0.4)
    status_text.empty()

    st.success(
        f"✅ Pipeline CRISP-DM selesai! **{optimal_k} cluster optimal** terbentuk "
        f"(Silhouette Score = {best_sil:.4f} — "
        f"{'Memenuhi' if best_sil >= 0.4 else 'Di bawah'} kriteria ≥ 0.4)"
    )

    if st.button("Lihat Hasil & Insight →", type="primary", use_container_width=True):
        st.session_state["step"] = "hasil"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — HASIL (FASE 5: Evaluation & Deployment)
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "hasil":
    df_proc    = gs("df_proc")
    wcss       = gs("wcss")
    sil_scores = gs("sil_scores")
    k_range    = gs("k_range")
    optimal_k  = gs("optimal_k")
    labels     = gs("labels")
    coords     = gs("coords")
    var_ratio  = gs("var_ratio")
    km_log     = gs("km_log")

    df_proc            = df_proc.iloc[:len(labels)].copy()
    df_proc["cluster"] = labels

    best_sil = max(sil_scores)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-number">{len(df_proc):,}</div>
            <div class="stat-label">Total Transaksi</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{optimal_k}</div>
            <div class="stat-label">Cluster Optimal</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{best_sil:.3f}</div>
            <div class="stat-label">Silhouette Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{sum(var_ratio)*100:.1f}%</div>
            <div class="stat-label">Variansi PCA 2D</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{km_log['n_iterations']}</div>
            <div class="stat-label">Iterasi K-Means</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{'✓' if best_sil >= 0.4 else '✗'}</div>
            <div class="stat-label">Kriteria ≥ 0.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Elbow + Silhouette Charts
    st.markdown("""<div class="section-title"><span class="section-title-icon">📈</span>
    Penentuan Cluster Optimal — Elbow Method + Silhouette Score</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=k_range, y=wcss, mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=8, color="#2563EB", line=dict(color="#08111E", width=2)),
            name="WCSS"
        ))
        fig_e.add_vline(x=optimal_k, line_dash="dot", line_color="#60A5FA",
                        line_width=1.5,
                        annotation_text=f"k = {optimal_k}  (optimal)",
                        annotation_font=dict(color="#60A5FA", size=11))
        fig_e.update_layout(title="Elbow Method — WCSS per k (fig.add_vline di k optimal)",
                            xaxis_title="Jumlah Cluster (k)",
                            yaxis_title="WCSS / Inertia", height=320, **DARK_LAYOUT)
        fig_e.update_xaxes(**DARK_GRID, tickvals=k_range)
        fig_e.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_e, use_container_width=True)

    with c2:
        bar_colors = [
            "#2563EB" if i == int(np.argmax(sil_scores)) else "rgba(37,99,235,0.22)"
            for i in range(len(k_range))
        ]
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=k_range, y=[round(s, 3) for s in sil_scores],
            marker=dict(color=bar_colors, cornerradius=6),
            text=[f"{s:.3f}" for s in sil_scores],
            textposition="outside",
            textfont=dict(size=11, color="#4E6380"),
            name="Silhouette"
        ))
        # Threshold line ≥ 0.4
        fig_s.add_hline(y=0.4, line_dash="dot", line_color="#34D399",
                        line_width=1.2,
                        annotation_text="threshold ≥ 0.4",
                        annotation_font=dict(color="#34D399", size=10))
        fig_s.update_layout(
            title="Silhouette Score per k — bar solid = np.argmax(sil_scores)",
            xaxis_title="Jumlah Cluster (k)",
            yaxis_title="Silhouette Score",
            yaxis_range=[0, max(sil_scores) * 1.28],
            height=320, **DARK_LAYOUT)
        fig_s.update_xaxes(showgrid=False, tickvals=k_range,
                           showline=True, linecolor="rgba(255,255,255,0.05)")
        fig_s.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Scatter PCA + Profil Tabel
    st.markdown("""<div class="section-title"><span class="section-title-icon">🗺</span>
    Sebaran Cluster — Proyeksi PCA 2D + Profil Statistik</div>""", unsafe_allow_html=True)

    c3, c4 = st.columns([1.3, 1])
    with c3:
        df_pca = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "Cluster": [f"Cluster {l}" for l in labels]
        })
        fig_sc = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=CLUSTER_COLORS,
            title=f"PCA Scatter — PC1 {var_ratio[0]*100:.1f}%  ·  PC2 {var_ratio[1]*100:.1f}%  ·  Total {sum(var_ratio)*100:.1f}%",
            opacity=0.65, height=420
        )
        fig_sc.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig_sc.update_layout(
            **DARK_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=-0.24,
                        xanchor="center", x=0.5, font=dict(color="#4E6380"))
        )
        fig_sc.update_xaxes(**DARK_GRID)
        fig_sc.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_sc, use_container_width=True)

    with c4:
        # FASE 5: Profiling — df.groupby("cluster").agg()
        profile_raw = fase5_profiling(df_proc)

        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#64748B;margin-bottom:0.5rem;">Profil Cluster — df.groupby("cluster").agg()</p>',
                    unsafe_allow_html=True)
        profile_display = profile_raw.copy()
        profile_display["nominal"]  = profile_display["nominal"].map("Rp{:,.0f}".format)
        profile_display["cashback"] = profile_display["cashback"].map("{:.1f}".format)
        profile_display["poin"]     = profile_display["poin"].map("{:.0f}".format)
        profile_display["jam"]      = profile_display["jam"].map(lambda x: f"{int(x):02d}:00")
        profile_display["cluster"]  = profile_display["cluster"].map(lambda x: f"C{x}")
        profile_display.columns = ["Cluster","N","Nominal","Cashback","Poin","Jam","Metode","Perangkat","Kategori"]
        st.dataframe(profile_display.set_index("Cluster"), use_container_width=True, height=360)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── FASE 5: Persona (get_persona_name + get_persona_desc)
    st.markdown("""<div class="section-title"><span class="section-title-icon">🎯</span>
    FASE 5: Evaluation — Persona Cluster</div>""", unsafe_allow_html=True)

    cols = st.columns(min(optimal_k, 4))
    for _, row in profile_raw.iterrows():
        c   = int(row["cluster"])
        clr = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        clt = CLUSTER_COLORS_TEXT[c % len(CLUSTER_COLORS_TEXT)]
        cll = CLUSTER_COLORS_LIGHT[c % len(CLUSTER_COLORS_LIGHT)]
        pct = row["jumlah"] / len(df_proc) * 100

        with cols[c % len(cols)]:
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-accent"
                     style="background:linear-gradient(90deg,{clr},{clr}44)"></div>
                <div class="persona-body">
                    <span class="persona-badge"
                          style="background:{cll};color:{clt}">
                        Cluster {c} &nbsp;·&nbsp; {pct:.1f}% &nbsp;·&nbsp; n={row['jumlah']:,}
                    </span>
                    <div class="persona-name">{get_persona_name(row)}</div>
                    <div class="persona-chips">
                        <span class="persona-chip">⏰ {int(row['jam']):02d}:00</span>
                        <span class="persona-chip">📱 {row['device']}</span>
                        <span class="persona-chip">Rp {row['nominal']:,.0f}</span>
                    </div>
                    <div class="persona-desc">{get_persona_desc(row)}</div>
                    <div class="persona-footer">
                        <span>💳 {row['metode']}</span>
                        <span>⭐ {row['poin']:.0f} poin</span>
                        <span>🛍 {row['kategori']}</span>
                        <span>💸 cb {row['cashback']:.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Rekomendasi Pemasaran
    st.markdown("""<div class="section-title"><span class="section-title-icon">💡</span>
    Rekomendasi Strategi Pemasaran per Cluster</div>""", unsafe_allow_html=True)

    reco_cols = st.columns(min(optimal_k, 4))
    for _, row in profile_raw.iterrows():
        c      = int(row["cluster"])
        clr    = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        # reward: cashback vs poin/10
        reward = "Cashback" if row["cashback"] > row["poin"] / 10 else "Poin Loyalitas"
        waktu  = f"{int(row['jam']):02d}:00 – {(int(row['jam']) + 2) % 24:02d}:00"

        with reco_cols[c % len(reco_cols)]:
            st.markdown(f"""
            <div class="reco-card">
                <div class="reco-header">
                    <span class="reco-dot" style="background:{clr}"></span>
                    Cluster {c} — {get_persona_name(row).split('·')[0].strip()}
                </div>
                <div class="reco-item"><strong>⏰ Waktu Promosi</strong>{waktu}</div>
                <div class="reco-item"><strong>🎁 Jenis Reward</strong>{reward}</div>
                <div class="reco-item"><strong>📲 Kanal Distribusi</strong>{row['device']} App</div>
                <div class="reco-item"><strong>🛒 Fokus Produk</strong>{row['kategori']}</div>
                <div class="reco-item"><strong>💳 Metode Favorit</strong>{row['metode']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Ringkasan Pipeline (untuk presentasi)
    st.markdown("""<div class="section-title"><span class="section-title-icon">📋</span>
    Ringkasan Pipeline CRISP-DM</div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="algo-phase">
        <div class="algo-phase-accent"
             style="background:linear-gradient(180deg,#3B82F6,#10B981,#F59E0B,#A78BFA,#F43F5E,#0EA5E9)"></div>
        <div class="algo-phase-body" style="line-height:2.1;">
        <b style="color:#CBD5E1;font-size:0.95rem">5 Fase CRISP-DM yang Dijalankan:</b><br><br>
        <b style="color:#60A5FA">Fase 1 · Business Understanding</b>
        → Tujuan: segmentasi 7 variabel gabungan.
        Kriteria: Silhouette ≥ 0.4 via <code>sklearn.metrics</code><br>
        <b style="color:#34D399">Fase 2 · Data Collection & Understanding</b>
        → EDA: <code>df.isnull().sum()</code>, distribusi kategorikal, validasi
        <code>set.issubset()</code><br>
        <b style="color:#FCD34D">Fase 3 · Data Preparation</b>
        → <code>pd.to_datetime</code> → <code>.dt.hour</code>,
        <code>LabelEncoder()</code> ×3,
        <code>StandardScaler().fit_transform()</code><br>
        <b style="color:#C4B5FD">Fase 4 · Modeling (K-Means Clustering)</b>
        → Elbow Method <code>go.Scatter + add_vline</code>,
        Silhouette <code>np.argmax</code> → optimal_k={optimal_k},
        <code>KMeans.fit_predict()</code>,
        <code>PCA(n_components=2)</code><br>
        <b style="color:#F87171">Fase 5 · Evaluation & Deployment</b>
        → <code>groupby().agg()</code> profiling,
        <code>get_persona_name()</code> &amp; <code>get_persona_desc()</code>,
        rekomendasi pemasaran per cluster<br><br>
        <b style="color:#94A3B8">Hasil:</b>
        {optimal_k} cluster optimal |
        Silhouette = {best_sil:.4f} {'✓ ≥ 0.4' if best_sil >= 0.4 else '✗ < 0.4'} |
        K-Means konvergen {km_log['n_iterations']} iterasi |
        PCA 2D {sum(var_ratio)*100:.1f}% variansi
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Analisis Ulang dengan File Baru", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()