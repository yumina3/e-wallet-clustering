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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=DM+Serif+Display&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
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
    font-family:'DM Mono',monospace; font-size:0.68rem;
    font-weight:500; letter-spacing:0.14em; text-transform:uppercase;
    color:rgba(147,197,253,0.55); margin-bottom:1rem;
}
.header-eyebrow::before {
    content:''; display:block; width:28px; height:1.5px;
    background:rgba(147,197,253,0.35);
}
.app-header h1 {
    font-family:'DM Serif Display', serif;
    font-size:2.6rem; font-weight:400; color:#FFFFFF !important;
    letter-spacing:-0.01em; line-height:1.1; margin:0 0 0.6rem;
}
.app-header p {
    font-size:0.875rem; color:rgba(186,215,250,0.6) !important;
    font-weight:300; line-height:1.7; max-width:520px; margin:0;
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
    font-family:'DM Mono',monospace; font-size:0.62rem;
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
.step-num { font-family:'DM Mono',monospace; font-size:0.62rem; opacity:0.5; }
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
.stat-number { font-size:2rem; font-weight:800; color:#F1F5F9 !important;
               line-height:1; margin-bottom:5px; letter-spacing:-0.03em; }
.stat-label { font-family:'DM Mono',monospace; font-size:0.65rem;
              color:#3D5070 !important; font-weight:500; text-transform:uppercase; letter-spacing:0.08em; }

/* ── Upload Info Box ── */
.upload-info {
    background:rgba(29,78,216,0.06); border:1px solid rgba(59,130,246,0.18);
    border-radius:16px; padding:1.3rem 1.5rem; height:100%; box-sizing:border-box;
}
.upload-info-header {
    display:flex; align-items:center; gap:6px;
    font-family:'DM Mono',monospace; font-size:0.65rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase; color:#60A5FA !important; margin-bottom:0.7rem;
}
.upload-info p { font-size:0.84rem; color:#4E6380 !important; line-height:1.6; margin:0 0 0.9rem; }
.col-list { display:flex; flex-direction:column; gap:6px; }
.col-item {
    display:flex; align-items:center; gap:8px; padding:0.5rem 0.8rem;
    background:rgba(7,16,31,0.7); border:1px solid rgba(255,255,255,0.04);
    border-radius:8px; transition:border-color 0.2s;
}
.col-item:hover { border-color:rgba(59,130,246,0.35); }
.col-dot { width:5px; height:5px; background:#3B82F6; border-radius:50%; flex-shrink:0; }
.col-name { font-family:'DM Mono',monospace; font-size:0.75rem; color:#94A3B8 !important; }

/* ── FILE UPLOADER — FIXED ── */
div[data-testid="stFileUploader"] {
    border:none !important; background:transparent !important; padding:0 !important;
    height:100% !important;
}
div[data-testid="stFileUploader"] > div {
    border:2px dashed rgba(59,130,246,0.22) !important;
    border-radius:20px !important;
    background:rgba(10,18,35,0.6) !important;
    transition:border-color 0.25s, background 0.25s;
    min-height:380px !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:stretch !important;
    justify-content:stretch !important;
    overflow:hidden !important;
}
div[data-testid="stFileUploader"] > div:hover {
    border-color:rgba(59,130,246,0.6) !important;
    background:rgba(29,78,216,0.06) !important;
}
div[data-testid="stFileUploader"] section {
    padding:0 !important;
    min-height:380px !important;
    height:100% !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
    width:100% !important;
}
div[data-testid="stFileUploader"] section > div {
    display:flex !important; flex-direction:column !important;
    align-items:center !important; justify-content:center !important;
    gap:10px !important;
}
div[data-testid="stFileUploaderDropzone"] {
    min-height:380px !important; height:100% !important;
    display:flex !important; flex-direction:column !important;
    align-items:center !important; justify-content:center !important;
    padding:2rem !important; text-align:center !important; box-sizing:border-box !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] {
    display:flex !important; flex-direction:column !important;
    align-items:center !important; justify-content:center !important;
    gap:4px !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] svg {
    width:48px !important; height:48px !important;
    color:#3B82F6 !important; margin-bottom:10px !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] span {
    color:#94A3B8 !important; font-size:1rem !important;
    font-weight:600 !important; line-height:1.3 !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] small {
    color:#3D5070 !important; font-size:0.8rem !important;
    margin-top:2px !important;
}
div[data-testid="stFileUploader"] label {
    color:#94A3B8 !important; font-size:1rem !important;
    font-weight:600 !important; text-align:center !important;
}
div[data-testid="stFileUploader"] button {
    background:rgba(59,130,246,0.13) !important; color:#60A5FA !important;
    border:1px solid rgba(59,130,246,0.28) !important; border-radius:10px !important;
    font-weight:600 !important; box-shadow:none !important; margin-top:0.5rem !important;
}
div[data-testid="stFileUploader"] button:hover {
    background:rgba(59,130,246,0.22) !important; transform:none !important;
}
div[data-testid="column"] { align-self:stretch !important; }

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
    font-family:'DM Mono',monospace; font-size:0.62rem; font-weight:500;
    color:#1D4ED8; background:rgba(29,78,216,0.12); padding:3px 8px;
    border-radius:5px; letter-spacing:0.1em; flex-shrink:0;
}
.algo-phase-title { font-size:0.9rem; font-weight:700; color:#CBD5E1 !important; }
.algo-phase-body { font-size:0.82rem; color:#4E6380 !important; line-height:1.7; }
.algo-phase-body code {
    background:rgba(255,255,255,0.05); color:#93C5FD !important;
    font-family:'DM Mono',monospace; font-size:0.78rem;
    padding:2px 6px; border-radius:4px;
}
.algo-phase-accent {
    position:absolute; left:0; top:0; bottom:0; width:3px;
    border-radius:16px 0 0 16px;
}
.algo-step-log {
    background:rgba(7,16,31,0.9); border:1px solid rgba(255,255,255,0.04);
    border-radius:12px; padding:1rem 1.2rem; margin-top:0.6rem;
    font-family:'DM Mono',monospace; font-size:0.75rem; line-height:1.9;
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
    font-family:'DM Mono',monospace; font-size:0.82rem;
    color:#93C5FD !important; text-align:center; letter-spacing:0.02em;
}
.formula-label {
    font-family:'DM Sans',sans-serif; font-size:0.72rem;
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
    display:inline-block; font-family:'DM Mono',monospace;
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
    "Jam Transaksi","Kategori Produk","Nominal Transaksi",
    "Cashback","Poin Loyalitas","Metode Pembayaran","Jenis Perangkat"
]

DARK_LAYOUT = dict(
    plot_bgcolor="#08111E", paper_bgcolor="#08111E",
    font=dict(family="DM Sans, sans-serif", color="#4E6380"),
    title_font=dict(size=13, color="#94A3B8", family="DM Sans"),
    margin=dict(t=50,b=40,l=45,r=20),
)
DARK_GRID = dict(
    showgrid=True, gridcolor="rgba(255,255,255,0.03)",
    showline=True, linecolor="rgba(255,255,255,0.05)"
)

# ══════════════════════════════════════════════════════════════════════════════
#  ╔═══════════════════════════════════════╗
#  ║   MODUL ALGORITMA — CRISP-DM + KMeans ║
#  ╚═══════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

def validate_columns(df: pd.DataFrame) -> bool:
    """
    FASE 1 — BUSINESS UNDERSTANDING
    Validasi apakah DataFrame memiliki semua kolom yang diperlukan.
    Kolom wajib sesuai dengan feature yang akan digunakan dalam clustering.
    """
    required = {
        "transaction_date", "product_category", "product_amount",
        "cashback", "loyalty_points", "payment_method", "device_type"
    }
    return required.issubset(set(df.columns))


def fase2_data_understanding(df: pd.DataFrame) -> dict:
    """
    FASE 2 — DATA UNDERSTANDING
    Menganalisis struktur data: jumlah baris, kolom, missing values,
    distribusi tipe data, dan statistik deskriptif dasar.

    Returns:
        dict berisi ringkasan pemahaman data
    """
    stats = {
        "n_rows"      : len(df),
        "n_cols"      : len(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "dtypes"      : df.dtypes.astype(str).to_dict(),
        "numeric_desc": df.describe().to_dict(),
        "unique_payment": df["payment_method"].nunique() if "payment_method" in df.columns else 0,
        "unique_device" : df["device_type"].nunique()    if "device_type"    in df.columns else 0,
        "unique_category": df["product_category"].nunique() if "product_category" in df.columns else 0,
    }
    return stats


def fase3_data_preparation(df: pd.DataFrame) -> tuple:
    """
    FASE 3 — DATA PREPARATION
    Tahapan pra-pemrosesan data sebelum clustering:

    Langkah 3a — Parsing tanggal & ekstraksi jam transaksi
        transaction_hour = hour(transaction_date)
        → Merepresentasikan pola waktu transaksi

    Langkah 3b — Label Encoding variabel kategorikal
        Mengubah string → integer agar bisa dihitung jarak Euclidean
        payment_method  → payment_method_enc   (LabelEncoder)
        device_type     → device_type_enc      (LabelEncoder)
        product_category→ product_category_enc (LabelEncoder)

    Langkah 3c — Feature Selection
        7 fitur dipilih:
        [transaction_hour, product_category_enc, product_amount,
         cashback, loyalty_points, payment_method_enc, device_type_enc]

    Langkah 3d — Feature Scaling (Standardisasi Z-Score)
        x' = (x - μ) / σ
        Tujuan: agar semua fitur berkontribusi setara pada perhitungan jarak.
        Tanpa standardisasi, fitur dengan skala besar (misal nominal Rp)
        akan mendominasi perhitungan jarak Euclidean.

    Returns:
        df_proc   : DataFrame yang sudah diproses
        X_scaled  : numpy array fitur yang telah distandarisasi
        features  : list nama kolom fitur
        le_info   : dict encoding info (untuk keperluan penjelasan)
        scaler    : objek StandardScaler (untuk inverse transform jika perlu)
    """
    df = df.copy()
    prep_log = {}

    # -- 3a: Parsing tanggal --
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True, errors="coerce")
    df["transaction_hour"] = df["transaction_date"].dt.hour
    prep_log["3a_parsing"] = f"Ekstrak jam dari transaction_date → transaction_hour"

    # -- 3b: Label Encoding --
    le = LabelEncoder()
    le_info = {}

    for col, new_col in [
        ("payment_method",   "payment_method_enc"),
        ("device_type",      "device_type_enc"),
        ("product_category", "product_category_enc"),
    ]:
        df[new_col] = le.fit_transform(df[col].astype(str))
        le_info[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        prep_log[f"3b_{col}"] = f"LabelEncoding: {dict(list(le_info[col].items())[:4])}"

    # -- 3c: Feature Selection --
    features = [
        "transaction_hour",
        "product_category_enc",
        "product_amount",
        "cashback",
        "loyalty_points",
        "payment_method_enc",
        "device_type_enc",
    ]
    prep_log["3c_features"] = f"7 fitur dipilih: {features}"

    # -- 3d: Standardisasi Z-Score --
    X_raw   = df[features].dropna()
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    prep_log["3d_scaling"] = (
        f"StandardScaler: μ={scaler.mean_.round(3)}, σ={scaler.scale_.round(3)}"
    )

    return df, X_scaled, features, le_info, scaler, prep_log


def fase4a_elbow_method(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple:
    """
    FASE 4a — ELBOW METHOD (Menentukan k optimal)

    Algoritma Elbow:
    Untuk setiap k dari k_min hingga k_max:
        1. Jalankan K-Means dengan k cluster
        2. Hitung WCSS (Within-Cluster Sum of Squares) = Inertia
           WCSS = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²
           di mana μᵢ adalah centroid cluster ke-i
        3. Plot WCSS vs k → cari "siku" (elbow) = penurunan WCSS
           mulai melambat ← itu k optimal

    Returns:
        k_range : list nilai k yang diuji
        wcss    : list nilai WCSS per k
        elbow_log : list log per iterasi
    """
    k_range   = list(range(k_min, k_max + 1))
    wcss      = []
    elbow_log = []

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            init="k-means++",   # Inisialisasi cerdas (bukan random murni)
            max_iter=300,
            n_init=10,
            random_state=42
        )
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        elbow_log.append({
            "k"    : k,
            "wcss" : round(km.inertia_, 2),
            "n_iter": km.n_iter_,
        })

    return k_range, wcss, elbow_log


def fase4b_silhouette_analysis(X_scaled: np.ndarray, k_range: list) -> tuple:
    """
    FASE 4b — SILHOUETTE ANALYSIS (Validasi kualitas cluster)

    Silhouette Score mengukur seberapa baik setiap titik data
    berada di clusternya sendiri dibanding cluster tetangga.

    Rumus Silhouette Score per titik data i:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

        di mana:
        a(i) = rata-rata jarak i ke semua titik dalam cluster yang sama
               (cohesion — seberapa dekat dengan sesama anggota cluster)
        b(i) = rata-rata jarak i ke semua titik di cluster terdekat lainnya
               (separation — seberapa jauh dari cluster lain)

    Interpretasi nilai s(i):
        +1 : sangat tepat berada di clusternya
         0 : berada di batas antara dua cluster
        -1 : kemungkinan salah cluster

    Silhouette Score keseluruhan = rata-rata s(i) semua titik
    Pilih k dengan Silhouette Score TERTINGGI

    Returns:
        sil_scores : list score per k
        sil_log    : list log per iterasi
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


def fase4c_kmeans_clustering(X_scaled: np.ndarray, k: int) -> tuple:
    """
    FASE 4c — K-MEANS CLUSTERING (Algoritma Utama)

    Algoritma K-Means dengan inisialisasi K-Means++:

    === INISIALISASI K-MEANS++ ===
    (lebih baik dari random — menghindari konvergensi ke local minimum)
    1. Pilih centroid pertama secara acak dari data
    2. Untuk setiap centroid berikutnya:
       Hitung D(x)² = jarak minimum ke centroid yang sudah ada
       Pilih titik berikutnya dengan probabilitas ∝ D(x)²
       (titik yang jauh dari centroid yang ada lebih mungkin dipilih)

    === ITERASI K-MEANS ===
    Ulangi hingga konvergen (centroid tidak berubah):

    Langkah E (Assignment Step):
        Untuk setiap titik data x:
            Assign x ke cluster Cᵢ dengan centroid μᵢ terdekat
            c(x) = argmin_i ‖x − μᵢ‖²   (Euclidean distance)

    Langkah M (Update Step):
        Perbarui centroid setiap cluster:
            μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x
            (centroid baru = rata-rata semua titik dalam cluster)

    Konvergensi: ketika Δμᵢ < ε untuk semua i
    (centroid tidak berpindah secara signifikan)

    Returns:
        labels    : array label cluster untuk setiap titik data
        km_model  : model KMeans yang sudah difit
        km_log    : dict informasi proses clustering
    """
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=300,
        n_init=10,         # 10 inisialisasi berbeda, ambil yang WCSS terkecil
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


def fase4d_pca_reduction(X_scaled: np.ndarray, labels: np.ndarray) -> tuple:
    """
    FASE 4d — PCA DIMENSIONALITY REDUCTION (Visualisasi)

    PCA (Principal Component Analysis) digunakan HANYA untuk visualisasi.
    Clustering dilakukan pada 7 dimensi asli (bukan hasil PCA).

    Algoritma PCA:
    1. Hitung matriks kovarians: Σ = (1/n) XᵀX
    2. Dekomposisi eigenvalue: Σ = VΛVᵀ
    3. Urutkan eigenvector berdasarkan eigenvalue (descending)
    4. Proyeksikan data ke 2 eigenvector pertama:
       X_reduced = X · V₂
       (V₂ = matrix 2 eigenvector dengan variance terbesar)

    Explained Variance Ratio:
       EVR = λᵢ / Σλ   (proporsi informasi yang dipertahankan)

    Returns:
        coords    : koordinat 2D setelah PCA (n_samples × 2)
        var_ratio : explained variance ratio [PC1, PC2]
        pca_log   : dict informasi PCA
    """
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pca_log = {
        "explained_var_ratio" : pca.explained_variance_ratio_.round(4).tolist(),
        "total_var_explained" : round(sum(pca.explained_variance_ratio_) * 100, 2),
        "n_components"        : 2,
        "note"                : "PCA hanya untuk visualisasi; clustering di 7D"
    }

    return coords, pca.explained_variance_ratio_, pca_log


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

def log_html(lines: list) -> str:
    """Render log box HTML dari list of (tag_type, text) tuples."""
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
    <p>Segmentasi perilaku transaksi digital menggunakan K-Means Clustering.
       Setiap tahap algoritma ditampilkan secara transparan untuk keperluan
       akademis dan presentasi.</p>
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

# ══════════════════════════════════════════════════════════════════════════════
#  STEP BAR
# ══════════════════════════════════════════════════════════════════════════════
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


#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if step_now == "upload":
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">🗄</span> Unggah File CSV Transaksi
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([4, 7], gap="large")

    with col_left:
        st.markdown("""
        <div class="upload-info">
            <div class="upload-info-header">ⓘ &nbsp;Persyaratan Data</div>
            <p>File CSV harus memiliki tepat <strong style="color:#60A5FA">7 kolom</strong>
               berikut. Separator kolom menggunakan titik koma <code style="color:#93C5FD;background:rgba(255,255,255,0.05);padding:1px 5px;border-radius:3px">;</code></p>
            <div class="col-list">
                <div class="col-item"><span class="col-dot"></span><span class="col-name">transaction_date</span></div>
                <div class="col-item"><span class="col-dot"></span><span class="col-name">product_category</span></div>
                <div class="col-item"><span class="col-dot"></span><span class="col-name">product_amount</span></div>
                <div class="col-item"><span class="col-dot"></span><span class="col-name">cashback</span></div>
                <div class="col-item"><span class="col-dot"></span><span class="col-name">loyalty_points</span></div>
                <div class="col-item"><span class="col-dot"></span><span class="col-name">payment_method</span></div>
                <div class="col-item"><span class="col-dot"></span><span class="col-name">device_type</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        uploaded = st.file_uploader(
            "☁  Pilih atau seret file CSV ke sini",
            type=["csv"],
            label_visibility="visible",
            help="Gunakan separator titik koma (;). Maksimal 200MB.",
            key="file_uploader"
        )
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, sep=";")
                if not validate_columns(df):
                    st.error("❌ Kolom tidak sesuai. Pastikan CSV memiliki 7 kolom yang diperlukan.")
                else:
                    st.session_state["df_raw"] = df
                    st.session_state["step"]   = "preview"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

        if st.button("🗄  Gunakan Data Contoh  →", use_container_width=True):
            try:
                df = pd.read_csv("/mnt/user-data/uploads/project_Data_ewallet.csv", sep=";")
                st.session_state["df_raw"] = df
                st.session_state["step"]   = "preview"
                st.rerun()
            except Exception as e:
                st.error(f"❌ File contoh tidak ditemukan: {e}")

# ═════════ STEP 2 — PREVIEW & VALIDASI═════════════════════════════════════════════════════════════════════
elif step_now == "preview":
    df = gs("df_raw")
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">⌕</span> Preview & Validasi Data
    </div>
    """, unsafe_allow_html=True)

    # Fase 2 — Data Understanding
    du = fase2_data_understanding(df)
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
            <div class="stat-label">Nilai Kosong</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{du['unique_category']}</div>
            <div class="stat-label">Kategori Produk</div>
        </div>
    </div>
    <div class="pill-row">
        <span class="pill-label">Fitur akan dianalisis :</span>
        {''.join(f'<span class="pill">{f}</span>' for f in FITUR_LABELS)}
    </div>
    """, unsafe_allow_html=True)

    # Penjelasan Fase 2 untuk dosen
    render_algo_phase(
        "FASE 02", "DATA UNDERSTANDING",
        f"""Tahap eksplorasi awal sebelum pemrosesan. Dataset berisi
        <code>{du['n_rows']:,}</code> baris transaksi dengan <code>{du['n_cols']}</code>
        kolom. Terdeteksi <code>{du['unique_payment']}</code> metode pembayaran unik,
        <code>{du['unique_device']}</code> jenis perangkat, dan
        <code>{du['unique_category']}</code> kategori produk.
        Missing values: <code>{missing}</code> — akan dihapus pada tahap preparasi.""",
        PHASE_COLORS[1],
        log_html([
            ("ok",  f"n_rows = {du['n_rows']:,}"),
            ("ok",  f"n_cols = {du['n_cols']}"),
            ("inf", f"unique payment_method = {du['unique_payment']}"),
            ("inf", f"unique device_type    = {du['unique_device']}"),
            ("inf", f"unique product_category = {du['unique_category']}"),
            ("run" if missing > 0 else "ok",
             f"missing_values = {missing} {'→ akan di-drop pada preprocessing' if missing > 0 else '✓ data bersih'}"),
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



# ═══════════STEP 3 — ANALISIS (dengan tampilan langkah algoritma)════════════════════════════════════════════════════
elif step_now == "analisis":
    df = gs("df_raw")
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">⚙</span> Proses K-Means Clustering — Langkah demi Langkah
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Memulai pipeline CRISP-DM…")
    status_text  = st.empty()

    # ── FASE 3: Data Preparation ──────────────────────────────────────────
    status_text.markdown("⚙️ **FASE 3** — Data Preparation: Encoding & Standardisasi…")
    time.sleep(0.3)
    progress_bar.progress(15, text="Fase 3: Preprocessing data…")

    df_proc, X_scaled, features, le_info, scaler, prep_log = fase3_data_preparation(df)

    render_algo_phase(
        "FASE 03", "DATA PREPARATION",
        """Tiga sub-langkah transformasi data sebelum algoritma clustering dijalankan:
        <br><br>
        <b>3a. Parsing Tanggal</b> — Ekstrak jam dari kolom <code>transaction_date</code>
        menghasilkan fitur numerik <code>transaction_hour</code> ∈ [0, 23].<br><br>
        <b>3b. Label Encoding</b> — Variabel kategorikal dikonversi ke integer:
        <code>payment_method</code>, <code>device_type</code>, <code>product_category</code>
        menggunakan <code>sklearn.LabelEncoder</code>.<br><br>
        <b>3c. Feature Selection</b> — 7 fitur dipilih sebagai input clustering.<br><br>
        <b>3d. Standardisasi Z-Score</b> — Setiap fitur dinormalisasi:
        <code>x' = (x − μ) / σ</code> sehingga mean=0 dan std=1.
        Tanpa ini, fitur berskala besar (misal <code>product_amount</code> dalam ribuan)
        akan mendominasi jarak Euclidean.""",
        PHASE_COLORS[2],
        log_html([
            ("ok",  f"transaction_hour diekstrak dari datetime"),
            ("ok",  f"LabelEncoding: payment_method → {list(le_info.get('payment_method',{}).items())[:3]}"),
            ("ok",  f"LabelEncoding: device_type    → {list(le_info.get('device_type',{}).items())[:3]}"),
            ("ok",  f"LabelEncoding: product_category → {list(le_info.get('product_category',{}).items())[:3]}"),
            ("ok",  f"X_scaled.shape = {X_scaled.shape}  (baris × fitur)"),
            ("inf", f"μ (mean) = {scaler.mean_.round(2)}"),
            ("inf", f"σ (std)  = {scaler.scale_.round(2)}"),
        ])
    )

    st.markdown("""
    <div class="formula-box">x' = (x − μ) / σ</div>
    <div class="formula-label">Rumus Standardisasi Z-Score — membuat semua fitur setara skalanya</div>
    """, unsafe_allow_html=True)

    # ── FASE 4a: Elbow Method ─────────────────────────────────────────────
    status_text.markdown("📈 **FASE 4a** — Elbow Method: Menghitung WCSS untuk k=2..10…")
    progress_bar.progress(30, text="Fase 4a: Elbow Method…")

    k_range, wcss, elbow_log = fase4a_elbow_method(X_scaled)

    render_algo_phase(
        "FASE 04a", "ELBOW METHOD — Mencari k Optimal",
        """Untuk setiap nilai k (jumlah cluster), jalankan K-Means dan hitung
        <b>WCSS (Within-Cluster Sum of Squares)</b> = total jarak kuadrat setiap titik
        ke centroidnya. Semakin banyak cluster, WCSS semakin kecil — namun ada titik
        di mana penurunan mulai melambat (bentuk "siku"). Titik siku itulah k optimal.""",
        PHASE_COLORS[0],
        log_html([
            ("run", f"Iterasi k = {k_range}"),
        ] + [
            ("ok", f"k={e['k']}: WCSS={e['wcss']:,.1f}  |  n_iter={e['n_iterations']}")
            for e in elbow_log
        ])
    )

    st.markdown("""
    <div class="formula-box">WCSS = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²</div>
    <div class="formula-label">Within-Cluster Sum of Squares — ukuran kepadatan cluster (semakin kecil semakin baik)</div>
    """, unsafe_allow_html=True)

    # ── FASE 4b: Silhouette ───────────────────────────────────────────────
    status_text.markdown("📊 **FASE 4b** — Silhouette Analysis: Validasi kualitas cluster…")
    progress_bar.progress(55, text="Fase 4b: Silhouette Analysis…")

    sil_scores, sil_log = fase4b_silhouette_analysis(X_scaled, k_range)
    optimal_k = k_range[int(np.argmax(sil_scores))]

    render_algo_phase(
        "FASE 04b", "SILHOUETTE ANALYSIS — Validasi Kualitas Cluster",
        f"""Silhouette Score mengukur dua hal sekaligus:
        <br>• <b>a(i)</b> — <i>cohesion</i>: rata-rata jarak titik i ke anggota clusternya sendiri
        <br>• <b>b(i)</b> — <i>separation</i>: rata-rata jarak titik i ke cluster terdekat lainnya
        <br><br>
        Score mendekati +1 = cluster sangat padat & terpisah dengan baik.
        Dipilih k dengan score tertinggi → <code>k = {optimal_k}</code>
        (score = {max(sil_scores):.4f}).""",
        PHASE_COLORS[3],
        log_html([
            ("run", "Menghitung s(i) = (b(i) − a(i)) / max(a(i), b(i)) untuk setiap titik"),
        ] + [
            ("ok" if e["best"] else "inf",
             f"k={e['k']}: silhouette={e['score']}  {'← TERPILIH ✓' if e['best'] else ''}")
            for e in sil_log
        ])
    )

    st.markdown("""
    <div class="formula-box">s(i) = (b(i) − a(i)) / max(a(i), b(i))</div>
    <div class="formula-label">Silhouette Score per titik data — rata-rata seluruh titik = Silhouette Score keseluruhan</div>
    """, unsafe_allow_html=True)

    # ── FASE 4c: K-Means Final ────────────────────────────────────────────
    status_text.markdown(f"🎯 **FASE 4c** — K-Means Clustering dengan k={optimal_k}…")
    progress_bar.progress(75, text=f"Fase 4c: K-Means k={optimal_k}…")

    labels, km_model, km_log = fase4c_kmeans_clustering(X_scaled, optimal_k)

    render_algo_phase(
        "FASE 04c", f"K-MEANS CLUSTERING — k = {optimal_k}",
        f"""Algoritma K-Means++ dijalankan dengan <code>n_init=10</code>
        (10 inisialisasi berbeda, ambil yang terbaik).
        <br><br>
        <b>Assignment Step (E-Step)</b>: setiap titik data di-assign ke centroid terdekat
        berdasarkan jarak Euclidean.<br>
        <b>Update Step (M-Step)</b>: centroid diperbarui ke rata-rata anggota clusternya.<br>
        <br>Proses konvergen setelah <code>{km_log['n_iterations']} iterasi</code>
        dengan WCSS akhir = <code>{km_log['final_inertia']:,.2f}</code>.""",
        PHASE_COLORS[4],
        log_html([
            ("run", f"KMeans(n_clusters={optimal_k}, init='k-means++', n_init=10, random_state=42)"),
            ("ok",  f"Konvergen dalam {km_log['n_iterations']} iterasi"),
            ("ok",  f"Final WCSS (inertia) = {km_log['final_inertia']:,.2f}"),
            ("ok",  f"Ukuran cluster: { {f'C{k}':v for k,v in km_log['cluster_sizes'].items()} }"),
        ])
    )

    st.markdown("""
    <div class="formula-box">c(x) = argminᵢ ‖x − μᵢ‖²  &nbsp;→&nbsp;  μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x</div>
    <div class="formula-label">Assignment Step (kiri) dan Update Step (kanan) — diulang hingga konvergen</div>
    """, unsafe_allow_html=True)

    # ── FASE 4d: PCA ──────────────────────────────────────────────────────
    status_text.markdown("🗺 **FASE 4d** — PCA 2D untuk visualisasi…")
    progress_bar.progress(90, text="Fase 4d: PCA Dimensionality Reduction…")

    coords, var_ratio, pca_log = fase4d_pca_reduction(X_scaled, labels)

    render_algo_phase(
        "FASE 04d", "PCA — Reduksi Dimensi untuk Visualisasi",
        f"""PCA (Principal Component Analysis) digunakan <b>hanya untuk visualisasi</b>.
        Clustering sesungguhnya terjadi di ruang 7 dimensi.
        PCA memproyeksikan 7D → 2D dengan mempertahankan arah variansi terbesar
        (eigenvector dengan eigenvalue terbesar).
        <br><br>
        PC1 menjelaskan <code>{var_ratio[0]*100:.1f}%</code> variansi,
        PC2 menjelaskan <code>{var_ratio[1]*100:.1f}%</code>.
        Total informasi dipertahankan: <code>{pca_log['total_var_explained']:.1f}%</code>.""",
        PHASE_COLORS[5],
        log_html([
            ("run", "PCA(n_components=2).fit_transform(X_scaled)"),
            ("ok",  f"PC1 explained variance = {var_ratio[0]*100:.2f}%"),
            ("ok",  f"PC2 explained variance = {var_ratio[1]*100:.2f}%"),
            ("inf", f"Total dipertahankan    = {pca_log['total_var_explained']:.1f}%"),
            ("inf", "Catatan: Label cluster dari K-Means 7D, bukan PCA"),
        ])
    )

    # Simpan semua ke session state
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
    time.sleep(0.5)
    status_text.empty()

    st.success(f"✅ Pipeline CRISP-DM selesai! Ditemukan **{optimal_k} cluster optimal** "
               f"(Silhouette Score = {max(sil_scores):.4f})")

    if st.button("Lihat Hasil & Insight →", type="primary", use_container_width=True):
        st.session_state["step"] = "hasil"; st.rerun()



# ════════════STEP 4 — HASIL══════════════════════════════════════════════════════════════════
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

    df_proc             = df_proc.iloc[:len(labels)].copy()
    df_proc["cluster"]  = labels

    # ── Summary stats ─────────────────────────────────────────────────────
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
            <div class="stat-number">{max(sil_scores):.3f}</div>
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
    </div>
    """, unsafe_allow_html=True)

    # ── Elbow + Silhouette Charts ─────────────────────────────────────────
    st.markdown("""<div class="section-title"><span class="section-title-icon">📈</span>
    Penentuan Cluster Optimal (Elbow + Silhouette)</div>""", unsafe_allow_html=True)

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
                        annotation_text=f"k = {optimal_k}",
                        annotation_font=dict(color="#60A5FA", size=12))
        fig_e.update_layout(title="Elbow Method — WCSS per k",
                            xaxis_title="Jumlah Cluster (k)",
                            yaxis_title="WCSS (Inertia)", height=320, **DARK_LAYOUT)
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
        fig_s.update_layout(title="Silhouette Score per k",
                            xaxis_title="Jumlah Cluster (k)",
                            yaxis_title="Score",
                            yaxis_range=[0, max(sil_scores) * 1.28],
                            height=320, **DARK_LAYOUT)
        fig_s.update_xaxes(showgrid=False, tickvals=k_range,
                           showline=True, linecolor="rgba(255,255,255,0.05)")
        fig_s.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── PCA Scatter + Profil ──────────────────────────────────────────────
    st.markdown("""<div class="section-title"><span class="section-title-icon">🗺</span>
    Sebaran Cluster dalam Ruang 2D (Proyeksi PCA)</div>""", unsafe_allow_html=True)

    c3, c4 = st.columns([1.3, 1])
    with c3:
        df_pca = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "Cluster": [f"Cluster {l}" for l in labels]
        })
        fig_sc = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=CLUSTER_COLORS,
            title=f"PCA 2D — PC1 {var_ratio[0]*100:.1f}%  ·  PC2 {var_ratio[1]*100:.1f}%",
            opacity=0.65, height=400
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
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#64748B;margin-bottom:0.5rem;">Profil Statistik Tiap Cluster</p>',
                    unsafe_allow_html=True)
        profile = df_proc.groupby("cluster").agg(
            Jumlah    =("cluster",          "count"),
            Nominal   =("product_amount",    "mean"),
            Cashback  =("cashback",          "mean"),
            Poin      =("loyalty_points",    "mean"),
            Jam_Puncak=("transaction_hour",  lambda x: x.mode()[0]),
            Metode    =("payment_method",    lambda x: x.mode()[0]),
            Perangkat =("device_type",       lambda x: x.mode()[0]),
        ).reset_index()
        profile["Nominal"]    = profile["Nominal"].map("Rp{:,.0f}".format)
        profile["Cashback"]   = profile["Cashback"].map("{:.1f}".format)
        profile["Poin"]       = profile["Poin"].map("{:.0f}".format)
        profile["Jam_Puncak"] = profile["Jam_Puncak"].map(lambda x: f"{x:02d}:00")
        profile["cluster"]    = profile["cluster"].map(lambda x: f"C{x}")
        profile.columns = ["Cluster","Jumlah","Nominal","Cashback","Poin","Jam","Metode","Perangkat"]
        st.dataframe(profile.set_index("Cluster"), use_container_width=True, height=360)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Persona Cards ─────────────────────────────────────────────────────
    st.markdown("""<div class="section-title"><span class="section-title-icon">🎯</span>
    Persona Tiap Cluster</div>""", unsafe_allow_html=True)

    profile_raw = df_proc.groupby("cluster").agg(
        jumlah  =("cluster",          "count"),
        nominal =("product_amount",    "mean"),
        cashback=("cashback",          "mean"),
        poin    =("loyalty_points",    "mean"),
        jam     =("transaction_hour",  lambda x: x.mode()[0]),
        metode  =("payment_method",    lambda x: x.mode()[0]),
        device  =("device_type",       lambda x: x.mode()[0]),
        kategori=("product_category",  lambda x: x.mode()[0]),
    ).reset_index()

    def p_name(row):
        sesi  = "Malam"   if row["jam"] >= 20 or row["jam"] < 5 else \
                "Pagi"    if row["jam"] < 12 else \
                "Siang"   if row["jam"] < 17 else "Sore"
        level = "Premium" if row["nominal"] > 6000 else \
                "Menengah" if row["nominal"] > 3000 else "Hemat"
        return f"Pengguna {level} · {row['device']} · {sesi}"

    def p_desc(row):
        return (f"Transaksi dominan pukul {row['jam']:02d}:00 melalui {row['device']}. "
                f"Metode favorit {row['metode']} dengan fokus kategori "
                f"<strong style='color:#CBD5E1'>{row['kategori']}</strong>.")

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
                        Cluster {c} &nbsp;·&nbsp; {pct:.1f}%
                    </span>
                    <div class="persona-name">{p_name(row)}</div>
                    <div class="persona-chips">
                        <span class="persona-chip">⏰ {row['jam']:02d}:00</span>
                        <span class="persona-chip">📱 {row['device']}</span>
                        <span class="persona-chip">Rp {row['nominal']:,.0f}</span>
                    </div>
                    <div class="persona-desc">{p_desc(row)}</div>
                    <div class="persona-footer">
                        <span>💳 {row['metode']}</span>
                        <span>⭐ {row['poin']:.0f} poin</span>
                        <span>🛍 {row['kategori']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Rekomendasi ───────────────────────────────────────────────────────
    st.markdown("""<div class="section-title"><span class="section-title-icon">💡</span>
    Rekomendasi Strategi Pemasaran per Cluster</div>""", unsafe_allow_html=True)

    reco_cols = st.columns(min(optimal_k, 4))
    for _, row in profile_raw.iterrows():
        c      = int(row["cluster"])
        clr    = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        reward = "Cashback" if row["cashback"] > row["poin"] / 10 else "Poin Loyalitas"
        waktu  = f"{row['jam']:02d}:00 – {(row['jam'] + 2) % 24:02d}:00"

        with reco_cols[c % len(reco_cols)]:
            st.markdown(f"""
            <div class="reco-card">
                <div class="reco-header">
                    <span class="reco-dot" style="background:{clr}"></span>
                    Cluster {c}
                </div>
                <div class="reco-item"><strong>⏰ Waktu</strong>{waktu}</div>
                <div class="reco-item"><strong>🎁 Reward</strong>{reward}</div>
                <div class="reco-item"><strong>📲 Kanal</strong>{row['device']} App</div>
                <div class="reco-item"><strong>🛒 Produk</strong>{row['kategori']}</div>
                <div class="reco-item"><strong>💳 Metode</strong>{row['metode']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Algoritma ───────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""<div class="section-title"><span class="section-title-icon">📋</span>
    Ringkasan Pipeline Algoritma (untuk Presentasi Dosen)</div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="algo-phase">
        <div class="algo-phase-accent" style="background:linear-gradient(180deg,#3B82F6,#10B981,#F59E0B,#A78BFA,#F43F5E,#0EA5E9)"></div>
        <div class="algo-phase-body" style="line-height:2;">
        <b style="color:#CBD5E1;font-size:0.95rem">Pipeline CRISP-DM yang Dijalankan:</b><br><br>
        <b style="color:#60A5FA">01 · Business Understanding</b>
            → Tujuan: segmentasi pengguna e-wallet berdasarkan perilaku transaksi<br>
        <b style="color:#34D399">02 · Data Understanding</b>
            → Eksplorasi {len(df_proc):,} record, 7 kolom, deteksi missing values & distribusi<br>
        <b style="color:#FCD34D">03 · Data Preparation</b>
            → Parsing datetime, Label Encoding (3 kolom kategorikal), Standardisasi Z-Score<br>
        <b style="color:#C4B5FD">04a · Modeling — Elbow Method</b>
            → Hitung WCSS untuk k=2..10, cari titik siku<br>
        <b style="color:#F87171">04b · Modeling — Silhouette</b>
            → Hitung s(i) tiap titik, pilih k dengan score tertinggi = <code>{optimal_k}</code><br>
        <b style="color:#38BDF8">04c · Modeling — K-Means++</b>
            → Clustering 7D, konvergen dalam {km_log['n_iterations']} iterasi, WCSS={km_log['final_inertia']:,.0f}<br>
        <b style="color:#94A3B8">04d · Evaluasi — PCA 2D</b>
            → Proyeksi 7D→2D untuk visualisasi, {sum(var_ratio)*100:.1f}% variansi dipertahankan<br>
        <b style="color:#CBD5E1">05 · Deployment</b>
            → Persona per cluster + rekomendasi strategi pemasaran
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Analisis Ulang dengan File Baru"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()