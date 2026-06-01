"""
============================================================
  Analisis Cluster E-Wallet — Revisi v3 FINAL
  Perbaikan utama dari v2:
  1. Semua 7 variabel asli TETAP dipakai dalam clustering
  2. payment_method & device_type → One-Hot Encoding (bukan Label Encoding)
  3. product_category → dikelompokkan 5 grup → One-Hot Encoding
  4. transaction_hour → tetap numerik (sudah ordinal bermakna)
  5. product_amount, cashback, loyalty_points → numerik murni
  6. StandardScaler SETELAH OHE → menyamakan bobot semua fitur
  Total fitur setelah OHE: 4 numerik + 5 OHE payment + 3 OHE device + 5 OHE kategori = 17 fitur
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
html,body,[class*="css"]{font-family:'Geist',sans-serif;-webkit-font-smoothing:antialiased;letter-spacing:-0.015em;}
.main{background-color:#07101F;background-image:radial-gradient(ellipse at 75% 0%,rgba(37,99,235,0.10) 0%,transparent 55%),radial-gradient(ellipse at 10% 85%,rgba(16,185,129,0.07) 0%,transparent 50%);min-height:100vh;}
.block-container{padding:2.5rem 3rem;max-width:1280px;}
h1,h2,h3,h4,p,span,div{color:inherit;}
.stMarkdown p{color:#94A3B8;}
.app-header{position:relative;overflow:hidden;background:linear-gradient(135deg,#0F2155 0%,#1A3A8F 45%,#0E2847 100%);border-radius:24px;padding:2.5rem 3rem;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.07);box-shadow:0 24px 64px rgba(0,0,0,0.55);}
.app-header::before{content:'';position:absolute;top:-80px;right:-80px;width:320px;height:320px;border-radius:50%;background:rgba(255,255,255,0.03);filter:blur(40px);}
.header-eyebrow{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;color:rgba(147,197,253,0.55);margin-bottom:1rem;}
.app-header h1{font-family:'Fraunces',serif;font-size:2.6rem;font-weight:400;color:#FFFFFF !important;letter-spacing:-0.02em;line-height:1.08;margin:0 0 0.6rem;font-style:italic;}
.app-header p{font-size:0.875rem;color:rgba(186,215,250,0.6) !important;font-weight:400;line-height:1.75;max-width:560px;margin:0;}
.step-bar{display:flex;gap:4px;margin-bottom:2rem;background:rgba(10,18,35,0.8);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:6px;}
.step{flex:1;padding:0.85rem 0.5rem;text-align:center;font-size:0.8rem;font-weight:500;color:#3D5070;border-radius:10px;transition:all 0.2s;display:flex;align-items:center;justify-content:center;gap:6px;}
.step-num{font-family:'JetBrains Mono',monospace;font-size:0.62rem;opacity:0.5;}
.step.done{color:#3B82F6;background:rgba(59,130,246,0.07);}
.step.active{background:#1D4ED8;color:white !important;font-weight:600;box-shadow:0 4px 18px rgba(29,78,216,0.4);}
.section-title{font-size:1.05rem;font-weight:700;color:#F1F5F9 !important;margin-bottom:1.2rem;display:flex;align-items:center;gap:10px;}
.section-title-icon{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;background:rgba(59,130,246,0.13);}
.stat-row{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap;}
.stat-card{background:rgba(10,18,35,0.8);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.4rem 1.6rem;flex:1;min-width:110px;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1D4ED8,#6366F1);border-radius:16px 16px 0 0;}
.stat-number{font-size:2rem;font-weight:700;color:#F1F5F9 !important;line-height:1;margin-bottom:5px;letter-spacing:-0.04em;}
.stat-label{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D5070 !important;font-weight:500;text-transform:uppercase;letter-spacing:0.08em;}
.upload-info{background:rgba(29,78,216,0.06);border:1px solid rgba(59,130,246,0.18);border-radius:16px;padding:1.3rem 1.5rem;box-sizing:border-box;display:flex;flex-direction:column;}
.upload-info-header{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#60A5FA !important;margin-bottom:0.7rem;}
.upload-info p{font-size:0.84rem;color:#4E6380 !important;line-height:1.6;margin:0 0 0.9rem;}
.col-list{display:flex;flex-direction:column;gap:6px;flex:1;}
.col-item{display:flex;align-items:center;gap:8px;padding:0.5rem 0.8rem;background:rgba(7,16,31,0.7);border:1px solid rgba(255,255,255,0.04);border-radius:8px;}
.col-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;}
.col-name{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#94A3B8 !important;}
.algo-phase{background:rgba(10,18,35,0.8);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;}
.algo-phase-header{display:flex;align-items:center;gap:10px;margin-bottom:0.8rem;}
.algo-phase-num{font-family:'JetBrains Mono',monospace;font-size:0.62rem;font-weight:500;color:#1D4ED8;background:rgba(29,78,216,0.12);padding:3px 8px;border-radius:5px;letter-spacing:0.1em;flex-shrink:0;}
.algo-phase-title{font-size:0.9rem;font-weight:700;color:#CBD5E1 !important;}
.algo-phase-body{font-size:0.82rem;color:#4E6380 !important;line-height:1.7;}
.algo-phase-body code{background:rgba(255,255,255,0.05);color:#93C5FD !important;font-family:'JetBrains Mono',monospace;font-size:0.78rem;padding:2px 6px;border-radius:4px;}
.algo-phase-accent{position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:16px 0 0 16px;}
.algo-step-log{background:rgba(7,16,31,0.9);border:1px solid rgba(255,255,255,0.04);border-radius:12px;padding:1rem 1.2rem;margin-top:0.6rem;font-family:'JetBrains Mono',monospace;font-size:0.75rem;line-height:1.9;}
.log-line{display:flex;gap:10px;align-items:flex-start;}
.log-tag{flex-shrink:0;padding:1px 7px;border-radius:4px;font-size:0.65rem;font-weight:600;letter-spacing:0.06em;}
.log-tag.ok{background:rgba(52,211,153,0.12);color:#34D399;}
.log-tag.run{background:rgba(251,191,36,0.12);color:#FBBF24;}
.log-tag.inf{background:rgba(99,102,241,0.12);color:#818CF8;}
.log-val{color:#E2E8F0 !important;}
.formula-box{background:rgba(7,16,31,0.9);border:1px solid rgba(59,130,246,0.15);border-radius:12px;padding:1rem 1.4rem;margin:0.6rem 0;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#93C5FD !important;text-align:center;}
.formula-label{font-family:'Geist',sans-serif;font-size:0.72rem;color:#3D5070 !important;text-align:center;margin-top:4px;}
.pill-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:1.5rem;align-items:center;}
.pill{background:rgba(59,130,246,0.09);color:#60A5FA !important;border:1px solid rgba(59,130,246,0.18);border-radius:7px;padding:4px 12px;font-size:0.76rem;font-weight:500;}
.pill-green{background:rgba(16,185,129,0.09);color:#34D399 !important;border:1px solid rgba(16,185,129,0.18);border-radius:7px;padding:4px 12px;font-size:0.76rem;font-weight:500;}
.pill-label{font-size:0.76rem;color:#3D5070 !important;font-weight:500;}
.info-banner{background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:1rem;font-size:0.82rem;color:#4E6380 !important;line-height:1.7;}
.info-banner b{color:#34D399 !important;}
.warn-banner{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:1rem;font-size:0.82rem;color:#4E6380 !important;line-height:1.7;}
.warn-banner b{color:#FBBF24 !important;}
.persona-card{background:rgba(10,18,35,0.85);border:1px solid rgba(255,255,255,0.05);border-radius:18px;overflow:hidden;margin-bottom:1rem;transition:border-color 0.25s,transform 0.2s,box-shadow 0.2s;}
.persona-card:hover{border-color:rgba(59,130,246,0.28);transform:translateY(-2px);box-shadow:0 12px 32px rgba(0,0,0,0.35);}
.persona-accent{height:3px;}
.persona-body{padding:1.3rem 1.4rem;}
.persona-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:3px 10px;border-radius:5px;margin-bottom:0.7rem;}
.persona-name{font-size:0.95rem;font-weight:700;color:#E2E8F0 !important;margin-bottom:0.9rem;line-height:1.3;}
.persona-chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0.8rem;}
.persona-chip{background:rgba(255,255,255,0.04);color:#64748B !important;border:1px solid rgba(255,255,255,0.05);border-radius:6px;padding:4px 10px;font-size:0.73rem;font-weight:500;}
.persona-desc{font-size:0.81rem;color:#4E6380 !important;line-height:1.6;}
.persona-footer{margin-top:0.9rem;padding-top:0.9rem;border-top:1px solid rgba(255,255,255,0.04);font-size:0.76rem;color:#3D5070 !important;display:flex;gap:0.9rem;flex-wrap:wrap;}
.reco-card{background:rgba(10,18,35,0.85);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.3rem 1.4rem;height:100%;}
.reco-header{font-weight:700;font-size:0.88rem;color:#E2E8F0 !important;margin-bottom:1rem;display:flex;align-items:center;gap:8px;}
.reco-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.reco-item{font-size:0.8rem;color:#4E6380 !important;margin:0.55rem 0;display:flex;gap:8px;align-items:flex-start;line-height:1.4;}
.reco-item strong{color:#64748B !important;font-weight:600;min-width:88px;flex-shrink:0;}
.section-divider{height:1px;background:rgba(255,255,255,0.04);border:none;margin:2rem 0;}
.stButton>button{background:rgba(255,255,255,0.04) !important;color:#94A3B8 !important;border:1px solid rgba(255,255,255,0.08) !important;border-radius:12px !important;font-weight:600 !important;font-size:0.875rem !important;padding:0.85rem 1.4rem !important;box-shadow:none !important;transition:all 0.2s !important;margin-top:0.75rem !important;}
.stButton>button:hover{background:rgba(255,255,255,0.07) !important;border-color:rgba(255,255,255,0.14) !important;}
.stButton>button[kind="primary"]{background:#1D4ED8 !important;color:white !important;border:none !important;box-shadow:0 4px 14px rgba(29,78,216,0.3) !important;}
.stButton>button[kind="primary"]:hover{background:#1E40AF !important;transform:translateY(-1px) !important;}
.stDataFrame{border-radius:12px !important;overflow:hidden;border:1px solid rgba(255,255,255,0.05) !important;}
[data-testid="stFileUploader"]{background:rgba(10,18,35,0.7);border:2px dashed rgba(59,130,246,0.35);border-radius:16px;padding:1rem;}
[data-testid="stFileUploaderDropzone"]{background:transparent !important;border:none !important;padding:1.5rem !important;}
[data-testid="stFileUploaderDropzone"] button{background:rgba(59,130,246,0.12) !important;border:1px solid rgba(59,130,246,0.28) !important;color:#60A5FA !important;border-radius:10px !important;font-weight:600 !important;margin-top:0.5rem !important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-thumb{background:rgba(59,130,246,0.28);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_COLORS       = ["#3B82F6","#10B981","#F59E0B","#A78BFA","#F43F5E","#0EA5E9","#8B5CF6","#EC4899"]
CLUSTER_COLORS_LIGHT = ["rgba(59,130,246,0.11)","rgba(16,185,129,0.11)",
                         "rgba(245,158,11,0.11)","rgba(167,139,250,0.11)",
                         "rgba(244,63,94,0.11)","rgba(14,165,233,0.11)"]
CLUSTER_COLORS_TEXT  = ["#60A5FA","#34D399","#FCD34D","#C4B5FD","#FB7185","#38BDF8"]
PHASE_COLORS         = ["#3B82F6","#10B981","#F59E0B","#A78BFA","#F43F5E","#0EA5E9"]

# Pemetaan 20 kategori → 5 grup
CATEGORY_GROUP_MAP = {
    "Food Delivery":"Lifestyle","Grocery Shopping":"Lifestyle",
    "Online Shopping":"Lifestyle","Gift Card":"Lifestyle",
    "Flight Booking":"Travel","Hotel Booking":"Travel",
    "Taxi Fare":"Travel","Bus Ticket":"Travel",
    "Streaming Service":"Hiburan","Gaming Credits":"Hiburan","Movie Ticket":"Hiburan",
    "Education Fee":"Pendidikan",
    "Electricity Bill":"Keuangan","Water Bill":"Keuangan","Gas Bill":"Keuangan",
    "Internet Bill":"Keuangan","Mobile Recharge":"Keuangan",
    "Rent Payment":"Keuangan","Loan Repayment":"Keuangan","Insurance Premium":"Keuangan",
}
DEFAULT_GROUP = "Lainnya"

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
#  ALGORITMA — CRISP-DM
# ══════════════════════════════════════════════════════════════════════════════

def validate_columns(df: pd.DataFrame) -> bool:
    required = {
        "transaction_date","product_category","product_amount",
        "cashback","loyalty_points","payment_method","device_type"
    }
    return required.issubset(set(df.columns))


def fase2_data_understanding(df: pd.DataFrame) -> dict:
    return {
        "n_rows"         : len(df),
        "n_cols"         : len(df.columns),
        "missing_total"  : int(df.isnull().sum().sum()),
        "unique_payment" : df["payment_method"].nunique() if "payment_method" in df.columns else 0,
        "unique_device"  : df["device_type"].nunique()    if "device_type"    in df.columns else 0,
        "unique_category": df["product_category"].nunique() if "product_category" in df.columns else 0,
        "payment_dist"   : (df["payment_method"].value_counts(normalize=True)*100).round(1).to_dict()
                            if "payment_method" in df.columns else {},
        "device_dist"    : (df["device_type"].value_counts(normalize=True)*100).round(1).to_dict()
                            if "device_type" in df.columns else {},
        "amount_stats"   : {
            "mean": df["product_amount"].mean() if "product_amount" in df.columns else 0,
            "std" : df["product_amount"].std()  if "product_amount" in df.columns else 0,
            "min" : df["product_amount"].min()  if "product_amount" in df.columns else 0,
            "max" : df["product_amount"].max()  if "product_amount" in df.columns else 0,
        },
    }


def fase3_data_preparation(df: pd.DataFrame):
    """
    FASE 3 — Data Preparation (v3 FINAL — 7 Variabel Lengkap):

    3a. Parsing datetime → transaction_hour (numerik 0-23)
    3b. product_category → 5 grup → One-Hot Encoding (5 kolom biner)
    3c. payment_method → One-Hot Encoding (5 kolom biner)  ← PERBAIKAN dari LabelEncoder
    3d. device_type → One-Hot Encoding (3 kolom biner)     ← PERBAIKAN dari LabelEncoder
    3e. Gabungkan semua: 4 numerik + 5 OHE payment + 3 OHE device + 5 OHE kategori = 17 fitur
    3f. StandardScaler → Z-Score normalisasi SETELAH OHE
        Ini kunci: StandardScaler menyamakan bobot semua fitur sehingga
        OHE tidak mendominasi jarak Euclidean K-Means

    Mengapa OHE lebih baik dari LabelEncoder untuk variabel nominal:
    - LabelEncoder: Android=0, iOS=1, Web=2 → K-Means menganggap iOS "lebih dekat" ke Android
    - OHE: setiap kategori jadi kolom biner terpisah → tidak ada urutan palsu
    """
    df = df.copy()
    prep_log = {}

    # 3a. Parsing datetime & ekstrak jam
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True, errors="coerce")
    df["transaction_hour"] = df["transaction_date"].dt.hour.fillna(0).astype(int)
    prep_log["3a"] = "pd.to_datetime → .dt.hour → transaction_hour ∈ [0,23]"

    # 3b. Pengelompokan product_category → 5 grup → OHE
    df["category_group"] = df["product_category"].map(CATEGORY_GROUP_MAP).fillna(DEFAULT_GROUP)
    cat_ohe = pd.get_dummies(df["category_group"], prefix="cat")
    cat_cols = sorted(cat_ohe.columns.tolist())
    cat_ohe  = cat_ohe[cat_cols]
    prep_log["3b"] = f"product_category → 5 grup → OHE: {cat_cols}"

    # 3c. payment_method → OHE (menggantikan LabelEncoder)
    pay_ohe  = pd.get_dummies(df["payment_method"], prefix="pay")
    pay_cols = sorted(pay_ohe.columns.tolist())
    pay_ohe  = pay_ohe[pay_cols]
    prep_log["3c"] = f"payment_method OHE: {pay_cols}"

    # 3d. device_type → OHE (menggantikan LabelEncoder)
    dev_ohe  = pd.get_dummies(df["device_type"], prefix="dev")
    dev_cols = sorted(dev_ohe.columns.tolist())
    dev_ohe  = dev_ohe[dev_cols]
    prep_log["3d"] = f"device_type OHE: {dev_cols}"

    # 3e. Gabungkan semua fitur
    numeric_cols = ["transaction_hour","product_amount","cashback","loyalty_points"]
    X_df = pd.concat([
        df[numeric_cols].reset_index(drop=True),
        pay_ohe.reset_index(drop=True),
        dev_ohe.reset_index(drop=True),
        cat_ohe.reset_index(drop=True),
    ], axis=1).dropna()

    all_features = numeric_cols + pay_cols + dev_cols + cat_cols
    prep_log["3e"] = (
        f"Gabungan fitur: {len(numeric_cols)} numerik + "
        f"{len(pay_cols)} OHE payment + {len(dev_cols)} OHE device + "
        f"{len(cat_cols)} OHE kategori = {len(all_features)} fitur total"
    )

    # 3f. StandardScaler — kunci agar OHE tidak mendominasi
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    prep_log["3f"] = (
        f"StandardScaler.fit_transform → semua {len(all_features)} fitur "
        f"dinormalisasi Z-Score (mean=0, std=1) → bobot seimbang"
    )

    # Simpan nama kolom untuk analisis
    feature_groups = {
        "numerik" : numeric_cols,
        "payment" : pay_cols,
        "device"  : dev_cols,
        "kategori": cat_cols,
    }

    return df, X_scaled, all_features, feature_groups, scaler, prep_log


def fase4a_elbow_method(X_scaled, k_min=2, k_max=10):
    k_range, wcss, logs = [], [], []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        km.fit(X_scaled)
        k_range.append(k); wcss.append(km.inertia_)
        logs.append({"k":k,"wcss":round(km.inertia_,2),"n_iterations":km.n_iter_})
    return k_range, wcss, logs


def fase4b_silhouette_analysis(X_scaled, k_range):
    sil_scores, sil_log = [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        sil_log.append({"k":k,"score":round(score,4),"best":False})
    best_idx = int(np.argmax(sil_scores))
    sil_log[best_idx]["best"] = True
    return sil_scores, sil_log


def fase4c_kmeans_final(X_scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    return labels, km, {
        "k"             : k,
        "n_iterations"  : km.n_iter_,
        "final_inertia" : round(km.inertia_, 4),
        "convergence"   : km.n_iter_ < km.max_iter,
        "cluster_sizes" : {int(i): int(np.sum(labels==i)) for i in range(k)},
        "centroids"     : km.cluster_centers_,
    }


def fase4d_pca(X_scaled, labels):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_, {
        "total_var": round(sum(pca.explained_variance_ratio_)*100, 2)
    }


def fase5_profiling(df_proc):
    return df_proc.groupby("cluster").agg(
        jumlah   = ("cluster",         "count"),
        nominal  = ("product_amount",   "mean"),
        cashback = ("cashback",         "mean"),
        poin     = ("loyalty_points",   "mean"),
        jam      = ("transaction_hour", lambda x: x.mode()[0]),
        metode   = ("payment_method",   lambda x: x.mode()[0]),
        device   = ("device_type",      lambda x: x.mode()[0]),
        kategori = ("category_group",   lambda x: x.mode()[0]),
    ).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
#  PERSONA — berbasis SEMUA 7 variabel
# ══════════════════════════════════════════════════════════════════════════════
def _level(nominal):
    if nominal >= 6558: return "Premium"
    if nominal >= 3233: return "Menengah"
    return "Hemat"

def _sesi(jam):
    if jam >= 20 or jam < 5: return "Malam"
    if jam < 12: return "Pagi"
    if jam < 17: return "Siang"
    return "Sore"

def _emoji_level(level):
    return {"Premium":"💎","Menengah":"⚡","Hemat":"🌱"}.get(level,"")

def _emoji_sesi(sesi):
    return {"Malam":"🌙","Pagi":"🌅","Siang":"☀️","Sore":"🌆"}.get(sesi,"")

def get_persona_name(row):
    level = _level(row["nominal"])
    sesi  = _sesi(int(row["jam"]))
    cb_hi = row["cashback"] > 50
    pt_hi = row["poin"] > 500

    if cb_hi and pt_hi:  reward = "Reward Hunter"
    elif cb_hi:          reward = "Cashback Hunter"
    elif pt_hi:          reward = "Kolektor Poin"
    else:                reward = "Reguler"

    return f"{_emoji_level(level)} {level} · {reward} · {_emoji_sesi(sesi)} {sesi}"


def get_persona_desc(row):
    level   = _level(row["nominal"])
    sesi    = _sesi(int(row["jam"]))
    cb_hi   = row["cashback"] > 50
    pt_hi   = row["poin"] > 500

    level_desc = {
        "Premium" : "Transaksi besar",
        "Menengah": "Transaksi menengah",
        "Hemat"   : "Transaksi kecil namun aktif",
    }[level]

    reward_desc = ""
    if cb_hi and pt_hi:
        reward_desc = "Segmen ini paling responsif terhadap program reward ganda — cashback dan poin sama-sama tinggi."
    elif cb_hi:
        reward_desc = "Fokus pada cashback — manfaatkan dengan promo cashback berlipat dan flash deal."
    elif pt_hi:
        reward_desc = "Akumulator poin setia — cocok untuk program membership tier dan hadiah redeem poin."
    else:
        reward_desc = "Pola reward standar — potensial ditingkatkan dengan program onboarding atau bonus loyalitas."

    return (
        f"{level_desc}, dominan pukul "
        f"<strong style='color:#93C5FD'>{int(row['jam']):02d}:00</strong> "
        f"({sesi}) via <strong style='color:#CBD5E1'>{row['device']}</strong>. "
        f"Metode favorit: <strong style='color:#93C5FD'>{row['metode']}</strong>. "
        f"{reward_desc}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def gs(k, d=None): return st.session_state.get(k, d)

def render_phase(num, title, body, color, log_html=""):
    st.markdown(f"""
    <div class="algo-phase">
        <div class="algo-phase-accent" style="background:{color}"></div>
        <div class="algo-phase-header">
            <span class="algo-phase-num">{num}</span>
            <span class="algo-phase-title">{title}</span>
        </div>
        <div class="algo-phase-body">{body}</div>
        {log_html}
    </div>""", unsafe_allow_html=True)

def make_log(lines):
    inner = "".join(
        f'<div class="log-line"><span class="log-tag {t}">{t.upper()}</span>'
        f'<span class="log-val">{txt}</span></div>'
        for t, txt in lines
    )
    return f'<div class="algo-step-log">{inner}</div>'


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-eyebrow">Fintech Analytics · CRISP-DM Framework · v3 Final</div>
    <h1>Analisis Cluster E-Wallet</h1>
    <p>Segmentasi pola perilaku transaksi digital menggunakan K-Means Clustering dengan
       pendekatan 5 fase CRISP-DM. Seluruh 7 variabel digunakan — variabel kategorikal
       dengan One-Hot Encoding + StandardScaler untuk bobot seimbang.</p>
</div>
""", unsafe_allow_html=True)

step_now    = gs("step", "upload")
steps       = ["upload","preview","analisis","hasil"]
step_icons  = ["↑","⌕","◎","⊞"]
step_labels = ["Unggah Data","Preview & Validasi","Proses Algoritma","Hasil & Insight"]
step_nums   = ["01","02","03","04"]

bar = '<div class="step-bar">'
for s, icon, label, num in zip(steps, step_icons, step_labels, step_nums):
    idx_now = steps.index(step_now); idx_s = steps.index(s)
    cls = "active" if s==step_now else ("done" if idx_s<idx_now else "step")
    bar += f'<div class="step {cls}"><span class="step-num">({num})</span> {icon} {label}</div>'
bar += "</div>"
st.markdown(bar, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if step_now == "upload":
    st.markdown('<div class="section-title"><span class="section-title-icon">🗄</span> Unggah File CSV Transaksi</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1,1], gap="large")

    with col_left:
        st.markdown("""
        <div class="upload-info">
            <div class="upload-info-header">ⓘ &nbsp;Persyaratan Data</div>
            <p>File CSV dengan separator <code style="color:#93C5FD;background:rgba(255,255,255,0.05);padding:1px 5px;border-radius:3px">;</code> — 7 kolom wajib:</p>
            <div class="col-list">
                <div class="col-item"><span class="col-dot" style="background:#3B82F6"></span>
                    <span class="col-name">transaction_date</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">DD/MM/YYYY HH:MM</span></div>
                <div class="col-item"><span class="col-dot" style="background:#10B981"></span>
                    <span class="col-name">product_category</span>
                    <span style="font-size:0.7rem;color:#34D399;margin-left:auto">→ 5 grup → OHE</span></div>
                <div class="col-item"><span class="col-dot" style="background:#3B82F6"></span>
                    <span class="col-name">product_amount</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Numerik</span></div>
                <div class="col-item"><span class="col-dot" style="background:#3B82F6"></span>
                    <span class="col-name">cashback</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Numerik</span></div>
                <div class="col-item"><span class="col-dot" style="background:#3B82F6"></span>
                    <span class="col-name">loyalty_points</span>
                    <span style="font-size:0.7rem;color:#3D5070;margin-left:auto">Numerik</span></div>
                <div class="col-item"><span class="col-dot" style="background:#10B981"></span>
                    <span class="col-name">payment_method</span>
                    <span style="font-size:0.7rem;color:#34D399;margin-left:auto">→ One-Hot Encoding</span></div>
                <div class="col-item"><span class="col-dot" style="background:#10B981"></span>
                    <span class="col-name">device_type</span>
                    <span style="font-size:0.7rem;color:#34D399;margin-left:auto">→ One-Hot Encoding</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("☁️  Seret & lepas file CSV", type=["csv"], key="file_uploader")
        if uploaded is not None:
            try:
                df_preview = pd.read_csv(uploaded, sep=";")
                if not validate_columns(df_preview):
                    st.error("❌ Kolom tidak sesuai. Pastikan 7 kolom wajib ada.")
                else:
                    st.session_state["df_pending"] = df_preview
                    n_rows = len(df_preview); missing = int(df_preview.isnull().sum().sum())
                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:1rem 1.2rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#34D399;margin-bottom:0.6rem;">✓ File Terdeteksi</div>
                        <div style="font-size:0.82rem;color:#64748B;line-height:1.8;">
                            📄 <b style="color:#94A3B8">{uploaded.name}</b><br>
                            📊 {n_rows:,} baris · {'<span style="color:#F87171">'+str(missing)+' missing</span>' if missing>0 else '<span style="color:#34D399">0 missing ✓</span>'}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.dataframe(df_preview.head(3), use_container_width=True, height=140)
                    if st.button("✅ Gunakan File Ini → Preview & Validasi", type="primary", use_container_width=True):
                        st.session_state["df_raw"] = df_preview
                        st.session_state["step"]   = "preview"
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗄  Gunakan Data Contoh (project_Data_ewallet.csv)", use_container_width=True):
            try:
                df = pd.read_csv("/mnt/user-data/uploads/project_Data_ewallet.csv", sep=";")
                st.session_state["df_raw"] = df
                st.session_state["step"]   = "preview"
                st.rerun()
            except Exception as e:
                st.error(f"❌ File contoh tidak ditemukan: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PREVIEW & VALIDASI
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "preview":
    df = gs("df_raw")
    st.markdown('<div class="section-title"><span class="section-title-icon">⌕</span> Preview & Validasi — FASE 2: Data Collection & Understanding</div>', unsafe_allow_html=True)
    du = fase2_data_understanding(df)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-number">{du['n_rows']:,}</div><div class="stat-label">Total Transaksi</div></div>
        <div class="stat-card"><div class="stat-number">{du['n_cols']}</div><div class="stat-label">Kolom Asli</div></div>
        <div class="stat-card"><div class="stat-number">{du['missing_total']}</div><div class="stat-label">Missing Values</div></div>
        <div class="stat-card"><div class="stat-number">17</div><div class="stat-label">Fitur Setelah OHE</div></div>
        <div class="stat-card"><div class="stat-number">{du['unique_payment']}</div><div class="stat-label">Metode Bayar</div></div>
        <div class="stat-card"><div class="stat-number">{du['unique_device']}</div><div class="stat-label">Jenis Perangkat</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <b>Strategi Encoding v3 (Perbaikan dari versi sebelumnya):</b><br>
        • <b>Masalah v1:</b> LabelEncoder untuk variabel nominal → menambahkan urutan palsu (Android=0 < iOS=1 < Web=2)<br>
        • <b>Masalah v2:</b> Membuang payment_method, device_type, transaction_hour dari clustering → hanya 3 fitur<br>
        • <b>Solusi v3:</b> One-Hot Encoding untuk semua variabel nominal + StandardScaler setelah OHE<br>
        &nbsp;&nbsp;→ <code>payment_method</code> (5 kolom biner) + <code>device_type</code> (3 kolom biner) + <code>category_group</code> (5 kolom biner) + 4 numerik = <b>17 fitur</b><br>
        &nbsp;&nbsp;→ StandardScaler menyamakan bobot → tidak ada variabel yang mendominasi jarak Euclidean
    </div>
    """, unsafe_allow_html=True)

    pay_str = "  |  ".join([f"{k}: {v:.1f}%" for k,v in list(du["payment_dist"].items())[:5]])
    dev_str = "  |  ".join([f"{k}: {v:.1f}%" for k,v in list(du["device_dist"].items())[:3]])

    render_phase(
        "FASE 02", "DATA COLLECTION & UNDERSTANDING",
        f"""EDA awal: <code>{du['n_rows']:,}</code> transaksi, <code>{du['n_cols']}</code> kolom.<br><br>
        <b>payment_method:</b> {pay_str}<br>
        <b>device_type:</b> {dev_str}<br>
        <b>product_amount:</b> mean=Rp{du['amount_stats']['mean']:,.0f} | std={du['amount_stats']['std']:,.0f} | range [{du['amount_stats']['min']:,.0f} – {du['amount_stats']['max']:,.0f}]<br><br>
        Validasi: <code>set.issubset()</code> · Missing: <code>df.isnull().sum()</code> → {du['missing_total']}""",
        PHASE_COLORS[1],
        make_log([
            ("ok",  "7 kolom wajib terverifikasi via set.issubset()"),
            ("ok",  f"n_rows={du['n_rows']:,} | missing={du['missing_total']}"),
            ("inf", f"payment_method: {du['unique_payment']} metode → OHE 5 kolom"),
            ("inf", f"device_type: {du['unique_device']} platform → OHE 3 kolom"),
            ("inf", f"product_category: {du['unique_category']} kategori → 5 grup → OHE 5 kolom"),
            ("inf", "Numerik: transaction_hour, product_amount, cashback, loyalty_points"),
            ("ok",  "Total setelah OHE: 4 + 5 + 3 + 5 = 17 fitur"),
        ])
    )

    st.markdown('<p style="font-size:0.85rem;color:#3D5070;margin:1rem 0 0.5rem;font-weight:600;">5 Baris Pertama Data</p>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True, height=220)

    c1, c2 = st.columns([1,4])
    with c1:
        if st.button("← Ganti File"): st.session_state["step"]="upload"; st.rerun()
    with c2:
        if st.button("Lanjut ke Proses Algoritma →", type="primary", use_container_width=True):
            st.session_state["step"]="analisis"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — ANALISIS
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "analisis":
    df = gs("df_raw")
    st.markdown('<div class="section-title"><span class="section-title-icon">⚙</span> Pipeline K-Means Clustering — CRISP-DM Fase 1, 3 & 4</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Memulai pipeline…")
    status_text  = st.empty()

    # FASE 1
    render_phase(
        "FASE 01", "BUSINESS UNDERSTANDING",
        """Tujuan: identifikasi pola perilaku pengguna e-wallet menggunakan
        <b>7 variabel lengkap</b> — waktu transaksi, kategori produk, nilai transaksi,
        cashback, poin loyalitas, metode pembayaran, dan jenis perangkat.
        <br><br>
        <b>Kriteria keberhasilan:</b> Silhouette Score <code>≥ 0.4</code> via
        <code>sklearn.metrics.silhouette_score</code>.<br><br>
        <b>Keputusan encoding:</b> Variabel nominal (<code>payment_method</code>,
        <code>device_type</code>, <code>product_category</code>) menggunakan
        <b>One-Hot Encoding</b> — tidak ada urutan palsu.
        StandardScaler setelah OHE memastikan semua 17 fitur memiliki bobot seimbang.""",
        PHASE_COLORS[0],
        make_log([
            ("inf", "7 variabel asli → semua masuk clustering"),
            ("inf", "Kriteria sukses: Silhouette ≥ 0.4"),
            ("inf", "Encoding: OHE (bukan LabelEncoder) untuk variabel nominal"),
            ("inf", "StandardScaler SETELAH OHE → bobot Euclidean seimbang"),
        ])
    )
    progress_bar.progress(10, text="Fase 1 selesai…")

    # FASE 3
    status_text.markdown("⚙️ **FASE 3** — Data Preparation…")
    progress_bar.progress(20, text="Fase 3: One-Hot Encoding + StandardScaler…")

    df_proc, X_scaled, all_features, feat_groups, scaler, prep_log = \
        fase3_data_preparation(df)

    n_num  = len(feat_groups["numerik"])
    n_pay  = len(feat_groups["payment"])
    n_dev  = len(feat_groups["device"])
    n_cat  = len(feat_groups["kategori"])
    n_tot  = len(all_features)

    render_phase(
        "FASE 03", f"DATA PREPARATION — One-Hot Encoding + StandardScaler ({n_tot} fitur)",
        f"""<b>3a. Parsing Datetime → transaction_hour</b><br>
        <code>pd.to_datetime(dayfirst=True)</code> → <code>.dt.hour</code> ∈ [0,23]<br><br>
        <b>3b. product_category → 5 Grup → OHE ({n_cat} kolom biner)</b><br>
        20 kategori asli dikelompokkan: Lifestyle / Travel / Hiburan / Pendidikan / Keuangan.
        Kemudian <code>pd.get_dummies(prefix='cat')</code> → {n_cat} kolom biner.<br><br>
        <b>3c. payment_method → OHE ({n_pay} kolom biner) ← PERBAIKAN dari v1</b><br>
        <code>pd.get_dummies(df['payment_method'], prefix='pay')</code>.<br>
        Alasan: LabelEncoder di v1 menambahkan urutan palsu (Credit Card=0 < Debit=1 < UPI=2).
        OHE mengubah setiap metode menjadi kolom biner terpisah, tanpa urutan.<br><br>
        <b>3d. device_type → OHE ({n_dev} kolom biner) ← PERBAIKAN dari v1</b><br>
        <code>pd.get_dummies(df['device_type'], prefix='dev')</code>.<br>
        Alasan: Android/iOS/Web tidak memiliki urutan bermakna.<br><br>
        <b>3e. Gabungkan → Matrix X shape: <code>{X_scaled.shape}</code></b><br>
        {n_num} numerik + {n_pay} OHE payment + {n_dev} OHE device + {n_cat} OHE kategori = {n_tot} fitur<br><br>
        <b>3f. StandardScaler — kunci keseimbangan bobot OHE</b><br>
        <code>StandardScaler().fit_transform(X)</code> → mean=0, std=1 untuk semua {n_tot} fitur.
        Tanpa ini, kolom numerik (product_amount ∈ [10–9997]) akan mendominasi
        kolom OHE biner (0/1). Dengan ini, semua fitur berkontribusi seimbang.""",
        PHASE_COLORS[2],
        make_log([
            ("ok",  f"3a. transaction_hour ∈ [0,23]"),
            ("ok",  f"3b. OHE category_group: {feat_groups['kategori']}"),
            ("ok",  f"3c. OHE payment_method: {feat_groups['payment']}"),
            ("ok",  f"3d. OHE device_type: {feat_groups['device']}"),
            ("ok",  f"3e. Matrix X shape = {X_scaled.shape} ({n_tot} fitur)"),
            ("ok",  f"3f. StandardScaler applied → semua fitur z-score"),
        ])
    )
    st.markdown(f"""
    <div class="formula-box">x' = (x − μ) / σ &nbsp;·&nbsp; diterapkan pada {n_tot} fitur (4 numerik + {n_pay} OHE payment + {n_dev} OHE device + {n_cat} OHE kategori)</div>
    <div class="formula-label">StandardScaler SETELAH OHE → bobot Euclidean seimbang → semua 7 variabel berkontribusi pada clustering</div>
    """, unsafe_allow_html=True)

    # Pill row fitur
    st.markdown('<div class="pill-row"><span class="pill-label">Numerik:</span>' +
        "".join(f'<span class="pill">{f}</span>' for f in feat_groups["numerik"]) +
        '<span class="pill-label" style="margin-left:8px">OHE:</span>' +
        "".join(f'<span class="pill-green">{f}</span>' for f in feat_groups["payment"]+feat_groups["device"]+feat_groups["kategori"]) +
        '</div>', unsafe_allow_html=True)

    # FASE 4a
    status_text.markdown("📈 **FASE 4a** — Elbow Method…")
    progress_bar.progress(35, text="Fase 4a: Elbow Method…")
    k_range, wcss, elbow_log = fase4a_elbow_method(X_scaled)

    render_phase(
        "FASE 04a", "MODELING — Elbow Method (k=2–10)",
        """Loop k=2 hingga 10. Tiap iterasi:
        <code>KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42).fit(X_scaled)</code>
        lalu simpan <code>.inertia_</code> (WCSS). Kurva Elbow diplot dengan garis vertikal
        <code>fig.add_vline</code> di k optimal.<br><br>
        <b>k-means++:</b> centroid awal dipilih proporsional terhadap jarak kuadrat dari
        centroid sebelumnya — lebih stabil dari random init.""",
        PHASE_COLORS[0],
        make_log([("run", f"Loop k={k_range}")] +
                 [("ok", f"k={e['k']}: WCSS={e['wcss']:,.1f} | iter={e['n_iterations']}") for e in elbow_log])
    )
    st.markdown('<div class="formula-box">WCSS = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²</div><div class="formula-label">Within-Cluster Sum of Squares — berbasis jarak Euclidean di ruang 17D ternormalisasi</div>', unsafe_allow_html=True)

    # FASE 4b
    status_text.markdown("📊 **FASE 4b** — Silhouette Score…")
    progress_bar.progress(55, text="Fase 4b: Silhouette Analysis…")
    sil_scores, sil_log = fase4b_silhouette_analysis(X_scaled, k_range)
    optimal_k = k_range[int(np.argmax(sil_scores))]
    best_sil  = max(sil_scores)
    met       = "✓ Memenuhi" if best_sil >= 0.4 else "⚠ Di bawah"

    render_phase(
        "FASE 04b", "MODELING — Silhouette Score (Validasi k Optimal)",
        f"""<code>silhouette_score(X_scaled, km.labels_)</code> per k.
        <code>np.argmax(sil_scores)</code> → <b>optimal_k = {optimal_k}</b>,
        score = <code>{best_sil:.4f}</code> — <b>{met} kriteria ≥ 0.4</b>.<br><br>
        <b>a(i)</b> = rata-rata jarak ke anggota cluster sendiri (cohesion)<br>
        <b>b(i)</b> = rata-rata jarak ke cluster terdekat lainnya (separation)<br>
        Score mendekati +1 → cluster padat dan terpisah baik.""",
        PHASE_COLORS[3],
        make_log(
            [("run","silhouette_score per k")] +
            [("ok" if e["best"] else "inf", f"k={e['k']}: {e['score']}  {'← OPTIMAL ✓' if e['best'] else ''}") for e in sil_log] +
            [("ok" if best_sil>=0.4 else "run", f"optimal_k={optimal_k} | {met} (≥0.4)")]
        )
    )
    st.markdown('<div class="formula-box">s(i) = (b(i) − a(i)) / max(a(i), b(i))</div><div class="formula-label">Silhouette Coefficient per titik — rata-rata semua titik = score keseluruhan</div>', unsafe_allow_html=True)

    # FASE 4c
    status_text.markdown(f"🎯 **FASE 4c** — Training K-Means Final k={optimal_k}…")
    progress_bar.progress(72, text=f"Fase 4c: K-Means k={optimal_k}…")
    labels, km_model, km_log = fase4c_kmeans_final(X_scaled, optimal_k)

    render_phase(
        "FASE 04c", f"MODELING — Training Model Final K-Means (k={optimal_k})",
        f"""<code>KMeans(n_clusters={optimal_k}, init='k-means++', max_iter=300, n_init=10, random_state=42).fit_predict(X_scaled)</code><br>
        Konvergen dalam <code>{km_log['n_iterations']} iterasi</code>,
        WCSS final = <code>{km_log['final_inertia']:,.2f}</code>.<br><br>
        <b>Jarak Euclidean di ruang {n_tot}D:</b> tepat digunakan karena semua fitur
        sudah dinormalisasi Z-Score — perbedaan skala tidak bias terhadap assignment cluster.""",
        PHASE_COLORS[4],
        make_log([
            ("run", f"KMeans(n_clusters={optimal_k}, init='k-means++', n_init=10, random_state=42)"),
            ("run", ".fit_predict(X_scaled) → array label integer"),
            ("ok",  f"Konvergen dalam {km_log['n_iterations']} iterasi"),
            ("ok",  f"WCSS final = {km_log['final_inertia']:,.2f}"),
            ("ok",  f"Distribusi: { {f'C{k}':v for k,v in km_log['cluster_sizes'].items()} }"),
        ])
    )
    st.markdown('<div class="formula-box">c(x) = argminᵢ ‖x − μᵢ‖²  →  μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x</div><div class="formula-label">Assignment (kiri) dan Update Step (kanan) — diulang hingga tidak ada perubahan label</div>', unsafe_allow_html=True)

    # FASE 4d
    status_text.markdown("🗺 **FASE 4d** — PCA 2D…")
    progress_bar.progress(88, text="Fase 4d: PCA Reduksi Dimensi…")
    coords, var_ratio, pca_log = fase4d_pca(X_scaled, labels)

    render_phase(
        "FASE 04d", f"MODELING — Reduksi PCA 2D (dari {n_tot}D → 2D, visualisasi)",
        f"""<code>PCA(n_components=2, random_state=42).fit_transform(X_scaled)</code><br>
        PC1 = <code>{var_ratio[0]*100:.1f}%</code> variansi,
        PC2 = <code>{var_ratio[1]*100:.1f}%</code>.
        Total = <code>{pca_log['total_var']:.1f}%</code>.<br><br>
        <b>Catatan:</b> Dengan 17 fitur, variansi yang dijelaskan 2 komponen PCA akan lebih rendah
        dibanding versi 3 fitur. Ini <b>normal</b> — label cluster tetap berasal dari K-Means
        17D, bukan dari PCA. PCA hanya untuk visualisasi 2D.""",
        PHASE_COLORS[5],
        make_log([
            ("run", f"PCA(n_components=2) dari ruang {n_tot}D → 2D"),
            ("ok",  f"PC1={var_ratio[0]*100:.2f}% | PC2={var_ratio[1]*100:.2f}%"),
            ("inf", f"Total variansi PCA 2D = {pca_log['total_var']:.1f}%"),
            ("inf", f"Label cluster = K-Means {n_tot}D (bukan dari PCA)"),
        ])
    )

    # Simpan state
    st.session_state.update({
        "df_proc"      : df_proc,
        "X_scaled"     : X_scaled,
        "all_features" : all_features,
        "feat_groups"  : feat_groups,
        "scaler"       : scaler,
        "k_range"      : k_range,
        "wcss"         : wcss,
        "sil_scores"   : sil_scores,
        "sil_log"      : sil_log,
        "optimal_k"    : optimal_k,
        "labels"       : labels,
        "km_model"     : km_model,
        "km_log"       : km_log,
        "coords"       : coords,
        "var_ratio"    : var_ratio,
    })

    progress_bar.progress(100, text="✅ Pipeline selesai!")
    time.sleep(0.3)
    status_text.empty()

    st.success(
        f"✅ Pipeline CRISP-DM selesai! **{optimal_k} cluster optimal** terbentuk "
        f"(Silhouette Score = {best_sil:.4f} — "
        f"{'Memenuhi' if best_sil >= 0.4 else 'Di bawah'} kriteria ≥ 0.4)"
    )
    if st.button("Lihat Hasil & Insight →", type="primary", use_container_width=True):
        st.session_state["step"] = "hasil"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — HASIL
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
    feat_groups= gs("feat_groups") or {}
    n_tot      = len(gs("all_features") or []) or 17

    df_proc            = df_proc.iloc[:len(labels)].copy()
    df_proc["cluster"] = labels
    best_sil           = max(sil_scores)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-number">{len(df_proc):,}</div><div class="stat-label">Total Transaksi</div></div>
        <div class="stat-card"><div class="stat-number">{optimal_k}</div><div class="stat-label">Cluster Optimal</div></div>
        <div class="stat-card"><div class="stat-number">{best_sil:.3f}</div><div class="stat-label">Silhouette Score</div></div>
        <div class="stat-card"><div class="stat-number">{n_tot}</div><div class="stat-label">Fitur (Setelah OHE)</div></div>
        <div class="stat-card"><div class="stat-number">{sum(var_ratio)*100:.1f}%</div><div class="stat-label">Variansi PCA 2D</div></div>
        <div class="stat-card"><div class="stat-number">{'✓' if best_sil>=0.4 else '✗'}</div><div class="stat-label">Kriteria ≥ 0.4</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-title"><span class="section-title-icon">📈</span> Penentuan Cluster Optimal</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=k_range, y=wcss, mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
            line=dict(color="#2563EB",width=2.5),
            marker=dict(size=8,color="#2563EB",line=dict(color="#08111E",width=2))
        ))
        fig_e.add_vline(x=optimal_k, line_dash="dot", line_color="#60A5FA", line_width=1.5,
                        annotation_text=f"k={optimal_k}",
                        annotation_font=dict(color="#60A5FA",size=11))
        fig_e.update_layout(title="Elbow Method — WCSS per k",
                            xaxis_title="k", yaxis_title="WCSS", height=300, **DARK_LAYOUT)
        fig_e.update_xaxes(**DARK_GRID, tickvals=k_range)
        fig_e.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_e, use_container_width=True)

    with c2:
        bar_colors = ["#2563EB" if i==int(np.argmax(sil_scores)) else "rgba(37,99,235,0.22)"
                      for i in range(len(k_range))]
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=k_range, y=[round(s,3) for s in sil_scores],
            marker=dict(color=bar_colors, cornerradius=6),
            text=[f"{s:.3f}" for s in sil_scores], textposition="outside",
            textfont=dict(size=11,color="#4E6380"),
        ))
        fig_s.add_hline(y=0.4, line_dash="dot", line_color="#34D399", line_width=1.2,
                        annotation_text="≥0.4", annotation_font=dict(color="#34D399",size=10))
        fig_s.update_layout(title="Silhouette Score per k",
                            xaxis_title="k", yaxis_title="Score",
                            yaxis_range=[0, max(sil_scores)*1.28], height=300, **DARK_LAYOUT)
        fig_s.update_xaxes(showgrid=False, tickvals=k_range)
        fig_s.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # PCA Scatter + Profil
    st.markdown('<div class="section-title"><span class="section-title-icon">🗺</span> Sebaran Cluster PCA 2D + Profil</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([1.3,1])

    with c3:
        df_pca = pd.DataFrame({
            "PC1": coords[:,0], "PC2": coords[:,1],
            "Cluster": [f"Cluster {l}" for l in labels]
        })
        fig_sc = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster",
                            color_discrete_sequence=CLUSTER_COLORS,
                            title=f"PCA 2D — {var_ratio[0]*100:.1f}%+{var_ratio[1]*100:.1f}%={sum(var_ratio)*100:.1f}% variansi",
                            opacity=0.65, height=420)
        fig_sc.update_traces(marker=dict(size=5,line=dict(width=0)))
        fig_sc.update_layout(**DARK_LAYOUT,
                             legend=dict(orientation="h",yanchor="bottom",y=-0.24,xanchor="center",x=0.5,font=dict(color="#4E6380")))
        fig_sc.update_xaxes(**DARK_GRID); fig_sc.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_sc, use_container_width=True)

    with c4:
        profile_raw = fase5_profiling(df_proc)
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#64748B;margin-bottom:0.5rem;">Profil Cluster — groupby().agg()</p>', unsafe_allow_html=True)
        pd_disp = profile_raw.copy()
        pd_disp["nominal"]  = pd_disp["nominal"].map("Rp{:,.0f}".format)
        pd_disp["cashback"] = pd_disp["cashback"].map("{:.1f}".format)
        pd_disp["poin"]     = pd_disp["poin"].map("{:.0f}".format)
        pd_disp["jam"]      = pd_disp["jam"].map(lambda x: f"{int(x):02d}:00")
        pd_disp["cluster"]  = pd_disp["cluster"].map(lambda x: f"C{x}")
        pd_disp.columns     = ["Cluster","N","Nominal","CB","Poin","Jam","Metode","Device","Kategori"]
        st.dataframe(pd_disp.set_index("Cluster"), use_container_width=True, height=380)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Persona Cards
    st.markdown('<div class="section-title"><span class="section-title-icon">🎯</span> FASE 5: Evaluation — Persona Cluster</div>', unsafe_allow_html=True)
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
                <div class="persona-accent" style="background:linear-gradient(90deg,{clr},{clr}44)"></div>
                <div class="persona-body">
                    <span class="persona-badge" style="background:{cll};color:{clt}">
                        Cluster {c} · {pct:.1f}% · n={row['jumlah']:,}
                    </span>
                    <div class="persona-name">{get_persona_name(row)}</div>
                    <div class="persona-chips">
                        <span class="persona-chip">⏰ {int(row['jam']):02d}:00</span>
                        <span class="persona-chip">📱 {row['device']}</span>
                        <span class="persona-chip">Rp {row['nominal']:,.0f}</span>
                        <span class="persona-chip">🛍 {row['kategori']}</span>
                    </div>
                    <div class="persona-desc">{get_persona_desc(row)}</div>
                    <div class="persona-footer">
                        <span>💳 {row['metode']}</span>
                        <span>⭐ {row['poin']:.0f} poin</span>
                        <span>💸 cb {row['cashback']:.1f}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Rekomendasi
    st.markdown('<div class="section-title"><span class="section-title-icon">💡</span> Rekomendasi Strategi Pemasaran per Cluster</div>', unsafe_allow_html=True)
    reco_cols = st.columns(min(optimal_k, 4))

    for _, row in profile_raw.iterrows():
        c      = int(row["cluster"])
        clr    = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        reward = "Cashback" if row["cashback"] > row["poin"]/10 else "Poin Loyalitas"
        waktu  = f"{int(row['jam']):02d}:00 – {(int(row['jam'])+2)%24:02d}:00"
        level  = _level(row["nominal"])

        with reco_cols[c % len(reco_cols)]:
            st.markdown(f"""
            <div class="reco-card">
                <div class="reco-header">
                    <span class="reco-dot" style="background:{clr}"></span>
                    Cluster {c} · {level}
                </div>
                <div class="reco-item"><strong>⏰ Waktu Promosi</strong>{waktu}</div>
                <div class="reco-item"><strong>🎁 Jenis Reward</strong>{reward}</div>
                <div class="reco-item"><strong>📲 Kanal</strong>{row['device']} App</div>
                <div class="reco-item"><strong>🛒 Kategori</strong>{row['kategori']}</div>
                <div class="reco-item"><strong>💳 Metode</strong>{row['metode']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Ringkasan Pipeline
    st.markdown('<div class="section-title"><span class="section-title-icon">📋</span> Ringkasan Pipeline CRISP-DM v3</div>', unsafe_allow_html=True)
    fg = gs("feat_groups") or {}
    st.markdown(f"""
    <div class="algo-phase">
        <div class="algo-phase-accent" style="background:linear-gradient(180deg,#3B82F6,#10B981,#F59E0B,#A78BFA,#F43F5E,#0EA5E9)"></div>
        <div class="algo-phase-body" style="line-height:2.1;">
        <b style="color:#CBD5E1;font-size:0.95rem">5 Fase CRISP-DM — v3 Final (7 Variabel Lengkap):</b><br><br>
        <b style="color:#60A5FA">Fase 1 · Business Understanding</b>
        → Segmentasi 7 variabel. Kriteria Silhouette ≥ 0.4.<br>
        <b style="color:#34D399">Fase 2 · Data Collection & Understanding</b>
        → EDA, distribusi, missing values, validasi 7 kolom.<br>
        <b style="color:#FCD34D">Fase 3 · Data Preparation (v3)</b>
        → <code>dt.hour</code> · category OHE({len(fg.get('kategori',[]))}) · payment OHE({len(fg.get('payment',[]))}) · device OHE({len(fg.get('device',[]))}) · StandardScaler → {n_tot} fitur<br>
        <b style="color:#C4B5FD">Fase 4 · Modeling</b>
        → Elbow WCSS · Silhouette np.argmax → optimal_k={optimal_k} · KMeans.fit_predict() · PCA 2D<br>
        <b style="color:#F87171">Fase 5 · Evaluation</b>
        → groupby().agg() · persona (level×reward×sesi) · rekomendasi per cluster<br><br>
        <b style="color:#94A3B8">Hasil:</b>
        {optimal_k} cluster | Silhouette={best_sil:.4f} {'✓≥0.4' if best_sil>=0.4 else '⚠<0.4'} |
        {km_log['n_iterations']} iterasi | {n_tot} fitur (OHE+StandardScaler) |
        PCA {sum(var_ratio)*100:.1f}% variansi
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Analisis Ulang dengan File Baru", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()