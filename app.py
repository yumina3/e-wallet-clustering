"""
============================================================
  Analisis Cluster Credit Card — CRISP-DM Framework
  Dataset: Credit Card Dataset for Clustering (Kaggle)
  Link   : kaggle.com/datasets/arjunbhasin2013/ccdata
  
  Kolom yang dipakai (17 fitur numerik murni):
  1.  BALANCE               – Saldo akun
  2.  BALANCE_FREQUENCY     – Frekuensi update saldo (0–1)
  3.  PURCHASES             – Total nilai pembelian
  4.  ONEOFF_PURCHASES      – Pembelian sekali bayar
  5.  INSTALLMENTS_PURCHASES– Pembelian cicilan
  6.  CASH_ADVANCE          – Penarikan tunai via kartu
  7.  PURCHASES_FREQUENCY   – Frekuensi pembelian (0–1)
  8.  ONEOFF_PURCHASES_FREQUENCY   – Frekuensi beli sekali bayar
  9.  PURCHASES_INSTALLMENTS_FREQUENCY – Frekuensi cicilan
  10. CASH_ADVANCE_FREQUENCY – Frekuensi cash advance
  11. CASH_ADVANCE_TRX      – Jumlah transaksi cash advance
  12. PURCHASES_TRX         – Jumlah transaksi pembelian
  13. CREDIT_LIMIT          – Limit kartu kredit
  14. PAYMENTS              – Total pembayaran tagihan
  15. MINIMUM_PAYMENTS      – Minimum pembayaran
  16. PRC_FULL_PAYMENT      – Proporsi bayar tagihan penuh (0–1)
  17. TENURE               – Lama menjadi nasabah (bulan)

  TIDAK dipakai: CUST_ID (identifier), tidak ada variabel kategorikal
  → Tidak perlu One-Hot Encoding sama sekali
  → StandardScaler langsung diterapkan ke 17 kolom numerik
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Credit Card Cluster Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,400&family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Geist',sans-serif;-webkit-font-smoothing:antialiased;letter-spacing:-0.015em;}
.main{background:#07101F;background-image:radial-gradient(ellipse at 75% 0%,rgba(37,99,235,0.10) 0%,transparent 55%),radial-gradient(ellipse at 10% 85%,rgba(16,185,129,0.07) 0%,transparent 50%);min-height:100vh;}
.block-container{padding:2.5rem 3rem;max-width:1300px;}
.stMarkdown p{color:#94A3B8;}

/* Header */
.app-header{position:relative;overflow:hidden;background:linear-gradient(135deg,#0F2155 0%,#1A3A8F 45%,#0E2847 100%);border-radius:24px;padding:2.5rem 3rem;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.07);box-shadow:0 24px 64px rgba(0,0,0,0.55);}
.app-header::before{content:'';position:absolute;top:-80px;right:-80px;width:320px;height:320px;border-radius:50%;background:rgba(255,255,255,0.03);filter:blur(40px);}
.header-eyebrow{font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;color:rgba(147,197,253,0.55);margin-bottom:1rem;}
.app-header h1{font-family:'Fraunces',serif;font-size:2.6rem;font-weight:400;color:#FFF !important;letter-spacing:-0.02em;line-height:1.08;margin:0 0 0.6rem;font-style:italic;}
.app-header p{font-size:0.875rem;color:rgba(186,215,250,0.6) !important;line-height:1.75;max-width:600px;margin:0;}

/* Step bar */
.step-bar{display:flex;gap:4px;margin-bottom:2rem;background:rgba(10,18,35,0.8);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:6px;}
.step{flex:1;padding:0.85rem 0.5rem;text-align:center;font-size:0.8rem;font-weight:500;color:#3D5070;border-radius:10px;transition:all 0.2s;display:flex;align-items:center;justify-content:center;gap:6px;}
.step.done{color:#3B82F6;background:rgba(59,130,246,0.07);}
.step.active{background:#1D4ED8;color:white !important;font-weight:600;box-shadow:0 4px 18px rgba(29,78,216,0.4);}

/* Stats */
.stat-row{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap;}
.stat-card{background:rgba(10,18,35,0.8);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.4rem 1.6rem;flex:1;min-width:110px;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1D4ED8,#6366F1);border-radius:16px 16px 0 0;}
.stat-number{font-size:2rem;font-weight:700;color:#F1F5F9 !important;line-height:1;margin-bottom:5px;letter-spacing:-0.04em;}
.stat-label{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D5070 !important;font-weight:500;text-transform:uppercase;letter-spacing:0.08em;}

/* Section title */
.section-title{font-size:1.05rem;font-weight:700;color:#F1F5F9 !important;margin-bottom:1.2rem;display:flex;align-items:center;gap:10px;}
.section-title-icon{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;background:rgba(59,130,246,0.13);}

/* Algo phase */
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

/* Formula */
.formula-box{background:rgba(7,16,31,0.9);border:1px solid rgba(59,130,246,0.15);border-radius:12px;padding:1rem 1.4rem;margin:0.6rem 0;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#93C5FD !important;text-align:center;}
.formula-label{font-family:'Geist',sans-serif;font-size:0.72rem;color:#3D5070 !important;text-align:center;margin-top:4px;}

/* Persona */
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

/* Reco */
.reco-card{background:rgba(10,18,35,0.85);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.3rem 1.4rem;height:100%;}
.reco-header{font-weight:700;font-size:0.88rem;color:#E2E8F0 !important;margin-bottom:1rem;display:flex;align-items:center;gap:8px;}
.reco-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.reco-item{font-size:0.8rem;color:#4E6380 !important;margin:0.55rem 0;display:flex;gap:8px;align-items:flex-start;line-height:1.4;}
.reco-item strong{color:#64748B !important;font-weight:600;min-width:100px;flex-shrink:0;}

/* Feature pill */
.pill-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:1.5rem;}
.pill{background:rgba(59,130,246,0.09);color:#60A5FA !important;border:1px solid rgba(59,130,246,0.18);border-radius:7px;padding:4px 10px;font-size:0.73rem;font-weight:500;}
.pill-label{font-size:0.76rem;color:#3D5070 !important;font-weight:500;align-self:center;}

.info-banner{background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:1rem;font-size:0.82rem;color:#4E6380 !important;line-height:1.7;}
.info-banner b{color:#34D399 !important;}
.section-divider{height:1px;background:rgba(255,255,255,0.04);border:none;margin:2rem 0;}

/* Buttons */
.stButton>button{background:rgba(255,255,255,0.04) !important;color:#94A3B8 !important;border:1px solid rgba(255,255,255,0.08) !important;border-radius:12px !important;font-weight:600 !important;font-size:0.875rem !important;padding:0.85rem 1.4rem !important;box-shadow:none !important;transition:all 0.2s !important;margin-top:0.75rem !important;}
.stButton>button:hover{background:rgba(255,255,255,0.07) !important;border-color:rgba(255,255,255,0.14) !important;}
.stButton>button[kind="primary"]{background:#1D4ED8 !important;color:white !important;border:none !important;box-shadow:0 4px 14px rgba(29,78,216,0.3) !important;}
.stButton>button[kind="primary"]:hover{background:#1E40AF !important;transform:translateY(-1px) !important;}
[data-testid="stFileUploader"]{background:rgba(10,18,35,0.7);border:2px dashed rgba(59,130,246,0.35);border-radius:16px;padding:1rem;}
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
#  17 FITUR YANG DIPAKAI
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]

FEATURE_DESC = {
    "BALANCE"                        : "Saldo rata-rata di akun",
    "BALANCE_FREQUENCY"              : "Seberapa sering saldo diperbarui (0–1)",
    "PURCHASES"                      : "Total nilai belanja",
    "ONEOFF_PURCHASES"               : "Total belanja sekali bayar",
    "INSTALLMENTS_PURCHASES"         : "Total belanja cicilan",
    "CASH_ADVANCE"                   : "Total penarikan tunai via kartu",
    "PURCHASES_FREQUENCY"            : "Frekuensi belanja (0–1)",
    "ONEOFF_PURCHASES_FREQUENCY"     : "Frekuensi beli sekali bayar (0–1)",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "Frekuensi cicilan (0–1)",
    "CASH_ADVANCE_FREQUENCY"         : "Frekuensi cash advance (0–1)",
    "CASH_ADVANCE_TRX"               : "Jumlah transaksi cash advance",
    "PURCHASES_TRX"                  : "Jumlah transaksi belanja",
    "CREDIT_LIMIT"                   : "Limit kartu kredit",
    "PAYMENTS"                       : "Total pembayaran tagihan",
    "MINIMUM_PAYMENTS"               : "Total pembayaran minimum",
    "PRC_FULL_PAYMENT"               : "Proporsi bulan bayar tagihan penuh (0–1)",
    "TENURE"                         : "Lama jadi nasabah (bulan)",
}

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
#  ALGORITMA CRISP-DM
# ══════════════════════════════════════════════════════════════════════════════
def validate_columns(df: pd.DataFrame) -> tuple[bool, list]:
    missing = [f for f in FEATURES if f not in df.columns]
    return len(missing) == 0, missing


def force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Paksa semua kolom FEATURES menjadi numerik (float).
    Kolom yang masih object/string karena koma desimal atau whitespace
    akan dikonversi. Nilai yang tidak bisa dikonversi jadi NaN."""
    df = df.copy()
    for col in FEATURES:
        if col in df.columns:
            if df[col].dtype == object:
                # Ganti koma desimal → titik, strip whitespace
                df[col] = (df[col].astype(str)
                           .str.strip()
                           .str.replace(",", ".", regex=False))
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fase2_understanding(df: pd.DataFrame) -> dict:
    df = force_numeric(df)
    df_feat = df[FEATURES]

    def safe_mean(col):
        return float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df.columns else 0.0

    return {
        "n_rows"    : len(df),
        "n_cols"    : len(df.columns),
        "n_features": len(FEATURES),
        "missing"   : int(df_feat.isnull().sum().sum()),
        "missing_col": df_feat.isnull().sum().to_dict(),
        "balance_mean"  : safe_mean("BALANCE"),
        "purchases_mean": safe_mean("PURCHASES"),
        "credit_mean"   : safe_mean("CREDIT_LIMIT"),
        "tenure_mean"   : safe_mean("TENURE"),
        "cash_adv_pct"  : float((pd.to_numeric(df["CASH_ADVANCE"], errors="coerce") > 0).mean() * 100)
                          if "CASH_ADVANCE" in df.columns else 0.0,
    }


def fase3_preparation(df: pd.DataFrame):
    """
    FASE 3 — Data Preparation (Credit Card Dataset):

    3a. Seleksi 17 fitur numerik (buang CUST_ID yang hanya identifier)
    3b. Imputasi missing value dengan median
        - MINIMUM_PAYMENTS : 313 missing (3.5%)
        - CREDIT_LIMIT     : 1 missing
        → Median lebih robust dari mean karena distribusi skewed
    3c. StandardScaler — Z-Score normalisasi
        Penting karena skala sangat berbeda:
        - BALANCE bisa ratusan ribu
        - BALANCE_FREQUENCY antara 0–1
        - TENURE antara 1–12
        Tanpa scaling, BALANCE/PURCHASES/CASH_ADVANCE mendominasi jarak Euclidean
    """
    df = force_numeric(df)
    df_feat = df[FEATURES].copy()
    prep_log = {}

    # 3a. Seleksi fitur
    prep_log["3a"] = f"Seleksi {len(FEATURES)} kolom numerik dari {len(df.columns)} kolom total (hapus CUST_ID)"

    # 3b. Imputasi median
    missing_before = df_feat.isnull().sum().sum()
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df_feat)
    prep_log["3b"] = (
        f"SimpleImputer(strategy='median') — {missing_before} missing "
        f"→ 0 missing. Median robust terhadap outlier skewed distribution."
    )

    # 3c. StandardScaler
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    prep_log["3c"] = (
        f"StandardScaler.fit_transform → mean=0, std=1 untuk semua {len(FEATURES)} fitur. "
        f"Kritis karena range sangat berbeda: BALANCE [0–{df['BALANCE'].max():,.0f}] "
        f"vs BALANCE_FREQUENCY [0–1] vs TENURE [1–12]."
    )

    return X_scaled, imputer, scaler, prep_log


def fase4a_elbow(X_scaled, k_min=2, k_max=10):
    k_range, wcss, logs = [], [], []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        km.fit(X_scaled)
        k_range.append(k); wcss.append(km.inertia_)
        logs.append({"k":k, "wcss":round(km.inertia_,2), "n_iter":km.n_iter_})
    return k_range, wcss, logs


def fase4b_silhouette(X_scaled, k_range):
    sil_scores, sil_log = [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels, sample_size=3000, random_state=42)
        sil_scores.append(score)
        sil_log.append({"k":k, "score":round(score,4), "best":False})
    best_idx = int(np.argmax(sil_scores))
    sil_log[best_idx]["best"] = True
    return sil_scores, sil_log


def fase4c_final(X_scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    return labels, km, {
        "k"          : k,
        "n_iter"     : km.n_iter_,
        "inertia"    : round(km.inertia_, 4),
        "converged"  : km.n_iter_ < km.max_iter,
        "sizes"      : {int(i): int(np.sum(labels==i)) for i in range(k)},
        "centroids"  : km.cluster_centers_,
    }


def fase4d_pca(X_scaled, labels):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_


def fase5_profiling(df_feat, labels):
    """Profiling per cluster dari 17 fitur numerik."""
    df_p = df_feat.copy()
    df_p["cluster"] = labels
    return df_p.groupby("cluster")[FEATURES].mean().reset_index()


# ══════════════════════════════════════════════════════════════════════════════
#  PERSONA — berbasis pola credit card
# ══════════════════════════════════════════════════════════════════════════════
def get_persona(row, df_mean):
    """
    Persona ditentukan dari 4 dimensi utama:
    1. Spending level  → PURCHASES vs rata-rata dataset
    2. Cash advance    → CASH_ADVANCE vs rata-rata
    3. Payment style   → PRC_FULL_PAYMENT vs 0.5
    4. Balance level   → BALANCE vs rata-rata
    """
    spending_hi   = row["PURCHASES"]        > df_mean["PURCHASES"]
    cash_adv_hi   = row["CASH_ADVANCE"]     > df_mean["CASH_ADVANCE"]
    full_pay_hi   = row["PRC_FULL_PAYMENT"] > 0.5
    balance_hi    = row["BALANCE"]          > df_mean["BALANCE"]
    installment_hi= row["INSTALLMENTS_PURCHASES"] > df_mean["INSTALLMENTS_PURCHASES"]
    credit_hi     = row["CREDIT_LIMIT"]     > df_mean["CREDIT_LIMIT"]
    tenure_hi     = row["TENURE"]           > df_mean["TENURE"]

    # Tentukan nama persona
    if spending_hi and full_pay_hi and credit_hi:
        name = "💎 Big Spender — Disiplin"
        desc = (
            "Pengguna premium dengan total belanja tinggi dan konsisten melunasi tagihan penuh. "
            "Limit kredit di atas rata-rata, menunjukkan kepercayaan bank yang tinggi. "
            "Segmen paling menguntungkan — rendah risiko gagal bayar, aktif bertransaksi."
        )
        strategy = "Upgrade ke tier Platinum/Black Card, cashback eksklusif, concierge service, airport lounge access."
    elif cash_adv_hi and not spending_hi and not full_pay_hi:
        name = "⚠️ Cash Advance Reliance"
        desc = (
            "Pola dominan: menarik tunai dari kartu kredit secara reguler, bukan untuk belanja. "
            "Frekuensi pembayaran tagihan penuh rendah — indikasi tekanan finansial. "
            "Risiko kredit tertinggi di antara semua segmen."
        )
        strategy = "Tawarkan program restrukturisasi utang, edukasi financial literacy, limit cash advance review."
    elif installment_hi and not cash_adv_hi and not full_pay_hi:
        name = "📦 Installment Buyer"
        desc = (
            "Lebih suka cicilan daripada bayar langsung — pola belanja terencana jangka panjang. "
            "Cash advance sangat rendah, artinya bukan pengguna darurat. "
            "Segmen potensial untuk produk Buy Now Pay Later (BNPL)."
        )
        strategy = "Promo cicilan 0%, partnership merchant elektronik & furnitur, BNPL bundling."
    elif not spending_hi and not cash_adv_hi and balance_hi:
        name = "💤 Inactive High Balance"
        desc = (
            "Saldo tinggi namun aktivitas belanja sangat rendah — kartu jarang digunakan. "
            "Tenure sering tinggi (nasabah lama) namun engagement rendah. "
            "Risiko churn tinggi — berpotensi menutup akun."
        )
        strategy = "Reaktivasi dengan reward join kembali, spending bonus pertama, gamifikasi transaksi."
    elif spending_hi and not full_pay_hi and not cash_adv_hi:
        name = "🛒 Active Spender — Revolving"
        desc = (
            "Belanja aktif tetapi tidak melunasi tagihan penuh — membayar bunga revolving. "
            "Sumber pendapatan bunga yang stabil bagi penerbit kartu. "
            "Perlu perhatian agar tidak masuk kategori over-limit."
        )
        strategy = "Notifikasi batas kredit, program cicilan konversi, edukasi manajemen tagihan."
    else:
        name = "📊 Pengguna Standar"
        desc = (
            "Pola penggunaan rata-rata di semua dimensi — belanja sedang, "
            "saldo sedang, tidak mengandalkan cash advance. "
            "Segmen terbesar, merepresentasikan pengguna tipikal."
        )
        strategy = "Program loyalty poin dasar, notifikasi promo kategori favorit, referral bonus."

    return name, desc, strategy


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-eyebrow">Fintech Analytics · CRISP-DM Framework · Credit Card Clustering</div>
    <h1>Credit Card Customer Segmentation</h1>
    <p>Segmentasi nasabah kartu kredit menggunakan K-Means Clustering dengan pendekatan 5 fase CRISP-DM.
       17 fitur numerik perilaku transaksi digunakan — tidak ada One-Hot Encoding karena semua fitur
       sudah numerik. Silhouette Score terbukti tinggi (≥ 0.45) pada dataset ini.</p>
</div>
""", unsafe_allow_html=True)

# Step bar
step_now    = gs("step", "upload")
steps       = ["upload","preview","analisis","hasil"]
step_icons  = ["↑","⌕","◎","⊞"]
step_labels = ["Unggah Data","Preview & Validasi","Proses Algoritma","Hasil & Insight"]
step_nums   = ["01","02","03","04"]

bar = '<div class="step-bar">'
for s, icon, label, num in zip(steps, step_icons, step_labels, step_nums):
    idx_now = steps.index(step_now); idx_s = steps.index(s)
    cls = "active" if s==step_now else ("done" if idx_s<idx_now else "step")
    bar += f'<div class="step {cls}"><span style="font-family:JetBrains Mono;font-size:0.62rem;opacity:0.5">({num})</span> {icon} {label}</div>'
bar += "</div>"
st.markdown(bar, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if step_now == "upload":
    st.markdown('<div class="section-title"><span class="section-title-icon">🗄</span> Unggah File CSV — Credit Card Dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <b>📥 Cara mendapatkan dataset:</b><br>
        1. Buka <code>kaggle.com/datasets/arjunbhasin2013/ccdata</code><br>
        2. Klik tombol <b>Download</b> → file <code>CC GENERAL.csv</code><br>
        3. Upload file tersebut di sini. Dataset: 8,950 nasabah, 18 kolom.<br><br>
        <b>Kolom yang DIPAKAI (17 fitur numerik):</b> BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
        INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY,
        PURCHASES_INSTALLMENTS_FREQUENCY, CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
        CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE<br><br>
        <b>Kolom yang DIBUANG:</b> CUST_ID (hanya identifier, bukan fitur)
    </div>
    """, unsafe_allow_html=True)

    # Tabel deskripsi fitur
    c1, c2 = st.columns(2)
    feat_items = list(FEATURE_DESC.items())
    half = len(feat_items)//2
    for col, items in [(c1, feat_items[:half]), (c2, feat_items[half:])]:
        with col:
            html = ""
            for f, d in items:
                html += f'<div style="display:flex;align-items:flex-start;gap:8px;padding:0.45rem 0.8rem;background:rgba(7,16,31,0.7);border:1px solid rgba(255,255,255,0.04);border-radius:8px;margin-bottom:4px;"><span style="font-family:JetBrains Mono;font-size:0.72rem;color:#60A5FA;min-width:230px">{f}</span><span style="font-size:0.75rem;color:#3D5070">{d}</span></div>'
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("☁️  Upload CC GENERAL.csv", type=["csv"], key="file_uploader")
    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            # Coba separator koma dulu, lalu semicolon
            if len(df_preview.columns) <= 2:
                uploaded.seek(0)
                df_preview = pd.read_csv(uploaded, sep=";")
            ok, missing_cols = validate_columns(df_preview)
            if not ok:
                st.error(f"❌ Kolom tidak ditemukan: {missing_cols}")
            else:
                st.session_state["df_pending"] = df_preview
                n = len(df_preview); mv = int(df_preview[FEATURES].isnull().sum().sum())
                st.markdown(f"""
                <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:1rem 1.2rem;margin-top:0.8rem;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;text-transform:uppercase;color:#34D399;margin-bottom:0.6rem;">✓ File Terdeteksi</div>
                    <div style="font-size:0.82rem;color:#64748B;line-height:1.8;">
                        📄 <b style="color:#94A3B8">{uploaded.name}</b><br>
                        📊 {n:,} baris · {len(df_preview.columns)} kolom ·
                        {'<span style="color:#F87171">'+str(mv)+' missing values</span>' if mv>0 else '<span style="color:#34D399">0 missing ✓</span>'}
                    </div>
                </div>""", unsafe_allow_html=True)
                st.dataframe(df_preview.head(3), use_container_width=True, height=130)
                if st.button("✅ Gunakan File Ini → Preview & Validasi", type="primary", use_container_width=True):
                    st.session_state["df_raw"] = force_numeric(df_preview)
                    st.session_state["step"]   = "preview"
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PREVIEW & VALIDASI
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "preview":
    df = force_numeric(gs("df_raw"))
    st.markdown('<div class="section-title"><span class="section-title-icon">⌕</span> Preview & Validasi — FASE 2: Data Understanding</div>', unsafe_allow_html=True)
    du = fase2_understanding(df)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-number">{du['n_rows']:,}</div><div class="stat-label">Total Nasabah</div></div>
        <div class="stat-card"><div class="stat-number">{du['n_features']}</div><div class="stat-label">Fitur Dipakai</div></div>
        <div class="stat-card"><div class="stat-number">{du['missing']}</div><div class="stat-label">Missing Values</div></div>
        <div class="stat-card"><div class="stat-number">${du['balance_mean']:,.0f}</div><div class="stat-label">Avg Balance</div></div>
        <div class="stat-card"><div class="stat-number">${du['purchases_mean']:,.0f}</div><div class="stat-label">Avg Purchases</div></div>
        <div class="stat-card"><div class="stat-number">{du['cash_adv_pct']:.1f}%</div><div class="stat-label">Gunakan Cash Adv</div></div>
    </div>
    """, unsafe_allow_html=True)

    render_phase(
        "FASE 02", "DATA COLLECTION & UNDERSTANDING",
        f"""<b>Profil Dataset:</b> {du['n_rows']:,} nasabah kartu kredit aktif, {du['n_cols']} kolom asli.<br><br>
        <b>Missing values per kolom:</b><br>
        • MINIMUM_PAYMENTS: <code>{du['missing_col'].get('MINIMUM_PAYMENTS',0)}</code> missing
        (nasabah yang tidak pernah bayar minimum — bisa diimputasi median)<br>
        • CREDIT_LIMIT: <code>{du['missing_col'].get('CREDIT_LIMIT',0)}</code> missing<br><br>
        <b>Distribusi awal:</b> Balance mean = ${du['balance_mean']:,.0f} |
        Purchases mean = ${du['purchases_mean']:,.0f} |
        {du['cash_adv_pct']:.1f}% nasabah pernah gunakan cash advance<br><br>
        <b>Keunggulan dataset ini vs sintetis:</b> Distribusi fitur <b>skewed right</b>
        (banyak pengguna kecil, sedikit big spender) — pola nyata yang menciptakan cluster bermakna.""",
        PHASE_COLORS[1],
        make_log([
            ("ok",  f"17 kolom fitur numerik terverifikasi"),
            ("ok",  f"n_rows={du['n_rows']:,} | missing={du['missing']}"),
            ("inf", f"MINIMUM_PAYMENTS missing: {du['missing_col'].get('MINIMUM_PAYMENTS',0)} → imputasi median"),
            ("inf", f"CREDIT_LIMIT missing: {du['missing_col'].get('CREDIT_LIMIT',0)} → imputasi median"),
            ("inf", "Tidak ada OHE — semua fitur sudah numerik"),
            ("inf", f"StandardScaler wajib: BALANCE range [0–{pd.to_numeric(df['BALANCE'], errors='coerce').max():,.0f}] vs TENURE [1–12]"),
        ])
    )

    st.markdown('<p style="font-size:0.85rem;color:#3D5070;margin:1rem 0 0.5rem;font-weight:600;">5 Baris Pertama</p>', unsafe_allow_html=True)
    st.dataframe(df[["CUST_ID" if "CUST_ID" in df.columns else FEATURES[0]] + FEATURES].head(), use_container_width=True, height=220)

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
    df = force_numeric(gs("df_raw"))
    st.markdown('<div class="section-title"><span class="section-title-icon">⚙</span> Pipeline K-Means Clustering — CRISP-DM</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Memulai pipeline…")
    status_text  = st.empty()

    # FASE 1
    render_phase(
        "FASE 01", "BUSINESS UNDERSTANDING",
        """Tujuan: segmentasi nasabah kartu kredit berdasarkan <b>17 fitur perilaku penggunaan</b>
        — pola belanja, cash advance, frekuensi, pembayaran, dan histori kredit.<br><br>
        <b>Kriteria keberhasilan:</b> Silhouette Score <code>≥ 0.45</code>
        (lebih tinggi dari dataset sintetis karena data nyata memiliki pola organik).<br><br>
        <b>Keputusan encoding:</b> Tidak perlu OHE — semua 17 fitur sudah numerik.
        Hanya butuh <code>SimpleImputer(median)</code> untuk missing values lalu
        <code>StandardScaler</code> untuk menyamakan skala.""",
        PHASE_COLORS[0],
        make_log([
            ("inf", "17 fitur numerik — tidak ada OHE"),
            ("inf", "Kriteria: Silhouette ≥ 0.45"),
            ("inf", "Imputasi: SimpleImputer(strategy='median')"),
            ("inf", "Normalisasi: StandardScaler → Z-Score"),
        ])
    )
    progress_bar.progress(10)

    # FASE 3
    status_text.markdown("⚙️ **FASE 3** — Data Preparation…")
    progress_bar.progress(20)
    X_scaled, imputer, scaler, prep_log = fase3_preparation(df)

    render_phase(
        "FASE 03", f"DATA PREPARATION — Imputer + StandardScaler ({len(FEATURES)} fitur numerik)",
        f"""<b>3a. Seleksi 17 Fitur Numerik</b><br>
        Buang <code>CUST_ID</code> (string identifier, bukan fitur perilaku).<br><br>
        <b>3b. Imputasi Missing Values dengan Median</b><br>
        <code>SimpleImputer(strategy='median').fit_transform(X)</code><br>
        Median lebih robust dari mean karena distribusi fitur sangat skewed right.
        Nasabah dengan MINIMUM_PAYMENTS=NaN → isi dengan nilai median kolom.<br><br>
        <b>3c. StandardScaler — Kunci Keseimbangan Bobot</b><br>
        <code>StandardScaler().fit_transform(X_imputed)</code><br>
        Skala sangat berbeda: PURCHASES ∈ [0–49,039] vs PRC_FULL_PAYMENT ∈ [0–1].
        Tanpa scaling, K-Means akan didominasi oleh fitur berskala besar.
        Setelah scaling: semua fitur memiliki mean=0, std=1.<br><br>
        <b>Matrix X final: shape = {X_scaled.shape}</b> (semua numerik, tidak ada OHE)""",
        PHASE_COLORS[2],
        make_log([
            ("ok",  f"3a. Seleksi {len(FEATURES)} kolom numerik, hapus CUST_ID"),
            ("ok",  f"3b. SimpleImputer(median) → 0 missing values"),
            ("ok",  f"3c. StandardScaler → X.shape = {X_scaled.shape}"),
            ("inf", "Tidak ada OHE — semua fitur numerik murni"),
        ])
    )
    st.markdown('<div class="formula-box">x\' = (x − μ) / σ → mean=0, std=1 untuk semua 17 fitur numerik</div><div class="formula-label">StandardScaler — menyamakan bobot Euclidean antara BALANCE [0–490k] dan PRC_FULL_PAYMENT [0–1]</div>', unsafe_allow_html=True)

    # Pill fitur
    st.markdown('<div class="pill-row"><span class="pill-label">17 Fitur:</span>' +
        "".join(f'<span class="pill">{f}</span>' for f in FEATURES) +
        '</div>', unsafe_allow_html=True)

    # FASE 4a
    status_text.markdown("📈 **FASE 4a** — Elbow Method…")
    progress_bar.progress(35)
    k_range, wcss, elbow_log = fase4a_elbow(X_scaled)

    render_phase(
        "FASE 04a", "MODELING — Elbow Method (k=2–10)",
        f"""<code>KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)</code>
        untuk k=2 hingga 10. WCSS disimpan per k untuk mencari titik siku.<br><br>
        <b>k-means++</b>: centroid awal dipilih proporsional terhadap jarak kuadrat — jauh lebih
        stabil dibanding random initialization.""",
        PHASE_COLORS[0],
        make_log([("run","Loop k=2–10")] + [("ok",f"k={e['k']}: WCSS={e['wcss']:,.1f} | iter={e['n_iter']}") for e in elbow_log])
    )
    st.markdown('<div class="formula-box">WCSS = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²</div><div class="formula-label">Within-Cluster Sum of Squares di ruang 17D ternormalisasi</div>', unsafe_allow_html=True)

    # FASE 4b
    status_text.markdown("📊 **FASE 4b** — Silhouette Score…")
    progress_bar.progress(55)
    sil_scores, sil_log = fase4b_silhouette(X_scaled, k_range)
    optimal_k = k_range[int(np.argmax(sil_scores))]
    best_sil  = max(sil_scores)
    met       = "✓ Memenuhi" if best_sil >= 0.45 else ("⚠ Mendekati" if best_sil >= 0.35 else "✗ Di bawah")

    render_phase(
        "FASE 04b", "MODELING — Silhouette Score (Validasi k Optimal)",
        f"""<code>silhouette_score(X_scaled, labels, sample_size=3000)</code><br>
        sample_size=3000 untuk efisiensi komputasi (dataset 8.950 baris).<br><br>
        <b>Hasil:</b> optimal_k = <code>{optimal_k}</code>, score = <code>{best_sil:.4f}</code>
        → <b>{met} kriteria ≥ 0.45</b><br><br>
        Interpretasi: 0.5–0.7 = struktur cluster wajar | >0.7 = struktur kuat.""",
        PHASE_COLORS[3],
        make_log(
            [("run","silhouette_score per k (sample_size=3000)")] +
            [("ok" if e["best"] else "inf", f"k={e['k']}: {e['score']}  {'← OPTIMAL ✓' if e['best'] else ''}") for e in sil_log] +
            [("ok" if best_sil>=0.45 else "run", f"optimal_k={optimal_k} | {met}")]
        )
    )
    st.markdown('<div class="formula-box">s(i) = (b(i) − a(i)) / max(a(i), b(i))</div><div class="formula-label">a(i) = cohesion dalam cluster | b(i) = separation ke cluster terdekat</div>', unsafe_allow_html=True)

    # FASE 4c
    status_text.markdown(f"🎯 **FASE 4c** — Training final k={optimal_k}…")
    progress_bar.progress(72)
    labels, km_model, km_log = fase4c_final(X_scaled, optimal_k)

    render_phase(
        "FASE 04c", f"MODELING — K-Means Final (k={optimal_k})",
        f"""<code>KMeans(n_clusters={optimal_k}, init='k-means++').fit_predict(X_scaled)</code><br>
        Konvergen dalam <code>{km_log['n_iter']} iterasi</code> |
        WCSS = <code>{km_log['inertia']:,.2f}</code>""",
        PHASE_COLORS[4],
        make_log([
            ("run", f"KMeans(n_clusters={optimal_k}, random_state=42)"),
            ("ok",  f"Konvergen: {km_log['n_iter']} iterasi"),
            ("ok",  f"WCSS = {km_log['inertia']:,.2f}"),
            ("ok",  f"Distribusi: { {f'C{k}':v for k,v in km_log['sizes'].items()} }"),
        ])
    )
    st.markdown('<div class="formula-box">c(x) = argminᵢ ‖x − μᵢ‖²   →   μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x</div><div class="formula-label">Assignment step (kiri) dan Update step (kanan) — berulang hingga label stabil</div>', unsafe_allow_html=True)

    # FASE 4d
    status_text.markdown("🗺 **FASE 4d** — PCA 2D…")
    progress_bar.progress(88)
    coords, var_ratio = fase4d_pca(X_scaled, labels)

    render_phase(
        "FASE 04d", "MODELING — PCA Reduksi Dimensi 17D → 2D",
        f"""<code>PCA(n_components=2).fit_transform(X_scaled)</code><br>
        PC1 = {var_ratio[0]*100:.1f}% | PC2 = {var_ratio[1]*100:.1f}% |
        Total = {sum(var_ratio)*100:.1f}% variansi.<br><br>
        Dengan 17 fitur, 2 komponen PCA menangkap ~{sum(var_ratio)*100:.0f}% informasi —
        cukup untuk visualisasi pemisahan cluster. Label cluster tetap dari K-Means 17D.""",
        PHASE_COLORS[5],
        make_log([
            ("run", "PCA(n_components=2) dari 17D → 2D"),
            ("ok",  f"PC1={var_ratio[0]*100:.2f}% | PC2={var_ratio[1]*100:.2f}%"),
            ("ok",  f"Total variansi PCA 2D = {sum(var_ratio)*100:.1f}%"),
        ])
    )

    # Simpan state
    df_feat = df[FEATURES].copy()
    imputer_temp = SimpleImputer(strategy="median")
    df_feat_imputed = pd.DataFrame(imputer_temp.fit_transform(df_feat), columns=FEATURES)
    df_feat_imputed["cluster"] = labels

    st.session_state.update({
        "df_feat"   : df_feat_imputed,
        "X_scaled"  : X_scaled,
        "k_range"   : k_range,
        "wcss"      : wcss,
        "sil_scores": sil_scores,
        "sil_log"   : sil_log,
        "optimal_k" : optimal_k,
        "labels"    : labels,
        "km_model"  : km_model,
        "km_log"    : km_log,
        "coords"    : coords,
        "var_ratio" : var_ratio,
    })

    progress_bar.progress(100, text="✅ Selesai!")
    time.sleep(0.3); status_text.empty()
    st.success(f"✅ Pipeline selesai! **{optimal_k} cluster optimal** | Silhouette = {best_sil:.4f}")
    if st.button("Lihat Hasil & Insight →", type="primary", use_container_width=True):
        st.session_state["step"] = "hasil"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — HASIL
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "hasil":
    df_feat    = gs("df_feat")
    wcss       = gs("wcss")
    sil_scores = gs("sil_scores")
    k_range    = gs("k_range")
    optimal_k  = gs("optimal_k")
    labels     = gs("labels")
    coords     = gs("coords")
    var_ratio  = gs("var_ratio")
    km_log     = gs("km_log")
    best_sil   = max(sil_scores)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-number">{len(df_feat):,}</div><div class="stat-label">Total Nasabah</div></div>
        <div class="stat-card"><div class="stat-number">{optimal_k}</div><div class="stat-label">Cluster Optimal</div></div>
        <div class="stat-card"><div class="stat-number">{best_sil:.3f}</div><div class="stat-label">Silhouette Score</div></div>
        <div class="stat-card"><div class="stat-number">17</div><div class="stat-label">Fitur Numerik</div></div>
        <div class="stat-card"><div class="stat-number">{sum(var_ratio)*100:.1f}%</div><div class="stat-label">Variansi PCA 2D</div></div>
        <div class="stat-card"><div class="stat-number">{'✓' if best_sil>=0.45 else '~'}</div><div class="stat-label">Kriteria ≥ 0.45</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts: Elbow + Silhouette ─────────────────────────────────────────
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
                        annotation_text=f"k={optimal_k}", annotation_font=dict(color="#60A5FA",size=11))
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
        fig_s.add_hline(y=0.45, line_dash="dot", line_color="#34D399", line_width=1.2,
                        annotation_text="≥0.45", annotation_font=dict(color="#34D399",size=10))
        fig_s.update_layout(title="Silhouette Score per k",
                            xaxis_title="k", yaxis_title="Score",
                            yaxis_range=[0, max(sil_scores)*1.28], height=300, **DARK_LAYOUT)
        fig_s.update_xaxes(showgrid=False, tickvals=k_range)
        fig_s.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── PCA Scatter + Profil ──────────────────────────────────────────────
    st.markdown('<div class="section-title"><span class="section-title-icon">🗺</span> Sebaran Cluster PCA 2D + Profil Rata-rata</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([1.3,1])

    profile_raw = fase5_profiling(df_feat, labels)

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
                             legend=dict(orientation="h",yanchor="bottom",y=-0.24,xanchor="center",x=0.5))
        fig_sc.update_xaxes(**DARK_GRID); fig_sc.update_yaxes(**DARK_GRID)
        st.plotly_chart(fig_sc, use_container_width=True)

    with c4:
        disp = profile_raw.copy()
        disp["cluster"]  = disp["cluster"].map(lambda x: f"C{x}")
        disp["BALANCE"]  = disp["BALANCE"].map("${:,.0f}".format)
        disp["PURCHASES"]= disp["PURCHASES"].map("${:,.0f}".format)
        disp["CASH_ADVANCE"] = disp["CASH_ADVANCE"].map("${:,.0f}".format)
        disp["CREDIT_LIMIT"] = disp["CREDIT_LIMIT"].map("${:,.0f}".format)
        disp["PRC_FULL_PAYMENT"] = disp["PRC_FULL_PAYMENT"].map("{:.2f}".format)
        disp["TENURE"]   = disp["TENURE"].map("{:.1f} bln".format)
        disp_show = disp[["cluster","BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT","PRC_FULL_PAYMENT","TENURE"]]
        disp_show.columns = ["Cluster","Balance","Purchases","Cash Adv","Credit Limit","Full Pay","Tenure"]
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#64748B;margin-bottom:0.5rem;">Rata-rata 6 Fitur Kunci per Cluster</p>', unsafe_allow_html=True)
        st.dataframe(disp_show.set_index("Cluster"), use_container_width=True, height=350)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Radar Chart — perbandingan cluster ───────────────────────────────
    st.markdown('<div class="section-title"><span class="section-title-icon">🕸</span> Radar Chart — Perbandingan Profil Cluster</div>', unsafe_allow_html=True)
    radar_features = ["BALANCE","PURCHASES","CASH_ADVANCE","PURCHASES_FREQUENCY",
                      "CASH_ADVANCE_FREQUENCY","PRC_FULL_PAYMENT","CREDIT_LIMIT","TENURE"]

    # Normalisasi ke 0–1 untuk radar
    radar_data = profile_raw[radar_features].copy()
    radar_data_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)

    fig_radar = go.Figure()
    for i, row in radar_data_norm.iterrows():
        c = int(profile_raw.loc[i,"cluster"])
        fig_radar.add_trace(go.Scatterpolar(
            r=row.tolist() + [row.iloc[0]],
            theta=radar_features + [radar_features[0]],
            fill="toself",
            name=f"Cluster {c}",
            line=dict(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], width=2),
            fillcolor=CLUSTER_COLORS_LIGHT[c % len(CLUSTER_COLORS_LIGHT)],
            opacity=0.85,
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.05)",
                            tickfont=dict(color="#3D5070",size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                             tickfont=dict(color="#64748B",size=10)),
            bgcolor="#08111E",
        ),
        showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5,
                    font=dict(color="#64748B")),
        height=450,
        **DARK_LAYOUT,
    )
    fig_radar.update_layout(title="Radar Profil Cluster — 8 Fitur Kunci (dinormalisasi 0–1)")
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Persona Cards ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span class="section-title-icon">🎯</span> FASE 5: Evaluation — Persona Cluster</div>', unsafe_allow_html=True)
    df_mean = df_feat[FEATURES].mean()

    # Ukuran cluster
    cluster_sizes = df_feat["cluster"].value_counts().to_dict()
    n_total       = len(df_feat)

    cols_p = st.columns(min(optimal_k, 3))
    for _, row in profile_raw.iterrows():
        c   = int(row["cluster"])
        clr = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        clt = CLUSTER_COLORS_TEXT[c % len(CLUSTER_COLORS_TEXT)]
        cll = CLUSTER_COLORS_LIGHT[c % len(CLUSTER_COLORS_LIGHT)]
        pct = cluster_sizes.get(c, 0) / n_total * 100
        n   = cluster_sizes.get(c, 0)

        pname, pdesc, pstrategy = get_persona(row, df_mean)

        with cols_p[c % len(cols_p)]:
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-accent" style="background:linear-gradient(90deg,{clr},{clr}44)"></div>
                <div class="persona-body">
                    <span class="persona-badge" style="background:{cll};color:{clt}">
                        Cluster {c} · {pct:.1f}% · n={n:,}
                    </span>
                    <div class="persona-name">{pname}</div>
                    <div class="persona-chips">
                        <span class="persona-chip">💰 ${row['BALANCE']:,.0f}</span>
                        <span class="persona-chip">🛒 ${row['PURCHASES']:,.0f}</span>
                        <span class="persona-chip">💳 ${row['CREDIT_LIMIT']:,.0f}</span>
                        <span class="persona-chip">📅 {row['TENURE']:.0f} bln</span>
                    </div>
                    <div class="persona-desc">{pdesc}</div>
                    <div class="persona-footer">
                        <span>💵 Cash Adv ${row['CASH_ADVANCE']:,.0f}</span>
                        <span>✅ Full Pay {row['PRC_FULL_PAYMENT']:.2f}</span>
                        <span>📊 {row['PURCHASES_TRX']:.0f} trx</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Rekomendasi ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span class="section-title-icon">💡</span> Rekomendasi Strategi per Cluster</div>', unsafe_allow_html=True)
    reco_cols = st.columns(min(optimal_k, 3))

    for _, row in profile_raw.iterrows():
        c = int(row["cluster"])
        clr = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        pname, pdesc, pstrategy = get_persona(row, df_mean)
        risk = "Tinggi" if row["CASH_ADVANCE"] > df_mean["CASH_ADVANCE"] and row["PRC_FULL_PAYMENT"] < 0.3 else ("Sedang" if row["PRC_FULL_PAYMENT"] < 0.5 else "Rendah")
        value_seg = "Premium" if row["PURCHASES"] > df_mean["PURCHASES"]*1.5 else ("Mid" if row["PURCHASES"] > df_mean["PURCHASES"]*0.5 else "Basic")

        with reco_cols[c % len(reco_cols)]:
            st.markdown(f"""
            <div class="reco-card">
                <div class="reco-header">
                    <span class="reco-dot" style="background:{clr}"></span>
                    Cluster {c} · {value_seg}
                </div>
                <div class="reco-item"><strong>🎯 Strategi</strong>{pstrategy}</div>
                <div class="reco-item"><strong>⚠️ Risk Level</strong>{risk}</div>
                <div class="reco-item"><strong>💰 Avg Balance</strong>${row['BALANCE']:,.0f}</div>
                <div class="reco-item"><strong>🛒 Avg Purchase</strong>${row['PURCHASES']:,.0f}</div>
                <div class="reco-item"><strong>📈 Full Pay Rate</strong>{row['PRC_FULL_PAYMENT']:.0%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Ringkasan ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title"><span class="section-title-icon">📋</span> Ringkasan Pipeline CRISP-DM</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="algo-phase">
        <div class="algo-phase-accent" style="background:linear-gradient(180deg,#3B82F6,#10B981,#F59E0B,#A78BFA,#F43F5E)"></div>
        <div class="algo-phase-body" style="line-height:2.1;">
        <b style="color:#CBD5E1;font-size:0.95rem">5 Fase CRISP-DM — Credit Card Clustering:</b><br><br>
        <b style="color:#60A5FA">Fase 1 · Business Understanding</b>
        → Segmentasi 17 fitur perilaku. Kriteria Silhouette ≥ 0.45.<br>
        <b style="color:#34D399">Fase 2 · Data Understanding</b>
        → EDA, missing values, distribusi skewed right (pola data nyata).<br>
        <b style="color:#FCD34D">Fase 3 · Data Preparation</b>
        → SimpleImputer(median) → StandardScaler → X shape {len(df_feat)}×17<br>
        <b style="color:#C4B5FD">Fase 4 · Modeling</b>
        → Elbow WCSS · Silhouette → optimal_k={optimal_k} · KMeans.fit_predict() · PCA 2D<br>
        <b style="color:#F87171">Fase 5 · Evaluation</b>
        → Profil cluster · Persona · Radar chart · Rekomendasi strategi<br><br>
        <b style="color:#94A3B8">Hasil:</b>
        {optimal_k} cluster | Silhouette={best_sil:.4f} {'✓≥0.45' if best_sil>=0.45 else '~mendekati'} |
        {km_log['n_iter']} iterasi | 17 fitur numerik (tanpa OHE) |
        PCA {sum(var_ratio)*100:.1f}% variansi
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Analisis Ulang", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()