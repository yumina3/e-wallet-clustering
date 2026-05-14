import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Cluster E-Wallet",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.main { background: #F8F7FF; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.app-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0; color: white; }
.app-header p { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }

/* Section title */
.section-title {
    font-size: 1.1rem; font-weight: 600;
    color: #1E1B4B; margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #C7D2FE;
    display: flex; align-items: center; gap: 8px;
}

/* Stat cards */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    background: white; border-radius: 12px;
    padding: 1.2rem 1.5rem; flex: 1;
    border: 1px solid #E0E7FF;
    box-shadow: 0 1px 4px rgba(67,56,202,0.06);
}
.stat-number { font-size: 2rem; font-weight: 700; color: #4338CA; }
.stat-label { font-size: 0.8rem; color: #6B7280; margin-top: 2px; font-weight: 500; }

/* Feature pills */
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem; }
.pill {
    background: #EEF2FF; color: #4338CA;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.8rem; font-weight: 500;
}

/* Cluster persona card */
.persona-card {
    background: white; border-radius: 14px;
    border: 1px solid #E0E7FF;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(67,56,202,0.06);
    margin-bottom: 1rem;
}
.persona-accent { height: 5px; }
.persona-body { padding: 1.2rem 1.4rem; }
.persona-badge {
    display: inline-block;
    font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: 0.5rem;
}
.persona-name { font-size: 1rem; font-weight: 700; color: #1E1B4B; margin-bottom: 0.8rem; }
.persona-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 0.8rem; }
.persona-chip {
    background: #F5F3FF; color: #5B21B6;
    border-radius: 8px; padding: 4px 10px;
    font-size: 0.75rem; font-weight: 500;
}
.persona-desc { font-size: 0.82rem; color: #6B7280; line-height: 1.5; }

/* Recommendation */
.reco-card {
    background: white; border-radius: 12px;
    border: 1px solid #E0E7FF; padding: 1.2rem;
    height: 100%;
}
.reco-cluster-dot {
    width: 12px; height: 12px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
}
.reco-item { font-size: 0.82rem; color: #374151; margin: 0.5rem 0; }
.reco-item strong { color: #1E1B4B; }

/* Upload area */
.upload-info {
    background: #EEF2FF; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 1rem;
    border-left: 4px solid #4338CA;
    font-size: 0.85rem; color: #3730A3;
}

/* Steps */
.step-bar {
    display: flex; gap: 0; margin-bottom: 2rem;
    background: white; border-radius: 10px;
    border: 1px solid #E0E7FF; overflow: hidden;
}
.step {
    flex: 1; padding: 0.7rem; text-align: center;
    font-size: 0.78rem; font-weight: 500; color: #9CA3AF;
    border-right: 1px solid #E0E7FF;
}
.step:last-child { border-right: none; }
.step.active { background: #4338CA; color: white; }
.step.done { background: #EEF2FF; color: #4338CA; }

div[data-testid="stFileUploader"] { border: 2px dashed #C7D2FE !important; border-radius: 12px !important; background: #F5F3FF !important; }
</style>
""", unsafe_allow_html=True)

# ── Cluster colors ─────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#4338CA", "#0D9488", "#D97706", "#DB2777"]
CLUSTER_COLORS_LIGHT = ["#EEF2FF", "#CCFBF1", "#FEF3C7", "#FCE7F3"]
CLUSTER_COLORS_TEXT = ["#3730A3", "#0F766E", "#B45309", "#BE185D"]

FITUR_LABELS = [
    "Jam Transaksi", "Kategori Produk", "Nominal Transaksi",
    "Cashback", "Poin Loyalitas", "Metode Pembayaran", "Jenis Perangkat"
]

# ── Helper functions ───────────────────────────────────────────────────────────
def validate_columns(df):
    required = {"transaction_date","product_category","product_amount",
                "cashback","loyalty_points","payment_method","device_type"}
    return required.issubset(set(df.columns))

def preprocess(df):
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True, errors="coerce")
    df["transaction_hour"] = df["transaction_date"].dt.hour

    le = LabelEncoder()
    df["payment_method_enc"] = le.fit_transform(df["payment_method"].astype(str))
    df["device_type_enc"] = le.fit_transform(df["device_type"].astype(str))
    df["product_category_enc"] = le.fit_transform(df["product_category"].astype(str))

    features = ["transaction_hour","product_category_enc","product_amount",
                "cashback","loyalty_points","payment_method_enc","device_type_enc"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, features

def find_optimal_k(X_scaled, k_min=2, k_max=10):
    wcss, sil = [], []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, km.labels_))
    return wcss, sil

def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    return labels

def get_pca(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_

def get_session(key, default=None):
    return st.session_state.get(key, default)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>💳 Analisis Cluster E-Wallet</h1>
    <p>K-Means Clustering · Pendekatan CRISP-DM · Pola Perilaku Transaksi Digital</p>
</div>
""", unsafe_allow_html=True)

# ── Step indicator ─────────────────────────────────────────────────────────────
step_now = get_session("step", "upload")
steps = ["upload","preview","analisis","hasil"]
step_labels = ["① Unggah Data","② Preview","③ Analisis","④ Hasil"]

bar_html = '<div class="step-bar">'
for i, (s, l) in enumerate(zip(steps, step_labels)):
    cls = "active" if s == step_now else ("done" if steps.index(s) < steps.index(step_now) else "step")
    bar_html += f'<div class="step {cls}">{l}</div>'
bar_html += "</div>"
st.markdown(bar_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if step_now == "upload":
    st.markdown('<div class="section-title">📂 Unggah File CSV Transaksi E-Wallet</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-info">
        File CSV harus memiliki 7 kolom: <strong>transaction_date, product_category, 
        product_amount, cashback, loyalty_points, payment_method, device_type</strong>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"], label_visibility="collapsed")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        use_sample = st.button("📎 Gunakan Data Contoh", use_container_width=True)

    if use_sample:
        df = pd.read_csv("/mnt/user-data/uploads/project_Data_ewallet.csv", sep=";")
        st.session_state["df_raw"] = df
        st.session_state["step"] = "preview"
        st.rerun()

    if uploaded:
        try:
            sep = ";" if uploaded.name.endswith(".csv") else ","
            df = pd.read_csv(uploaded, sep=sep)
            if not validate_columns(df):
                st.error("❌ Kolom tidak sesuai. Pastikan CSV memiliki 7 kolom yang benar.")
            else:
                st.session_state["df_raw"] = df
                st.session_state["step"] = "preview"
                st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "preview":
    df = get_session("df_raw")
    st.markdown('<div class="section-title">🔍 Preview & Validasi Data</div>', unsafe_allow_html=True)

    missing = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-number">{len(df):,}</div><div class="stat-label">Total Transaksi</div></div>
        <div class="stat-card"><div class="stat-number">{len(df.columns)}</div><div class="stat-label">Kolom Terdeteksi</div></div>
        <div class="stat-card"><div class="stat-number">{missing}</div><div class="stat-label">Nilai Kosong</div></div>
    </div>
    <div class="pill-row">
        <span style="font-size:0.82rem;color:#6B7280;font-weight:500;align-self:center;">Fitur yang Dianalisis:</span>
        {''.join(f'<span class="pill">{f}</span>' for f in FITUR_LABELS)}
    </div>
    """, unsafe_allow_html=True)

    if missing > 0:
        st.warning(f"⚠️ Terdapat {missing} nilai kosong. Akan diabaikan saat analisis.")

    st.markdown("**5 Baris Pertama Data:**")
    st.dataframe(df.head(), use_container_width=True, height=220)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Ganti File"):
            st.session_state["step"] = "upload"
            st.rerun()
    with col2:
        if st.button("🚀 Mulai Analisis Clustering →", type="primary", use_container_width=True):
            st.session_state["step"] = "analisis"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ANALISIS
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "analisis":
    df = get_session("df_raw")
    st.markdown('<div class="section-title">⚙️ Proses K-Means Clustering</div>', unsafe_allow_html=True)

    k_min, k_max = 2, 10

    with st.spinner("Memproses data dan menjalankan K-Means..."):
        df_proc, X_scaled, features = preprocess(df)
        wcss, sil = find_optimal_k(X_scaled, k_min, k_max)
        k_range = list(range(k_min, k_max+1))
        optimal_k = k_range[np.argmax(sil)]
        labels = run_kmeans(X_scaled, optimal_k)
        coords, var_ratio = get_pca(X_scaled, labels)

        st.session_state.update({
            "df_proc": df_proc,
            "X_scaled": X_scaled,
            "wcss": wcss,
            "sil": sil,
            "k_range": k_range,
            "optimal_k": optimal_k,
            "labels": labels,
            "coords": coords,
            "var_ratio": var_ratio,
            "step": "hasil"
        })

    st.success(f"✅ Analisis selesai! K optimal ditemukan: **{optimal_k} cluster**")
    st.button("Lihat Hasil →", on_click=lambda: st.session_state.update({"step": "hasil"}))
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — HASIL
# ══════════════════════════════════════════════════════════════════════════════
elif step_now == "hasil":
    df_proc  = get_session("df_proc")
    wcss     = get_session("wcss")
    sil      = get_session("sil")
    k_range  = get_session("k_range")
    optimal_k = get_session("optimal_k")
    labels   = get_session("labels")
    coords   = get_session("coords")
    var_ratio = get_session("var_ratio")
    X_scaled = get_session("X_scaled")

    df_proc = df_proc.iloc[:len(labels)].copy()
    df_proc["cluster"] = labels

    # ── Stat summary ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-number">{len(df_proc):,}</div><div class="stat-label">Total Transaksi</div></div>
        <div class="stat-card"><div class="stat-number">{optimal_k}</div><div class="stat-label">Jumlah Cluster Optimal</div></div>
        <div class="stat-card"><div class="stat-number">{max(sil):.3f}</div><div class="stat-label">Silhouette Score Terbaik</div></div>
        <div class="stat-card"><div class="stat-number">{sum(var_ratio)*100:.1f}%</div><div class="stat-label">Variansi PCA (2 komponen)</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Elbow + Silhouette ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Penentuan Jumlah Cluster Optimal</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_range, y=wcss, mode="lines+markers",
            line=dict(color="#4338CA", width=2.5),
            marker=dict(size=8, color="#4338CA"),
            name="WCSS"
        ))
        fig_elbow.add_vline(
            x=optimal_k, line_dash="dash", line_color="#DB2777",
            annotation_text=f"k optimal = {optimal_k}",
            annotation_position="top right",
            annotation_font_color="#DB2777"
        )
        fig_elbow.update_layout(
            title="Metode Elbow", xaxis_title="Jumlah Cluster (k)",
            yaxis_title="WCSS (Inersia)", plot_bgcolor="white",
            paper_bgcolor="white", height=320,
            font=dict(family="Plus Jakarta Sans"),
            margin=dict(t=50, b=40, l=40, r=20)
        )
        fig_elbow.update_xaxes(showgrid=True, gridcolor="#F3F4F6", tickvals=k_range)
        fig_elbow.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        bar_colors = ["#C7D2FE"] * len(k_range)
        bar_colors[np.argmax(sil)] = "#4338CA"
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=k_range, y=[round(s, 3) for s in sil],
            marker_color=bar_colors, name="Silhouette Score",
            text=[f"{s:.3f}" for s in sil], textposition="outside"
        ))
        fig_sil.update_layout(
            title="Silhouette Score per k", xaxis_title="Jumlah Cluster (k)",
            yaxis_title="Score", plot_bgcolor="white",
            paper_bgcolor="white", height=320,
            font=dict(family="Plus Jakarta Sans"),
            margin=dict(t=50, b=40, l=40, r=20),
            yaxis_range=[0, max(sil) * 1.25]
        )
        fig_sil.update_xaxes(showgrid=False, tickvals=k_range)
        fig_sil.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
        st.plotly_chart(fig_sil, use_container_width=True)

    # ── Scatter PCA + Tabel Profil ────────────────────────────────────────────
    st.markdown('<div class="section-title">🗺️ Visualisasi Cluster</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([1.2, 1])

    with c3:
        df_pca = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "Cluster": [f"Cluster {l}" for l in labels]
        })
        fig_scatter = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=CLUSTER_COLORS,
            title=f"Sebaran Cluster (PCA 2D · {var_ratio[0]*100:.1f}% & {var_ratio[1]*100:.1f}% variansi)",
            opacity=0.7, height=380
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Plus Jakarta Sans"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
        fig_scatter.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        st.markdown("**Profil Tiap Cluster**")
        profile = df_proc.groupby("cluster").agg(
            Jumlah=("cluster", "count"),
            Rata2_Nominal=("product_amount", "mean"),
            Rata2_Cashback=("cashback", "mean"),
            Rata2_Poin=("loyalty_points", "mean"),
            Jam_Puncak=("transaction_hour", lambda x: x.mode()[0]),
            Metode_Utama=("payment_method", lambda x: x.mode()[0]),
            Perangkat=("device_type", lambda x: x.mode()[0]),
        ).reset_index()

        profile["Rata2_Nominal"] = profile["Rata2_Nominal"].map("Rp{:,.0f}".format)
        profile["Rata2_Cashback"] = profile["Rata2_Cashback"].map("{:.1f}".format)
        profile["Rata2_Poin"] = profile["Rata2_Poin"].map("{:.0f}".format)
        profile["Jam_Puncak"] = profile["Jam_Puncak"].map(lambda x: f"{x:02d}:00")
        profile["cluster"] = profile["cluster"].map(lambda x: f"Cluster {x}")
        profile.columns = ["Cluster","Jumlah","Nominal","Cashback","Poin","Jam Puncak","Metode","Perangkat"]
        st.dataframe(profile.set_index("Cluster"), use_container_width=True, height=320)

    # ── Persona Cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Persona Tiap Cluster</div>', unsafe_allow_html=True)

    profile_raw = df_proc.groupby("cluster").agg(
        jumlah=("cluster","count"),
        nominal=("product_amount","mean"),
        cashback=("cashback","mean"),
        poin=("loyalty_points","mean"),
        jam=("transaction_hour", lambda x: x.mode()[0]),
        metode=("payment_method", lambda x: x.mode()[0]),
        device=("device_type", lambda x: x.mode()[0]),
        kategori=("product_category", lambda x: x.mode()[0]),
    ).reset_index()

    def get_persona_name(row):
        jam = row["jam"]
        device = row["device"]
        nominal = row["nominal"]
        sesi = "Malam" if jam >= 20 or jam < 5 else ("Pagi" if jam < 12 else ("Siang" if jam < 17 else "Sore"))
        level = "Premium" if nominal > 6000 else ("Menengah" if nominal > 3000 else "Hemat")
        return f"Pengguna {level} {device} — {sesi} Hari"

    def get_persona_desc(row):
        return (f"Bertransaksi dominan pukul {row['jam']:02d}:00 via {row['device']}. "
                f"Metode pembayaran favorit {row['metode']} dengan kategori utama {row['kategori']}.")

    cols = st.columns(min(optimal_k, 4))
    for i, row in profile_raw.iterrows():
        c = int(row["cluster"])
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        color_light = CLUSTER_COLORS_LIGHT[c % len(CLUSTER_COLORS_LIGHT)]
        color_text = CLUSTER_COLORS_TEXT[c % len(CLUSTER_COLORS_TEXT)]
        pct = row["jumlah"] / len(df_proc) * 100

        with cols[c % len(cols)]:
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-accent" style="background:{color}"></div>
                <div class="persona-body">
                    <div>
                        <span class="persona-badge" style="background:{color_light};color:{color_text}">
                            Cluster {c} · {pct:.1f}%
                        </span>
                    </div>
                    <div class="persona-name">{get_persona_name(row)}</div>
                    <div class="persona-chips">
                        <span class="persona-chip">⏰ {row['jam']:02d}:00</span>
                        <span class="persona-chip">📱 {row['device']}</span>
                        <span class="persona-chip">Rp{row['nominal']:,.0f}</span>
                    </div>
                    <div class="persona-desc">{get_persona_desc(row)}</div>
                    <div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #F3F4F6;
                                font-size:0.78rem;color:#6B7280;">
                        💳 {row['metode']} &nbsp;·&nbsp; 🛍️ {row['kategori']}
                        &nbsp;·&nbsp; ⭐ {row['poin']:.0f} poin
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Rekomendasi ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Rekomendasi Strategi Pemasaran</div>', unsafe_allow_html=True)

    reco_cols = st.columns(min(optimal_k, 4))
    for i, row in profile_raw.iterrows():
        c = int(row["cluster"])
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        jam = row["jam"]
        reward = "Cashback" if row["cashback"] > row["poin"] / 10 else "Poin Loyalitas"
        waktu = f"{jam:02d}:00–{(jam+2)%24:02d}:00"

        with reco_cols[c % len(reco_cols)]:
            st.markdown(f"""
            <div class="reco-card">
                <div style="font-weight:600;font-size:0.9rem;color:#1E1B4B;margin-bottom:0.8rem;">
                    <span class="reco-cluster-dot" style="background:{color}"></span>Cluster {c}
                </div>
                <div class="reco-item">⏰ <strong>Waktu Promosi:</strong> {waktu}</div>
                <div class="reco-item">🎁 <strong>Jenis Reward:</strong> {reward}</div>
                <div class="reco-item">📲 <strong>Kanal:</strong> {row['device']} App</div>
                <div class="reco-item">🛒 <strong>Fokus Produk:</strong> {row['kategori']}</div>
                <div class="reco-item">💳 <strong>Metode:</strong> {row['metode']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Analisis Ulang dengan File Baru"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
