import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# KONFIG HALAMAN
# -------------------------------
st.set_page_config(
    page_title="AI Prediksi Gaji IT Indonesia - Edisi Profesional",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# THEME / STYLE
# -------------------------------
st.markdown(
    """
<style>
.main { background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); }
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white; font-weight: bold; border-radius: 10px;
    padding: 15px; border: none; font-size: 16px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px; border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}
h1, h2, h3 { color: #667eea; font-weight: 700; }
.info-box {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    padding: 15px; border-radius: 5px; margin: 10px 0;
}
.warning-box {
    background: rgba(255, 167, 38, 0.1);
    border-left: 4px solid #ffa726;
    padding: 15px; border-radius: 5px; margin: 10px 0;
}
.success-box {
    background: rgba(76, 175, 80, 0.1);
    border-left: 4px solid #4caf50;
    padding: 15px; border-radius: 5px; margin: 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------
# SESSION STATE
# -------------------------------
def init_session_state():
    defaults = {
        "riwayat_prediksi": [],
        "feedback_diberikan": False,
        "ada_prediksi": False,
        "prediksi": 0,
        "parameter_user": {},
        "analisis": {
            "persentil": 0,
            "ci": {"bawah": 0, "atas": 0, "margin": 0},
            "jalur_karir": [],
            "gaji_disesuaikan": 0,
            "indeks_col": 100,
            "pengali": {"role": 1.0, "lokasi": 1.0, "tech": 1.0},
            "rekomendasi_skill": [],
            "metrik": {"mae": 0, "rmse": 0, "r2": 0, "akurasi": 0},
        },
        "waktu_prediksi_terakhir": None,
        "cache_df": None,
        "cache_df_name": None,
        "mode_model": "Random Forest",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# -------------------------------
# KONSTANTA
# -------------------------------
REQUIRED_COLS = {"Pengalaman", "Bahasa", "Lokasi", "Role", "Gaji"}

DAFTAR_ROLE = [
    "Backend Developer","Frontend Developer","Full Stack Developer","Data Scientist","Data Engineer",
    "Data Analyst","Machine Learning Engineer","AI Engineer","Deep Learning Engineer","DevOps Engineer",
    "Cloud Engineer","Site Reliability Engineer","System Administrator","Database Administrator",
    "Mobile Developer","iOS Developer","Android Developer","QA Engineer","Test Automation Engineer",
    "QA Lead","Security Engineer","Cybersecurity Analyst","Penetration Tester","Network Engineer",
    "Network Administrator","UI/UX Designer","Product Designer","Graphic Designer","Product Manager",
    "Project Manager","Scrum Master","Software Architect","Solutions Architect","Enterprise Architect",
    "Technical Lead","Engineering Manager","CTO",
]

DAFTAR_TECH_STACK = [
    "Python","Java","JavaScript","TypeScript","Go","Rust","PHP","Ruby","C++","C#",
    "Swift","Kotlin","Scala","React","Vue.js","Angular","Svelte","Next.js","Node.js",
    "Django","Flask","FastAPI","Spring Boot","Laravel",".NET","Express.js","NestJS",
    "TensorFlow","PyTorch","Scikit-learn","AWS","Google Cloud","Azure","Docker","Kubernetes",
]

KOTA_INDONESIA = [
    "Jakarta","Bandung","Surabaya","Semarang","Yogyakarta","Tangerang",
    "Tangerang Selatan","Bekasi","Depok","Bogor","Malang","Denpasar","Bali",
    "Mataram","Lombok","Sumbawa","Kupang","Medan","Palembang","Pekanbaru",
    "Padang","Bandar Lampung","Batam","Jambi","Bengkulu","Balikpapan","Samarinda",
    "Pontianak","Banjarmasin","Palangkaraya","Makassar","Manado","Palu","Kendari",
    "Gorontalo","Jayapura","Ambon","Sorong","Ternate","Remote","Hybrid",
]

PENGALI_ROLE = {
    "Backend Developer": 1.0, "Frontend Developer": 0.95, "Full Stack Developer": 1.15,
    "Data Scientist": 1.4, "Data Engineer": 1.3, "Data Analyst": 1.05,
    "Machine Learning Engineer": 1.45, "AI Engineer": 1.5, "Deep Learning Engineer": 1.55,
    "DevOps Engineer": 1.25, "Cloud Engineer": 1.3, "Site Reliability Engineer": 1.35,
    "System Administrator": 0.9, "Database Administrator": 1.1,
    "Mobile Developer": 1.1, "iOS Developer": 1.2, "Android Developer": 1.15,
    "QA Engineer": 0.85, "Test Automation Engineer": 1.0, "QA Lead": 1.2,
    "Security Engineer": 1.35, "Cybersecurity Analyst": 1.3, "Penetration Tester": 1.4,
    "Network Engineer": 1.0, "Network Administrator": 0.95,
    "UI/UX Designer": 1.0, "Product Designer": 1.1, "Graphic Designer": 0.85,
    "Product Manager": 1.4, "Project Manager": 1.25, "Scrum Master": 1.2,
    "Software Architect": 1.6, "Solutions Architect": 1.55, "Enterprise Architect": 1.65,
    "Technical Lead": 1.45, "Engineering Manager": 1.5, "CTO": 2.2,
}

PENGALI_LOKASI = {
    "Jakarta": 1.3, "Bandung": 1.05, "Surabaya": 1.15, "Denpasar": 1.1, "Bali": 1.1,
    "Tangerang": 1.2, "Tangerang Selatan": 1.25, "Bekasi": 1.1, "Depok": 1.05, "Bogor": 1.0,
    "Semarang": 0.95, "Yogyakarta": 0.95, "Malang": 0.9, "Medan": 0.95, "Makassar": 0.9,
    "Balikpapan": 1.05, "Batam": 1.1, "Palembang": 0.85, "Mataram": 0.8, "Lombok": 0.78,
    "Manado": 0.85, "Pekanbaru": 0.8, "Padang": 0.75, "Banjarmasin": 0.78, "Pontianak": 0.75,
    "Palu": 0.7, "Kendari": 0.7, "Jayapura": 0.85, "Ambon": 0.75, "Kupang": 0.7,
    "Sumbawa": 0.65, "Ternate": 0.7, "Remote": 1.15, "Hybrid": 1.1,
}

PENGALI_TECH_STACK = {
    "Python": 1.15, "Java": 1.05, "JavaScript": 1.0, "TypeScript": 1.2, "Go": 1.3, "Rust": 1.35,
    "PHP": 0.85, "Ruby": 0.95, "C++": 1.2, "C#": 1.15, "Swift": 1.25, "Kotlin": 1.2,
    "Scala": 1.3, "React": 1.15, "Vue.js": 1.1, "Angular": 1.12, "Svelte": 1.15, "Next.js": 1.2,
    "Node.js": 1.15, "Django": 1.2, "Flask": 1.1, "FastAPI": 1.25, "Spring Boot": 1.18,
    "Laravel": 0.9, ".NET": 1.15, "Express.js": 1.1, "NestJS": 1.2,
    "TensorFlow": 1.3, "PyTorch": 1.35, "Scikit-learn": 1.2,
    "AWS": 1.25, "Google Cloud": 1.2, "Azure": 1.2, "Docker": 1.2, "Kubernetes": 1.25,
}

DAMPAK_SKILL = {
    "Python": {"boost_gaji": 18, "permintaan": "Sangat Tinggi"},
    "Machine Learning": {"boost_gaji": 30, "permintaan": "Sangat Tinggi"},
    "Cloud (AWS/GCP/Azure)": {"boost_gaji": 25, "permintaan": "Sangat Tinggi"},
    "Docker/Kubernetes": {"boost_gaji": 22, "permintaan": "Tinggi"},
    "React/Vue/Angular": {"boost_gaji": 15, "permintaan": "Tinggi"},
    "Data Engineering": {"boost_gaji": 28, "permintaan": "Sangat Tinggi"},
    "Cybersecurity": {"boost_gaji": 32, "permintaan": "Sangat Tinggi"},
    "Blockchain": {"boost_gaji": 35, "permintaan": "Sedang"},
    "Go/Rust": {"boost_gaji": 28, "permintaan": "Tinggi"},
    "Leadership": {"boost_gaji": 25, "permintaan": "Tinggi"},
    "DevOps": {"boost_gaji": 24, "permintaan": "Sangat Tinggi"},
    "AI/Deep Learning": {"boost_gaji": 35, "permintaan": "Sangat Tinggi"},
}


# -------------------------------
# NORMALISASI / AUTOFILL / CLEANING
# -------------------------------
def normalisasi_kolom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # support header bawaan kamu + header aplikasi
    rename_map = {
        # versi aplikasi
        "pengalaman": "Pengalaman", "experience": "Pengalaman", "exp": "Pengalaman",
        "bahasa": "Bahasa", "tech": "Bahasa", "stack": "Bahasa",
        "lokasi": "Lokasi", "location": "Lokasi", "kota": "Lokasi",
        "role": "Role", "posisi": "Role", "jabatan": "Role",
        "gaji": "Gaji", "salary": "Gaji", "income": "Gaji",

        # versi dataset kamu (snake_case)
        "years_experience": "Pengalaman",
        "job_title": "Role",
    }

    lower_to_actual = {c.lower(): c for c in df.columns}
    for low, target in rename_map.items():
        if low in lower_to_actual and target not in df.columns:
            df = df.rename(columns={lower_to_actual[low]: target})
    return df


def lengkapi_kolom_wajib_upload(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    default_cols = {
        "Pengalaman": 0.0,
        "Bahasa": "Python",          # dataset kamu memang tidak punya tech stack [file:1]
        "Lokasi": "Remote",
        "Role": "Backend Developer",
        "Gaji": 5_000_000,
    }
    for col, default_val in default_cols.items():
        if col not in df.columns:
            df[col] = default_val
    return df


def bersihkan_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # rapikan string kosong jadi NaN (lebih konsisten)
    for col in ["Bahasa", "Lokasi", "Role"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "None", "nan", "NaN"]), col] = np.nan

    # convert numeric + drop NaN di target supaya tidak error "Input y contains NaN"
    df["Pengalaman"] = pd.to_numeric(df["Pengalaman"], errors="coerce")
    df["Gaji"] = pd.to_numeric(df["Gaji"], errors="coerce")

    # drop rows: target NaN wajib dibuang
    df = df.dropna(subset=["Gaji"])

    # optional: drop rows yang fiturnya NaN (atau bisa diimpute, tapi versi cepat drop)
    df = df.dropna(subset=["Pengalaman", "Bahasa", "Lokasi", "Role"])

    # guard: pengalaman minimal 0
    df["Pengalaman"] = df["Pengalaman"].clip(lower=0)

    return df


def validasi_dataframe(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"Kolom CSV wajib belum ada: {sorted(list(missing))}")
        st.write("Kolom yang terbaca dari file:", list(df.columns))
        st.info("Header minimal: Pengalaman, Bahasa, Lokasi, Role, Gaji")
        st.stop()

    # validasi tambahan: pastikan target tidak kosong
    if df["Gaji"].isna().any():
        st.error("Kolom Gaji masih mengandung NaN. Pastikan cleaning berjalan.")
        st.stop()

    if len(df) < 50:
        st.warning("Data terlalu sedikit setelah cleaning; hasil model bisa tidak stabil.")


# -------------------------------
# DATA SIMULASI
# -------------------------------
@st.cache_data(ttl=3600)
def dapatkan_data_simulasi():
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame(
        {
            "Pengalaman": np.random.randint(0, 20, n),
            "Bahasa": np.random.choice(DAFTAR_TECH_STACK, n),
            "Lokasi": np.random.choice(KOTA_INDONESIA, n),
            "Role": np.random.choice(DAFTAR_ROLE, n),
        }
    )

    gaji_dasar = 5_000_000
    daftar_gaji = []
    for i in range(n):
        gaji = gaji_dasar + (df["Pengalaman"][i] * 1_800_000)
        pengali_role = PENGALI_ROLE.get(df["Role"][i], 1.0)
        pengali_lok = PENGALI_LOKASI.get(df["Lokasi"][i], 0.8)
        pengali_tech = PENGALI_TECH_STACK.get(df["Bahasa"][i], 1.0)

        pengali_senior = 1.0
        if df["Pengalaman"][i] > 10:
            pengali_senior = 1 + ((df["Pengalaman"][i] - 10) * 0.05)

        gaji_final = gaji * pengali_role * pengali_lok * pengali_tech * pengali_senior
        gaji_final = int(gaji_final + np.random.normal(0, 1_000_000))
        gaji_final = max(gaji_final, 4_500_000)
        daftar_gaji.append(gaji_final)

    df["Gaji"] = daftar_gaji
    return df


# -------------------------------
# HELPER LAIN
# -------------------------------
def hitung_pengalaman_kerja(tanggal_mulai, tanggal_akhir=None):
    if tanggal_akhir is None:
        tanggal_akhir = date.today()

    if isinstance(tanggal_mulai, datetime):
        tanggal_mulai = tanggal_mulai.date()
    if isinstance(tanggal_akhir, datetime):
        tanggal_akhir = tanggal_akhir.date()

    diff = relativedelta(tanggal_akhir, tanggal_mulai)
    tahun = diff.years
    bulan = diff.months
    hari = diff.days
    total_tahun = tahun + (bulan / 12) + (hari / 365)

    return {
        "tahun": tahun,
        "bulan": bulan,
        "hari": hari,
        "total_tahun": round(total_tahun, 2),
        "tampilan": f"{tahun} tahun, {bulan} bulan, {hari} hari",
    }


def hitung_persentil(nilai, data):
    return (data < nilai).sum() / len(data) * 100


def prediksi_jalur_karir(pengalaman_sekarang, gaji_sekarang, role):
    proyeksi = []
    peningkatan_dasar = 0.10
    for tahun in range(1, 6):
        pengalaman = pengalaman_sekarang + tahun
        tingkat_pertumbuhan = peningkatan_dasar * (1 - (pengalaman * 0.008))
        proyeksi_gaji = gaji_sekarang * ((1 + tingkat_pertumbuhan) ** tahun)
        proyeksi.append({"tahun": tahun, "pengalaman": round(pengalaman, 1), "gaji": int(proyeksi_gaji)})
    return proyeksi


def generate_rekomendasi_skill(role, gaji_sekarang, gaji_target):
    rekomendasi = []

    if "Data" in role or "ML" in role or "AI" in role:
        relevan = ["Python", "Machine Learning", "Data Engineering", "Cloud (AWS/GCP/Azure)", "AI/Deep Learning"]
    elif "DevOps" in role or "Cloud" in role:
        relevan = ["Docker/Kubernetes", "Cloud (AWS/GCP/Azure)", "Python", "Go/Rust", "DevOps"]
    elif "Security" in role or "Cybersecurity" in role:
        relevan = ["Cybersecurity", "Cloud (AWS/GCP/Azure)", "Python", "DevOps"]
    elif "Frontend" in role or "UI" in role:
        relevan = ["React/Vue/Angular", "Python", "Cloud (AWS/GCP/Azure)"]
    else:
        relevan = ["Python", "Cloud (AWS/GCP/Azure)", "Docker/Kubernetes", "Leadership", "DevOps"]

    for skill in relevan[:5]:
        if skill in DAMPAK_SKILL:
            dampak = DAMPAK_SKILL[skill]
            rekomendasi.append(
                {
                    "skill": skill,
                    "boost": dampak["boost_gaji"],
                    "permintaan": dampak["permintaan"],
                    "dampak_rp": int(gaji_sekarang * (dampak["boost_gaji"] / 100)),
                }
            )
    return rekomendasi


def hitung_confidence_interval(prediksi, model, X_test, y_test):
    residual = y_test - model.predict(X_test)
    std_residual = np.std(residual)
    margin = 1.44 * std_residual
    return {"bawah": max(prediksi - margin, 0), "atas": prediksi + margin, "margin": margin}


def buat_radar_chart(params, pengali):
    kategori = ["Dampak Role", "Faktor Lokasi", "Tech Stack", "Pengalaman"]
    pengalaman_normal = min(params["Pengalaman"] / 10, 2.0)

    nilai = [pengali["role"], pengali["lokasi"], pengali["tech"], pengalaman_normal]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=nilai,
            theta=kategori,
            fill="toself",
            line_color="#667eea",
            fillcolor="rgba(102, 126, 234, 0.3)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=400,
        title="Analisis Faktor Gaji",
    )
    return fig


def buat_gauge_chart(persentil):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=persentil,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Persentil Gaji Anda", "font": {"color": "white", "size": 20}},
            delta={"reference": 50, "increasing": {"color": "#4caf50"}},
            gauge={
                "axis": {"range": [None, 100], "tickcolor": "white"},
                "bar": {"color": "#667eea"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [
                    {"range": [0, 25], "color": "#ff5252"},
                    {"range": [25, 50], "color": "#ffa726"},
                    {"range": [50, 75], "color": "#66bb6a"},
                    {"range": [75, 100], "color": "#4caf50"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": persentil},
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, height=350)
    return fig


def buat_chart_perbandingan_gaji(df, role):
    df_filtered = df[df["Role"] == role].groupby("Lokasi")["Gaji"].median().reset_index()
    df_filtered = df_filtered.sort_values("Gaji", ascending=False).head(10)

    fig = px.bar(
        df_filtered,
        x="Lokasi",
        y="Gaji",
        title=f"Perbandingan Median Gaji {role} di Berbagai Kota",
        labels={"Gaji": "Median Gaji (Rp)", "Lokasi": "Kota"},
        color="Gaji",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=400,
    )
    return fig


def buat_chart_tren_pengalaman(df):
    df_grouped = df.groupby("Pengalaman")["Gaji"].agg(["mean", "median"]).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_grouped["Pengalaman"],
            y=df_grouped["mean"],
            mode="lines+markers",
            name="Rata-rata",
            line=dict(color="#667eea", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_grouped["Pengalaman"],
            y=df_grouped["median"],
            mode="lines+markers",
            name="Median",
            line=dict(color="#f093fb", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title="Tren Gaji Berdasarkan Pengalaman Kerja",
        xaxis_title="Pengalaman (Tahun)",
        yaxis_title="Gaji (Rp)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=400,
        hovermode="x unified",
    )
    return fig


def build_pipeline_random_forest():
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["Bahasa", "Lokasi", "Role"])],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=25,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# -------------------------------
# UI
# -------------------------------
st.title("💰 AI Prediksi Gaji IT Indonesia - Edisi Profesional")
st.markdown("### 🤖 Prediksi Gaji IT Berbasis Machine Learning")


# -------------------------------
# SIDEBAR: upload langsung diproses
# -------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.caption("🧠 Model ML: Random Forest (fixed)")
    st.markdown("---")

    sumber_data = st.radio("📊 Sumber Data", ["Simulasi (Offline)", "Upload File CSV"], key="sumber_data")

    if sumber_data == "Upload File CSV":
        file_upload = st.file_uploader(
            "Upload file CSV",
            type=["csv"],
            help="Support: (1) format aplikasi: Pengalaman,Bahasa,Lokasi,Role,Gaji; (2) format dataset kamu: job_title,years_experience,location,salary",
            key="uploader_csv",
        )

        if file_upload is not None:
            with st.spinner("Memproses file CSV..."):
                try:
                    df_up = pd.read_csv(file_upload, skipinitialspace=True)
                    df_up = normalisasi_kolom(df_up)
                    df_up = lengkapi_kolom_wajib_upload(df_up)
                    df_up = bersihkan_dataset(df_up)
                    validasi_dataframe(df_up)

                    st.session_state.cache_df = df_up
                    st.session_state.cache_df_name = file_upload.name
                    st.success(f"✅ CSV diproses: {len(df_up)} baris (setelah cleaning)")
                except Exception as e:
                    st.session_state.cache_df = None
                    st.session_state.cache_df_name = None
                    st.error(f"❌ Gagal memproses CSV: {str(e)}")

        if st.session_state.cache_df is not None:
            df = st.session_state.cache_df
            st.caption(f"📌 Dataset aktif: {st.session_state.cache_df_name}")
        else:
            df = dapatkan_data_simulasi()
            st.caption("📌 Dataset aktif: Simulasi (fallback)")
    else:
        st.session_state.cache_df = None
        st.session_state.cache_df_name = None
        df = dapatkan_data_simulasi()
        st.caption("📌 Dataset aktif: Simulasi")

    # Safety (untuk semua mode)
    df = normalisasi_kolom(df)
    df = lengkapi_kolom_wajib_upload(df)
    df = bersihkan_dataset(df)
    validasi_dataframe(df)

    st.markdown("---")
    st.markdown("### 📊 Info Dataset")
    st.info(
        f"""**Total Data:** {len(df):,} record
**Gaji Min:** Rp {df['Gaji'].min():,}
**Gaji Max:** Rp {df['Gaji'].max():,}
**Gaji Rata-rata:** Rp {df['Gaji'].mean():,.0f}"""
    )


tab1, tab2, tab3, tab4 = st.tabs(["Prediksi Gaji", "Analisis Pasar", "Riwayat", "Tips Karir"])


# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    st.markdown("## Prediksi Gaji Anda")

    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox("Posisi Role", DAFTAR_ROLE)
        techstack = st.selectbox("Tech Stack Utama", DAFTAR_TECH_STACK)
        lokasi = st.selectbox("Lokasi Kerja", KOTA_INDONESIA)

    with col2:
        metode_pengalaman = st.radio("Cara Input Pengalaman", ["Input Tahun Langsung", "Dari Tanggal Mulai Kerja"])
        if metode_pengalaman == "Input Tahun Langsung":
            pengalaman = st.number_input("Pengalaman Kerja (Tahun)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
            detail_pengalaman = f"{pengalaman} tahun"
        else:
            tanggal_mulai = st.date_input("Tanggal Mulai Bekerja", value=date.today() - timedelta(days=730), max_value=date.today())
            pengalaman_obj = hitung_pengalaman_kerja(tanggal_mulai)
            pengalaman = pengalaman_obj["total_tahun"]
            detail_pengalaman = pengalaman_obj["tampilan"]
            st.info(f"Pengalaman Kerja: {detail_pengalaman} (~{pengalaman} tahun)")

    st.markdown("---")

    if st.button("Prediksi Gaji Saya", type="primary", use_container_width=True):
        with st.spinner("Sedang menganalisis data dan memprediksi gaji Anda..."):
            try:
                validasi_dataframe(df)

                X = df[["Pengalaman", "Bahasa", "Lokasi", "Role"]]
                y = df["Gaji"]

                pipeline = build_pipeline_random_forest()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                pipeline.fit(X_train, y_train)

                input_data = pd.DataFrame({"Pengalaman": [pengalaman], "Bahasa": [techstack], "Lokasi": [lokasi], "Role": [role]})
                prediksi_final = int(float(pipeline.predict(input_data)[0]))

                y_pred = pipeline.predict(X_test)
                mae = float(mean_absolute_error(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))
                akurasi_persen = float(max(0, (1 - mae / df["Gaji"].mean()) * 100))

                persentil = float(hitung_persentil(prediksi_final, df["Gaji"]))
                ci = hitung_confidence_interval(prediksi_final, pipeline, X_test, y_test)
                jalur_karir = prediksi_jalur_karir(pengalaman, prediksi_final, role)
                rekomendasi_skill = generate_rekomendasi_skill(role, prediksi_final, prediksi_final * 1.3)

                pengali = {
                    "role": PENGALI_ROLE.get(role, 1.0),
                    "lokasi": PENGALI_LOKASI.get(lokasi, 1.0),
                    "tech": PENGALI_TECH_STACK.get(techstack, 1.0),
                }

                st.session_state.ada_prediksi = True
                st.session_state.prediksi = prediksi_final
                st.session_state.parameter_user = {
                    "Role": role,
                    "Pengalaman": float(pengalaman),
                    "DetailPengalaman": detail_pengalaman,
                    "Bahasa": techstack,
                    "Lokasi": lokasi,
                }
                st.session_state.analisis = {
                    "persentil": persentil,
                    "ci": ci,
                    "jalur_karir": jalur_karir,
                    "rekomendasi_skill": rekomendasi_skill,
                    "pengali": pengali,
                    "metrik": {"mae": mae, "rmse": rmse, "r2": r2, "akurasi": akurasi_persen},
                }

                st.session_state.riwayat_prediksi.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "role": role,
                        "lokasi": lokasi,
                        "tech": techstack,
                        "pengalaman": float(pengalaman),
                        "prediksi": int(prediksi_final),
                    }
                )

                st.success("✅ Prediksi berhasil!")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")

    if st.session_state.ada_prediksi:
        st.markdown("---")
        st.markdown("## Hasil Prediksi")

        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Estimasi Gaji", f"Rp {st.session_state.prediksi:,.0f}")
        with colB:
            st.metric("Persentil", f"{st.session_state.analisis['persentil']:.1f}%")
        with colC:
            st.metric("Akurasi Model", f"{st.session_state.analisis['metrik']['akurasi']:.1f}%")
        with colD:
            st.metric("Gaji Tahunan", f"Rp {st.session_state.prediksi*12:,.0f}")

        pctl = st.session_state.analisis["persentil"]
        if pctl >= 75:
            st.markdown(
                f"<div class='success-box'><h4>Selamat! Gaji Anda di Atas Rata-rata</h4>"
                f"<p>Gaji Anda berada di <strong>persentil {pctl:.1f}%</strong>.</p></div>",
                unsafe_allow_html=True,
            )
        elif pctl >= 50:
            st.markdown(
                f"<div class='info-box'><h4>Gaji Anda Kompetitif</h4>"
                f"<p>Gaji Anda berada di <strong>persentil {pctl:.1f}%</strong>.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='warning-box'><h4>Ada Ruang untuk Peningkatan</h4>"
                f"<p>Gaji Anda berada di <strong>persentil {pctl:.1f}%</strong>.</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Range Gaji yang Wajar (85% CI)")
        ci = st.session_state.analisis["ci"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Minimum", f"Rp {int(ci['bawah']):,.0f}")
        with c2:
            st.metric("Prediksi", f"Rp {st.session_state.prediksi:,.0f}")
        with c3:
            st.metric("Maximum", f"Rp {int(ci['atas']):,.0f}")

        colL, colR = st.columns(2)
        with colL:
            st.plotly_chart(buat_radar_chart(st.session_state.parameter_user, st.session_state.analisis["pengali"]),
                            use_container_width=True)
        with colR:
            st.plotly_chart(buat_gauge_chart(st.session_state.analisis["persentil"]), use_container_width=True)

        laporan = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameter": st.session_state.parameter_user,
            "prediksi_gaji": st.session_state.prediksi,
            "analisis": {
                "persentil": st.session_state.analisis["persentil"],
                "confidence_interval": st.session_state.analisis["ci"],
                "metrik_model": st.session_state.analisis["metrik"],
            },
            "proyeksi_karir": st.session_state.analisis["jalur_karir"],
            "rekomendasi_skill": st.session_state.analisis["rekomendasi_skill"],
            "model": "Random Forest",
        }

        st.download_button(
            label="⬇️ Download Laporan JSON",
            data=json.dumps(laporan, indent=2, ensure_ascii=False),
            file_name=f"laporan_prediksi_gaji_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# -------------------------------
# TAB 2
# -------------------------------
with tab2:
    st.markdown("## Analisis Pasar Gaji IT")

    col1, col2 = st.columns(2)
    with col1:
        role_filter = st.selectbox("Pilih Role untuk Analisis", DAFTAR_ROLE, key="role_analisis")
    with col2:
        lokasi_filter = st.selectbox("Pilih Lokasi", KOTA_INDONESIA, key="lokasi_analisis")

    st.markdown("---")
    st.markdown("### Tech Stack dengan Gaji Tertinggi (Median)")

    df_role = df[df["Role"] == role_filter]
    if len(df_role) == 0:
        st.info("Data untuk role ini tidak tersedia pada dataset aktif.")
    else:
        top_tech = df_role.groupby("Bahasa")["Gaji"].median().sort_values(ascending=False).head(10)

        fig_tech = px.bar(
            x=top_tech.values,
            y=top_tech.index,
            orientation="h",
            title=f"Top 10 Tech Stack untuk {role_filter}",
            labels={"x": "Median Gaji (Rp)", "y": "Tech Stack"},
            color=top_tech.values,
            color_continuous_scale="Plasma",
        )
        fig_tech.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500)
        st.plotly_chart(fig_tech, use_container_width=True)

        st.markdown("---")
        st.markdown("### Statistik Deskriptif (Role Terpilih)")
        a, b, c, d = st.columns(4)
        with a:
            st.metric("Gaji Minimum", f"Rp {df_role['Gaji'].min():,.0f}")
        with b:
            st.metric("Gaji Median", f"Rp {df_role['Gaji'].median():,.0f}")
        with c:
            st.metric("Gaji Rata-rata", f"Rp {df_role['Gaji'].mean():,.0f}")
        with d:
            st.metric("Gaji Maximum", f"Rp {df_role['Gaji'].max():,.0f}")

        st.markdown("---")
        st.plotly_chart(buat_chart_perbandingan_gaji(df, role_filter), use_container_width=True)


# -------------------------------
# TAB 3
# -------------------------------
with tab3:
    st.markdown("## Riwayat Prediksi")

    if st.session_state.riwayat_prediksi:
        st.info(f"Total Prediksi: {len(st.session_state.riwayat_prediksi)} kali")

        df_riwayat = pd.DataFrame(st.session_state.riwayat_prediksi)
        df_riwayat["prediksi_format"] = df_riwayat["prediksi"].apply(lambda x: f"Rp {x:,.0f}")

        st.dataframe(
            df_riwayat[["timestamp", "role", "lokasi", "tech", "pengalaman", "prediksi_format"]],
            use_container_width=True,
        )

        st.markdown("---")
        last_role = df_riwayat.iloc[-1]["role"]
        st.plotly_chart(buat_chart_tren_pengalaman(df[df["Role"] == last_role]), use_container_width=True)

        st.markdown("---")
        st.download_button(
            label="⬇️ Download Riwayat CSV",
            data=df_riwayat.to_csv(index=False),
            file_name=f"riwayat_prediksi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Belum ada riwayat prediksi. Silakan lakukan prediksi terlebih dahulu.")


# -------------------------------
# TAB 4
# -------------------------------
with tab4:
    st.markdown("## Tips Meningkatkan Gaji IT")

    tips_data = [
        {"icon": "📚", "judul": "Tingkatkan Skill Teknis", "deskripsi": "Pelajari teknologi trending seperti AI/ML, Cloud, atau Cybersecurity untuk meningkatkan nilai pasar."},
        {"icon": "🌍", "judul": "Pertimbangkan Remote/Hybrid", "deskripsi": "Remote/hybrid sering memberi akses ke pasar lebih luas dengan kompensasi lebih tinggi."},
        {"icon": "🎖️", "judul": "Dapatkan Sertifikasi", "deskripsi": "Sertifikasi AWS/GCP/Azure atau sertifikasi security bisa menaikkan kredibilitas."},
        {"icon": "🧠", "judul": "Networking & Personal Branding", "deskripsi": "Bangun portofolio GitHub, LinkedIn, dan ikut komunitas."},
        {"icon": "🚀", "judul": "Cari Peluang di Startup/Unicorn", "deskripsi": "Startup tertentu menawarkan kompensasi kompetitif plus peluang growth cepat."},
        {"icon": "🎯", "judul": "Spesialisasi Niche Skill", "deskripsi": "Menjadi expert di niche Rust/Go/Kubernetes dapat meningkatkan bargaining power."},
        {"icon": "💼", "judul": "Kuatkan Business Acumen", "deskripsi": "Memahami bisnis/product membantu naik ke role leadership dengan gaji lebih tinggi."},
    ]

    for i in range(0, len(tips_data), 2):
        c1, c2 = st.columns(2)
        with c1:
            tip = tips_data[i]
            st.markdown(
                f"<div class='metric-card'><h3>{tip['icon']} {tip['judul']}</h3><p>{tip['deskripsi']}</p></div>",
                unsafe_allow_html=True,
            )
        with c2:
            if i + 1 < len(tips_data):
                tip = tips_data[i + 1]
                st.markdown(
                    f"<div class='metric-card'><h3>{tip['icon']} {tip['judul']}</h3><p>{tip['deskripsi']}</p></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### Rekomendasi Skill untuk Meningkatkan Gaji")
    if st.session_state.ada_prediksi and st.session_state.analisis["rekomendasi_skill"]:
        for i, rec in enumerate(st.session_state.analisis["rekomendasi_skill"], 1):
            with st.expander(f"{i}. {rec['skill']} — Potensi Peningkatan {rec['boost']}%"):
                a, b, c = st.columns(3)
                with a:
                    st.metric("Boost Gaji", f"{rec['boost']}%")
                with b:
                    st.metric("Dampak Finansial", f"Rp {rec['dampak_rp']:,.0f}")
                with c:
                    st.metric("Permintaan Pasar", rec["permintaan"])
    else:
        st.info("Lakukan prediksi dulu untuk melihat rekomendasi skill yang dipersonalisasi.")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 20px;'>"
    "<p><strong>AI Prediksi Gaji IT Indonesia - Edisi Profesional</strong></p>"
    "<p>Dibuat Oleh Andrian & Sulthan</p>"
    "<p>Data bersifat estimasi dan untuk tujuan referensi</p>"
    "</div>",
    unsafe_allow_html=True,
)
