import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================
# 1. Konfigurasi Halaman
# ============================================================
st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Rain Prediction App (Ensemble Model)")
st.write("""
Aplikasi ini memprediksi apakah **akan turun hujan** berdasarkan data cuaca:
- Curah hujan
- Suhu maksimum
- Suhu minimum
- Kecepatan angin

Model menggunakan **Ensemble Voting** (Naive Bayes + Random Forest).
""")


# ============================================================
# 2. Load Model
# ============================================================
@st.cache_resource
def load_models():
    try:
        model_nb = joblib.load("model_nb.pkl")
        model_rf = joblib.load("model_rf.pkl")
        model_ensemble = joblib.load("model_ensemble.pkl")
        features = joblib.load("model_features.pkl")
        return model_nb, model_rf, model_ensemble, features

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None


model_nb, model_rf, model_ensemble, features = load_models()

if not all([model_nb, model_rf, model_ensemble, features]):
    st.error("⚠️ File model .pkl tidak ditemukan!\n\nUpload semua file .pkl ke repository Streamlit Anda.")
    st.stop()


# ============================================================
# 3. Sidebar — Model Accuracy (isi sesuai hasil training Anda)
# ============================================================
st.sidebar.header("📊 Model Accuracy")

st.sidebar.success("Naive Bayes: 0.88")
st.sidebar.success("Random Forest: 0.96")
st.sidebar.success("Ensemble Voting: 0.97")


# ============================================================
# 4. Input Form
# ============================================================
st.subheader("Masukkan Data Cuaca")

precipitation = st.number_input("Curah Hujan (mm)", min_value=0.0, value=0.0, step=0.1)
temp_max = st.number_input("Suhu Maksimum (°C)", min_value=-10.0, value=12.0, step=0.1)
temp_min = st.number_input("Suhu Minimum (°C)", min_value=-20.0, value=5.0, step=0.1)
wind = st.number_input("Kecepatan Angin (m/s)", min_value=0.0, value=3.0, step=0.1)

input_data = pd.DataFrame(
    [[precipitation, temp_max, temp_min, wind]],
    columns=features
)


# ============================================================
# 5. Predict Button
# ============================================================
st.write("---")

if st.button("🔮 Prediksi Hujan / Tidak"):
    try:
        pred = model_ensemble.predict(input_data)[0]

        label = "Akan Turun Hujan" if pred == 1 else "Tidak Akan Hujan"
        color = "🌧️" if pred == 1 else "☀️"

        st.subheader("Hasil Prediksi")
        st.success(f"{color} **{label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
