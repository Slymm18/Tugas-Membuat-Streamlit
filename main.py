import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================
# 1. Page Configuration
# ============================================================
st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Rain Prediction App")
st.write("""
Aplikasi ini memprediksi apakah **akan turun hujan** berdasarkan data cuaca:
- Curah hujan (mm)
- Suhu maksimum (°C)
- Suhu minimum (°C)
- Kecepatan angin (m/s)

Sekarang Anda dapat memilih model yang digunakan!
""")


# ============================================================
# 2. Load Models
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
    st.error("⚠️ File model .pkl tidak ditemukan! Pastikan file ada di repository.")
    st.stop()


# ============================================================
# 3. Sidebar – Model Selection & Accuracy Display
# ============================================================
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Pilih Model untuk Prediksi",
    ["Naive Bayes", "Random Forest", "Ensemble Voting"]
)

st.sidebar.header("📊 Akurasi Model")
st.sidebar.info("Naive Bayes: 0.88")
st.sidebar.info("Random Forest: 0.96")
st.sidebar.success("Ensemble Voting: 0.97")


# ============================================================
# 4. Input Form Section (Use Columns)
# ============================================================
st.subheader("Masukkan Data Cuaca")

col1, col2 = st.columns(2)

with col1:
    precipitation = st.number_input(
        "Curah Hujan (mm)", min_value=0.0, value=0.0, step=0.1
    )
    temp_max = st.number_input(
        "Suhu Maksimum (°C)", min_value=-10.0, value=12.0, step=0.1
    )

with col2:
    temp_min = st.number_input(
        "Suhu Minimum (°C)", min_value=-20.0, value=5.0, step=0.1
    )
    wind = st.number_input(
        "Kecepatan Angin (m/s)", min_value=0.0, value=3.0, step=0.1
    )

input_data = pd.DataFrame(
    [[precipitation, temp_max, temp_min, wind]],
    columns=features
)


# ============================================================
# 5. Predict Button
# ============================================================
st.write("---")

if st.button("🔮 Prediksi Hujan / Tidak"):

    # Pilih model sesuai user
    if model_choice == "Naive Bayes":
        model = model_nb
    elif model_choice == "Random Forest":
        model = model_rf
    else:
        model = model_ensemble

    try:
        pred = model.predict(input_data)[0]
        label = "Akan Turun Hujan" if pred == 1 else "Tidak Akan Hujan"
        icon = "🌧️" if pred == 1 else "☀️"

        st.subheader(f"Hasil Prediksi ({model_choice})")
        st.success(f"{icon} **{label}**")

        # Show (pseudo) probabilities if available
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                st.write("### Probabilitas:")
                st.write(f"- Tidak Hujan: **{proba[0]*100:.2f}%**")
                st.write(f"- Akan Hujan: **{proba[1]*100:.2f}%**")
        except:
            st.info("Model ini tidak mendukung probabilitas.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")


# ============================================================
# 6. Explanation Section
# ============================================================
st.write("---")
st.write("## ℹ️ Tentang Model")
st.write("""
Aplikasi ini menggunakan 3 model Machine Learning:

### 🔹 Gaussian Naive Bayes
Model probabilistik sederhana namun efektif.

### 🔹 Random Forest
Model berbasis banyak decision tree, akurat dan stabil.

### 🔹 Ensemble Voting
Menggabungkan suara dari NB + RF  
→ Paling stabil dan mencapai akurasi tertinggi.

Model dilatih menggunakan **Seattle Weather Dataset**.
""")
