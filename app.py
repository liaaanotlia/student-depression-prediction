import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('svm_model_depression.pkl')
scaler = joblib.load('scaler_depression.pkl')

st.title("Prediksi Risiko Depresi Mahasiswa")

with st.form("form_depresi"):
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.slider("Usia", 15, 40, 20)
    academic_pressure = st.slider("Tekanan Akademik (1 = rendah, 5 = tinggi)", 1, 5, 3)
    work_pressure = st.slider("Tekanan Pekerjaan (0 = tidak ada, 5 = sangat tinggi)", 0, 5, 0)
    cgpa = st.number_input("Nilai IPK / CGPA", 0.0, 10.0, 7.0)
    study_satisfaction = st.slider("Kepuasan dalam Belajar (1-5)", 1, 5, 3)
    job_satisfaction = st.slider("Kepuasan terhadap Pekerjaan (0-5)", 0, 5, 0)
    sleep_duration = st.selectbox("Durasi Tidur", [
        "Kurang dari 5 jam", "5-6 jam", "7-8 jam", "Lebih dari 8 jam"])
    dietary_habits = st.selectbox("Kebiasaan Pola Makan", ["Sehat", "Cukup Sehat", "Tidak Sehat"])
    suicidal_thoughts = st.selectbox("Pernah Memiliki Pikiran Bunuh Diri?", ["Ya", "Tidak"])
    work_study_hours = st.slider("Jam Kerja / Belajar per Hari", 0, 16, 4)
    financial_stress = st.slider("Tingkat Stres Finansial (0 = tidak stres, 5 = sangat stres)", 0, 5, 2)
    family_history = st.selectbox("Ada Riwayat Gangguan Mental dalam Keluarga?", ["Ya", "Tidak"])

    submit = st.form_submit_button("Lakukan Prediksi")

if submit:
    gender_map = {"Laki-laki": 0, "Perempuan": 1}
    sleep_map = {
        "5-6 jam": 0,
        "Kurang dari 5 jam": 1,
        "7-8 jam": 2,
        "Lebih dari 8 jam": 3
    }
    diet_map = {
        "Sehat": 0,
        "Cukup Sehat": 1,
        "Tidak Sehat": 2
    }
    binary_map = {"Tidak": 0, "Ya": 1}

    # Encode input user
    encoded_input = [
        gender_map[gender],
        age,
        academic_pressure,
        work_pressure,
        cgpa,
        study_satisfaction,
        job_satisfaction,
        sleep_map[sleep_duration],
        diet_map[dietary_habits],
        binary_map[suicidal_thoughts],
        work_study_hours,
        financial_stress,
        binary_map[family_history]
    ]

    input_array = np.array([encoded_input])
    input_scaled = scaler.transform(input_array)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Anda terindikasi memiliki risiko **depresi**. Disarankan untuk berbicara dengan profesional kesehatan mental.")
    else:
        st.success("✅ Anda tidak terindikasi mengalami depresi berdasarkan data yang diberikan.")
