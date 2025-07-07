import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('svm_model_depression.pkl')
scaler = joblib.load('scaler_depression.pkl')

st.title("Prediksi Risiko Depresi Mahasiswa")

with st.form("form_depresi"):
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.slider("Usia", 15, 40, 20)
    academic_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
    work_pressure = st.slider("Tekanan Pekerjaan (1-5)", 0, 5, 0)
    cgpa = st.number_input("Nilai IPK / CGPA", 0.0, 10.0, 7.0)
    study_satisfaction = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    job_satisfaction = st.slider("Kepuasan Kerja (1-5)", 0, 5, 0)
    sleep_duration = st.selectbox("Durasi Tidur", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Kebiasaan Makan", ["Healthy", "Moderate", "Unhealthy"])
    suicidal_thoughts = st.selectbox("Pernah Memiliki Pikiran Bunuh Diri?", ["Yes", "No"])
    work_study_hours = st.slider("Jam Kerja/Belajar per Hari", 0, 16, 4)
    financial_stress = st.slider("Stres Finansial (1-5)", 0, 5, 2)
    family_history = st.selectbox("Riwayat Gangguan Mental dalam Keluarga?", ["Yes", "No"])

    submit = st.form_submit_button("Prediksi")

if submit:
    # Label Encoding manual (pastikan mapping ini sesuai hasil LabelEncoder kamu)
    gender = 1 if gender == "Female" else 0
    sleep_map = {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }
    sleep_duration = sleep_map[sleep_duration]

    diet_map = {
        "Healthy": 0,
        "Moderate": 1,
        "Unhealthy": 2
    }
    dietary_habits = diet_map[dietary_habits]

    suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0

    # Susun input sesuai urutan training
    input_data = np.array([[gender, age, academic_pressure, work_pressure, cgpa,
                            study_satisfaction, job_satisfaction, sleep_duration,
                            dietary_habits, suicidal_thoughts, work_study_hours,
                            financial_stress, family_history]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Hasil: Terindikasi Risiko Depresi")
    else:
        st.success("Hasil: Tidak Terindikasi Risiko Depresi")
