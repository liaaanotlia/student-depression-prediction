import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('svm_model_depression.pkl')
scaler = joblib.load('scaler_depression.pkl')

st.title("Aplikasi Prediksi Risiko Depresi Mahasiswa")

st.markdown("""
Aplikasi ini membantu memprediksi kemungkinan depresi berdasarkan faktor-faktor seperti tekanan akademik, tidur, dan kebiasaan hidup.
Silakan isi formulir berikut:
""")

with st.form("form_depresi"):
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.slider("Usia", 15, 40, 20)
    academic_pressure = st.slider("Tekanan Akademik (1 = rendah, 5 = tinggi)", 1, 5, 3)
    work_pressure = st.slider("Tekanan Pekerjaan (0 = tidak ada, 5 = sangat tinggi)", 0, 5, 0)
    cgpa = st.number_input("Nilai IPK / CGPA", 0.0, 10.0, 7.0)
    study_satisfaction = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    job_satisfaction = st.slider("Kepuasan Pekerjaan (0-5)", 0, 5, 0)
    sleep_duration = st.selectbox("Durasi Tidur", [
        "Kurang dari 5 jam", "5–6 jam", "7–8 jam", "Lebih dari 8 jam"])
    dietary_habits = st.selectbox("Kebiasaan Makan", ["Sehat", "Cukup Sehat", "Tidak Sehat"])
    suicidal_thoughts = st.selectbox("Pernah Memiliki Pikiran Bunuh Diri?", ["Ya", "Tidak"])
    work_study_hours = st.slider("Jam Belajar / Kerja per Hari", 0, 16, 4)
    financial_stress = st.slider("Stres Finansial (0-5)", 0, 5, 2)
    family_history = st.selectbox("Riwayat Gangguan Mental dalam Keluarga?", ["Ya", "Tidak"])

    submit = st.form_submit_button("Lakukan Prediksi")

if submit:
    # Mapping Bahasa Indonesia → Label Model
    gender = "Female" if gender == "Perempuan" else "Male"

    sleep_map = {
        "Kurang dari 5 jam": "Less than 5 hours",
        "5–6 jam": "5-6 hours",
        "7–8 jam": "7-8 hours",
        "Lebih dari 8 jam": "More than 8 hours"
    }
    sleep_duration = sleep_map[sleep_duration]

    diet_map = {
        "Sehat": "Healthy",
        "Cukup Sehat": "Moderate",
        "Tidak Sehat": "Unhealthy"
    }
    dietary_habits = diet_map[dietary_habits]

    suicidal_thoughts = "Yes" if suicidal_thoughts == "Ya" else "No"
    family_history = "Yes" if family_history == "Ya" else "No"

    # Urutan fitur sama seperti saat training
    input_text = {
        "Gender": gender,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history
    }

    # Convert to DataFrame → lalu transform pakai scaler
    import pandas as pd
    input_df = pd.DataFrame([input_text])
    input_encoded = input_df.copy()

    # Apply LabelEncoder ke kolom-kolom tertentu (sama kayak training)
    label_cols = ['Gender', 'Sleep Duration', 'Dietary Habits',
                  'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for col in label_cols:
        le = joblib.load(f'labelencoder_{col}.pkl')  # kamu harus simpan encoder saat training
        input_encoded[col] = le.transform(input_encoded[col])

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Anda terindikasi memiliki risiko depresi. Silakan pertimbangkan untuk berkonsultasi dengan profesional.")
    else:
        st.success("✅ Tidak terindikasi mengalami depresi berdasarkan data yang Anda berikan.")
