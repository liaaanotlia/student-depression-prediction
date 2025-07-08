import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('svm_model_depression.pkl')
scaler = joblib.load('scaler_depression.pkl')

st.title("Student Depression Risk Prediction")

st.markdown("""
Please fill out the form below according to your condition.
""")

with st.form("depression_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 40, 20)
    academic_pressure = st.slider("Academic Pressure (1 = low, 5 = high)", 1, 5, 3)
    work_pressure = st.slider("Work Pressure (0 = none, 5 = very high)", 0, 5, 0)
    cgpa = st.number_input("GPA / CGPA Score", 0.0, 10.0, 7.0)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
    job_satisfaction = st.slider("Job Satisfaction (0-5)", 0, 5, 0)
    sleep_duration = st.selectbox("Sleep Duration", [
        "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderately Healthy", "Unhealthy"])
    suicidal_thoughts = st.selectbox("Ever Had Suicidal Thoughts?", ["Yes", "No"])
    work_study_hours = st.slider("Work / Study Hours per Day", 0, 16, 4)
    financial_stress = st.slider("Financial Stress Level (0 = not stressed, 5 = very stressed)", 0, 5, 2)
    family_history = st.selectbox("Family History of Mental Disorder?", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

if submit:
    # Map English labels to numerical labels based on LabelEncoder results
    gender_map = {"Male": 0, "Female": 1}
    sleep_map = {
        "5-6 hours": 0,
        "Less than 5 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }
    diet_map = {
        "Healthy": 0,
        "Moderately Healthy": 1,
        "Unhealthy": 2
    }
    binary_map = {"No": 0, "Yes": 1}

    # Encode user input
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

    # Convert to 2D array and scale
    input_array = np.array([encoded_input])
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Display prediction results
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ You are indicated to be at risk of **depression**. It is recommended to speak with a mental health professional.")
    else:
        st.success("✅ You are not indicated to be experiencing depression based on the provided data.")
