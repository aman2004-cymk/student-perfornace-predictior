import streamlit as st
import joblib
import numpy as np

# Load saved models
linear_model = joblib.load("linear_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student's academic and study details below:")

# Input fields
G1 = st.slider("Grade 1 (G1)", 0, 20, 10)
G2 = st.slider("Grade 2 (G2)", 0, 20, 10)
studytime = st.selectbox("Weekly Study Time (hours)", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.slider("Number of Absences", 0, 30, 5)

# Combine into array
input_data = np.array([[G1, G2, studytime, failures, absences]])

# Predict button
if st.button("Predict Performance"):
    pred_grade = linear_model.predict(input_data)[0]
    pred_pass = logistic_model.predict(input_data)[0]

    st.success(f"🎯 Predicted Final Grade (G3): **{pred_grade:.2f}**")
    st.info(f"✅ Prediction: {'Pass' if pred_pass == 1 else 'Fail'}")
