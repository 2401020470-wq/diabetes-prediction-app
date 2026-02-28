import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        color:#00B4D8;
    }
    .sub-text {
        font-size:18px;
        color:gray;
    }
    .result-box {
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Machine Learning Based Health Risk Analysis</div>', unsafe_allow_html=True)

st.write("")

# Two column layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0.0)
    glucose = st.number_input("Glucose Level", min_value=0.0)
    bp = st.number_input("Blood Pressure", min_value=0.0)
    skin = st.number_input("Skin Thickness", min_value=0.0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0.0)

st.write("")

if st.button("🔍 Predict Now"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.markdown(
            f'<div class="result-box" style="background-color:#ff4b4b;color:white;">⚠ High Risk of Diabetes<br>Confidence: {round(probability[0][1]*100,2)}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background-color:#00c853;color:white;">✅ Low Risk of Diabetes<br>Confidence: {round(probability[0][0]*100,2)}%</div>',
            unsafe_allow_html=True
        )
