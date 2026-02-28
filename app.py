import streamlit as st
import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(layout="wide")

# Hide default Streamlit menu
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
.big-title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #cccccc;
}
.card {
    background-color: #1e2a38;
    padding: 30px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Landing Section
st.markdown('<div class="big-title">Diabetes Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Health Prediction Platform</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# Form Card
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")

with col2:
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

st.write("")

if st.button("Start Prediction"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

st.markdown('</div>', unsafe_allow_html=True)
