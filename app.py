import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes AI", layout="wide")

# Hide default Streamlit UI
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.hero-title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
}

.hero-sub {
    font-size: 22px;
    text-align: center;
    color: #cccccc;
}

.section-title {
    font-size: 35px;
    font-weight: bold;
    margin-top: 40px;
}

.card {
    background-color: #1e2a38;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown('<div class="hero-title">🩺 AI Powered Diabetes Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Smart Risk Assessment Using Machine Learning</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------- ABOUT SECTION ----------------
st.markdown('<div class="section-title">📌 About This System</div>', unsafe_allow_html=True)

st.write("""
This platform uses a trained Machine Learning model to predict whether a patient is at risk of diabetes 
based on medical parameters such as glucose level, BMI, insulin, and age.

The model is trained using Logistic Regression with feature scaling for improved accuracy.
""")

# ---------------- HOW IT WORKS ----------------
st.markdown('<div class="section-title">⚙ How It Works</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Enter Data")
    st.write("Provide patient medical parameters.")

with col2:
    st.markdown("### 2️⃣ AI Processing")
    st.write("Model scales and analyzes data.")

with col3:
    st.markdown("### 3️⃣ Get Prediction")
    st.write("Instant diabetes risk result.")

# ---------------- PREDICTION SECTION ----------------
st.markdown('<div class="section-title">🔍 Risk Prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

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

if st.button("🚀 Analyze Risk"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Diabetes (Confidence: {round(probability[0][1]*100,2)}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes (Confidence: {round(probability[0][0]*100,2)}%)")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("")
st.markdown("---")
st.markdown("© 2026 Diabetes AI System | Developed for Academic Project")
