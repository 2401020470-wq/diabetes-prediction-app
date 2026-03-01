import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Risk Dashboard", layout="wide")

# ----------- CUSTOM STYLE -----------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #00c6ff;
}
.subtitle {
    font-size: 18px;
    color: #cfcfcf;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e1e2f;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------- HEADER -----------
st.markdown('<div class="big-title">🩺 Diabetes Risk Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Logistic Regression — Real-time Health Risk Prediction</div>', unsafe_allow_html=True)

st.markdown("---")

# ----------- SIDEBAR -----------
st.sidebar.title("📊 Dashboard Info")

st.sidebar.markdown("### 🤖 Model")
st.sidebar.write("Logistic Regression")

st.sidebar.markdown("### 📈 Performance")
st.sidebar.write("Accuracy: 78%")
st.sidebar.write("Precision: 74%")
st.sidebar.write("Recall: 72%")
st.sidebar.write("F1 Score: 73%")

st.sidebar.markdown("### 📁 Dataset")
st.sidebar.write("Total Records: 768")
st.sidebar.write("Diabetic: 268 (34%)")
st.sidebar.write("Non-Diabetic: 500 (66%)")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit & Scikit-learn")

# ----------- INPUT SECTION -----------
st.markdown("## Enter Patient Details")

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

st.markdown("---")

# ----------- PREDICTION -----------
if st.button("🔍 Predict Risk"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1] * 100
    percentage = round(probability, 2)

    st.markdown("## 🧾 Prediction Result")

    if percentage < 30:
        st.success(f"🟢 Low Risk\n\nPercentage: {percentage}%")
    elif percentage < 70:
        st.warning(f"🟡 Medium Risk\n\nPercentage: {percentage}%")
    else:
        st.error(f"🔴 High Risk\n\nPercentage: {percentage}%")
