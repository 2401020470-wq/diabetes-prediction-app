import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes AI System", layout="wide")

# Hide default UI
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.hero {
    text-align:center;
    padding:60px;
}

.hero h1 {
    font-size:60px;
}

.section-title {
    font-size:35px;
    font-weight:bold;
    margin-top:30px;
}

.card {
    background-color:#1e2a38;
    padding:30px;
    border-radius:15px;
    box-shadow:0px 4px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# Navigation
menu = st.sidebar.radio("Navigation", ["Home", "About", "Prediction", "Model Info"])

# ---------------- HOME ----------------
if menu == "Home":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1>🩺 AI Powered Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p>Smart Healthcare Risk Assessment Platform</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("### Why This System?")
    st.write("""
    - Uses Logistic Regression  
    - Feature Scaling Applied  
    - Instant Risk Prediction  
    - Academic ML Deployment Project  
    """)

# ---------------- ABOUT ----------------
elif menu == "About":
    st.markdown('<div class="section-title">About The Project</div>', unsafe_allow_html=True)
    st.write("""
    This system predicts diabetes risk using Machine Learning.
    The model was trained on medical diagnostic data and optimized 
    using Logistic Regression with feature scaling.
    """)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", min_value=0.0)
        glucose = st.number_input("Glucose Level", min_value=0.0)
        bp = st.number_input("Blood Pressure", min_value=0.0)
        skin = st.number_input("Skin Thickness", min_value=0.0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=0.0)

    if st.button("Analyze Risk"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        prob_diabetic = probability[0][1] * 100
        prob_not = probability[0][0] * 100

        if prediction[0] == 1:
            st.error(f"⚠ High Risk of Diabetes ({round(prob_diabetic,2)}%)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({round(prob_not,2)}%)")

        # Probability chart
        fig, ax = plt.subplots()
        ax.bar(["Not Diabetic", "Diabetic"], [prob_not, prob_diabetic])
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if prediction[0]==1 else 'Low Risk'}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {round(max(prob_not, prob_diabetic),2)}%", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as file:
            st.download_button("Download Report", file, "Diabetes_Report.pdf")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------
elif menu == "Model Info":
    st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)
    st.write("""
    Algorithm: Logistic Regression  
    Feature Scaling: StandardScaler  
    Deployment: Streamlit Cloud  
    """)

st.markdown("---")
st.markdown("© 2026 AI Diabetes System | Academic ML Deployment")
