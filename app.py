import streamlit as st
import numpy as np
import joblib
from fpdf import FPDF

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

st.title("🩺 Diabetes Prediction System")
st.write("Machine Learning Based Health Risk Analysis")

# Layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0)
    bp = st.number_input("Blood Pressure", min_value=0.0)
    skin = st.number_input("Skin Thickness", min_value=0.0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

# Prediction
if st.button("🔍 Predict Now"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Correct percentage calculation
    if prediction == 1:
        percentage = round(probability * 100, 2)
    else:
        percentage = round((1 - probability) * 100, 2)

    # Risk category
    if percentage < 40:
        risk = "Low Risk"
        color = "green"
    elif percentage < 70:
        risk = "Moderate Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # Result display
    st.markdown(f"""
        <div style='
            background-color:#111827;
            padding:30px;
            border-radius:15px;
            border-left:8px solid {color};
            text-align:center;
            margin-top:20px'>
            <h2 style='color:{color};'>{risk}</h2>
            <h3>Percentage: {percentage:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- PDF REPORT --------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Diabetes Risk Assessment Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Risk Level: {risk}", ln=True)
    pdf.cell(200, 10, txt=f"Percentage: {percentage:.2f}%", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt="Input Details:", ln=True)
    pdf.cell(200, 10, txt=f"Pregnancies: {preg}", ln=True)
    pdf.cell(200, 10, txt=f"Glucose: {glucose}", ln=True)
    pdf.cell(200, 10, txt=f"Blood Pressure: {bp}", ln=True)
    pdf.cell(200, 10, txt=f"Skin Thickness: {skin}", ln=True)
    pdf.cell(200, 10, txt=f"Insulin: {insulin}", ln=True)
    pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=True)
    pdf.cell(200, 10, txt=f"DPF: {dpf}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as f:
        st.download_button(
            "📄 Download PDF Report",
            f,
            file_name="Diabetes_Report.pdf"
        )
