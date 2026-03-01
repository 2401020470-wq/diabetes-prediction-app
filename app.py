import streamlit as st
import numpy as np
import joblib
from fpdf import FPDF
from datetime import datetime

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Risk Analyzer", layout="wide")

st.title("🩺 Diabetes Risk Analyzer")
st.write("AI Powered Health Risk Assessment System")

st.markdown("---")

# Input section
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

if st.button("🔍 Predict Risk"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100
    percentage = round(probability, 2)

    st.markdown("## Result")

    # Risk logic
    if percentage < 30:
        risk_level = "Low Risk"
        st.success(f"{risk_level}")
    elif percentage < 70:
        risk_level = "Medium Risk"
        st.warning(f"{risk_level}")
    else:
        risk_level = "High Risk"
        st.error(f"{risk_level}")

    st.markdown(f"### Percentage: {percentage}%")

    st.markdown("---")

    # ========== PROFESSIONAL PDF ==========
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "DIABETES RISK REPORT", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, "Patient Inputs:", ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, f"Pregnancies: {preg}", ln=True)
    pdf.cell(0, 8, f"Glucose Level: {glucose}", ln=True)
    pdf.cell(0, 8, f"Blood Pressure: {bp}", ln=True)
    pdf.cell(0, 8, f"Skin Thickness: {skin}", ln=True)
    pdf.cell(0, 8, f"Insulin Level: {insulin}", ln=True)
    pdf.cell(0, 8, f"BMI: {bmi}", ln=True)
    pdf.cell(0, 8, f"Diabetes Pedigree Function: {dpf}", ln=True)
    pdf.cell(0, 8, f"Age: {age}", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Prediction Result:", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Risk Level: {risk_level}", ln=True)
    pdf.cell(0, 8, f"Percentage: {percentage}%", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 6, "Disclaimer: This prediction is generated using a machine learning model "
                         "and should not replace professional medical consultation.")

    pdf_file = "diabetes_report.pdf"
    pdf.output(pdf_file)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Professional PDF Report",
            data=f,
            file_name="Diabetes_Risk_Report.pdf",
            mime="application/pdf"
        )
