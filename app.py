st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# ----------- WHITE BACKGROUND + DARK TEXT THEME -----------
st.markdown("""
<style>

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #f4f6f9;
}

/* Make all text dark */
html, body, p, label, div {
    color: #111827 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Headings dark */
h1, h2, h3 {
    color: #1f2937 !important;
    font-weight: 700;
}

/* Sidebar text dark */
[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* Button style */
.stButton>button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 15px;
}

.stButton>button:hover {
    background-color: #1e40af;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])
@@ -17,17 +64,18 @@
# Permanent About Section in Sidebar
st.sidebar.subheader("About Project")
st.sidebar.write("""
📊 Dataset: PIMA Indian Diabetes Dataset  
📊 Dataset: PIMA Indian Diabetes Dataset
    Total Records: 768
🤖 Model: Logistic Regression  
🎯 Accuracy: 77.6%  
⚙ Deployment: Streamlit Cloud  
👩‍💻 Developed by: Aliya Afzal
👥 Developed as a Group Project
""")

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.markdown(
        "<h1 style='text-align: center; color: #1e3a8a;'>Welcome to Diabetes Prediction System</h1>",
        "<h1 style='text-align: center;'>Welcome to Diabetes Prediction System</h1>",
        unsafe_allow_html=True
    )

@@ -72,4 +120,4 @@
        else:
            st.success("✅ Low Risk: The patient is Not Diabetic.")

    st.caption("Developed by streanmlit | BTech AI Project")
    st.caption("BTech AI/ML Group Project | Streamlit Deployment")
