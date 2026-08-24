import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioAI",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #fff4f4, #f5f8ff);
    border: 1px solid #e8e8e8;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 19px;
    color: #666666;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 15px;
}

.result-card {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    background-color: white;
    text-align: center;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
}

.card-label {
    font-size: 15px;
    color: #777777;
}

.card-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 7px;
}

.footer {
    text-align: center;
    color: #777777;
    font-size: 13px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)

except Exception as e:
    st.error("❌ Unable to load the machine learning model.")
    st.write(e)
    st.stop()


# ============================================================
# PDF GENERATION FUNCTION
# ============================================================

def generate_pdf(
    age_years,
    gender,
    height,
    weight,
    bmi,
    ap_hi,
    ap_lo,
    cholesterol_text,
    glucose_text,
    smoke_text,
    alcohol_text,
    active_text,
    risk_score,
    result
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    content = []

    # Report title
    content.append(
        Paragraph(
            "CardioAI - Cardiovascular Risk Assessment Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    # Patient information
    content.append(
        Paragraph(
            "Patient Information",
            styles["Heading2"]
        )
    )

    patient_data = [
        ["Parameter", "Value"],
        ["Age", f"{age_years} Years"],
        ["Gender", "Female" if gender == 1 else "Male"],
        ["Height", f"{height:.1f} cm"],
        ["Weight", f"{weight:.1f} kg"],
        ["BMI", f"{bmi:.2f}"],
        ["Systolic Blood Pressure", str(ap_hi)],
        ["Diastolic Blood Pressure", str(ap_lo)],
        ["Cholesterol", cholesterol_text],
        ["Glucose", glucose_text],
        ["Smoking", smoke_text],
        ["Alcohol Consumption", alcohol_text],
        ["Physical Activity", active_text]
    ]

    table = Table(
        patient_data,
        colWidths=[200, 250]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    content.append(table)

    content.append(Spacer(1, 20))

    # Risk result
    content.append(
        Paragraph(
            "Risk Assessment",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Estimated Risk Score:</b> {risk_score:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Prediction Result:</b> {result}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 15))

    # Recommendations
    content.append(
        Paragraph(
            "Health Recommendations",
            styles["Heading2"]
        )
    )

    if result == "High Risk":

        recommendations = [
            "Monitor blood pressure regularly.",
            "Follow a balanced and nutritious diet.",
            "Limit excessive salt and added sugar.",
            "Maintain regular physical activity.",
            "Maintain healthy lifestyle habits.",
            "Discuss health concerns with a qualified healthcare professional."
        ]

    else:

        recommendations = [
            "Continue maintaining healthy lifestyle habits.",
            "Maintain regular physical activity.",
            "Follow a balanced and nutritious diet.",
            "Stay adequately hydrated.",
            "Continue routine health check-ups."
        ]

    for item in recommendations:
        content.append(
            Paragraph(
                f"- {item}",
                styles["Normal"]
            )
        )

    content.append(Spacer(1, 20))

    # Model information
    content.append(
        Paragraph(
            "Model Information",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            "Algorithm: Random Forest Classifier",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            "Dataset Size: 70,000 patient records",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            "Task Type: Binary Classification",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    # Disclaimer
    content.append(
        Paragraph(
            "<b>Disclaimer:</b> CardioAI is an educational "
            "machine-learning project and is not a medical diagnostic "
            "tool. The generated prediction should not replace "
            "professional medical advice.",
            styles["Normal"]
        )
    )

    document.build(content)

    buffer.seek(0)

    return buffer


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("❤️ CardioAI")

    st.caption("AI Healthcare Project")

    st.markdown("---")

    st.subheader("📌 Project Information")

    st.write("**Developer**")
    st.write("Tushita Bhardwaj")

    st.write("**Project**")
    st.write("Cardiovascular Disease Risk Prediction")

    st.write("**Final ML Model**")
    st.write("Random Forest Classifier")

    st.write("**Dataset Size**")
    st.write("70,000 Patient Records")

    st.markdown("---")

    st.subheader("🛠 Technology Stack")

    st.write("• Python")
    st.write("• Pandas")
    st.write("• NumPy")
    st.write("• Scikit-Learn")
    st.write("• Streamlit")
    st.write("• Plotly")
    st.write("• ReportLab")

    st.markdown("---")

    st.info(
        "This application is an educational ML project "
        "and is not a medical diagnostic system."
    )



# ============================================================
# HERO SECTION
# ============================================================

st.title("❤️ CardioAI")

st.subheader(
    "AI-Powered Cardiovascular Disease Risk Assessment System"
)

st.info(
    "This system uses a trained machine learning model to analyze "
    "health and lifestyle parameters and estimate cardiovascular "
    "disease risk."
)

# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    age_years = st.number_input(
        "Age (Years)",
        min_value=18,
        max_value=100,
        value=25
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [1, 2],
        format_func=lambda x:
        "Female" if x == 1 else "Male"
    )


with col3:

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=170.0
    )


with col4:

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0
    )


# Calculate BMI
height_m = height / 100

bmi = weight / (height_m ** 2)


# ============================================================
# MEDICAL PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">🩺 Medical Parameters</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    ap_hi = st.number_input(
        "Systolic BP",
        min_value=70,
        max_value=250,
        value=120
    )


with col2:

    ap_lo = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=150,
        value=80
    )


with col3:

    cholesterol = st.selectbox(
        "Cholesterol Level",
        [1, 2, 3],
        format_func=lambda x:
        "Normal"
        if x == 1
        else "Above Normal"
        if x == 2
        else "Well Above Normal"
    )


with col4:

    gluc = st.selectbox(
        "Glucose Level",
        [1, 2, 3],
        format_func=lambda x:
        "Normal"
        if x == 1
        else "Above Normal"
        if x == 2
        else "Well Above Normal"
    )


# ============================================================
# LIFESTYLE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🏃 Lifestyle Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    smoke = st.selectbox(
        "Smoking",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col2:

    alco = st.selectbox(
        "Alcohol Consumption",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


with col3:

    active = st.selectbox(
        "Physically Active",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


# ============================================================
# CURRENT BMI
# ============================================================

st.markdown("---")

bmi_col1, bmi_col2, bmi_col3 = st.columns([1, 1, 1])

with bmi_col2:

    st.metric(
        "Calculated BMI",
        f"{bmi:.2f}"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

button_columns = st.columns([1, 2, 1])

with button_columns[1]:

    predict = st.button(
        "🔍 Analyze Cardiovascular Risk",
        use_container_width=True
    )


# ============================================================
# EVERYTHING BELOW RUNS ONLY AFTER PREDICT IS CLICKED
# ============================================================

if predict:

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    # Original dataset stores age in days.
    # User enters age in years, so we convert it back to days.
    age = age_years * 365

    input_data = np.array([[
        age,
        gender,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active
    ]])


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    risk_score = probability[0][1] * 100


    if prediction[0] == 1:
        result = "High Risk"

    else:
        result = "Low Risk"



    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.markdown("---")
    st.subheader("📊 Risk Assessment Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            label="BMI",
            value=f"{bmi:.2f}"
        )

    with result_col2:
        st.metric(
            label="Estimated Risk Score",
            value=f"{risk_score:.2f}%"
        )

    with result_col3:
        st.metric(
            label="Prediction",
            value=result
        )
   
       

    # ========================================================
    # CARDIOVASCULAR RISK GAUGE
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Cardiovascular Risk Gauge</div>',
        unsafe_allow_html=True
    )

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=risk_score,

            number={
                "suffix": "%",
                "font": {
                    "size": 38
                }
            },

            title={
                "text": "Estimated Cardiovascular Risk"
            },

            gauge={

                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1
                },

                "bar": {
                    "thickness": 0.30
                },

                "steps": [

                    {
                        "range": [0, 35],
                        "color": "#d8f3dc"
                    },

                    {
                        "range": [35, 65],
                        "color": "#fff3bf"
                    },

                    {
                        "range": [65, 100],
                        "color": "#ffd6d6"
                    }
                ],

                "threshold": {

                    "line": {
                        "color": "#333333",
                        "width": 4
                    },

                    "thickness": 0.75,

                    "value": risk_score
                }
            }
        )
    )


    gauge.update_layout(
        height=340,
        margin=dict(
            l=40,
            r=40,
            t=60,
            b=20
        )
    )


    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 Health Recommendations</div>',
        unsafe_allow_html=True
    )


    if result == "High Risk":

        st.error(
            "⚠️ The model predicts a higher cardiovascular disease risk."
        )

        recommendations = [
            "Monitor blood pressure regularly.",
            "Follow a balanced and nutritious diet.",
            "Limit excessive salt and added sugar.",
            "Maintain regular physical activity.",
            "Maintain healthy lifestyle habits.",
            "Discuss health concerns with a qualified healthcare professional."
        ]


    else:

        st.success(
            "✅ The model predicts a lower cardiovascular disease risk."
        )

        recommendations = [
            "Continue maintaining healthy lifestyle habits.",
            "Maintain regular physical activity.",
            "Follow a balanced and nutritious diet.",
            "Stay adequately hydrated.",
            "Continue routine health check-ups."
        ]


    for recommendation in recommendations:

        st.write(
            f"✅ {recommendation}"
        )


    # ========================================================
    # TEXT VALUES FOR PDF
    # ========================================================

    if cholesterol == 1:
        cholesterol_text = "Normal"

    elif cholesterol == 2:
        cholesterol_text = "Above Normal"

    else:
        cholesterol_text = "Well Above Normal"


    if gluc == 1:
        glucose_text = "Normal"

    elif gluc == 2:
        glucose_text = "Above Normal"

    else:
        glucose_text = "Well Above Normal"


    smoke_text = "Yes" if smoke == 1 else "No"

    alcohol_text = "Yes" if alco == 1 else "No"

    active_text = "Yes" if active == 1 else "No"


    # ========================================================
    # GENERATE PDF
    # ========================================================

    pdf = generate_pdf(
        age_years,
        gender,
        height,
        weight,
        bmi,
        ap_hi,
        ap_lo,
        cholesterol_text,
        glucose_text,
        smoke_text,
        alcohol_text,
        active_text,
        risk_score,
        result
    )


    st.markdown("---")


    st.download_button(
        label="📄 Download Complete Health Report",
        data=pdf,
        file_name="CardioAI_Health_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    "⚠️ Disclaimer: CardioAI is an educational machine-learning "
    "project and is not intended to diagnose cardiovascular disease. "
    "Its predictions should not replace evaluation or advice from a "
    "qualified healthcare professional."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

❤️ <b>CardioAI</b> | Machine Learning Healthcare Project

<br><br>

Developed by Tushita Bhardwaj

</div>
""", unsafe_allow_html=True)