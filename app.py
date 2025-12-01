import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

# The exact feature columns model expects
feature_columns = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# ------------------- UI STYLE -------------------
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .title {font-size: 32px; font-weight: bold; color:#333;}
    .sub {font-size:18px; color:#666;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='title'>📊 Customer Churn Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Enter customer details below to predict churn.</p>", unsafe_allow_html=True)

# ------------------- INPUT FORM -------------------
col1, col2 = st.columns(2)

with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", [0, 1])
    Dependents = st.selectbox("Dependents", [0, 1])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
    PhoneService = st.selectbox("Phone Service", [0, 1])
    PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 1000.0, 75.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Credit card (automatic)"])

# ------------------- BUILD INPUT -------------------
input_data = {
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'PaperlessBilling': PaperlessBilling,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'gender_Male': 1 if gender == "Male" else 0,
    'MultipleLines_No phone service': 1 if MultipleLines == "No phone service" else 0,
    'MultipleLines_Yes': 1 if MultipleLines == "Yes" else 0,
    'InternetService_Fiber optic': 1 if InternetService == "Fiber optic" else 0,
    'InternetService_No': 1 if InternetService == "No" else 0,
    'OnlineSecurity_No internet service': 1 if OnlineSecurity == "No internet service" else 0,
    'OnlineSecurity_Yes': 1 if OnlineSecurity == "Yes" else 0,
    'OnlineBackup_No internet service': 1 if OnlineBackup == "No internet service" else 0,
    'OnlineBackup_Yes': 1 if OnlineBackup == "Yes" else 0,
    'DeviceProtection_No internet service': 1 if DeviceProtection == "No internet service" else 0,
    'DeviceProtection_Yes': 1 if DeviceProtection == "Yes" else 0,
    'TechSupport_No internet service': 1 if TechSupport == "No internet service" else 0,
    'TechSupport_Yes': 1 if TechSupport == "Yes" else 0,
    'StreamingTV_No internet service': 1 if StreamingTV == "No internet service" else 0,
    'StreamingTV_Yes': 1 if StreamingTV == "Yes" else 0,
    'StreamingMovies_No internet service': 1 if StreamingMovies == "No internet service" else 0,
    'StreamingMovies_Yes': 1 if StreamingMovies == "Yes" else 0,
    'Contract_One year': 1 if Contract == "One year" else 0,
    'Contract_Two year': 1 if Contract == "Two year" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == "Mailed check" else 0
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ------------------- PREDICT -------------------
st.markdown("---")

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # churn probability

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("⚠ The customer is likely to CHURN.")
    else:
        st.success("✔ The customer is NOT likely to churn.")

    st.write("")
    st.subheader("Churn Probability")
    st.progress(float(prob))

    st.write(f"**Probability: {prob*100:.2f}%**")
