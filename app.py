import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load Trained Model
# -------------------------------
with open("model/fraud_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

THRESHOLD = 0.3

st.set_page_config(page_title="Fraud Risk Scoring System", layout="centered")

st.title("💳 Financial Transaction Risk Scoring System")
st.markdown("Enter transaction details below to evaluate fraud risk.")

st.divider()

# -------------------------------
# User-Friendly Inputs
# -------------------------------

transaction_id = st.number_input("Transaction ID", value=1)
amount = st.number_input("Transaction Amount", value=0.0)
transaction_hour = st.slider("Transaction Hour", 0, 23, 12)

foreign_transaction = st.selectbox("Foreign Transaction?", ["No", "Yes"])
location_mismatch = st.selectbox("Location Mismatch?", ["No", "Yes"])

device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
velocity_last_24h = st.number_input("Transactions in Last 24h", value=0)
cardholder_age = st.number_input("Cardholder Age", value=30)

merchant_category = st.selectbox(
    "Merchant Category",
    ["Electronics", "Food", "Grocery", "Travel"]
)

st.divider()

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Analyze Transaction"):

    # Convert Yes/No to numeric
    foreign_transaction_num = 1 if foreign_transaction == "Yes" else 0
    location_mismatch_num = 1 if location_mismatch == "Yes" else 0

    # Initialize all model features to 0
    input_data = dict.fromkeys(model.feature_names_in_, 0)

    # Fill numeric fields
    input_data["transaction_id"] = transaction_id
    input_data["amount"] = amount
    input_data["transaction_hour"] = transaction_hour
    input_data["foreign_transaction"] = foreign_transaction_num
    input_data["location_mismatch"] = location_mismatch_num
    input_data["device_trust_score"] = device_trust_score
    input_data["velocity_last_24h"] = velocity_last_24h
    input_data["cardholder_age"] = cardholder_age

    # Handle Merchant Category Encoding
    category_column = f"merchant_category_{merchant_category}"
    if category_column in input_data:
        input_data[category_column] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Model Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk_score = round(prob * 100, 2)

    st.subheader("📊 Risk Analysis Result")

    if prob >= THRESHOLD:
        st.error("🚨 FRAUD ALERT")
    else:
        st.success("✅ Normal Transaction")

    st.write(f"**Fraud Probability:** {prob:.4f}")
    st.write(f"**Risk Score (0–100):** {risk_score}")
    st.progress(min(int(risk_score), 100))

    # -------------------------------
    # Show Explanation ONLY for Fraud
    # -------------------------------

    if prob >= THRESHOLD:

        st.subheader("🧠 Why Flagged as Fraud?")

        explanations = []

        if foreign_transaction_num == 1:
            explanations.append("• Foreign transaction detected.")

        if location_mismatch_num == 1:
            explanations.append("• Location mismatch detected.")

        if device_trust_score < 0.3:
            explanations.append("• Low device trust score.")

        if velocity_last_24h > 5:
            explanations.append("• High transaction velocity in last 24 hours.")

        if amount > 2000:
            explanations.append("• High transaction amount.")

        if transaction_hour < 6 or transaction_hour > 22:
            explanations.append("• Unusual transaction hour.")

        if merchant_category in ["Electronics", "Travel"]:
            explanations.append("• High-risk merchant category.")

        if explanations:
            for explanation in explanations:
                st.write(explanation)
        else:
            st.write("• Multiple behavioral indicators triggered risk threshold.")