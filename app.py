import streamlit as st
import requests
import json

st.set_page_config(
    page_title="GNN Fraud Intelligence Hub",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Enterprise GNN Fraud & Ring Intelligence Platform")
st.markdown("Real-time graph neural network telemetry and vector similarity screening.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💳 Test Live Transaction")
    user_id = st.text_input("User ID", "U_00123")
    merchant_id = st.text_input("Merchant ID", "MCH_045")
    amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=2450.75)
    age = st.slider("User Age", 18, 80, 24)
    credit_score = st.slider("Credit Score", 300, 850, 410)

    if st.button("Evaluate Transaction Risk", type="primary"):
        payload = {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "amount": amount,
            "age": age,
            "credit_score": credit_score
        }
        
        try:
            # Call local FastAPI backend
            response = requests.post("http://127.0.0.1:8000/predict/fraud", json=payload)
            if response.status_code == 200:
                st.session_state['result'] = response.json()
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to backend server. Make sure FastAPI is running (`uvicorn api.main:app --reload`). Error: {e}")

with col2:
    st.subheader("📊 Intelligence & Risk Evaluation")
    if 'result' in st.session_state:
        res = st.session_state['result']
        analysis = res["fraud_analysis"]
        
        risk_level = analysis["risk_classification"]
        prob = analysis["fraud_probability"]
        
        if risk_level == "CRITICAL_FRAUD_RING":
            st.error(f"🚨 **CRITICAL RISK DETECTED** (Probability: {prob * 100}%)")
        elif risk_level == "SUSPICIOUS":
            st.warning(f"⚠️ **SUSPICIOUS ACTIVITY** (Probability: {prob * 100}%)")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION** (Probability: {prob * 100}%)")
            
        st.json(res)
    else:
        st.info("Configure transaction parameters on the left and click **Evaluate Transaction Risk**.")