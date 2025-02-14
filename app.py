import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and scaler
with open('Models/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Credit Card Fraud Detection 🔍")
st.markdown("**Enter transaction details to analyze fraud probability**")

# Main input form
with st.form("transaction_form"):
    st.subheader("Transaction Details")
    
    # Amount input
    amount = st.number_input("Transaction Amount", 
                           min_value=0.0, 
                           value=0.0,
                           step=0.01,
                           format="%.2f")
    
    # Create columns for V features
    cols = st.columns(4)
    v_features = {}
    
    # Generate inputs for V1-V28
    for i in range(1, 29):
        with cols[(i-1)%4]:
            v_features[f'V{i}'] = st.number_input(
                f"V{i}",
                value=0.0,
                step=0.1,
                format="%.1f",
                help=f"PCA component V{i}"
            )
    
    # Prediction button
    submitted = st.form_submit_button("Analyze Transaction")
    
    if submitted:
        # Prepare input data in correct order
        input_data = [v_features[f'V{i}'] for i in range(1, 29)] + [amount]
        
        # Scale features
        scaled_data = scaler.transform([input_data])
        
        # Make prediction
        proba = model.predict_proba(scaled_data)[0]
        prediction = model.predict(scaled_data)[0]
        
        # Display results
        st.subheader("Analysis Results")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Legitimate Probability", f"{proba[0]*100:.2f}%")
        with result_col2:
            st.metric("Fraud Probability", f"{proba[1]*100:.2f}%")
        
        st.markdown("---")
        if prediction == 1:
            st.error("🚨 Fraud Alert: This transaction is classified as fraudulent")
        else:
            st.success("✅ Legitimate Transaction: This transaction appears normal")