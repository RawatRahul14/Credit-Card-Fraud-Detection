import streamlit as st
import os
import pickle
import pandas as pd

def load_model():
    """Load the trained model from disk."""
    model_path = os.path.join("Model", "logistic_model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def data_preprocess(input_data) -> pd.DataFrame:
    """Preprocess the input data to match the model's expected format."""
    # Load the expected columns
    columns_path = os.path.join("Lists", "col_names.pkl")
    with open(columns_path, "rb") as file:
        expected_columns = pickle.load(file)
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply one-hot encoding
    categorical_cols = ["category", "gender", "state", "job"]
    df_encoded = pd.get_dummies(df, columns = categorical_cols)
    
    # Add missing columns with 0s
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Drop any extra columns
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded

def load_lists_cols():
    """Load necessary lists for categorical features and column names."""
    cat_path = os.path.join("Lists", "categories.pkl")
    col_names_path = os.path.join("Lists", "col_names.pkl")
    jobs_path = os.path.join("Lists", "jobs.pkl")
    state_path = os.path.join("Lists", "states.pkl")

    with open(cat_path, "rb") as file:
        cat_names = pickle.load(file)
    with open(col_names_path, "rb") as file:
        col_names = pickle.load(file)
    with open(jobs_path, "rb") as file:
        job_names = pickle.load(file)
    with open(state_path, "rb") as file:
        state_names = pickle.load(file)

    return cat_names, col_names, job_names, state_names

def predict_fraud(model, processed_data: pd.DataFrame):
    """Make prediction using the loaded model."""
    prediction = model.predict_proba(processed_data)[0][1]
    return prediction

def main():
    st.title("Credit Card Fraud Detection")
    st.write("This application predicts the probability of a credit card transaction being fraudulent. Please fill in the transaction details below.")

    # Load necessary lists and model
    cat_names, _, job_names, state_names = load_lists_cols()
    model = load_model()

    # Create input widgets in two columns
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Transaction Category", sorted(cat_names))
        gender = st.selectbox("Cardholder Gender", ["M", "F"])
        state = st.selectbox("Cardholder State", sorted(state_names))

    with col2:
        job = st.selectbox("Cardholder Occupation", sorted(job_names))
        amount = st.number_input("Transaction Amount ($)", min_value = 0.0, format = "%.2f")
        city_pop = st.number_input("City Population (in thousands)", min_value = 0)

    # Create input data dictionary
    input_data = {
        "category": category,
        "amt": amount,
        "gender": gender,
        "state": state,
        "city_pop": city_pop*1000,
        "job": job
    }

    # Predict button
    if st.button("Predict Fraud Probability"):
        try:
            # Preprocess and predict
            processed_data = data_preprocess(input_data)
            probability = predict_fraud(model, processed_data)
            
            # Display results
            st.subheader(f"Fraud Probability: {probability:.2%}")
            st.subheader(f"Not-Fraud Probability: {1-probability:.2%}")
            
            # Add interpretation
            if probability > 0.7:
                st.error("High risk transaction! Recommend further investigation.")
            elif probability > 0.3:
                st.warning("Medium risk transaction. Review recommended.")
            else:
                st.success("Low risk transaction. Likely legitimate.")

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()