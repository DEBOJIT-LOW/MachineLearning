import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraudDetection.sav")
load = pickle.load(open(MODEL_PATH, 'rb'))


def predict_fraud(input_data):
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = load.predict(input_data)
    return prediction[0]


def main():
    st.title("Credit Card Fraud Detection System")

    st.write(
        "Upload a CSV file with transaction data (same format as Kaggle dataset).")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.write(data.head())
            predictions = load.predict(
                data.drop(columns=["Class"], errors="ignore"))
    
            data["Prediction"] = predictions
            data["Prediction"] = data["Prediction"].map(
                {0: "Not Fraud", 1: "Fraud"})
    
            st.subheader("Prediction Results")
            st.write(data.head(20))
    
            fraud_count = (data["Prediction"] == "Fraud").sum()
            st.success(f"✅ Detected {fraud_count} fraudulent transactions.")
        except:
            st.error("Please ensure the format of data is correct!")

if __name__ == "__main__":
    main()

