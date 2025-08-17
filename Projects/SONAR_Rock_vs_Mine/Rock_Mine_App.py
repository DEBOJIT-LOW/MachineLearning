import numpy as np
import pickle
import streamlit as st
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Rock_Mine.sav")
load = pickle.load(open(MODEL_PATH, "rb"))

# Create a function for Prediction
def predict(input_data):
    try:
        # Convert to list of floats
        input_data = [float(x.strip()) for x in input_data.split(",")]
        
        # Check feature length
        if len(input_data) != 60:
            return "Error: Please enter exactly 60 features."
        
        # Convert to numpy and reshape
        input_data = np.asarray(input_data).reshape(1, -1)
        
        prediction = load.predict(input_data)
        return "The object is a Rock" if prediction[0] == 'R' else "The object is a Mine"
    
    except ValueError:
        return "Error: Please enter only numeric values separated by commas."

# Streamlit app
def main():
    st.title("Mine vs Rock Predictor")
    features = st.text_input("Enter 60 comma-separated features")
    
    Prediction = ""
    if st.button("Predict"):
        Prediction = predict(features)
    st.success(Prediction)

if __name__ == '__main__':
    main()
