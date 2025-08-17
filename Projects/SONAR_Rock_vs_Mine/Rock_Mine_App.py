import numpy as np
import pickle
import streamlit as st

load = pickle.load(open(
    'Rock_Mine.sav', 'rb'))

# Create a function for Prediction


def predict(input_data):
    # Convert comma-separated string to list of floats
    input_data = [float(x) for x in input_data.split(",")]
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)
    prediction = load.predict(input_data)
    if (prediction[0] == 'R'):
        return "The object is a Rock"
    return "The object is a Mine"


def main():
    # title
    st.title("Mine vs Rock Predictor")

    # data
    features = st.text_input("Enter 60 comma-separated features")

    # prediction
    Prediction = ""

    # Creating a button for Prediction
    if (st.button("Predict")):
        Prediction = predict(features)
    st.success(Prediction)


if __name__ == '__main__':
    main()
