import numpy as np
import pickle
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.sav")
load = pickle.load(open(MODEL_PATH, "rb"))

def predict_diabetes(input_data):
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = load.predict(input_data)
    print(prediction[0])
    if (prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    st.title("Diabetes Prediction App")

    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("Body Mass Index")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    diagnosis = ''
    try:
        if st.button("Diabetes Test Result"):
            diagnosis = predict_diabetes([Pregnancies, Glucose, BloodPressure,
                                          SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

        st.success(diagnosis)
    except:
        st.error("Please enter the values correctly.")


if __name__ == '__main__':
    main()


