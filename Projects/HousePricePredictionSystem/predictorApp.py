import streamlit as st
import pandas as pd
import pickle
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_model.sav")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

original_order = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
top_features = ['LSTAT', 'RM', 'DIS', 'TAX', 'PTRATIO']
other_features = [f for f in original_order if f not in top_features]

df = pd.read_csv(
    "Projects/HousePricePredictionSystem/housingDataset_clean.csv")

st.title("House Price Predictor")
st.subheader("Enter the following details:")

feature_fullnames = {
    "LSTAT": "Lower Status Population (LSTAT) (%)",
    "RM": "Average Number of Rooms (RM)",
    "DIS": "Weighted Distance to Employment Centers (DIS)",
    "TAX": "Property Tax Rate (TAX)",
    "PTRATIO": "Pupil-Teacher Ratio (PTRATIO)"
}

user_input_values = {}
for feature in top_features:
    user_input_values[feature] = st.number_input(
        f"{feature_fullnames[feature]}:", value=None, step=0.00001, format="%.5f")

user_input = pd.DataFrame(
    [list(user_input_values.values())], columns=top_features)

for feature in other_features:
    user_input[feature] = df[feature].median()

final_input = user_input[original_order]

if st.button("Predict Price"):
    predicted_price = model.predict(final_input)
    st.success(f"Predicted House Price: ${predicted_price[0]*1e5:,.2f}")
