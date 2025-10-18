import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open('linear_regression_fuel_data.pkl', 'rb'))

# Title of the web app
st.title("Fuel Efficiency Prediction App 🚗")

st.write("Enter the car details below to predict its fuel efficiency (or mileage).")

# Example input fields (adjust these to match your dataset features)
engine_size = st.number_input('Engine Size (L)', min_value=0.0, value=2.0)
cylinders = st.number_input('Number of Cylinders', min_value=1, value=4)
fuel_consumption = st.number_input('Fuel Consumption (L/100 km)', min_value=0.0, value=8.0)

# Make prediction button
if st.button('Predict'):
    # Create a 2D array (model expects this format)
    features = np.array([[engine_size, cylinders, fuel_consumption]])
    prediction = model.predict(features)

    st.success(f"Predicted Fuel Efficiency: {prediction[0]:.2f} km/L")
