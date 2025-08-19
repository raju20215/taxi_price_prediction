# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Load your trained model
# -------------------------------
@st.cache_resource
def load_model():
    with open("model_pipeline.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("🚕 Taxi Fare Prediction App")

st.markdown("Enter trip details below to estimate the **Trip Price**:")

# -------------------------------
# Input fields
# -------------------------------
trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, step=0.1)

time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_of_week = st.selectbox("Day of Week", ['Weekday', 'Weekend'])

passenger_count = st.number_input("Passenger Count", min_value=1, step=1)

traffic_conditions = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
weather = st.selectbox("Weather", ["Clear", "Rainy", "Snowy"])

base_fare = st.number_input("Base Fare", min_value=0.0, step=1.0)
per_km_rate = st.number_input("Per Km Rate", min_value=0.0, step=0.1)
per_minute_rate = st.number_input("Per Minute Rate", min_value=0.0, step=0.1)
trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0, step=1.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Fare"):
    input_data = pd.DataFrame([{
        "Trip_Distance_km": trip_distance,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Passenger_Count": passenger_count,
        "Traffic_Conditions": traffic_conditions,
        "Weather": weather,
        "Base_Fare": base_fare,
        "Per_Km_Rate": per_km_rate,
        "Per_Minute_Rate": per_minute_rate,
        "Trip_Duration_Minutes": trip_duration
    }])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.success(f"💰 Estimated Trip Price: ₹{prediction[0]:.2f}")
