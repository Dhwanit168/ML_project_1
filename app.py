import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="🏢 Building Energy Predictor", page_icon="⚡", layout="centered")
st.title("🏢 Building Energy Consumption Prediction")

# -------------------------------
# Load saved model, scaler, encoders, and feature columns
# -------------------------------
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # saved from training

# -------------------------------
# User Inputs
# -------------------------------
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
square_footage = st.number_input("Square Footage", value=1500.0)
occupancy = st.number_input("Occupancy", value=5)
hvac = st.selectbox("HVAC Usage", ["On", "Off"])
lighting = st.selectbox("Lighting Usage", ["On", "Off"])
renewable = st.number_input("Renewable Energy (kWh)", value=5.0)
holiday = st.selectbox("Holiday", ["Yes", "No"])
hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1)
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Energy Consumption ⚡"):

    # Encode binary inputs
    hvac_val = label_encoders['HVACUsage'].transform([hvac])[0]
    lighting_val = label_encoders['LightingUsage'].transform([lighting])[0]
    holiday_val = label_encoders['Holiday'].transform([holiday])[0]

    # Handle one-hot encoding for DayOfWeek
    dayofweek_columns = [col for col in feature_columns if col.startswith("DayOfWeek_")]
    dayofweek_array = np.zeros(len(dayofweek_columns))
    day_map = { "Tuesday":0, "Wednesday":1, "Thursday":2, "Friday":3, "Saturday":4, "Sunday":5 }
    if day_of_week != "Monday":  # Monday was dropped in training
        idx = day_map[day_of_week]
        dayofweek_array[idx] = 1

    # Combine all features
    input_features = np.array([[temperature, humidity, square_footage, occupancy,
                                hvac_val, lighting_val, renewable, holiday_val, hour, month]])
    input_features = np.hstack([input_features, dayofweek_array.reshape(1, -1)])

    # Scale features
    input_scaled = scaler.transform(input_features)

    # Predict
    predicted_energy = model.predict(input_scaled)[0]

    st.success(f"⚡ Predicted Energy Consumption: {predicted_energy:.2f} kWh")