import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏢 Building Energy Prediction")

# --- User Inputs with defaults ---
temperature = st.number_input("Temperature (°C)", value=25.0)        # default 25°C
humidity = st.number_input("Humidity (%)", value=50.0)               # default 50%
square_footage = st.number_input("Building Size (sq ft)", value=1000) # default 1000 sq ft
occupancy = st.number_input("Number of Occupants", value=10)          # default 10

hvac = st.selectbox("HVAC Usage", ["Low", "Medium", "High"], index=1)       # default Medium
lighting = st.selectbox("Lighting Usage", ["Low", "Medium", "High"], index=1) # default Medium
renewable = st.number_input("Renewable Energy Contribution (kW)", value=5.0)  # default 5 kW
dayofweek = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], index=0)  # default Mon
holiday = st.selectbox("Holiday", ["No","Yes"], index=0)  # default No
hour = st.number_input("Hour of Day", min_value=0, max_value=23, value=12)  # default 12 PM
month = st.number_input("Month", min_value=1, max_value=12, value=6)        # default June

# --- Map categorical to numeric ---
hvac_map = {"Low":0,"Medium":1,"High":2}
lighting_map = {"Low":0,"Medium":1,"High":2}
day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
holiday_map = {"No":0,"Yes":1}

hvac = hvac_map[hvac]
lighting = lighting_map[lighting]
dayofweek = day_map[dayofweek]
holiday = holiday_map[holiday]

# --- Predict ---
if st.button("Predict"):
    features = np.array([[temperature, humidity, square_footage, occupancy,
                          hvac, lighting, renewable, dayofweek, holiday, hour, month]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f} kWh")