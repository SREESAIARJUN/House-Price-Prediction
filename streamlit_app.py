import streamlit as st
import requests
import json

# API URL hosted on Render
API_URL = "https://house-price-prediction-oo91.onrender.com/predict"

# Streamlit UI Configuration
st.set_page_config(page_title="California House Price Prediction", page_icon="🏡", layout="centered")

# App Title
st.title("🏡 California House Price Prediction")
st.markdown("Enter the house details below to get a predicted price.")

# Sidebar for Input Features
st.sidebar.header("📊 Enter House Features")

# User Input Fields
MedInc = st.sidebar.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
HouseAge = st.sidebar.slider("House Age (in years)", 1, 100, 15)
AveRooms = st.sidebar.slider("Average Rooms per House", 1.0, 10.0, 5.4)
AveBedrms = st.sidebar.slider("Average Bedrooms per House", 0.5, 5.0, 1.2)
Population = st.sidebar.slider("Population of the Area", 100, 40000, 1500)
AveOccup = st.sidebar.slider("Average Occupants per Household", 1.0, 6.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.5)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -122.0)

# Create input data dictionary
input_data = {
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
}

# Button to make prediction
if st.button("💰 Predict House Price"):
    try:
        # Send request to API
        response = requests.post(API_URL, json=input_data)
        
        if response.status_code == 200:
            predicted_price = response.json().get("predicted_price", "N/A")
            st.success(f"🏠 Predicted House Price: **${predicted_price:,.2f}**")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection Error: {e}")

# Footer
st.markdown("---")
st.markdown("🔹 **Powered by FastAPI & Streamlit** | 🌍 Hosted on **Render**")
