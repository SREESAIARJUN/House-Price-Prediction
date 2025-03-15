import os
import requests
import pickle
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths
MODEL_PATH = "house_price_model.pkl"
SCALER_X_PATH = "scaler_X.pkl"   # on GitHub
# scaler_Y is on GitHub (if needed) but not used in this approach

# Google Drive direct download URL for the model file
MODEL_URL = "https://drive.google.com/uc?export=download&id=1rYn4wRbG-FRty0UIAVZG_ek0BLrJ9b2L"

# Download the model file from Google Drive if it does not exist locally
if not os.path.exists(MODEL_PATH):
    try:
        logging.info("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logging.info("Model downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise RuntimeError("Failed to download model.")

# Load the trained model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

# Load the feature scaler (scaler_X) from local repo (small file on GitHub)
try:
    with open(SCALER_X_PATH, "rb") as f:
        scaler_X = pickle.load(f)
    logging.info("Feature scaler (scaler_X) loaded successfully.")
except Exception as e:
    logging.error(f"Error loading feature scaler: {e}")
    raise RuntimeError("Failed to load feature scaler")

# Define FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices based on given features",
    version="1.0"
)

# Input schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running! Use /docs for Swagger UI."}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        data = data.astype(float)

        # Scale the input features using the loaded scaler_X
        scaled_input = scaler_X.transform(data)

        # Predict using the scaled input. The model output is in units of 100,000 dollars.
        prediction_units = model.predict(scaled_input)[0]

        # Convert prediction to actual dollars
        predicted_price = prediction_units * 100000

        return {"predicted_price": predicted_price}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run locally for debugging (not needed for deployment)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
