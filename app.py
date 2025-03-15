import os
import logging
import joblib  # For loading compressed model and scaler
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import gdown to download files from Google Drive
try:
    import gdown
except ImportError:
    raise ImportError("gdown is required to download the model from Google Drive. Install it via 'pip install gdown'.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths
MODEL_PATH = "house_price_model_compressed.pkl"
SCALER_X_PATH = "scaler_X.pkl"   # This file is stored on GitHub

# Google Drive direct download URL for the compressed model file.
# Provided link: https://drive.google.com/file/d/1huYgFS6yUleTzl5WJaHLjKOfCR7f-WDy/view?usp=sharing
# Converted to direct download URL:
MODEL_URL = "https://drive.google.com/uc?export=download&id=1huYgFS6yUleTzl5WJaHLjKOfCR7f-WDy"

# Download the compressed model file using gdown if it does not exist locally
if not os.path.exists(MODEL_PATH):
    try:
        logging.info("Downloading compressed model from Google Drive using gdown...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        logging.info("Compressed model downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise RuntimeError("Failed to download compressed model.")

# Load the compressed model using joblib
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Compressed model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load compressed model.")

# Load the feature scaler (scaler_X) from local repository (stored on GitHub)
try:
    with open(SCALER_X_PATH, "rb") as f:
        scaler_X = joblib.load(f)
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

        # Predict using the scaled input.
        # The model output is assumed to be in units of 100,000 dollars.
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
