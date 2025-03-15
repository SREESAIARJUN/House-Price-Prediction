import pickle
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the trained model
try:
    with open("house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

# Define FastAPI app
app = FastAPI(title="House Price Prediction API",
              description="Predict house prices based on given features",
              version="1.0")

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
        
        # Ensure correct dtype
        data = data.astype(float)

        # Make prediction
        prediction = model.predict(data)
        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run locally for debugging (not needed for deployment)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
