import os
import logging
import pandas as pd
import numpy as np
import onnxruntime as rt  # For ONNX inference
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file path for the ONNX model (stored locally in your repo)
MODEL_PATH = "model.onnx"

# Load the ONNX model using onnxruntime
try:
    session = rt.InferenceSession(MODEL_PATH)
    logging.info("ONNX model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading ONNX model: {e}")
    raise RuntimeError("Failed to load ONNX model.")

# Define FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices based on given features using an ONNX model (no scaler)",
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

# Prediction endpoint using ONNX inference (no scaler applied)
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        data = data.astype(float)

        # ONNX Runtime requires input as a numpy array of type float32
        input_array = data.to_numpy().astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Run inference using ONNX Runtime
        pred_onx = session.run(None, {input_name: input_array})
        prediction_units = pred_onx[0][0]

        # Convert prediction to actual dollars.
        # (Assuming the model output is in units of 100,000 dollars)
        predicted_price = float(prediction_units) * 100000

        return {"predicted_price": predicted_price}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run locally for debugging (not needed for deployment)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
