import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define FastAPI app
app = FastAPI()

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

# API endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}
