from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Crime Prediction & Analysis API", version="1.0")

# Load model on startup
MODEL_PATH = '../models/best_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PredictionRequest(BaseModel):
    # This accepts a dictionary of features. 
    # In a strict production environment, you would define every expected feature here.
    features: dict

@app.get("/")
def home():
    return {"message": "Crime Prediction API is running. Send POST requests to /predict."}

@app.post("/predict")
def predict_crime(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train the model first.")
    
    try:
        # Convert incoming JSON dict to DataFrame
        input_data = pd.DataFrame([request.features])
        
        # Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None
        
        return {
            "highCrime_prediction": bool(prediction[0]),
            "probability": float(probability) if probability is not None else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
