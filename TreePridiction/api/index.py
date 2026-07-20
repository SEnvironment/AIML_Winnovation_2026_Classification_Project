import os
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS so your front-end can talk to your back-end safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your voting classifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Define the expected structure for incoming data
class PredictionInput(BaseModel):
    features: list[float]  # Needs to contain exactly 73 values matching feature_names_in_

@app.post("/api/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert incoming list into a 2D numpy array row
        data_array = np.array([input_data.features])
        
        # Make prediction
        prediction = model.predict(data_array)
        
        # Convert numpy type to standard Python integer for JSON response
        result = int(prediction[0])
        
        return {"status": "success", "prediction": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}