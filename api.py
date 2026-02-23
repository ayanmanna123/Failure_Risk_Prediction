from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from predict import predict_risk
import uvicorn

app = FastAPI(title="Electric Machine Failure Risk API", 
              description="Real-time prediction of machine failure risk using XGBoost and sensor data.")

class SensorReading(BaseModel):
    voltage: float
    current: float
    temperature: float
    vibration: float
    speed: float
    load_percent: float

class PredictionRequest(BaseModel):
    # Needs at least 6 readings to calculate lag features accurately (lag5)
    history: List[SensorReading]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Machine Risk Prediction API. Send POST requests to /predict"}

@app.post("/predict")
def get_prediction(request: PredictionRequest):
    if len(request.history) < 6:
        raise HTTPException(status_code=400, detail="At least 6 historical readings are required for accurate lag-feature calculation.")
    
    # Convert request to DataFrame
    data_list = []
    for r in request.history:
        data_list.append({
            'Actual Voltage (V)': r.voltage,
            'Actual Current (A)': r.current,
            'Temperature (°C)': r.temperature,
            'Vibration (mm/s)': r.vibration,
            'Speed (RPM)': r.speed,
            'Load %': r.load_percent
        })
    
    df = pd.DataFrame(data_list)
    
    try:
        result = predict_risk(df)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts not found on server. Please run training first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
