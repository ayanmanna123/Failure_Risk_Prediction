# ⚡ AI-Based Real-Time Failure Risk Prediction System for Electric Machines

This project is an industry-grade predictive maintenance system designed to monitor electric machines and predict their failure risk in real-time using sensor data (Voltage, Current, Temperature, Vibration, Speed, and Load).

## 🚀 Key Features

*   **Real-Time Prediction**: Uses a high-performance XGBoost regressor to estimate failure probability (0-100%).
*   **Advanced Feature Engineering**: Incorporates physical deviations, rolling averages, and 1/3/5-step lag features for highly accurate temporal analysis.
*   **Production Safety Guards**: Real-time sanity checks to prevent sensor glitches from triggering false alarms.
*   **Explainable AI (SHAP)**: Understand which sensor factors are driving the risk assessment.
*   **FastAPI Deployment**: Ready-to-use API endpoint for production integration.

## 📁 Project Structure

*   `train.py`: The core training pipeline (Preprocessing -> Feature Engineering -> Tuning -> Evaluation).
*   `predict.py`: Prediction logic with built-in safety guards for production use.
*   `api.py`: FastAPI implementation for real-time risk estimation.
*   `Final_Notebook.ipynb`: A Jupyter Notebook for visual exploration, training, and SHAP analysis.
*   `equipment_monitoring_1000.xlsx`: The training dataset.
*   `advanced_machine_risk_model.pkl`: The trained XGBoost model.
*   `robust_scaler.pkl`: Feature scaler (fitted on training data only).
*   `feature_names.pkl`: List of features used by the model.

## 🛠️ How to Use

### 1. Installation
Ensure you have Python 3.8+ and install the required dependencies:
```bash
pip install pandas numpy xgboost scikit-learn joblib shap fastapi uvicorn matplotlib openpyxl
```

### 2. Training the Model
To retrain the model on the latest data:
```bash
python train.py
```
This will generate `advanced_machine_risk_model.pkl`, `feature_importance.csv`, and `shap_summary.png`.

### 3. Running Predictions (CLI)
Test the model locally with predefined scenarios:
```bash
python predict.py
```

### 4. Starting the API
Deploy the model as a web service:
```bash
python -m uvicorn api:app --reload
```
The API will be available at `http://127.0.0.1:8000`. Use the `/docs` endpoint for interactive documentation.

### 5. API Usage Example
Send a POST request to `/predict` with the last 6 sensor readings:
```json
{
  "history": [
    {"voltage": 230, "current": 11.2, "temperature": 42, "vibration": 1.1, "speed": 1485, "load_percent": 85},
    ... (6 readings total)
  ]
}
```

## 📊 Model Performance
*   **MAE**: ~1.58
*   **R2 Score**: ~0.99
*   **Stability**: Time-Series cross-validated.

---
**Developed for Predictive Maintenance Hackathons & Industry Applications.**
