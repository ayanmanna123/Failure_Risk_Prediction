import joblib
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

def apply_safety_guards(df):
    """
    Production Touch: Real-time safety guards to prevent sensor glitches
    from triggering false alarms.
    """
    ranges = {
        'Actual Voltage (V)': (150, 300),
        'Actual Current (A)': (0, 50),
        'Temperature (°C)': (-20, 180),
        'Vibration (mm/s)': (0, 25),
        'Speed (RPM)': (0, 5000),
        'Load %': (0, 150)
    }
    
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            # Check for impossible values
            count_invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if count_invalid > 0:
                print(f"Warning: {count_invalid} invalid sensor readings detected in {col}. Clipping to safety range.")
                df[col] = df[col].clip(min_val, max_val)
    return df

def predict_risk(recent_data):
    """
    recent_data: pd.DataFrame with at least 6 rows of sensor data to support lag features (lag5).
    Columns: ['Actual Voltage (V)', 'Actual Current (A)', 'Temperature (°C)', 'Vibration (mm/s)', 'Speed (RPM)', 'Load %']
    """
    # 0. Minimum History Check
    if len(recent_data) < 6:
        raise ValueError("Need at least 6 rows of sensor data for lag features (lag5).")

    # Load artifacts
    artifacts = ['advanced_machine_risk_model.pkl', 'risk_classifier.pkl', 'robust_scaler.pkl', 'feature_names.pkl']
    if not all(os.path.exists(f) for f in artifacts):
        raise FileNotFoundError("Model artifacts not found. Please run train.py first.")

    model = joblib.load('advanced_machine_risk_model.pkl')
    clf = joblib.load('risk_classifier.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Static Specs
    specs = {
        'Rated_Voltage': 230,
        'Rated_Current': 11.5,
        'Max_Winding_Temp': 155,
    }
    
    df = recent_data.copy()
    
    # 1. Apply Safety Guards
    df = apply_safety_guards(df)

    # Sensor Drift Detection (Bonus)
    if df['Temperature (°C)'].iloc[-1] - df['Temperature (°C)'].iloc[-6] > 20:
        print("Warning: Sudden temperature spike detected (>20°C in 5 steps)!")
    
    # 2. Physical Deviations
    df['Voltage_Deviation'] = abs(df['Actual Voltage (V)'] - specs['Rated_Voltage'])
    df['Temp_Margin'] = specs['Max_Winding_Temp'] - df['Temperature (°C)']
    df['Load_Stress'] = (df['Load %'] / 100) * (df['Actual Current (A)'] / specs['Rated_Current'])

    # 3. Rolling & ROC features
    df['Vibration_Smooth'] = df['Vibration (mm/s)'].rolling(window=5, min_periods=1).mean()
    df['Temp_Smooth'] = df['Temperature (°C)'].rolling(window=5, min_periods=1).mean()
    df['Vibration_Trend'] = df['Vibration (mm/s)'].diff().rolling(window=5, min_periods=1).mean()
    
    df['Temp_ROC'] = df['Temperature (°C)'].pct_change().fillna(0)
    df['Vibration_ROC'] = df['Vibration (mm/s)'].pct_change().fillna(0)

    # 4. Lag Features
    sensor_cols = ['Actual Voltage (V)', 'Actual Current (A)', 'Temperature (°C)', 'Vibration (mm/s)', 'Speed (RPM)', 'Load %']
    for col in sensor_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_lag5'] = df[col].shift(5)

    # Fill NaNs handles first rows
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Select last row for prediction
    X = df[feature_names].iloc[[-1]]
    X_scaled = scaler.transform(X)
    
    # Predict Regression (Risk %)
    risk_percent = model.predict(X_scaled)[0]
    final_risk = np.clip(risk_percent, 0, 100)
    
    # Predict Classification (Status)
    class_idx = clf.predict(X_scaled)[0]
    status_map = {0: "Safe", 1: "Warning", 2: "High Risk"}
    status = status_map.get(class_idx, "Unknown")
        
    return {
        "Failure Risk %": round(float(final_risk), 2),
        "Status": status,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    # Scenario: Recent history of sensor readings (at least 6 rows for lag5)
    print("Testing Prediction with Safety Guards, Drift Detection, and ML Classifier...")
    
    history_data = pd.DataFrame([
        {'Actual Voltage (V)': 220, 'Actual Current (A)': 14, 'Temperature (°C)': 80, 'Vibration (mm/s)': 3.5, 'Speed (RPM)': 1420, 'Load %': 100},
        {'Actual Voltage (V)': 215, 'Actual Current (A)': 15, 'Temperature (°C)': 85, 'Vibration (mm/s)': 4.2, 'Speed (RPM)': 1410, 'Load %': 105},
        {'Actual Voltage (V)': 210, 'Actual Current (A)': 16, 'Temperature (°C)': 90, 'Vibration (mm/s)': 5.1, 'Speed (RPM)': 1400, 'Load %': 110},
        {'Actual Voltage (V)': 205, 'Actual Current (A)': 17, 'Temperature (°C)': 95, 'Vibration (mm/s)': 6.0, 'Speed (RPM)': 1390, 'Load %': 115},
        {'Actual Voltage (V)': 200, 'Actual Current (A)': 18, 'Temperature (°C)': 100, 'Vibration (mm/s)': 7.2, 'Speed (RPM)': 1380, 'Load %': 120},
        {'Actual Voltage (V)': 195, 'Actual Current (A)': 19, 'Temperature (°C)': 125, 'Vibration (mm/s)': 8.5, 'Speed (RPM)': 1370, 'Load %': 125}, # Induced spike
    ])
    
    try:
        result = predict_risk(history_data)
        print(f"\nPrediction Result: {result}")
    except Exception as e:
        print(f"Error: {e}. (Make sure to run train.py first to generate models)")
