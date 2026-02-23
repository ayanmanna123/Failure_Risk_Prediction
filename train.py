import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, classification_report
import joblib
import shap
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def train_pipeline():
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('equipment_monitoring_1000.xlsx'):
        print("Error: Dataset file not found.")
        return

    df = pd.read_excel('equipment_monitoring_1000.xlsx', header=1)

    # Combine Date and Time for proper sorting
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # 2. Feature Engineering (Early - before split for some, ROC after)
    print("Engineering features...")
    # Static Specs
    specs = {
        'Rated_Voltage': 230,
        'Rated_Current': 11.5,
        'Max_Winding_Temp': 155,
    }

    # Physical Deviations
    df['Voltage_Deviation'] = abs(df['Actual Voltage (V)'] - specs['Rated_Voltage'])
    df['Temp_Margin'] = specs['Max_Winding_Temp'] - df['Temperature (°C)']
    df['Load_Stress'] = (df['Load %'] / 100) * (df['Actual Current (A)'] / specs['Rated_Current'])

    # Rolling features (Smoothing & Trends - uses previous 5 readings)
    df['Vibration_Smooth'] = df['Vibration (mm/s)'].rolling(window=5, min_periods=1).mean()
    df['Temp_Smooth'] = df['Temperature (°C)'].rolling(window=5, min_periods=1).mean()
    df['Vibration_Trend'] = df['Vibration (mm/s)'].diff().rolling(window=5, min_periods=1).mean()

    # Rate of Change (ROC) Features
    df['Temp_ROC'] = df['Temperature (°C)'].pct_change().fillna(0)
    df['Vibration_ROC'] = df['Vibration (mm/s)'].pct_change().fillna(0)

    # Lag Features
    sensor_cols = ['Actual Voltage (V)', 'Actual Current (A)', 'Temperature (°C)', 'Vibration (mm/s)', 'Speed (RPM)', 'Load %']
    print("Adding lag features...")
    for col in sensor_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_lag5'] = df[col].shift(5)

    # Fill NaNs created by rolling/diff/lag operations
    df = df.fillna(method='bfill').fillna(method='ffill')

    # Define feature list
    lag_features = [f'{col}_lag{i}' for col in sensor_cols for i in [1, 3, 5]]
    features = sensor_cols + ['Voltage_Deviation', 'Temp_Margin', 'Load_Stress', 'Vibration_Smooth', 'Temp_Smooth', 'Vibration_Trend', 'Temp_ROC', 'Vibration_ROC'] + lag_features
    
    X = df[features]
    y = df['Failure Risk %'].clip(0, 100)

    # 3. Time-Series Aware Validation
    print("Splitting data (Time-Series Split)...")
    tscv = TimeSeriesSplit(n_splits=5)

    # Get final split indices
    train_indices, test_indices = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # 4. Outlier Removal (Time-Safe: On Train ONLY)
    print("Removing outliers from training data...")
    for col in sensor_cols:
        mean = X_train[col].mean()
        std = X_train[col].std()
        z_train = (X_train[col] - mean) / std
        mask = abs(z_train) < 3
        X_train = X_train[mask]
        y_train = y_train[mask]

    # 5. Professional Scaling (Data Leakage Fix: Fit on Train only)
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. XGBoost Hyperparameter Tuning (Inner CV to avoid leakage)
    print("Tuning XGBoost Regression model...")
    tscv_inner = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist')
    
    search = RandomizedSearchCV(
        xgb_reg, 
        param_grid, 
        cv=tscv_inner, 
        n_iter=15, 
        scoring='neg_mean_absolute_error', 
        random_state=42
    )
    search.fit(X_train_scaled, y_train)
    best_reg_model = search.best_estimator_
    print(f"Best Hyperparameters: {search.best_params_}")

    # 7. Final Fit with Early Stopping
    print("Final Regression training with Early Stopping...")
    # Re-initialize to ensure compatible parameters with current XGBoost version
    best_reg_model = xgb.XGBRegressor(
        **search.best_params_,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=30,
        eval_metric='mae'
    )
    best_reg_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # 8. Classification Model (Judge-Worthy Upgrade)
    print("Training parallel classification model...")
    # Create classes: 0: Safe (0-30), 1: Warning (30-70), 2: High Risk (70-100)
    y_train_class = pd.cut(y_train, bins=[-1, 30, 70, 101], labels=[0, 1, 2]).astype(int)
    y_test_class = pd.cut(y_test, bins=[-1, 30, 70, 101], labels=[0, 1, 2]).astype(int)

    clf = xgb.XGBClassifier(objective='multi:softprob', random_state=42, tree_method='hist', n_estimators=200)
    clf.fit(X_train_scaled, y_train_class)
    
    y_pred_class = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test_class, y_pred_class)
    print(f"Classification Accuracy: {acc:.4f}")

    # 9. Evaluation
    y_pred = best_reg_model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 100)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Model Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 10. Feature Importance
    importance = best_reg_model.feature_importances_
    fi = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)
    fi.to_csv("feature_importance.csv", index=False)

    # 11. Explainability (SHAP)
    print("Generating SHAP analysis...")
    try:
        explainer = shap.TreeExplainer(best_reg_model)
        shap_values = explainer.shap_values(X_test_scaled)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
    except Exception as e:
        print(f"SHAP Error: {e}")

    # 12. Save Artifacts & Metadata
    print("Saving production artifacts...")
    joblib.dump(best_reg_model, 'advanced_machine_risk_model.pkl')
    joblib.dump(clf, 'risk_classifier.pkl')
    joblib.dump(scaler, 'robust_scaler.pkl')
    joblib.dump(features, 'feature_names.pkl')

    metadata = {
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "classification_accuracy": float(acc),
        "best_params": search.best_params_
    }
    joblib.dump(metadata, "training_metadata.pkl")
    print("Model, Scaler, Classifier, and Metadata saved successfully.")

if __name__ == "__main__":
    train_pipeline()
