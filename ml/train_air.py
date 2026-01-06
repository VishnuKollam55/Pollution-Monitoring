"""
train_air.py - Air Pollution ML Model Training Script
Trains a Random Forest Regressor for AQI prediction.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_csv_data, clean_air_data, prepare_features_air


def train_air_model():
    """
    Train Random Forest model for AQI prediction.
    
    Model predicts AQI value based on:
    - PM2.5 concentration
    - PM10 concentration
    - CO2 concentration
    
    Returns:
        tuple: (trained model, metrics dictionary)
    """
    print("=" * 50)
    print("AIR POLLUTION MODEL TRAINING")
    print("=" * 50)
    
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'air_data.csv')
    print(f"\nLoading data from: {data_path}")
    
    df = load_csv_data(data_path)
    print(f"Loaded {len(df)} records")
    
    df = clean_air_data(df)
    print("Data cleaned and validated")
    
    # Prepare features
    X, y = prepare_features_air(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize and train Random Forest model
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'feature_importance': dict(zip(
            ['pm25', 'pm10', 'co2'],
            model.feature_importances_
        ))
    }
    
    print("\n" + "=" * 30)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 30)
    print(f"Training RMSE: {metrics['train_rmse']:.2f}")
    print(f"Testing RMSE: {metrics['test_rmse']:.2f}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Testing R²: {metrics['test_r2']:.4f}")
    
    print("\nFeature Importance:")
    for feature, importance in metrics['feature_importance'].items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'air_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    return model, metrics


def predict_aqi(model, pm25, pm10, co2):
    """
    Predict AQI using the trained model.
    
    Args:
        model: Trained Random Forest model
        pm25 (float): PM2.5 concentration
        pm10 (float): PM10 concentration
        co2 (float): CO2 concentration
        
    Returns:
        int: Predicted AQI value
    """
    features = np.array([[pm25, pm10, co2]])
    prediction = model.predict(features)[0]
    return round(prediction)


def batch_predict(model, data):
    """
    Perform batch AQI predictions.
    
    Args:
        model: Trained Random Forest model
        data (pd.DataFrame): DataFrame with pm25, pm10, co2 columns
        
    Returns:
        np.array: Array of predicted AQI values
    """
    features = data[['pm25', 'pm10', 'co2']].values
    predictions = model.predict(features)
    return np.round(predictions).astype(int)


if __name__ == '__main__':
    # Train the model
    model, metrics = train_air_model()
    
    # Test predictions
    print("\n" + "=" * 30)
    print("SAMPLE PREDICTIONS")
    print("=" * 30)
    
    test_cases = [
        (25, 40, 400),   # Low pollution
        (55, 85, 480),   # Moderate pollution
        (100, 150, 550)  # High pollution
    ]
    
    for pm25, pm10, co2 in test_cases:
        aqi = predict_aqi(model, pm25, pm10, co2)
        print(f"PM2.5={pm25}, PM10={pm10}, CO2={co2} → AQI={aqi}")
