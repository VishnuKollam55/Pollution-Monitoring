"""
train_air.py - Air Pollution ML Model Training Script
Trains a Random Forest Regressor for AQI prediction.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # ✅ Added for graph
plt.switch_backend('TkAgg')


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_csv_data, clean_air_data, prepare_features_air


def train_air_model():

    print("=" * 50)
    print("AIR POLLUTION MODEL TRAINING")
    print("=" * 50)

    # Load and preprocess data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'air_data.csv'
    )

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

    # Train model
    print("\nTraining Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Model training completed!")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
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

    # ===============================
    # ✅ Accuracy Graph (IMPORTANT)
    # ===============================
    print("\nGenerating accuracy graph...")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_test)

    # Best fit line
    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test))

    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI - Random Forest")
    plt.grid(True)

    # Save graph
    graph_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models',
        'aqi_accuracy.png'
    )

    plt.savefig(graph_path)
    plt.show()

    print(f"Graph saved to: {graph_path}")

    # Save model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models',
        'air_model.pkl'
    )

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path}")

    return model, metrics


def predict_aqi(model, pm25, pm10, co2):
    features = np.array([[pm25, pm10, co2]])
    prediction = model.predict(features)[0]
    return round(prediction)


if __name__ == '__main__':
    model, metrics = train_air_model()

    print("\n" + "=" * 30)
    print("SAMPLE PREDICTIONS")
    print("=" * 30)

    test_cases = [
        (25, 40, 400),
        (55, 85, 480),
        (100, 150, 550)
    ]

    for pm25, pm10, co2 in test_cases:
        aqi = predict_aqi(model, pm25, pm10, co2)
        print(f"PM2.5={pm25}, PM10={pm10}, CO2={co2} → AQI={aqi}")
