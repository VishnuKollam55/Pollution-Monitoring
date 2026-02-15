import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fallback model
from ml.lstm_model import SimpleLSTMPredictor

# Try TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Using simplified LSTM implementation.")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================
# Helper Functions
# ============================================

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length, n_features=1):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(30, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ============================================
# Training Function
# ============================================

def train_lstm_model(data_path=None, sequence_length=7, epochs=50):

    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'air_data.csv'
        )

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if 'aqi' in df.columns:
        aqi_values = df['aqi'].values.astype(float)
    else:
        aqi_values = df['pm25'].values * 2

    aqi_values = pd.Series(aqi_values).fillna(method='ffill').fillna(method='bfill').values

    scaler = MinMaxScaler()
    aqi_scaled = scaler.fit_transform(aqi_values.reshape(-1, 1))

    X, y = create_sequences(aqi_scaled, sequence_length)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if TENSORFLOW_AVAILABLE:
        print("Training TensorFlow LSTM model...")

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = build_lstm_model(sequence_length)

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)

    else:
        print("Training fallback Simple LSTM predictor...")

        model = SimpleLSTMPredictor(sequence_length)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train).reshape(-1, 1)
        y_pred_test = model.predict(X_test).reshape(-1, 1)

    # Inverse transform
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_train_inv = scaler.inverse_transform(y_pred_train)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))

    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    if TENSORFLOW_AVAILABLE:
        model_path = os.path.join(models_dir, 'lstm_model.keras')
        model.save(model_path)
    else:
        model_path = os.path.join(models_dir, 'lstm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    with open(os.path.join(models_dir, 'lstm_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("Model saved successfully.")

    return model, scaler


# ============================================
# Load Function (FIXED HERE 🔥)
# ============================================

def load_lstm_model():

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    model = None
    scaler = None
    metrics = None

    try:
        if TENSORFLOW_AVAILABLE:
            model_path = os.path.join(models_dir, 'lstm_model.keras')
            if os.path.exists(model_path):
                model = load_model(model_path, compile=False)
        else:
            model_path = os.path.join(models_dir, 'lstm_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

    # Load scaler
    scaler_path = os.path.join(models_dir, 'lstm_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    # Load metrics
    metrics_path = os.path.join(models_dir, 'lstm_metrics.pkl')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

    return model, scaler, metrics



# ============================================
# Forecast Function
# ===============================

def forecast_aqi(model, scaler, recent_data, days_ahead=7):

    sequence_length = len(recent_data)
    forecasts = []

    current_sequence = scaler.transform(np.array(recent_data).reshape(-1, 1))

    for i in range(days_ahead):

        if TENSORFLOW_AVAILABLE:
            input_seq = current_sequence[-sequence_length:].reshape(1, sequence_length, 1)
            prediction = model.predict(input_seq, verbose=0)
            pred_value = prediction[0][0]
        else:
            input_seq = current_sequence[-sequence_length:].reshape(1, sequence_length)
            pred_value = model.predict(input_seq)[0]

        pred_aqi = scaler.inverse_transform([[pred_value]])[0][0]
        pred_aqi = max(0, min(500, pred_aqi))

        forecast_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')

        forecasts.append({
            'date': forecast_date,
            'day': i + 1,
            'aqi': round(pred_aqi)
        })

        current_sequence = np.append(current_sequence, [[pred_value]], axis=0)

    return forecasts


# ============================================
# Run Directly
# ============================================

if __name__ == "__main__":
    train_lstm_model()
