"""
train_water.py - Water Pollution ML Model Training Script
Trains a Support Vector Machine (SVM) for water quality classification.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_csv_data, clean_water_data, prepare_features_water


def train_water_model():
    """
    Train SVM model for water quality classification.
    
    Model classifies water quality based on:
    - pH level
    - Turbidity (NTU)
    - Dissolved Oxygen (mg/L)
    
    Classes:
    - Safe (0)
    - Moderate (1)
    - Polluted (2)
    
    Returns:
        tuple: (trained model, scaler, metrics dictionary)
    """
    print("=" * 50)
    print("WATER POLLUTION MODEL TRAINING")
    print("=" * 50)
    
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'water_data.csv')
    print(f"\nLoading data from: {data_path}")
    
    df = load_csv_data(data_path)
    print(f"Loaded {len(df)} records")
    
    df = clean_water_data(df)
    print("Data cleaned and validated")
    
    # Prepare features
    X, y, quality_mapping = prepare_features_water(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Quality mapping: {quality_mapping}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler")
    
    # Initialize and train SVM model
    print("\nTraining Support Vector Machine (SVM)...")
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model training completed!")
    
    # Evaluate model
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Reverse mapping for classification report
    reverse_mapping = {v: k for k, v in quality_mapping.items()}
    class_names = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'quality_mapping': quality_mapping,
        'classification_report': classification_report(
            y_test, y_pred_test, 
            target_names=class_names,
            output_dict=True
        )
    }
    
    print("\n" + "=" * 30)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 30)
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Save model and scaler
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'water_model.pkl')
    model_data = {
        'model': model,
        'scaler': scaler,
        'quality_mapping': quality_mapping
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to: {model_path}")
    
    return model, scaler, metrics


def predict_water_quality(model_data, ph, turbidity, dissolved_oxygen):
    """
    Predict water quality using the trained SVM model.
    
    Args:
        model_data (dict): Dictionary containing model, scaler, and mappings
        ph (float): Water pH level
        turbidity (float): Turbidity in NTU
        dissolved_oxygen (float): Dissolved oxygen in mg/L
        
    Returns:
        tuple: (predicted class label, probability scores)
    """
    model = model_data['model']
    scaler = model_data['scaler']
    mapping = model_data['quality_mapping']
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    features = np.array([[ph, turbidity, dissolved_oxygen]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    quality_label = reverse_mapping[prediction]
    prob_dict = {reverse_mapping[i]: prob for i, prob in enumerate(probabilities)}
    
    return quality_label, prob_dict


def classify_water_quality(ph, turbidity, dissolved_oxygen):
    """
    Rule-based water quality classification (fallback method).
    
    Args:
        ph (float): Water pH level
        turbidity (float): Turbidity in NTU
        dissolved_oxygen (float): Dissolved oxygen in mg/L
        
    Returns:
        str: Water quality classification
    """
    # Check pH - should be between 6.5 and 8.5 for safe water
    if ph < 6.0 or ph > 9.0:
        return 'Polluted'
    
    # Check turbidity - lower is better
    if turbidity > 10:
        return 'Polluted'
    elif turbidity > 5:
        if ph >= 6.5 and ph <= 8.5:
            return 'Moderate'
        else:
            return 'Polluted'
    
    # Check dissolved oxygen - higher is better (> 5 mg/L for healthy water)
    if dissolved_oxygen < 4:
        return 'Polluted'
    elif dissolved_oxygen < 6:
        return 'Moderate'
    
    # All parameters within safe limits
    if 6.5 <= ph <= 8.5 and turbidity <= 5 and dissolved_oxygen >= 6:
        return 'Safe'
    
    return 'Moderate'


if __name__ == '__main__':
    # Train the model
    model, scaler, metrics = train_water_model()
    
    # Load saved model for testing
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'water_model.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Test predictions
    print("\n" + "=" * 30)
    print("SAMPLE PREDICTIONS")
    print("=" * 30)
    
    test_cases = [
        (7.2, 3.0, 8.5),   # Safe water
        (6.8, 6.0, 6.5),   # Moderate quality
        (5.8, 15.0, 4.5)   # Polluted water
    ]
    
    for ph, turbidity, do in test_cases:
        quality, probs = predict_water_quality(model_data, ph, turbidity, do)
        print(f"pH={ph}, Turbidity={turbidity}, DO={do} → Quality: {quality}")
        print(f"  Probabilities: {probs}")
