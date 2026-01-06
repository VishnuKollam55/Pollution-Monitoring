"""
train_noise.py - Noise Pollution ML Model Training Script
Trains a Random Forest Classifier for noise level classification.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_csv_data, clean_noise_data, prepare_features_noise


def train_noise_model():
    """
    Train Random Forest Classifier for noise level classification.
    
    Model classifies noise levels based on:
    - Sound level (dB)
    - Zone type (Residential, Commercial, Industrial)
    
    Classes:
    - Low (0)
    - Medium (1)
    - High (2)
    
    Returns:
        tuple: (trained model, mappings, metrics dictionary)
    """
    print("=" * 50)
    print("NOISE POLLUTION MODEL TRAINING")
    print("=" * 50)
    
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'noise_data.csv')
    print(f"\nLoading data from: {data_path}")
    
    df = load_csv_data(data_path)
    print(f"Loaded {len(df)} records")
    
    df = clean_noise_data(df)
    print("Data cleaned and validated")
    
    # Prepare features
    X, y, zone_mapping, level_mapping = prepare_features_noise(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Zone mapping: {zone_mapping}")
    print(f"Level mapping: {level_mapping}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize and train Random Forest model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
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
    
    # Reverse mapping for classification report
    reverse_level_mapping = {v: k for k, v in level_mapping.items()}
    class_names = [reverse_level_mapping[i] for i in sorted(reverse_level_mapping.keys())]
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'zone_mapping': zone_mapping,
        'level_mapping': level_mapping,
        'feature_importance': dict(zip(
            ['sound_level', 'zone'],
            model.feature_importances_
        )),
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
    
    print("\nFeature Importance:")
    for feature, importance in metrics['feature_importance'].items():
        print(f"  {feature}: {importance:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Save model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'noise_model.pkl')
    model_data = {
        'model': model,
        'zone_mapping': zone_mapping,
        'level_mapping': level_mapping
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to: {model_path}")
    
    return model, model_data, metrics


def predict_noise_level(model_data, sound_level, zone):
    """
    Predict noise level classification using the trained model.
    
    Args:
        model_data (dict): Dictionary containing model and mappings
        sound_level (float): Sound level in dB
        zone (str): Zone type (Residential, Commercial, Industrial)
        
    Returns:
        tuple: (predicted level, probability scores)
    """
    model = model_data['model']
    zone_mapping = model_data['zone_mapping']
    level_mapping = model_data['level_mapping']
    reverse_level_mapping = {v: k for k, v in level_mapping.items()}
    
    zone_encoded = zone_mapping.get(zone, 0)
    features = np.array([[sound_level, zone_encoded]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    level_label = reverse_level_mapping[prediction]
    prob_dict = {reverse_level_mapping[i]: prob for i, prob in enumerate(probabilities)}
    
    return level_label, prob_dict


def classify_noise_level(sound_level, zone='General'):
    """
    Rule-based noise level classification (fallback method).
    
    Zone-based limits (dB):
    - Residential: Day 55, Night 45
    - Commercial: Day 65, Night 55
    - Industrial: Day 75, Night 70
    
    Args:
        sound_level (float): Sound level in dB
        zone (str): Zone type
        
    Returns:
        tuple: (level classification, risk description)
    """
    zone_limits = {
        'Residential': {'low': 45, 'medium': 55, 'high': 75},
        'Commercial': {'low': 55, 'medium': 65, 'high': 85},
        'Industrial': {'low': 65, 'medium': 75, 'high': 95}
    }
    
    limits = zone_limits.get(zone, {'low': 50, 'medium': 70, 'high': 85})
    
    if sound_level <= limits['low']:
        return 'Low', 'Acceptable noise level for the zone'
    elif sound_level <= limits['medium']:
        return 'Medium', 'Moderate noise, monitoring recommended'
    elif sound_level <= limits['high']:
        return 'High', 'Exceeds zone limits, action required'
    else:
        return 'Critical', 'Dangerous noise level, immediate action required'


def get_zone_risk_level(zone, sound_level):
    """
    Determine health risk based on zone and sound level.
    
    Args:
        zone (str): Zone type
        sound_level (float): Sound level in dB
        
    Returns:
        dict: Risk assessment with level and recommendations
    """
    # General noise health impact thresholds
    thresholds = {
        55: {'risk': 'Low', 'description': 'Minimal health impact'},
        70: {'risk': 'Moderate', 'description': 'May cause annoyance and sleep disturbance'},
        85: {'risk': 'High', 'description': 'Risk of hearing damage with prolonged exposure'},
        100: {'risk': 'Severe', 'description': 'Immediate hearing damage risk'},
        120: {'risk': 'Critical', 'description': 'Pain threshold, immediate danger'}
    }
    
    risk_level = {'risk': 'Low', 'description': 'Minimal health impact'}
    
    for threshold, info in sorted(thresholds.items()):
        if sound_level >= threshold:
            risk_level = info
    
    # Zone-specific adjustments
    if zone == 'Residential' and sound_level > 55:
        risk_level['zone_note'] = 'Exceeds residential zone limits'
    elif zone == 'Commercial' and sound_level > 65:
        risk_level['zone_note'] = 'Exceeds commercial zone limits'
    elif zone == 'Industrial' and sound_level > 75:
        risk_level['zone_note'] = 'Exceeds industrial zone limits'
    
    return risk_level


if __name__ == '__main__':
    # Train the model
    model, model_data, metrics = train_noise_model()
    
    # Test predictions
    print("\n" + "=" * 30)
    print("SAMPLE PREDICTIONS")
    print("=" * 30)
    
    test_cases = [
        (45, 'Residential'),
        (65, 'Commercial'),
        (92, 'Industrial')
    ]
    
    for sound_level, zone in test_cases:
        level, probs = predict_noise_level(model_data, sound_level, zone)
        print(f"Sound={sound_level}dB, Zone={zone} → Level: {level}")
        print(f"  Probabilities: {probs}")
        
        risk = get_zone_risk_level(zone, sound_level)
        print(f"  Risk: {risk}")
