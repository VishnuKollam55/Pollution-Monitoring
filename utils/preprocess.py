"""
preprocess.py - Data Preprocessing Utility Module
Handles data loading, cleaning, and preprocessing for pollution monitoring system.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_csv_data(filepath):
    """
    Load CSV data from the specified filepath.
    Handles missing values and invalid data.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
        # Convert date column if exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        # Handle missing values - fill with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def clean_air_data(df):
    """
    Clean and validate air pollution data.
    Ensures PM2.5, PM10, and CO2 values are within valid ranges.
    
    Args:
        df (pandas.DataFrame): Raw air pollution data
        
    Returns:
        pandas.DataFrame: Cleaned air pollution data
    """
    # Define valid ranges for air pollution parameters
    valid_ranges = {
        'pm25': (0, 500),
        'pm10': (0, 600),
        'co2': (300, 1000)
    }
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            # Clip values to valid range
            df[col] = df[col].clip(min_val, max_val)
    
    return df


def clean_water_data(df):
    """
    Clean and validate water pollution data.
    Ensures pH, turbidity, and dissolved oxygen are within valid ranges.
    
    Args:
        df (pandas.DataFrame): Raw water pollution data
        
    Returns:
        pandas.DataFrame: Cleaned water pollution data
    """
    # Define valid ranges for water pollution parameters
    valid_ranges = {
        'ph': (0, 14),
        'turbidity': (0, 100),
        'dissolved_oxygen': (0, 14)
    }
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(min_val, max_val)
    
    return df


def clean_noise_data(df):
    """
    Clean and validate noise pollution data.
    Ensures sound levels are within valid ranges (0-150 dB).
    
    Args:
        df (pandas.DataFrame): Raw noise pollution data
        
    Returns:
        pandas.DataFrame: Cleaned noise pollution data
    """
    if 'sound_level' in df.columns:
        df['sound_level'] = df['sound_level'].clip(0, 150)
    
    return df


def prepare_features_air(df):
    """
    Prepare feature matrix for air pollution ML model.
    
    Args:
        df (pandas.DataFrame): Cleaned air data
        
    Returns:
        tuple: (X features, y target)
    """
    feature_cols = ['pm25', 'pm10', 'co2']
    target_col = 'aqi'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y


def prepare_features_water(df):
    """
    Prepare feature matrix for water pollution ML model.
    Encodes quality labels for classification.
    
    Args:
        df (pandas.DataFrame): Cleaned water data
        
    Returns:
        tuple: (X features, y target encoded)
    """
    feature_cols = ['ph', 'turbidity', 'dissolved_oxygen']
    
    # Encode quality labels
    quality_mapping = {'Safe': 0, 'Moderate': 1, 'Polluted': 2}
    
    X = df[feature_cols].values
    y = df['quality'].map(quality_mapping).values
    
    return X, y, quality_mapping


def prepare_features_noise(df):
    """
    Prepare feature matrix for noise pollution ML model.
    Encodes zone and level labels.
    
    Args:
        df (pandas.DataFrame): Cleaned noise data
        
    Returns:
        tuple: (X features, y target encoded)
    """
    # Encode zone labels
    zone_mapping = {'Residential': 0, 'Commercial': 1, 'Industrial': 2}
    level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    
    df_encoded = df.copy()
    df_encoded['zone_encoded'] = df['zone'].map(zone_mapping)
    
    X = df_encoded[['sound_level', 'zone_encoded']].values
    y = df_encoded['level'].map(level_mapping).values
    
    return X, y, zone_mapping, level_mapping


def generate_simulated_data(data_type='air', num_records=100):
    """
    Generate simulated real-time pollution data for demonstration.
    
    Args:
        data_type (str): Type of pollution data ('air', 'water', 'noise')
        num_records (int): Number of records to generate
        
    Returns:
        pandas.DataFrame: Simulated pollution data
    """
    np.random.seed(42)
    dates = pd.date_range(start='2026-01-01', periods=num_records, freq='D')
    
    if data_type == 'air':
        pm25 = np.random.uniform(20, 150, num_records)
        pm10 = pm25 * 1.5 + np.random.uniform(-10, 10, num_records)
        co2 = 380 + pm25 + np.random.uniform(-20, 20, num_records)
        
        return pd.DataFrame({
            'date': dates,
            'pm25': pm25.round(1),
            'pm10': pm10.clip(0, 600).round(1),
            'co2': co2.clip(300, 1000).round(1)
        })
    
    elif data_type == 'water':
        ph = np.random.uniform(5.5, 8.5, num_records)
        turbidity = np.random.uniform(1, 25, num_records)
        do = 10 - (turbidity / 5) + np.random.uniform(-1, 1, num_records)
        
        return pd.DataFrame({
            'date': dates,
            'ph': ph.round(2),
            'turbidity': turbidity.round(1),
            'dissolved_oxygen': do.clip(3, 12).round(1)
        })
    
    elif data_type == 'noise':
        zones = np.random.choice(['Residential', 'Commercial', 'Industrial'], num_records)
        base_levels = {'Residential': 45, 'Commercial': 60, 'Industrial': 75}
        sound_levels = [base_levels[z] + np.random.uniform(-10, 25) for z in zones]
        
        return pd.DataFrame({
            'date': dates,
            'zone': zones,
            'sound_level': np.array(sound_levels).clip(30, 110).round(1)
        })
    
    return pd.DataFrame()
