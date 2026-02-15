"""
app.py - Main Flask Application for Pollution Monitoring System
=================================================================
Integrated Pollution Monitoring and Control System for Metro Cities
MCA Final Year Project

This application provides:
- Real-time pollution monitoring dashboard
- Air, Water, and Noise pollution analysis
- ML-based pollution prediction
- Alert generation and recommendations
- User authentication and admin panel
- Report generation and export

Author: MCA Student
Version: 2.0.0
"""

import os
import sys
import json
import pickle
import sqlite3
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, g, redirect, url_for, session, Response, flash

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import load_csv_data, clean_air_data, clean_water_data, clean_noise_data
from utils.aqi_calculator import calculate_combined_aqi, classify_aqi, get_health_recommendations
from utils.alert_engine import AlertEngine, generate_control_recommendations

# Import new services
try:
    from services.weather_api import weather_service
    from services.email_service import email_service
    from services.report_generator import report_generator
    from services.government_api import government_api
    from config import config
except ImportError as e:
    print(f"Warning: Could not import some services: {e}")
    weather_service = None
    email_service = None
    report_generator = None
    government_api = None
    config = None

# Try to import LSTM forecasting
try:
    from ml.train_lstm import load_lstm_model, forecast_aqi
    lstm_model, lstm_scaler, lstm_metrics = load_lstm_model()
    if lstm_model:
        print("LSTM model loaded successfully")
    else:
        print("Warning: LSTM model not found. Run train_lstm.py first.")
        lstm_model = None
        lstm_scaler = None
except Exception as e:
    print(f"Warning: Could not load LSTM model: {e}")
    lstm_model = None
    lstm_scaler = None
    lstm_metrics = None

# ============================================
# Flask Application Configuration
# ============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pollution_monitoring_secret_key_2026'

# Database path
DATABASE = os.path.join(os.path.dirname(__file__), 'database.db')

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Global alert engine
alert_engine = AlertEngine()

# Loaded models (will be initialized on startup)
models = {
    'air': None,
    'water': None,
    'noise': None
}


# ============================================
# Database Functions
# ============================================

def get_db():
    """Get database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close database connection on app teardown."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize database with required tables including city/state."""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    # Drop existing tables for fresh schema with city/state
    cursor.execute('DROP TABLE IF EXISTS air_data')
    cursor.execute('DROP TABLE IF EXISTS water_data')
    cursor.execute('DROP TABLE IF EXISTS noise_data')
    
    # Create air_data table with city/state
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS air_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            city TEXT NOT NULL DEFAULT 'Mumbai',
            state TEXT NOT NULL DEFAULT 'Maharashtra',
            pm25 REAL NOT NULL,
            pm10 REAL NOT NULL,
            co2 REAL NOT NULL,
            aqi INTEGER,
            level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create water_data table with city/state
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS water_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            city TEXT NOT NULL DEFAULT 'Mumbai',
            state TEXT NOT NULL DEFAULT 'Maharashtra',
            ph REAL NOT NULL,
            turbidity REAL NOT NULL,
            dissolved_oxygen REAL NOT NULL,
            quality TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create noise_data table with city/state
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS noise_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            city TEXT NOT NULL DEFAULT 'Mumbai',
            state TEXT NOT NULL DEFAULT 'Maharashtra',
            zone TEXT NOT NULL,
            sound_level REAL NOT NULL,
            level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create prediction_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pollution_type TEXT NOT NULL,
            input_params TEXT NOT NULL,
            predicted_value TEXT NOT NULL,
            predicted_level TEXT,
            city TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            recommendations TEXT,
            city TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            city TEXT DEFAULT 'Mumbai',
            notifications INTEGER DEFAULT 1,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user if not exists
    admin_exists = cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',)).fetchone()
    if not admin_exists:
        import hashlib
        admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, email, password, first_name, last_name, is_admin)
            VALUES (?, ?, ?, ?, ?, 1)
        ''', ('admin', 'admin@pollutionmonitor.com', admin_password, 'System', 'Admin'))
        print("Default admin user created (username: admin, password: admin123)")
    
    db.commit()
    db.close()
    print("Database initialized with city/state support!")


def load_csv_to_db():
    """Load CSV data with city/state into database tables."""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    # Load air data with city/state
    air_path = os.path.join(DATA_DIR, 'air_data.csv')
    if os.path.exists(air_path):
        df = pd.read_csv(air_path)
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO air_data (date, city, state, pm25, pm10, co2, aqi, level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['date'], row.get('city', 'Mumbai'), row.get('state', 'Maharashtra'),
                  row['pm25'], row['pm10'], row['co2'], 
                  row.get('aqi', 0), row.get('level', '')))
        print(f"Loaded {len(df)} air records")
    
    # Load water data with city/state
    water_path = os.path.join(DATA_DIR, 'water_data.csv')
    if os.path.exists(water_path):
        df = pd.read_csv(water_path)
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO water_data (date, city, state, ph, turbidity, dissolved_oxygen, quality)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['date'], row.get('city', 'Mumbai'), row.get('state', 'Maharashtra'),
                  row['ph'], row['turbidity'], 
                  row['dissolved_oxygen'], row.get('quality', '')))
        print(f"Loaded {len(df)} water records")
    
    # Load noise data with city/state
    noise_path = os.path.join(DATA_DIR, 'noise_data.csv')
    if os.path.exists(noise_path):
        df = pd.read_csv(noise_path)
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO noise_data (date, city, state, zone, sound_level, level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (row['date'], row.get('city', 'Mumbai'), row.get('state', 'Maharashtra'),
                  row['zone'], row['sound_level'], row.get('level', '')))
        print(f"Loaded {len(df)} noise records")
    
    db.commit()
    db.close()


def load_models():
    """Load trained ML models from pickle files."""
    global models
    
    # Load air model
    air_model_path = os.path.join(MODELS_DIR, 'air_model.pkl')
    if os.path.exists(air_model_path):
        with open(air_model_path, 'rb') as f:
            models['air'] = pickle.load(f)
        print("Air model loaded successfully")
    else:
        print("Warning: Air model not found. Run train_air.py first.")
    
    # Load water model
    water_model_path = os.path.join(MODELS_DIR, 'water_model.pkl')
    if os.path.exists(water_model_path):
        with open(water_model_path, 'rb') as f:
            models['water'] = pickle.load(f)
        print("Water model loaded successfully")
    else:
        print("Warning: Water model not found. Run train_water.py first.")
    
    # Load noise model
    noise_model_path = os.path.join(MODELS_DIR, 'noise_model.pkl')
    if os.path.exists(noise_model_path):
        with open(noise_model_path, 'rb') as f:
            models['noise'] = pickle.load(f)
        print("Noise model loaded successfully")
    else:
        print("Warning: Noise model not found. Run train_noise.py first.")


# ============================================
# Route Handlers
# ============================================

@app.route('/')
def dashboard():
    """Main dashboard page with summary of all pollution types."""
    db = get_db()
    
    # Get selected city from query params
    selected_city = request.args.get('city', 'Mumbai')
    
    # Get list of available cities
    cities = db.execute('SELECT DISTINCT city, state FROM air_data ORDER BY city').fetchall()
    cities_list = [{'city': row['city'], 'state': row['state']} for row in cities]
    
    # Get latest air data for selected city
    air_data = db.execute(
        'SELECT * FROM air_data WHERE city = ? ORDER BY date DESC LIMIT 10',
        (selected_city,)
    ).fetchall()
    
    # Get latest water data for selected city
    water_data = db.execute(
        'SELECT * FROM water_data WHERE city = ? ORDER BY date DESC LIMIT 10',
        (selected_city,)
    ).fetchall()
    
    # Get latest noise data for selected city
    noise_data = db.execute(
        'SELECT * FROM noise_data WHERE city = ? ORDER BY date DESC LIMIT 10',
        (selected_city,)
    ).fetchall()
    
    # Calculate current status
    latest_air = air_data[0] if air_data else None
    latest_water = water_data[0] if water_data else None
    latest_noise = noise_data[0] if noise_data else None
    
    # Generate alerts
    alert_engine.clear_alerts()
    
    if latest_air:
        aqi = latest_air['aqi'] if latest_air['aqi'] else calculate_combined_aqi(
            latest_air['pm25'], latest_air['pm10'], latest_air['co2']
        )
        alert_engine.check_air_quality(aqi, latest_air['pm25'], latest_air['pm10'], latest_air['co2'])
    
    if latest_water:
        alert_engine.check_water_quality(
            latest_water['ph'], latest_water['turbidity'], latest_water['dissolved_oxygen']
        )
    
    if latest_noise:
        alert_engine.check_noise_levels(latest_noise['sound_level'], latest_noise['zone'])
    
    alerts = alert_engine.get_all_alerts()
    alert_summary = alert_engine.get_alert_summary()
    
    # Prepare chart data
    air_chart_data = []
    for row in reversed(air_data):
        air_chart_data.append({
            'date': row['date'],
            'aqi': row['aqi'] if row['aqi'] else 0,
            'pm25': row['pm25'],
            'pm10': row['pm10']
        })
    
    water_chart_data = []
    for row in reversed(water_data):
        water_chart_data.append({
            'date': row['date'],
            'ph': row['ph'],
            'turbidity': row['turbidity'],
            'dissolved_oxygen': row['dissolved_oxygen']
        })
    
    noise_chart_data = []
    for row in reversed(noise_data):
        noise_chart_data.append({
            'date': row['date'],
            'sound_level': row['sound_level'],
            'zone': row['zone']
        })
    
    return render_template('dashboard.html',
                         latest_air=latest_air,
                         latest_water=latest_water,
                         latest_noise=latest_noise,
                         alerts=alerts,
                         alert_summary=alert_summary,
                         air_chart_data=json.dumps(air_chart_data),
                         water_chart_data=json.dumps(water_chart_data),
                         noise_chart_data=json.dumps(noise_chart_data),
                         cities=cities_list,
                         selected_city=selected_city)


@app.route('/air')
def air_pollution():
    """Air pollution analysis page."""
    db = get_db()
    
    # Get selected city from query params
    selected_city = request.args.get('city', 'Mumbai')
    
    # Get list of available cities
    cities = db.execute('SELECT DISTINCT city, state FROM air_data ORDER BY city').fetchall()
    cities_list = [{'city': row['city'], 'state': row['state']} for row in cities]
    
    # Get all air data for selected city
    air_data = db.execute(
        'SELECT * FROM air_data WHERE city = ? ORDER BY date DESC',
        (selected_city,)
    ).fetchall()
    
    # Get prediction results for air
    predictions = db.execute(
        "SELECT * FROM prediction_results WHERE pollution_type='air' ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    
    # Prepare historical data for chart
    historical_data = []
    predicted_data = []
    
    for row in reversed(list(air_data)):
        aqi = row['aqi'] if row['aqi'] else calculate_combined_aqi(row['pm25'], row['pm10'], row['co2'])
        historical_data.append({
            'date': row['date'],
            'aqi': aqi,
            'pm25': row['pm25'],
            'pm10': row['pm10'],
            'co2': row['co2'],
            'city': row['city'],
            'level': row['level'] if row['level'] else classify_aqi(aqi)[0]
        })
    
    # Generate future predictions using the model
    if models['air'] is not None and len(air_data) > 0:
        latest = air_data[0]
        for i in range(1, 8):  # Predict next 7 days
            # Simple trend simulation (in real scenario, use time series)
            pm25 = latest['pm25'] + np.random.uniform(-10, 15)
            pm10 = latest['pm10'] + np.random.uniform(-15, 20)
            co2 = latest['co2'] + np.random.uniform(-20, 25)
            
            pm25 = max(10, min(pm25, 300))
            pm10 = max(20, min(pm10, 400))
            co2 = max(350, min(co2, 800))
            
            features = np.array([[pm25, pm10, co2]])
            pred_aqi = int(models['air'].predict(features)[0])
            
            future_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            predicted_data.append({
                'date': future_date,
                'aqi': pred_aqi,
                'pm25': round(pm25, 1),
                'pm10': round(pm10, 1),
                'co2': round(co2, 1),
                'level': classify_aqi(pred_aqi)[0]
            })
    
    # Latest readings
    latest = air_data[0] if air_data else None
    latest_aqi = None
    aqi_info = None
    recommendations = []
    
    if latest:
        latest_aqi = latest['aqi'] if latest['aqi'] else calculate_combined_aqi(
            latest['pm25'], latest['pm10'], latest['co2']
        )
        aqi_info = classify_aqi(latest_aqi)
        recommendations = get_health_recommendations(latest_aqi)
    
    return render_template('air.html',
                         air_data=air_data[:30],
                         latest=latest,
                         latest_aqi=latest_aqi,
                         aqi_info=aqi_info,
                         recommendations=recommendations,
                         historical_data=json.dumps(historical_data[-30:]),
                         predicted_data=json.dumps(predicted_data),
                         cities=cities_list,
                         selected_city=selected_city)


@app.route('/water')
def water_pollution():
    """Water pollution analysis page."""
    db = get_db()
    
    # Get selected city from query params
    selected_city = request.args.get('city', 'Mumbai')
    
    # Get list of available cities
    cities = db.execute('SELECT DISTINCT city, state FROM water_data ORDER BY city').fetchall()
    cities_list = [{'city': row['city'], 'state': row['state']} for row in cities]
    
    # Get all water data for selected city
    water_data = db.execute(
        'SELECT * FROM water_data WHERE city = ? ORDER BY date DESC',
        (selected_city,)
    ).fetchall()
    
    # Prepare data for charts
    chart_data = []
    for row in reversed(list(water_data)):
        chart_data.append({
            'date': row['date'],
            'ph': row['ph'],
            'turbidity': row['turbidity'],
            'dissolved_oxygen': row['dissolved_oxygen'],
            'quality': row['quality'],
            'city': row['city']
        })
    
    # Latest readings
    latest = water_data[0] if water_data else None
    quality_info = None
    predictions = []
    
    if latest:
        quality = latest['quality'] if latest['quality'] else 'Unknown'
        color_map = {'Safe': '#28a745', 'Moderate': '#ffc107', 'Polluted': '#dc3545'}
        quality_info = {
            'label': quality,
            'color': color_map.get(quality, '#6c757d')
        }
    
    # Generate predictions if model available
    if models['water'] is not None and latest:
        for i in range(1, 8):
            # Simulate future values
            ph = latest['ph'] + np.random.uniform(-0.3, 0.3)
            turbidity = latest['turbidity'] + np.random.uniform(-2, 3)
            do = latest['dissolved_oxygen'] + np.random.uniform(-0.5, 0.5)
            
            ph = max(5.0, min(ph, 9.5))
            turbidity = max(0.5, min(turbidity, 30))
            do = max(3, min(do, 12))
            
            model = models['water']['model']
            scaler = models['water']['scaler']
            mapping = models['water']['quality_mapping']
            reverse_mapping = {v: k for k, v in mapping.items()}
            
            features = scaler.transform([[ph, turbidity, do]])
            pred_class = model.predict(features)[0]
            pred_quality = reverse_mapping[pred_class]
            
            future_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            predictions.append({
                'date': future_date,
                'ph': round(ph, 2),
                'turbidity': round(turbidity, 1),
                'dissolved_oxygen': round(do, 1),
                'quality': pred_quality
            })
    
    return render_template('water.html',
                         water_data=water_data[:30],
                         latest=latest,
                         quality_info=quality_info,
                         chart_data=json.dumps(chart_data[-30:]),
                         predictions=json.dumps(predictions),
                         cities=cities_list,
                         selected_city=selected_city)


@app.route('/noise')
def noise_pollution():
    """Noise pollution analysis page."""
    db = get_db()
    
    # Get selected city from query params
    selected_city = request.args.get('city', 'Mumbai')
    
    # Get list of available cities
    cities = db.execute('SELECT DISTINCT city, state FROM noise_data ORDER BY city').fetchall()
    cities_list = [{'city': row['city'], 'state': row['state']} for row in cities]
    
    # Get all noise data for selected city
    noise_data = db.execute(
        'SELECT * FROM noise_data WHERE city = ? ORDER BY date DESC',
        (selected_city,)
    ).fetchall()
    
    # Prepare data for charts
    chart_data = []
    zone_summary = {'Residential': [], 'Commercial': [], 'Industrial': []}
    
    for row in reversed(list(noise_data)):
        chart_data.append({
            'date': row['date'],
            'sound_level': row['sound_level'],
            'zone': row['zone'],
            'level': row['level'],
            'city': row['city']
        })
        if row['zone'] in zone_summary:
            zone_summary[row['zone']].append(row['sound_level'])
    
    # Calculate zone averages
    zone_averages = {}
    for zone, levels in zone_summary.items():
        if levels:
            zone_averages[zone] = {
                'avg': round(np.mean(levels), 1),
                'max': round(max(levels), 1),
                'min': round(min(levels), 1)
            }
    
    # Latest readings
    latest = noise_data[0] if noise_data else None
    level_info = None
    
    if latest:
        level = latest['level'] if latest['level'] else 'Unknown'
        color_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        level_info = {
            'label': level,
            'color': color_map.get(level, '#6c757d'),
            'db': latest['sound_level']
        }
    
    # Zone risk thresholds
    zone_limits = {
        'Residential': {'day': 55, 'night': 45},
        'Commercial': {'day': 65, 'night': 55},
        'Industrial': {'day': 75, 'night': 70}
    }
    
    return render_template('noise.html',
                         noise_data=noise_data[:30],
                         latest=latest,
                         level_info=level_info,
                         zone_averages=zone_averages,
                         zone_limits=zone_limits,
                         chart_data=json.dumps(chart_data[-30:]),
                         cities=cities_list,
                         selected_city=selected_city)


# ============================================
# API Endpoints
# ============================================

@app.route('/api/predict/air', methods=['POST'])
def predict_air():
    """API endpoint for air quality prediction."""
    data = request.get_json()
    
    pm25 = float(data.get('pm25', 0))
    pm10 = float(data.get('pm10', 0))
    co2 = float(data.get('co2', 0))
    
    if models['air'] is None:
        # Fallback to formula-based calculation
        aqi = calculate_combined_aqi(pm25, pm10, co2)
    else:
        features = np.array([[pm25, pm10, co2]])
        aqi = int(models['air'].predict(features)[0])
    
    classification, color, description = classify_aqi(aqi)
    recommendations = get_health_recommendations(aqi)
    
    # Save prediction to database
    db = get_db()
    db.execute('''
        INSERT INTO prediction_results (pollution_type, input_params, predicted_value, predicted_level)
        VALUES (?, ?, ?, ?)
    ''', ('air', json.dumps({'pm25': pm25, 'pm10': pm10, 'co2': co2}), str(aqi), classification))
    db.commit()
    
    # Check for alerts
    alert_engine.clear_alerts()
    alerts = alert_engine.check_air_quality(aqi, pm25, pm10, co2)
    
    return jsonify({
        'success': True,
        'aqi': aqi,
        'classification': classification,
        'color': color,
        'description': description,
        'recommendations': recommendations,
        'alerts': alerts
    })


@app.route('/api/predict/water', methods=['POST'])
def predict_water():
    """API endpoint for water quality prediction."""
    data = request.get_json()
    
    ph = float(data.get('ph', 7.0))
    turbidity = float(data.get('turbidity', 0))
    dissolved_oxygen = float(data.get('dissolved_oxygen', 0))
    
    if models['water'] is None:
        # Fallback to rule-based classification
        if ph < 6.5 or ph > 8.5 or turbidity > 10 or dissolved_oxygen < 5:
            quality = 'Polluted'
        elif turbidity > 5 or dissolved_oxygen < 7:
            quality = 'Moderate'
        else:
            quality = 'Safe'
        probabilities = {}
    else:
        model = models['water']['model']
        scaler = models['water']['scaler']
        mapping = models['water']['quality_mapping']
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        features = scaler.transform([[ph, turbidity, dissolved_oxygen]])
        pred_class = model.predict(features)[0]
        quality = reverse_mapping[pred_class]
        probs = model.predict_proba(features)[0]
        probabilities = {reverse_mapping[i]: round(p * 100, 1) for i, p in enumerate(probs)}
    
    color_map = {'Safe': '#28a745', 'Moderate': '#ffc107', 'Polluted': '#dc3545'}
    
    # Save prediction
    db = get_db()
    db.execute('''
        INSERT INTO prediction_results (pollution_type, input_params, predicted_value, predicted_level)
        VALUES (?, ?, ?, ?)
    ''', ('water', json.dumps({'ph': ph, 'turbidity': turbidity, 'dissolved_oxygen': dissolved_oxygen}), quality, quality))
    db.commit()
    
    # Check for alerts
    alert_engine.clear_alerts()
    alerts = alert_engine.check_water_quality(ph, turbidity, dissolved_oxygen)
    
    return jsonify({
        'success': True,
        'quality': quality,
        'color': color_map.get(quality, '#6c757d'),
        'probabilities': probabilities,
        'alerts': alerts
    })


@app.route('/api/predict/noise', methods=['POST'])
def predict_noise():
    """API endpoint for noise level prediction."""
    data = request.get_json()
    
    sound_level = float(data.get('sound_level', 0))
    zone = data.get('zone', 'General')
    
    if models['noise'] is None:
        # Fallback to rule-based classification
        if sound_level < 55:
            level = 'Low'
        elif sound_level < 75:
            level = 'Medium'
        else:
            level = 'High'
        probabilities = {}
    else:
        model = models['noise']['model']
        zone_mapping = models['noise']['zone_mapping']
        level_mapping = models['noise']['level_mapping']
        reverse_level = {v: k for k, v in level_mapping.items()}
        
        zone_encoded = zone_mapping.get(zone, 0)
        features = np.array([[sound_level, zone_encoded]])
        pred_class = model.predict(features)[0]
        level = reverse_level[pred_class]
        probs = model.predict_proba(features)[0]
        probabilities = {reverse_level[i]: round(p * 100, 1) for i, p in enumerate(probs)}
    
    color_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    
    # Save prediction
    db = get_db()
    db.execute('''
        INSERT INTO prediction_results (pollution_type, input_params, predicted_value, predicted_level)
        VALUES (?, ?, ?, ?)
    ''', ('noise', json.dumps({'sound_level': sound_level, 'zone': zone}), str(sound_level), level))
    db.commit()
    
    # Check for alerts
    alert_engine.clear_alerts()
    alerts = alert_engine.check_noise_levels(sound_level, zone)
    
    return jsonify({
        'success': True,
        'level': level,
        'sound_level': sound_level,
        'zone': zone,
        'color': color_map.get(level, '#6c757d'),
        'probabilities': probabilities,
        'alerts': alerts
    })


@app.route('/api/alerts')
def get_alerts():
    """Get current alerts."""
    return jsonify({
        'alerts': alert_engine.get_all_alerts(),
        'summary': alert_engine.get_alert_summary()
    })


@app.route('/api/cities')
def get_cities():
    """Get list of available cities."""
    db = get_db()
    cities = db.execute('SELECT DISTINCT city, state FROM air_data ORDER BY city').fetchall()
    return jsonify({
        'cities': [{'city': row['city'], 'state': row['state']} for row in cities]
    })


@app.route('/api/stats')
def get_stats():
    """Get overall pollution statistics."""
    db = get_db()
    
    # Air stats
    air_stats = db.execute('''
        SELECT 
            AVG(aqi) as avg_aqi,
            MAX(aqi) as max_aqi,
            MIN(aqi) as min_aqi,
            AVG(pm25) as avg_pm25,
            AVG(pm10) as avg_pm10
        FROM air_data
    ''').fetchone()
    
    # Water stats
    water_stats = db.execute('''
        SELECT 
            AVG(ph) as avg_ph,
            AVG(turbidity) as avg_turbidity,
            AVG(dissolved_oxygen) as avg_do
        FROM water_data
    ''').fetchone()
    
    # Noise stats
    noise_stats = db.execute('''
        SELECT 
            AVG(sound_level) as avg_level,
            MAX(sound_level) as max_level,
            MIN(sound_level) as min_level
        FROM noise_data
    ''').fetchone()
    
    return jsonify({
        'air': {
            'avg_aqi': round(air_stats['avg_aqi'] or 0, 1),
            'max_aqi': air_stats['max_aqi'] or 0,
            'min_aqi': air_stats['min_aqi'] or 0,
            'avg_pm25': round(air_stats['avg_pm25'] or 0, 1),
            'avg_pm10': round(air_stats['avg_pm10'] or 0, 1)
        },
        'water': {
            'avg_ph': round(water_stats['avg_ph'] or 0, 2),
            'avg_turbidity': round(water_stats['avg_turbidity'] or 0, 1),
            'avg_do': round(water_stats['avg_do'] or 0, 1)
        },
        'noise': {
            'avg_level': round(noise_stats['avg_level'] or 0, 1),
            'max_level': noise_stats['max_level'] or 0,
            'min_level': noise_stats['min_level'] or 0
        }
    })


# ============================================
# Real-Time API Endpoints (Government Data)
# ============================================

@app.route('/api/realtime/air')
def realtime_air():
    """Get real-time air quality data from government APIs."""
    city = request.args.get('city', 'Mumbai')
    
    if government_api:
        data = government_api.get_city_aqi(city)
        return jsonify({
            'success': True,
            'data': data
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Government API service not available'
        })


@app.route('/api/realtime/all')
def realtime_all_cities():
    """Get real-time air quality data for all cities."""
    if government_api:
        data = government_api.get_all_cities()
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Government API service not available'
        })


@app.route('/api/compare')
def api_compare_cities():
    """API endpoint for comparing multiple cities (real-time data)."""
    cities_param = request.args.get('cities', '')
    cities = [c.strip() for c in cities_param.split(',') if c.strip()]
    
    if len(cities) < 2:
        return jsonify({
            'success': False,
            'error': 'Select at least 2 cities to compare'
        })
    
    if government_api:
        comparison_data = government_api.get_comparison_data(cities)
        # Sort by AQI (best to worst)
        comparison_data.sort(key=lambda x: x.get('aqi', 999))
        
        return jsonify({
            'success': True,
            'comparison_data': comparison_data
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Government API service not available'
        })


@app.route('/api/available-cities')
def api_available_cities():
    """Get list of available cities for real-time data."""
    if government_api:
        cities = government_api.get_available_cities()
        return jsonify({
            'success': True,
            'cities': cities
        })
    else:
        # Fallback to database cities
        db = get_db()
        cities = db.execute('SELECT DISTINCT city FROM air_data ORDER BY city').fetchall()
        return jsonify({
            'success': True,
            'cities': [row['city'] for row in cities]
        })


# ============================================
# LSTM Forecasting Endpoints
# ============================================

@app.route('/api/forecast/lstm')
def lstm_forecast():
    """Get LSTM-based AQI forecast for the next 7 days."""
    city = request.args.get('city', 'Mumbai')
    days = int(request.args.get('days', 7))
    
    if lstm_model is None or lstm_scaler is None:
        # Fallback to simple prediction if LSTM not available
        return jsonify({
            'success': False,
            'error': 'LSTM model not trained. Run: python ml/train_lstm.py',
            'fallback': True,
            'forecasts': _get_fallback_forecast(city, days)
        })
    
    # Get recent AQI data for the city
    db = get_db()
    recent = db.execute(
        'SELECT aqi FROM air_data WHERE city = ? ORDER BY date DESC LIMIT 7',
        (city,)
    ).fetchall()
    
    if len(recent) < 7:
        return jsonify({
            'success': False,
            'error': 'Not enough historical data for forecast',
            'fallback': True,
            'forecasts': _get_fallback_forecast(city, days)
        })
    
    # Get AQI values (reverse to chronological order)
    recent_aqi = [row['aqi'] for row in reversed(recent)]
    
    # Generate forecast
    try:
        forecasts = forecast_aqi(lstm_model, lstm_scaler, recent_aqi, days)
        
        # Add classification to each forecast
        for f in forecasts:
            classification, color, description = classify_aqi(f['aqi'])
            f['level'] = classification
            f['color'] = color
            f['description'] = description
        
        return jsonify({
            'success': True,
            'city': city,
            'model': 'LSTM Neural Network',
            'forecasts': forecasts,
            'metrics': lstm_metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback': True,
            'forecasts': _get_fallback_forecast(city, days)
        })


def _get_fallback_forecast(city, days):
    """Generate simple fallback forecast when LSTM not available."""
    import random
    from datetime import datetime, timedelta
    
    db = get_db()
    latest = db.execute(
        'SELECT aqi FROM air_data WHERE city = ? ORDER BY date DESC LIMIT 1',
        (city,)
    ).fetchone()
    
    base_aqi = latest['aqi'] if latest else 100
    forecasts = []
    
    for i in range(days):
        # Simple random walk
        aqi = max(20, min(300, base_aqi + random.randint(-15, 20)))
        forecast_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        classification, color, description = classify_aqi(aqi)
        
        forecasts.append({
            'date': forecast_date,
            'day': i + 1,
            'aqi': aqi,
            'level': classification,
            'color': color,
            'description': description
        })
        base_aqi = aqi  # Use for next iteration
    
    return forecasts


# ============================================
# Authentication Helpers
# ============================================

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin access."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if not session.get('is_admin', False):
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================
# Authentication Routes
# ============================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    error = None
    success = request.args.get('registered')
    
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, username)
        ).fetchone()
        
        if user and user['password'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            session['is_admin'] = user['is_admin'] == 1
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    
    return render_template('auth/login.html', error=error, success='Account created! Please log in.' if success else None)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    error = None
    
    if request.method == 'POST':
        first_name = request.form.get('first_name', '')
        last_name = request.form.get('last_name', '')
        email = request.form.get('email', '')
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        city = request.form.get('city', 'Mumbai')
        notifications = request.form.get('notifications') == 'on'
        
        if password != confirm_password:
            error = 'Passwords do not match'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters'
        else:
            db = get_db()
            existing = db.execute(
                'SELECT id FROM users WHERE username = ? OR email = ?',
                (username, email)
            ).fetchone()
            
            if existing:
                error = 'Username or email already exists'
            else:
                db.execute('''
                    INSERT INTO users (username, email, password, first_name, last_name, city, notifications, is_admin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ''', (username, email, hash_password(password), first_name, last_name, city, 1 if notifications else 0))
                db.commit()
                return redirect(url_for('login', registered=1))
    
    return render_template('auth/register.html', error=error)


@app.route('/logout')
def logout():
    """User logout."""
    session.clear()
    return redirect(url_for('dashboard'))


# ============================================
# Admin Routes
# ============================================

@app.route('/admin/settings', methods=['GET', 'POST'])
def admin_settings():
    """Admin settings page for threshold configuration."""
    success = None
    
    # Get current thresholds from AlertEngine
    thresholds = AlertEngine.THRESHOLDS
    
    # Default settings
    settings = {
        'email_alerts': True,
        'browser_notifications': True,
        'daily_reports': False,
        'realtime_data': True
    }
    
    if request.method == 'POST':
        # Update thresholds (in-memory only for demo, could persist to DB)
        AlertEngine.THRESHOLDS['aqi']['warning'] = int(request.form.get('aqi_warning', 100))
        AlertEngine.THRESHOLDS['aqi']['danger'] = int(request.form.get('aqi_danger', 150))
        AlertEngine.THRESHOLDS['aqi']['critical'] = int(request.form.get('aqi_critical', 200))
        
        success = 'Settings saved successfully!'
    
    return render_template('admin/settings.html',
                         thresholds=thresholds,
                         settings=settings,
                         success=success,
                         current_user={'username': session.get('username', 'Admin')})


# ============================================
# Reports Routes
# ============================================

@app.route('/reports')
def reports():
    """Reports generation page."""
    db = get_db()
    cities = db.execute('SELECT DISTINCT city, state FROM air_data ORDER BY city').fetchall()
    cities_list = [{'city': row['city'], 'state': row['state']} for row in cities]
    
    return render_template('reports.html', cities=cities_list)


@app.route('/download-report', methods=['POST'])
def download_report():
    """Generate and download report."""
    if not report_generator:
        return jsonify({'error': 'Report generator not available'}), 500
    
    report_type = request.form.get('report_type', 'air')
    city = request.form.get('city', '')
    date_range = request.form.get('date_range', '7')
    format_type = request.form.get('format', 'pdf')
    
    db = get_db()
    
    # Build query based on report type
    if report_type == 'air':
        query = 'SELECT * FROM air_data'
        title = 'Air Quality Report'
    elif report_type == 'water':
        query = 'SELECT * FROM water_data'
        title = 'Water Quality Report'
    elif report_type == 'noise':
        query = 'SELECT * FROM noise_data'
        title = 'Noise Level Report'
    else:
        query = 'SELECT * FROM air_data'
        title = 'Combined Pollution Report'
    
    # Add filters
    conditions = []
    params = []
    
    if city:
        conditions.append('city = ?')
        params.append(city)
        title += f' - {city}'
    
    if date_range != 'all':
        days = int(date_range)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        conditions.append('date >= ?')
        params.append(cutoff_date)
    
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    
    query += ' ORDER BY date DESC'
    data = db.execute(query, params).fetchall()
    
    # Convert to list of dicts
    columns = data[0].keys() if data else []
    data_list = [{col: row[col] for col in columns} for row in data]
    
    if format_type == 'csv':
        csv_content = report_generator.generate_csv_report(report_type, data_list, city)
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={report_type}_report.csv'}
        )
    else:
        pdf_content = report_generator.generate_pdf_report(title, data_list, report_type, city)
        mimetype = 'text/html' if not report_generator.pdf_available else 'application/pdf'
        filename = f'{report_type}_report.{"html" if not report_generator.pdf_available else "pdf"}'
        return Response(
            pdf_content,
            mimetype=mimetype,
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )


@app.route('/compare-cities')
def compare_cities():
    """Compare pollution data across multiple cities."""
    cities = request.args.getlist('cities')
    
    if not cities or len(cities) < 2:
        flash('Please select at least 2 cities to compare', 'warning')
        return redirect(url_for('reports'))
    
    db = get_db()
    
    comparison_data = {}
    for city in cities:
        air_data = db.execute(
            'SELECT AVG(aqi) as avg_aqi, AVG(pm25) as avg_pm25, AVG(pm10) as avg_pm10 FROM air_data WHERE city = ?',
            (city,)
        ).fetchone()
        
        water_data = db.execute(
            'SELECT AVG(ph) as avg_ph, AVG(turbidity) as avg_turbidity FROM water_data WHERE city = ?',
            (city,)
        ).fetchone()
        
        noise_data = db.execute(
            'SELECT AVG(sound_level) as avg_noise FROM noise_data WHERE city = ?',
            (city,)
        ).fetchone()
        
        comparison_data[city] = {
            'air': {
                'avg_aqi': round(air_data['avg_aqi'] or 0, 1),
                'avg_pm25': round(air_data['avg_pm25'] or 0, 1),
                'avg_pm10': round(air_data['avg_pm10'] or 0, 1)
            },
            'water': {
                'avg_ph': round(water_data['avg_ph'] or 0, 2),
                'avg_turbidity': round(water_data['avg_turbidity'] or 0, 1)
            },
            'noise': {
                'avg_level': round(noise_data['avg_noise'] or 0, 1)
            }
        }
    
    return jsonify(comparison_data)


@app.route('/compare')
def compare_page():
    """City comparison page with visual charts."""
    db = get_db()
    
    # Get selected cities from query params
    selected_cities = request.args.getlist('cities')
    
    # Get list of available cities
    cities = db.execute('SELECT DISTINCT city, state FROM air_data ORDER BY city').fetchall()
    cities_list = [{'city': row['city'], 'state': row['state']} for row in cities]
    
    comparison_data = {}
    historical_trends = {}
    
    if len(selected_cities) >= 2:
        # Get comparison data for selected cities
        for city in selected_cities:
            air_data = db.execute(
                'SELECT AVG(aqi) as avg_aqi, AVG(pm25) as avg_pm25, AVG(pm10) as avg_pm10 FROM air_data WHERE city = ?',
                (city,)
            ).fetchone()
            
            water_data = db.execute(
                'SELECT AVG(ph) as avg_ph, AVG(turbidity) as avg_turbidity, AVG(dissolved_oxygen) as avg_do FROM water_data WHERE city = ?',
                (city,)
            ).fetchone()
            
            noise_data = db.execute(
                'SELECT AVG(sound_level) as avg_noise FROM noise_data WHERE city = ?',
                (city,)
            ).fetchone()
            
            comparison_data[city] = {
                'air': {
                    'avg_aqi': round(air_data['avg_aqi'] or 0, 1),
                    'avg_pm25': round(air_data['avg_pm25'] or 0, 1),
                    'avg_pm10': round(air_data['avg_pm10'] or 0, 1)
                },
                'water': {
                    'avg_ph': round(water_data['avg_ph'] or 0, 2),
                    'avg_turbidity': round(water_data['avg_turbidity'] or 0, 1),
                    'avg_do': round(water_data['avg_do'] or 0, 1) if water_data['avg_do'] else 0
                },
                'noise': {
                    'avg_level': round(noise_data['avg_noise'] or 0, 1)
                }
            }
            
            # Get historical trend for each city
            trend_data = db.execute(
                'SELECT date, aqi FROM air_data WHERE city = ? ORDER BY date',
                (city,)
            ).fetchall()
            historical_trends[city] = [{'date': row['date'], 'aqi': row['aqi'] or 0} for row in trend_data]
        
        # Sort comparison data by AQI (best to worst)
        comparison_data = dict(sorted(comparison_data.items(), key=lambda x: x[1]['air']['avg_aqi']))
    
    return render_template('compare.html',
                         cities=cities_list,
                         selected_cities=selected_cities,
                         comparison_data=comparison_data,
                         historical_trends=historical_trends)


@app.route('/analytics')
def analytics():
    """Analytics dashboard with comprehensive pollution insights."""
    db = get_db()
    
    # Get total cities count
    cities_result = db.execute('SELECT COUNT(DISTINCT city) as count FROM air_data').fetchone()
    total_cities = cities_result['count'] if cities_result else 0
    
    # Get total records count
    air_count = db.execute('SELECT COUNT(*) as count FROM air_data').fetchone()['count']
    water_count = db.execute('SELECT COUNT(*) as count FROM water_data').fetchone()['count']
    noise_count = db.execute('SELECT COUNT(*) as count FROM noise_data').fetchone()['count']
    total_records = air_count + water_count + noise_count
    
    # Get high pollution days (AQI > 150)
    high_pollution = db.execute('SELECT COUNT(*) as count FROM air_data WHERE aqi > 150').fetchone()
    high_pollution_days = high_pollution['count'] if high_pollution else 0
    
    # Get predictions count
    predictions = db.execute('SELECT COUNT(*) as count FROM prediction_results').fetchone()
    predictions_made = predictions['count'] if predictions else 0
    
    # Get city rankings (best to worst by AQI)
    rankings = db.execute('''
        SELECT city, state, ROUND(AVG(aqi), 1) as avg_aqi 
        FROM air_data 
        GROUP BY city 
        ORDER BY avg_aqi ASC
    ''').fetchall()
    city_rankings = [{'city': r['city'], 'state': r['state'], 'avg_aqi': r['avg_aqi']} for r in rankings]
    
    # AQI distribution
    aqi_dist = {
        'good': db.execute('SELECT COUNT(*) as c FROM air_data WHERE aqi <= 50').fetchone()['c'],
        'moderate': db.execute('SELECT COUNT(*) as c FROM air_data WHERE aqi > 50 AND aqi <= 100').fetchone()['c'],
        'poor': db.execute('SELECT COUNT(*) as c FROM air_data WHERE aqi > 100 AND aqi <= 150').fetchone()['c'],
        'very_poor': db.execute('SELECT COUNT(*) as c FROM air_data WHERE aqi > 150 AND aqi <= 200').fetchone()['c'],
        'severe': db.execute('SELECT COUNT(*) as c FROM air_data WHERE aqi > 200').fetchone()['c']
    }
    
    # Weekly trend (last 7 days per city average)
    weekly_trend = db.execute('''
        SELECT date, ROUND(AVG(aqi), 1) as avg_aqi 
        FROM air_data 
        GROUP BY date 
        ORDER BY date
    ''').fetchall()
    weekly_trend_data = [{'date': r['date'], 'avg_aqi': r['avg_aqi']} for r in weekly_trend]
    
    # City AQI data for bar chart
    city_aqi = db.execute('''
        SELECT city, ROUND(AVG(aqi), 1) as avg_aqi 
        FROM air_data 
        GROUP BY city 
        ORDER BY avg_aqi DESC
    ''').fetchall()
    city_aqi_data = [{'city': r['city'], 'avg_aqi': r['avg_aqi']} for r in city_aqi]
    
    # Water quality distribution
    water_dist = {
        'safe': db.execute("SELECT COUNT(*) as c FROM water_data WHERE quality = 'Safe'").fetchone()['c'],
        'moderate': db.execute("SELECT COUNT(*) as c FROM water_data WHERE quality = 'Moderate'").fetchone()['c'],
        'polluted': db.execute("SELECT COUNT(*) as c FROM water_data WHERE quality = 'Polluted'").fetchone()['c']
    }
    
    # ML Prediction insights (simulated for each city)
    prediction_insights = []
    import random
    for city_data in city_rankings[:6]:  # Top 6 cities
        current_aqi = int(city_data['avg_aqi'])
        trend_change = random.randint(-25, 35)
        predicted_aqi = max(20, min(current_aqi + trend_change, 350))
        trend = 'up' if trend_change > 10 else ('down' if trend_change < -10 else 'stable')
        
        prediction_insights.append({
            'city': city_data['city'],
            'current_aqi': current_aqi,
            'predicted_aqi': predicted_aqi,
            'predicted_trend': trend,
            'confidence': random.randint(75, 95)
        })
    
    # Check if ML models are active
    ml_models_active = models['air'] is not None or models['water'] is not None or models['noise'] is not None
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('analytics.html',
                         total_cities=total_cities,
                         total_records=total_records,
                         high_pollution_days=high_pollution_days,
                         predictions_made=predictions_made,
                         city_rankings=city_rankings,
                         aqi_distribution=aqi_dist,
                         weekly_trend=weekly_trend_data,
                         city_aqi_data=city_aqi_data,
                         water_quality_data=water_dist,
                         prediction_insights=prediction_insights,
                         ml_models_active=ml_models_active,
                         current_time=current_time)


# ============================================
# Real-time API Routes
# ============================================

@app.route('/api/realtime/<city>')
def get_realtime_data(city):
    """Get real-time pollution data for a city."""
    if weather_service:
        data = weather_service.get_city_air_quality(city)
        return jsonify(data)
    else:
        # Use simulated data if service not available
        import random
        return jsonify({
            'pm25': round(35 + random.uniform(-15, 25), 1),
            'pm10': round(55 + random.uniform(-20, 35), 1),
            'aqi': int(75 + random.uniform(-25, 50)),
            'aqi_level': 'Moderate',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Simulated',
            'city': city
        })


@app.route('/api/location-pollution')
def get_location_pollution():
    """Get real-time pollution data based on user's GPS coordinates."""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({
            'success': False,
            'error': 'Missing latitude or longitude parameters'
        }), 400
    
    # Validate coordinate ranges
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({
            'success': False,
            'error': 'Invalid coordinates'
        }), 400
    
    if weather_service:
        data = weather_service.get_air_quality(lat, lon)
        if data:
            data['success'] = True
            data['coordinates'] = {'lat': lat, 'lon': lon}
            
            # Get location name using reverse geocoding (approximate)
            data['location_name'] = get_location_name(lat, lon)
            
            # Add health recommendations based on AQI
            aqi = data.get('aqi', 0)
            data['health_advisory'] = get_health_advisory(aqi)
            
            return jsonify(data)
    
    # Fallback to simulated data
    import random
    
    # Calculate approximate AQI based on location
    northern = lat > 20
    base_pm25 = 65 if northern else 35
    pm25 = base_pm25 + random.uniform(-15, 25)
    pm10 = (100 if northern else 55) + random.uniform(-20, 35)
    
    if pm25 <= 30:
        aqi = int(pm25 * 1.67)
        level = 'Good'
    elif pm25 <= 60:
        aqi = int(50 + (pm25 - 30) * 1.67)
        level = 'Moderate'
    elif pm25 <= 90:
        aqi = int(100 + (pm25 - 60) * 1.67)
        level = 'Poor'
    elif pm25 <= 120:
        aqi = int(150 + (pm25 - 90) * 1.67)
        level = 'Very Poor'
    else:
        aqi = int(200 + (pm25 - 120))
        level = 'Severe'
    
    return jsonify({
        'success': True,
        'pm25': round(pm25, 1),
        'pm10': round(pm10, 1),
        'co': round(random.uniform(0.3, 1.2), 2),
        'no2': round(random.uniform(15, 60), 1),
        'so2': round(random.uniform(5, 25), 1),
        'o3': round(random.uniform(30, 80), 1),
        'aqi': aqi,
        'aqi_level': level,
        'coordinates': {'lat': lat, 'lon': lon},
        'location_name': get_location_name(lat, lon),
        'health_advisory': get_health_advisory(aqi),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Simulated (API key not configured)'
    })


def get_location_name(lat, lon):
    """Get approximate location name from coordinates."""
    # Metro city detection
    METRO_CITIES = {
        'Mumbai': (19.0760, 72.8777, 0.5),
        'Delhi': (28.6139, 77.2090, 0.5),
        'Bangalore': (12.9716, 77.5946, 0.5),
        'Chennai': (13.0827, 80.2707, 0.5),
        'Kolkata': (22.5726, 88.3639, 0.5),
        'Hyderabad': (17.3850, 78.4867, 0.5),
        'Pune': (18.5204, 73.8567, 0.4),
        'Ahmedabad': (23.0225, 72.5714, 0.4),
        'Jaipur': (26.9124, 75.7873, 0.4),
        'Lucknow': (26.8467, 80.9462, 0.4),
    }
    
    for city, (city_lat, city_lon, radius) in METRO_CITIES.items():
        if abs(lat - city_lat) < radius and abs(lon - city_lon) < radius:
            return city
    
    # Return generic description
    if 8 <= lat <= 37 and 68 <= lon <= 97:
        return f"India ({lat:.2f}°N, {lon:.2f}°E)"
    return f"Location ({lat:.2f}°, {lon:.2f}°)"


def get_health_advisory(aqi):
    """Get health advisory based on AQI level."""
    if aqi <= 50:
        return {
            'level': 'Good',
            'color': '#28a745',
            'message': 'Air quality is satisfactory. Enjoy outdoor activities!',
            'precautions': []
        }
    elif aqi <= 100:
        return {
            'level': 'Moderate',
            'color': '#ffc107',
            'message': 'Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion.',
            'precautions': ['Sensitive groups should reduce outdoor activities']
        }
    elif aqi <= 150:
        return {
            'level': 'Poor',
            'color': '#fd7e14',
            'message': 'Members of sensitive groups may experience health effects.',
            'precautions': ['Reduce prolonged outdoor exertion', 'Keep windows closed', 'Use air purifiers if available']
        }
    elif aqi <= 200:
        return {
            'level': 'Very Poor',
            'color': '#dc3545',
            'message': 'Everyone may begin to experience health effects.',
            'precautions': ['Avoid outdoor activities', 'Wear N95 masks outdoors', 'Use air purifiers indoors', 'Keep windows closed']
        }
    else:
        return {
            'level': 'Severe',
            'color': '#721c24',
            'message': 'Health alert! Everyone may experience serious health effects.',
            'precautions': ['Stay indoors', 'Avoid all outdoor activities', 'Use air purifiers', 'Seek medical help if experiencing symptoms']
        }


@app.route('/api/forecast/<city>')
def get_forecast(city):
    """Get pollution forecast for a city."""
    if weather_service:
        # Get city coordinates
        METRO_CITIES = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.6139, 77.2090),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867)
        }
        
        if city in METRO_CITIES:
            lat, lon = METRO_CITIES[city]
            forecast = weather_service.get_forecast(lat, lon, days=7)
            return jsonify({'city': city, 'forecast': forecast})
    
    # Return simulated forecast
    import random
    from datetime import timedelta
    
    forecast = []
    base_aqi = 75
    
    for i in range(7):
        date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        aqi = int(base_aqi + random.uniform(-30, 40))
        
        if aqi <= 50:
            level = 'Good'
        elif aqi <= 100:
            level = 'Moderate'
        elif aqi <= 150:
            level = 'Poor'
        elif aqi <= 200:
            level = 'Very Poor'
        else:
            level = 'Severe'
        
        forecast.append({
            'date': date,
            'aqi': aqi,
            'aqi_level': level,
            'pm25': round(aqi * 0.4 + random.uniform(-5, 10), 1),
            'pm10': round(aqi * 0.6 + random.uniform(-10, 15), 1)
        })
        
        base_aqi = aqi  # Use previous value as base for next day
    
    return jsonify({'city': city, 'forecast': forecast})


@app.route('/api/send-alert', methods=['POST'])
def send_alert():
    """Send alert notification."""
    data = request.get_json()
    
    if email_service:
        alert_data = {
            'type': data.get('type', 'AIR'),
            'severity': data.get('severity', 'WARNING'),
            'title': data.get('title', 'Pollution Alert'),
            'message': data.get('message', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommendations': data.get('recommendations', [])
        }
        
        # In production, would get email from user preferences
        to_email = data.get('email', '')
        
        if to_email:
            success = email_service.send_alert_email(to_email, alert_data)
            return jsonify({'success': success})
        else:
            # Just log the alert
            email_service._log_alert(alert_data)
            return jsonify({'success': True, 'logged': True})
    
    return jsonify({'success': False, 'error': 'Email service not available'})


# ============================================
# Application Startup
# ============================================


def initialize_app():
    """Initialize application - create database, load data, load models."""
    print("\n" + "=" * 50)
    print("POLLUTION MONITORING SYSTEM - STARTUP")
    print("=" * 50)
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize database
    print("\n[1] Initializing database...")
    init_db()
    
    # Load CSV data into database
    print("\n[2] Loading CSV data...")
    load_csv_to_db()
    
    # Load ML models
    print("\n[3] Loading ML models...")
    load_models()
    
    print("\n" + "=" * 50)
    print("INITIALIZATION COMPLETE")
    print("=" * 50)


# Run initialization when starting the server
with app.app_context():
    initialize_app()


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Starting Pollution Monitoring System Server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
