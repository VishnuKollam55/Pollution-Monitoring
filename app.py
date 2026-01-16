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

Author: MCA Student
Version: 1.0.0
"""

import os
import sys
import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, g

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import load_csv_data, clean_air_data, clean_water_data, clean_noise_data
from utils.aqi_calculator import calculate_combined_aqi, classify_aqi, get_health_recommendations
from utils.alert_engine import AlertEngine, generate_control_recommendations

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
