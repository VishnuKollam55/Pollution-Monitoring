"""
config.py - Configuration Management for Pollution Monitoring System
====================================================================
Centralized configuration for API keys, thresholds, and application settings.
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class."""
    
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'pollution_monitoring_secret_key_2026')
    
    # Database
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'database.db')
    
    # Session Settings
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # API Keys (set via environment variables for security)
    OPENWEATHERMAP_API_KEY = os.environ.get('OPENWEATHERMAP_API_KEY', '')
    
    # Government API Keys (optional - falls back to Open-Meteo which is free)
    CPCB_API_KEY = os.environ.get('CPCB_API_KEY', '')  # Central Pollution Control Board
    DATA_GOV_API_KEY = os.environ.get('DATA_GOV_API_KEY', '')  # data.gov.in API
    
    # Email Settings (for notifications)
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@pollutionmonitor.com')
    
    # Alert Thresholds (can be modified via admin panel)
    THRESHOLDS = {
        'aqi': {
            'good': 50,
            'moderate': 100,
            'poor': 150,
            'very_poor': 200,
            'severe': 300
        },
        'water': {
            'ph_min': 6.5,
            'ph_max': 8.5,
            'turbidity_warning': 5,
            'turbidity_danger': 10,
            'dissolved_oxygen_min': 5
        },
        'noise': {
            'residential_day': 55,
            'residential_night': 45,
            'commercial_day': 65,
            'commercial_night': 55,
            'industrial_day': 75,
            'industrial_night': 70
        }
    }
    
    # Real-time Data Settings
    DATA_REFRESH_INTERVAL = 300  # 5 minutes in seconds
    API_CACHE_TIMEOUT = 600  # 10 minutes
    
    # Geolocation Default (Mumbai)
    DEFAULT_LATITUDE = 19.0760
    DEFAULT_LONGITUDE = 72.8777
    
    # Indian Metro Cities with coordinates
    METRO_CITIES = {
        'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra'},
        'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'state': 'Delhi'},
        'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
        'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu'},
        'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal'},
        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana'}
    }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


# Active configuration
config = DevelopmentConfig()
