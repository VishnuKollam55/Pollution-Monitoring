"""
government_api.py - Government Pollution Data API Service
===========================================================
Integrates with Indian government APIs for real-time pollution data:
- CPCB (Central Pollution Control Board) via APISetu
- data.gov.in Open Government Data Platform
- Fallback to Open-Meteo (free, no API key needed)
"""

import requests
import time
from datetime import datetime
from functools import lru_cache

# Try to import config
try:
    from config import config
except ImportError:
    config = None


class CPCBService:
    """
    Service for fetching data from CPCB (Central Pollution Control Board).
    Uses APISetu gateway for accessing CPCB CAAQMS data.
    """
    
    BASE_URL = "https://api.apisetu.gov.in/cpcb/v1"
    
    # Major CPCB monitoring stations
    STATIONS = {
        'Delhi': ['ITO', 'Anand Vihar', 'RK Puram', 'Punjabi Bagh'],
        'Mumbai': ['Bandra', 'Worli', 'Colaba', 'Navi Mumbai'],
        'Bangalore': ['BTM Layout', 'Silk Board', 'Peenya'],
        'Chennai': ['Alandur', 'Manali', 'Velachery'],
        'Kolkata': ['Victoria', 'Rabindra Bharati', 'Jadavpur'],
        'Hyderabad': ['Sanathnagar', 'Zoo Park', 'Bollaram'],
        'Pune': ['Karve Road', 'Bhosari', 'Hadapsar'],
        'Ahmedabad': ['Maninagar', 'Chandkheda', 'Navrangpura'],
    }
    
    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key
        if not self.api_key and config:
            self.api_key = getattr(config, 'CPCB_API_KEY', None)
        self._cache = {}
        self.CACHE_TIMEOUT = 900  # 15 minutes
    
    def get_city_aqi(self, city_name):
        """
        Get real-time AQI data for a city from CPCB.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            dict: AQI data or None if unavailable
        """
        if not self.api_key:
            return None
        
        cache_key = f"cpcb_{city_name}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.CACHE_TIMEOUT:
                return data
        
        try:
            headers = {
                'Accept': 'application/json',
                'X-APISETU-APIKEY': self.api_key
            }
            
            # Try to get data from CPCB
            url = f"{self.BASE_URL}/aqi/city/{city_name}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                parsed_data = self._parse_cpcb_response(api_data, city_name)
                self._cache[cache_key] = (parsed_data, time.time())
                return parsed_data
                
        except requests.RequestException as e:
            print(f"CPCB API error: {e}")
        
        return None
    
    def _parse_cpcb_response(self, api_data, city_name):
        """Parse CPCB API response."""
        # Extract data from CPCB response format
        records = api_data.get('records', [])
        if not records:
            return None
        
        # Get latest record
        latest = records[0] if records else {}
        
        return {
            'city': city_name,
            'aqi': int(latest.get('aqi', 0)),
            'pm25': float(latest.get('pm25', 0)),
            'pm10': float(latest.get('pm10', 0)),
            'no2': float(latest.get('no2', 0)),
            'so2': float(latest.get('so2', 0)),
            'co': float(latest.get('co', 0)),
            'o3': float(latest.get('ozone', 0)),
            'station': latest.get('station', 'Unknown'),
            'timestamp': latest.get('last_update', datetime.now().isoformat()),
            'source': 'CPCB Official'
        }


class DataGovService:
    """
    Service for fetching data from data.gov.in Open Government Data Platform.
    """
    
    BASE_URL = "https://api.data.gov.in/resource"
    
    # Known resource IDs for pollution data
    RESOURCE_IDS = {
        'realtime_aqi': '3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69',
        'historical_aqi': '9402465b-d112-4573-9aef-6c8c6bbf7c8f',
    }
    
    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key
        if not self.api_key and config:
            self.api_key = getattr(config, 'DATA_GOV_API_KEY', None)
        self._cache = {}
        self.CACHE_TIMEOUT = 900  # 15 minutes
    
    def get_realtime_aqi(self, city_name=None):
        """
        Get real-time AQI data from data.gov.in.
        
        Args:
            city_name (str, optional): Filter by city name
            
        Returns:
            list: List of AQI records
        """
        if not self.api_key:
            return None
        
        cache_key = f"datagov_{city_name or 'all'}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.CACHE_TIMEOUT:
                return data
        
        try:
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 100
            }
            
            if city_name:
                params['filters[city]'] = city_name
            
            url = f"{self.BASE_URL}/{self.RESOURCE_IDS['realtime_aqi']}"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                parsed_data = self._parse_response(api_data)
                self._cache[cache_key] = (parsed_data, time.time())
                return parsed_data
                
        except requests.RequestException as e:
            print(f"data.gov.in API error: {e}")
        
        return None
    
    def _parse_response(self, api_data):
        """Parse data.gov.in API response."""
        records = api_data.get('records', [])
        parsed = []
        
        for record in records:
            parsed.append({
                'city': record.get('city', 'Unknown'),
                'state': record.get('state', 'Unknown'),
                'aqi': int(record.get('pollutant_avg', 0)),
                'pm25': float(record.get('pm25', 0) or 0),
                'pm10': float(record.get('pm10', 0) or 0),
                'no2': float(record.get('no2', 0) or 0),
                'so2': float(record.get('so2', 0) or 0),
                'station': record.get('station', 'Unknown'),
                'timestamp': record.get('last_update', datetime.now().isoformat()),
                'source': 'data.gov.in'
            })
        
        return parsed


class OpenMeteoService:
    """
    Service for fetching data from Open-Meteo Air Quality API.
    FREE, no API key required - this is the primary fallback.
    """
    
    BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # All major Indian cities with coordinates
    INDIAN_CITIES = {
        'Mumbai': (19.0760, 72.8777, 'Maharashtra'),
        'Delhi': (28.6139, 77.2090, 'Delhi'),
        'Bangalore': (12.9716, 77.5946, 'Karnataka'),
        'Chennai': (13.0827, 80.2707, 'Tamil Nadu'),
        'Kolkata': (22.5726, 88.3639, 'West Bengal'),
        'Hyderabad': (17.3850, 78.4867, 'Telangana'),
        'Pune': (18.5204, 73.8567, 'Maharashtra'),
        'Ahmedabad': (23.0225, 72.5714, 'Gujarat'),
        'Jaipur': (26.9124, 75.7873, 'Rajasthan'),
        'Lucknow': (26.8467, 80.9462, 'Uttar Pradesh'),
        'Kanpur': (26.4499, 80.3319, 'Uttar Pradesh'),
        'Nagpur': (21.1458, 79.0882, 'Maharashtra'),
        'Visakhapatnam': (17.6868, 83.2185, 'Andhra Pradesh'),
        'Bhopal': (23.2599, 77.4126, 'Madhya Pradesh'),
        'Patna': (25.5941, 85.1376, 'Bihar'),
        'Indore': (22.7196, 75.8577, 'Madhya Pradesh'),
        'Surat': (21.1702, 72.8311, 'Gujarat'),
        'Kochi': (9.9312, 76.2673, 'Kerala'),
        'Coimbatore': (11.0168, 76.9558, 'Tamil Nadu'),
        'Guwahati': (26.1445, 91.7362, 'Assam'),
        'Chandigarh': (30.7333, 76.7794, 'Punjab'),
        'Thiruvananthapuram': (8.5241, 76.9366, 'Kerala'),
        'Ranchi': (23.3441, 85.3096, 'Jharkhand'),
        'Shimla': (31.1048, 77.1734, 'Himachal Pradesh'),
        'Noida': (28.5355, 77.3910, 'Uttar Pradesh'),
        'Gurugram': (28.4595, 77.0266, 'Haryana'),
        'Faridabad': (28.4089, 77.3178, 'Haryana'),
        'Ghaziabad': (28.6692, 77.4538, 'Uttar Pradesh'),
        'Varanasi': (25.3176, 82.9739, 'Uttar Pradesh'),
        'Agra': (27.1767, 78.0081, 'Uttar Pradesh'),
        'Amritsar': (31.6340, 74.8723, 'Punjab'),
        'Ludhiana': (30.9010, 75.8573, 'Punjab'),
        'Mysore': (12.2958, 76.6394, 'Karnataka'),
        'Mangalore': (12.9141, 74.8560, 'Karnataka'),
        'Madurai': (9.9252, 78.1198, 'Tamil Nadu'),
        'Jodhpur': (26.2389, 73.0243, 'Rajasthan'),
        'Udaipur': (24.5854, 73.7125, 'Rajasthan'),
        'Dehradun': (30.3165, 78.0322, 'Uttarakhand'),
        'Bhubaneswar': (20.2961, 85.8245, 'Odisha'),
        'Raipur': (21.2514, 81.6296, 'Chhattisgarh'),
    }
    
    def __init__(self):
        """Initialize the service."""
        self._cache = {}
        self.CACHE_TIMEOUT = 600  # 10 minutes
    
    def get_city_aqi(self, city_name):
        """
        Get real-time AQI data for a city from Open-Meteo.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            dict: AQI data
        """
        if city_name not in self.INDIAN_CITIES:
            # Default to Mumbai if city not found
            city_name = 'Mumbai'
        
        lat, lon, state = self.INDIAN_CITIES[city_name]
        
        cache_key = f"openmeteo_{city_name}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.CACHE_TIMEOUT:
                return data
        
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi',
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                parsed_data = self._parse_response(api_data, city_name, state)
                self._cache[cache_key] = (parsed_data, time.time())
                return parsed_data
                
        except requests.RequestException as e:
            print(f"Open-Meteo API error: {e}")
        
        # Return simulated data as last resort
        return self._get_fallback_data(city_name, state)
    
    def _parse_response(self, api_data, city_name, state):
        """Parse Open-Meteo API response."""
        current = api_data.get('current', {})
        
        pm25 = current.get('pm2_5', 35)
        pm10 = current.get('pm10', 55)
        european_aqi = current.get('european_aqi', 50)
        
        # Convert European AQI to Indian AQI scale
        if european_aqi <= 20:
            aqi = int(european_aqi * 2.5)
            level = 'Good'
        elif european_aqi <= 40:
            aqi = int(50 + (european_aqi - 20) * 2.5)
            level = 'Moderate'
        elif european_aqi <= 60:
            aqi = int(100 + (european_aqi - 40) * 2.5)
            level = 'Poor'
        elif european_aqi <= 80:
            aqi = int(150 + (european_aqi - 60) * 2.5)
            level = 'Very Poor'
        else:
            aqi = int(200 + (european_aqi - 80))
            level = 'Severe'
        
        return {
            'city': city_name,
            'state': state,
            'aqi': aqi,
            'aqi_level': level,
            'pm25': round(pm25, 1),
            'pm10': round(pm10, 1),
            'co': round(current.get('carbon_monoxide', 500) / 1000, 2),
            'no2': round(current.get('nitrogen_dioxide', 20), 1),
            'so2': round(current.get('sulphur_dioxide', 10), 1),
            'o3': round(current.get('ozone', 60), 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Open-Meteo API (Real-time)'
        }
    
    def _get_fallback_data(self, city_name, state):
        """Generate fallback data when API fails."""
        import random
        
        # Northern cities typically have higher pollution
        northern = city_name in ['Delhi', 'Noida', 'Gurugram', 'Ghaziabad', 'Lucknow', 'Kanpur', 'Patna']
        base_pm25 = 75 if northern else 40
        
        pm25 = base_pm25 + random.uniform(-15, 25)
        pm10 = pm25 * 1.5 + random.uniform(-10, 20)
        
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
        
        return {
            'city': city_name,
            'state': state,
            'aqi': aqi,
            'aqi_level': level,
            'pm25': round(pm25, 1),
            'pm10': round(pm10, 1),
            'co': round(random.uniform(0.3, 1.2), 2),
            'no2': round(random.uniform(15, 60), 1),
            'so2': round(random.uniform(5, 25), 1),
            'o3': round(random.uniform(30, 80), 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Simulated (API Fallback)'
        }
    
    def get_all_cities(self):
        """Get AQI data for all configured cities."""
        results = []
        for city_name in self.INDIAN_CITIES.keys():
            data = self.get_city_aqi(city_name)
            if data:
                results.append(data)
        return results
    
    def get_cities_for_comparison(self, city_list):
        """Get AQI data for specific cities for comparison."""
        results = []
        for city_name in city_list:
            data = self.get_city_aqi(city_name)
            if data:
                results.append(data)
        return results


class GovernmentAPIService:
    """
    Unified service that tries government APIs first, then falls back to Open-Meteo.
    This is the main service to use throughout the application.
    """
    
    def __init__(self, cpcb_key=None, datagov_key=None):
        """Initialize all services."""
        self.cpcb = CPCBService(cpcb_key)
        self.datagov = DataGovService(datagov_key)
        self.openmeteo = OpenMeteoService()
    
    def get_city_aqi(self, city_name):
        """
        Get AQI data for a city, trying government APIs first.
        
        Priority:
        1. CPCB (if API key configured)
        2. data.gov.in (if API key configured)
        3. Open-Meteo (always available, free)
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            dict: AQI data with source information
        """
        # Try CPCB first
        if self.cpcb.api_key:
            data = self.cpcb.get_city_aqi(city_name)
            if data:
                return data
        
        # Try data.gov.in
        if self.datagov.api_key:
            records = self.datagov.get_realtime_aqi(city_name)
            if records:
                return records[0]
        
        # Fallback to Open-Meteo (always works)
        return self.openmeteo.get_city_aqi(city_name)
    
    def get_all_cities(self):
        """Get AQI data for all cities."""
        return self.openmeteo.get_all_cities()
    
    def get_comparison_data(self, cities):
        """Get data for multiple cities for comparison."""
        return self.openmeteo.get_cities_for_comparison(cities)
    
    def get_available_cities(self):
        """Get list of all available cities."""
        return list(OpenMeteoService.INDIAN_CITIES.keys())


# Singleton instance for use throughout the application
government_api = GovernmentAPIService()
