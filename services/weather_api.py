"""
weather_api.py - Real-time Weather & Pollution API Service
===========================================================
Integrates with OpenWeatherMap Air Pollution API for real-time data.
Provides fallback to local data when API is unavailable.
"""

import requests
import time
from datetime import datetime
from functools import lru_cache

# Try to import config, use defaults if not available
try:
    from config import config
except ImportError:
    config = None


class WeatherAPIService:
    """
    Service for fetching real-time air quality data from external APIs.
    Uses OpenWeatherMap Air Pollution API.
    """
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
    GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/direct"
    
    # Cache for API responses (city -> (data, timestamp))
    _cache = {}
    CACHE_TIMEOUT = 600  # 10 minutes
    
    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key
        if not self.api_key and config:
            self.api_key = config.OPENWEATHERMAP_API_KEY
    
    def get_air_quality(self, lat, lon):
        """
        Get current air quality data for given coordinates.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Air quality data or None if unavailable
        """
        if not self.api_key:
            return self._get_simulated_data(lat, lon)
        
        cache_key = f"{lat:.2f},{lon:.2f}"
        
        # Check cache
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.CACHE_TIMEOUT:
                return data
        
        try:
            url = f"{self.BASE_URL}?lat={lat}&lon={lon}&appid={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                parsed_data = self._parse_api_response(api_data)
                self._cache[cache_key] = (parsed_data, time.time())
                return parsed_data
            else:
                print(f"API Error: {response.status_code}")
                return self._get_simulated_data(lat, lon)
                
        except requests.RequestException as e:
            print(f"API Request failed: {e}")
            return self._get_simulated_data(lat, lon)
    
    def _parse_api_response(self, api_data):
        """Parse OpenWeatherMap API response."""
        if not api_data.get('list'):
            return None
        
        item = api_data['list'][0]
        components = item.get('components', {})
        main = item.get('main', {})
        
        # AQI mapping: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
        aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
        aqi_level_mapping = {1: 'Good', 2: 'Moderate', 3: 'Poor', 4: 'Very Poor', 5: 'Severe'}
        
        aqi_index = main.get('aqi', 2)
        
        return {
            'pm25': round(components.get('pm2_5', 35), 1),
            'pm10': round(components.get('pm10', 55), 1),
            'co': round(components.get('co', 500) / 1000, 2),  # Convert to mg/m³
            'no2': round(components.get('no2', 20), 1),
            'so2': round(components.get('so2', 10), 1),
            'o3': round(components.get('o3', 60), 1),
            'aqi': aqi_mapping.get(aqi_index, 100),
            'aqi_level': aqi_level_mapping.get(aqi_index, 'Moderate'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'OpenWeatherMap API'
        }
    
    def _get_simulated_data(self, lat, lon):
        """
        Get REAL pollution data from Open-Meteo Air Quality API (FREE, no API key needed).
        Falls back to simulated data only if API fails.
        """
        # Open-Meteo Air Quality API - FREE, no API key required!
        OPENMETEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
        
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi',
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(OPENMETEO_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                
                pm25 = current.get('pm2_5', 35)
                pm10 = current.get('pm10', 55)
                european_aqi = current.get('european_aqi', 50)
                
                # Convert European AQI to Indian AQI scale (approximate)
                # European AQI: 0-20 Good, 20-40 Fair, 40-60 Moderate, 60-80 Poor, 80-100 Very Poor, 100+ Extremely Poor
                if european_aqi <= 20:
                    aqi = int(european_aqi * 2.5)  # 0-50 Good
                    level = 'Good'
                elif european_aqi <= 40:
                    aqi = int(50 + (european_aqi - 20) * 2.5)  # 51-100 Moderate
                    level = 'Moderate'
                elif european_aqi <= 60:
                    aqi = int(100 + (european_aqi - 40) * 2.5)  # 101-150 Poor
                    level = 'Poor'
                elif european_aqi <= 80:
                    aqi = int(150 + (european_aqi - 60) * 2.5)  # 151-200 Very Poor
                    level = 'Very Poor'
                else:
                    aqi = int(200 + (european_aqi - 80))  # 200+ Severe
                    level = 'Severe'
                
                return {
                    'pm25': round(pm25, 1),
                    'pm10': round(pm10, 1),
                    'co': round(current.get('carbon_monoxide', 500) / 1000, 2),  # μg/m³ to mg/m³
                    'no2': round(current.get('nitrogen_dioxide', 20), 1),
                    'so2': round(current.get('sulphur_dioxide', 10), 1),
                    'o3': round(current.get('ozone', 60), 1),
                    'aqi': aqi,
                    'aqi_level': level,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'Real-time API (Open-Meteo)'
                }
            else:
                print(f"Open-Meteo API Error: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"Open-Meteo API Request failed: {e}")
        
        # Fallback to simulated data only if API fails
        import random
        northern = lat > 20
        base_pm25 = 65 if northern else 35
        base_pm10 = 100 if northern else 55
        
        pm25 = base_pm25 + random.uniform(-15, 25)
        pm10 = base_pm10 + random.uniform(-20, 35)
        
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
            'pm25': round(pm25, 1),
            'pm10': round(pm10, 1),
            'co': round(random.uniform(0.3, 1.2), 2),
            'no2': round(random.uniform(15, 60), 1),
            'so2': round(random.uniform(5, 25), 1),
            'o3': round(random.uniform(30, 80), 1),
            'aqi': aqi,
            'aqi_level': level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Simulated (API fallback)'
        }
    
    def get_city_air_quality(self, city_name):
        """
        Get air quality for a city by name.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            dict: Air quality data
        """
        # All major Indian cities with coordinates (covering all states/UTs)
        INDIAN_CITIES = {
            # Metro Cities
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.6139, 77.2090),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867),
            
            # Andhra Pradesh
            'Visakhapatnam': (17.6868, 83.2185),
            'Vijayawada': (16.5062, 80.6480),
            'Tirupati': (13.6288, 79.4192),
            'Guntur': (16.3067, 80.4365),
            
            # Assam
            'Guwahati': (26.1445, 91.7362),
            'Dibrugarh': (27.4728, 94.9120),
            
            # Bihar
            'Patna': (25.5941, 85.1376),
            'Gaya': (24.7914, 85.0002),
            
            # Chhattisgarh
            'Raipur': (21.2514, 81.6296),
            'Bhilai': (21.2094, 81.4285),
            
            # Goa
            'Panaji': (15.4909, 73.8278),
            'Margao': (15.2832, 73.9862),
            
            # Gujarat
            'Ahmedabad': (23.0225, 72.5714),
            'Surat': (21.1702, 72.8311),
            'Vadodara': (22.3072, 73.1812),
            'Rajkot': (22.3039, 70.8022),
            
            # Haryana
            'Gurugram': (28.4595, 77.0266),
            'Faridabad': (28.4089, 77.3178),
            'Chandigarh': (30.7333, 76.7794),
            
            # Himachal Pradesh
            'Shimla': (31.1048, 77.1734),
            'Dharamshala': (32.2190, 76.3234),
            
            # Jharkhand
            'Ranchi': (23.3441, 85.3096),
            'Jamshedpur': (22.8046, 86.2029),
            
            # Karnataka
            'Mysore': (12.2958, 76.6394),
            'Mangalore': (12.9141, 74.8560),
            'Hubli': (15.3647, 75.1240),
            
            # Kerala
            'Kochi': (9.9312, 76.2673),
            'Thiruvananthapuram': (8.5241, 76.9366),
            'Kozhikode': (11.2588, 75.7804),
            
            # Madhya Pradesh
            'Bhopal': (23.2599, 77.4126),
            'Indore': (22.7196, 75.8577),
            'Gwalior': (26.2183, 78.1828),
            'Jabalpur': (23.1815, 79.9864),
            
            # Maharashtra
            'Pune': (18.5204, 73.8567),
            'Nagpur': (21.1458, 79.0882),
            'Nashik': (19.9975, 73.7898),
            'Aurangabad': (19.8762, 75.3433),
            'Thane': (19.2183, 72.9781),
            
            # Odisha
            'Bhubaneswar': (20.2961, 85.8245),
            'Cuttack': (20.4625, 85.8830),
            
            # Punjab
            'Ludhiana': (30.9010, 75.8573),
            'Amritsar': (31.6340, 74.8723),
            'Jalandhar': (31.3260, 75.5762),
            
            # Rajasthan
            'Jaipur': (26.9124, 75.7873),
            'Jodhpur': (26.2389, 73.0243),
            'Udaipur': (24.5854, 73.7125),
            'Kota': (25.2138, 75.8648),
            
            # Tamil Nadu
            'Coimbatore': (11.0168, 76.9558),
            'Madurai': (9.9252, 78.1198),
            'Tiruchirappalli': (10.7905, 78.7047),
            'Salem': (11.6643, 78.1460),
            
            # Telangana
            'Warangal': (17.9784, 79.5941),
            'Nizamabad': (18.6725, 78.0941),
            
            # Uttar Pradesh
            'Lucknow': (26.8467, 80.9462),
            'Kanpur': (26.4499, 80.3319),
            'Agra': (27.1767, 78.0081),
            'Varanasi': (25.3176, 82.9739),
            'Prayagraj': (25.4358, 81.8463),
            'Noida': (28.5355, 77.3910),
            'Ghaziabad': (28.6692, 77.4538),
            'Meerut': (28.9845, 77.7064),
            
            # Uttarakhand
            'Dehradun': (30.3165, 78.0322),
            'Haridwar': (29.9457, 78.1642),
            
            # West Bengal
            'Howrah': (22.5958, 88.2636),
            'Durgapur': (23.5204, 87.3119),
            'Siliguri': (26.7271, 88.3953),
            
            # Union Territories
            'Srinagar': (34.0837, 74.7973),
            'Jammu': (32.7266, 74.8570),
            'Pondicherry': (11.9416, 79.8083),
        }
        
        if city_name in INDIAN_CITIES:
            lat, lon = INDIAN_CITIES[city_name]
            data = self.get_air_quality(lat, lon)
            if data:
                data['city'] = city_name
            return data
        
        # Try geocoding for other cities
        if self.api_key:
            try:
                url = f"{self.GEOCODING_URL}?q={city_name},IN&limit=1&appid={self.api_key}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    locations = response.json()
                    if locations:
                        lat = locations[0]['lat']
                        lon = locations[0]['lon']
                        data = self.get_air_quality(lat, lon)
                        if data:
                            data['city'] = city_name
                        return data
            except requests.RequestException:
                pass
        
        # Fallback to simulated data for Mumbai
        data = self._get_simulated_data(19.0760, 72.8777)
        data['city'] = city_name
        return data
    
    def get_forecast(self, lat, lon, days=5):
        """
        Get air quality forecast for next few days.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            days (int): Number of forecast days
            
        Returns:
            list: Forecast data for each day
        """
        if not self.api_key:
            return self._get_simulated_forecast(lat, lon, days)
        
        try:
            url = f"{self.BASE_URL}/forecast?lat={lat}&lon={lon}&appid={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                return self._parse_forecast_response(api_data, days)
        except requests.RequestException:
            pass
        
        return self._get_simulated_forecast(lat, lon, days)
    
    def _parse_forecast_response(self, api_data, days):
        """Parse forecast API response."""
        forecasts = []
        items_per_day = 24  # Hourly data
        
        for i, item in enumerate(api_data.get('list', [])[:days * items_per_day:items_per_day]):
            forecasts.append(self._parse_api_response({'list': [item]}))
        
        return forecasts
    
    def _get_simulated_forecast(self, lat, lon, days):
        """Generate simulated forecast data."""
        import random
        from datetime import timedelta
        
        forecasts = []
        current = self._get_simulated_data(lat, lon)
        
        for i in range(days):
            forecast = current.copy()
            # Add some variation for each day
            forecast['pm25'] = round(current['pm25'] + random.uniform(-10, 15), 1)
            forecast['pm10'] = round(current['pm10'] + random.uniform(-15, 20), 1)
            forecast['aqi'] = int(current['aqi'] + random.uniform(-20, 25))
            forecast['date'] = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
            forecast['timestamp'] = forecast['date']
            forecasts.append(forecast)
        
        return forecasts


# Singleton instance
weather_service = WeatherAPIService()
