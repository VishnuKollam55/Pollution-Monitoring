"""
aqi_calculator.py - Air Quality Index Calculator Module
Calculates AQI based on PM2.5, PM10, and CO2 values using standard formulas.
"""


def calculate_aqi_pm25(pm25):
    """
    Calculate AQI based on PM2.5 concentration.
    Uses EPA breakpoints for PM2.5 (μg/m³).
    
    Breakpoints:
    - Good (0-50): 0-12 μg/m³
    - Moderate (51-100): 12.1-35.4 μg/m³
    - Unhealthy for Sensitive (101-150): 35.5-55.4 μg/m³
    - Unhealthy (151-200): 55.5-150.4 μg/m³
    - Very Unhealthy (201-300): 150.5-250.4 μg/m³
    - Hazardous (301-500): 250.5-500.4 μg/m³
    
    Args:
        pm25 (float): PM2.5 concentration in μg/m³
        
    Returns:
        int: AQI value
    """
    breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return round(aqi)
    
    # If PM2.5 exceeds breakpoints, return max AQI
    return 500 if pm25 > 500 else 0


def calculate_aqi_pm10(pm10):
    """
    Calculate AQI based on PM10 concentration.
    Uses EPA breakpoints for PM10 (μg/m³).
    
    Args:
        pm10 (float): PM10 concentration in μg/m³
        
    Returns:
        int: AQI value
    """
    breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm10 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm10 - bp_lo) + aqi_lo
            return round(aqi)
    
    return 500 if pm10 > 604 else 0


def calculate_combined_aqi(pm25, pm10, co2=None):
    """
    Calculate combined AQI by taking the maximum of individual pollutant AQIs.
    This follows the EPA methodology where the dominant pollutant determines the AQI.
    
    Args:
        pm25 (float): PM2.5 concentration
        pm10 (float): PM10 concentration
        co2 (float, optional): CO2 concentration (not used in standard AQI but tracked)
        
    Returns:
        int: Combined AQI value
    """
    aqi_pm25 = calculate_aqi_pm25(pm25)
    aqi_pm10 = calculate_aqi_pm10(pm10)
    
    return max(aqi_pm25, aqi_pm10)


def classify_aqi(aqi):
    """
    Classify AQI into health categories.
    
    Categories:
    - Good: 0-50 (Green)
    - Moderate: 51-100 (Yellow)
    - Poor: 101-150 (Orange) - Unhealthy for Sensitive Groups
    - Very Poor: 151-200 (Red) - Unhealthy
    - Severe: 201+ (Dark Red) - Very Unhealthy to Hazardous
    
    Args:
        aqi (int): AQI value
        
    Returns:
        tuple: (classification string, color code, description)
    """
    if aqi <= 50:
        return ('Good', '#28a745', 'Air quality is satisfactory, pollution poses little or no risk.')
    elif aqi <= 100:
        return ('Moderate', '#ffc107', 'Air quality is acceptable. Sensitive individuals may experience issues.')
    elif aqi <= 150:
        return ('Poor', '#fd7e14', 'Sensitive groups may experience health effects. General public less likely to be affected.')
    elif aqi <= 200:
        return ('Very Poor', '#dc3545', 'Everyone may begin to experience health effects. Sensitive groups may experience serious effects.')
    else:
        return ('Severe', '#721c24', 'Health alert: everyone may experience serious health effects.')


def get_aqi_badge_html(aqi):
    """
    Generate HTML badge for AQI display.
    
    Args:
        aqi (int): AQI value
        
    Returns:
        str: HTML string for the badge
    """
    classification, color, description = classify_aqi(aqi)
    
    return f'''
    <div class="aqi-badge" style="background-color: {color}; color: white; padding: 10px 20px; border-radius: 5px; text-align: center;">
        <span class="aqi-value" style="font-size: 24px; font-weight: bold;">{aqi}</span>
        <span class="aqi-label" style="display: block; font-size: 14px;">{classification}</span>
    </div>
    '''


def get_health_recommendations(aqi):
    """
    Get health recommendations based on AQI level.
    
    Args:
        aqi (int): AQI value
        
    Returns:
        list: List of health recommendations
    """
    recommendations = []
    
    if aqi <= 50:
        recommendations = [
            "Air quality is ideal for outdoor activities.",
            "No special precautions needed.",
            "Enjoy the fresh air!"
        ]
    elif aqi <= 100:
        recommendations = [
            "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "Keep windows open for ventilation.",
            "Monitor air quality for changes."
        ]
    elif aqi <= 150:
        recommendations = [
            "People with respiratory diseases should limit outdoor exertion.",
            "Children and elderly should reduce outdoor activities.",
            "Consider using air purifiers indoors."
        ]
    elif aqi <= 200:
        recommendations = [
            "Everyone should reduce prolonged outdoor exertion.",
            "People with heart or lung disease, children, and older adults should avoid outdoor activities.",
            "Keep windows closed and use air purifiers."
        ]
    else:
        recommendations = [
            "Everyone should avoid all outdoor exertion.",
            "Stay indoors with air filtration.",
            "Wear N95 masks if going outside is unavoidable.",
            "Consider evacuation if levels persist."
        ]
    
    return recommendations
