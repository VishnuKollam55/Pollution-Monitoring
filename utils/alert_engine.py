"""
alert_engine.py - Alert and Recommendation Engine Module
Monitors pollution levels and generates alerts with actionable recommendations.
"""

from datetime import datetime


class AlertEngine:
    """
    Alert Engine for pollution monitoring system.
    Monitors thresholds and generates alerts with recommendations.
    """
    
    # Alert thresholds
    THRESHOLDS = {
        'aqi': {
            'warning': 100,
            'danger': 150,
            'critical': 200
        },
        'ph': {
            'low_danger': 6.5,
            'high_danger': 8.5
        },
        'turbidity': {
            'warning': 5,
            'danger': 10
        },
        'dissolved_oxygen': {
            'danger': 5
        },
        'noise': {
            'warning': 70,
            'danger': 85,
            'critical': 100
        }
    }
    
    def __init__(self):
        """Initialize the Alert Engine with empty alert list."""
        self.alerts = []
        
    def clear_alerts(self):
        """Clear all current alerts."""
        self.alerts = []
    
    def check_air_quality(self, aqi, pm25=None, pm10=None, co2=None):
        """
        Check air quality and generate alerts if thresholds exceeded.
        
        Args:
            aqi (int): Current AQI value
            pm25 (float, optional): PM2.5 concentration
            pm10 (float, optional): PM10 concentration
            co2 (float, optional): CO2 concentration
            
        Returns:
            list: Generated alerts
        """
        air_alerts = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if aqi >= self.THRESHOLDS['aqi']['critical']:
            alert = {
                'type': 'AIR',
                'severity': 'CRITICAL',
                'title': 'Critical Air Quality Alert',
                'message': f'AQI has reached critical level: {aqi}',
                'timestamp': timestamp,
                'recommendations': [
                    'Implement immediate traffic restrictions',
                    'Halt all non-essential industrial operations',
                    'Issue public health advisory',
                    'Deploy emergency air quality response teams',
                    'Close schools and outdoor activities'
                ]
            }
            air_alerts.append(alert)
        elif aqi >= self.THRESHOLDS['aqi']['danger']:
            alert = {
                'type': 'AIR',
                'severity': 'DANGER',
                'title': 'High Air Pollution Alert',
                'message': f'AQI exceeds safe levels: {aqi}',
                'timestamp': timestamp,
                'recommendations': [
                    'Implement odd-even vehicle restrictions',
                    'Increase industrial emission monitoring',
                    'Advise sensitive groups to stay indoors',
                    'Enhance public transport services'
                ]
            }
            air_alerts.append(alert)
        elif aqi >= self.THRESHOLDS['aqi']['warning']:
            alert = {
                'type': 'AIR',
                'severity': 'WARNING',
                'title': 'Moderate Air Quality Warning',
                'message': f'AQI in moderate range: {aqi}',
                'timestamp': timestamp,
                'recommendations': [
                    'Monitor air quality trends',
                    'Sensitive individuals should limit outdoor exposure',
                    'Consider carpooling and public transport'
                ]
            }
            air_alerts.append(alert)
        
        self.alerts.extend(air_alerts)
        return air_alerts
    
    def check_water_quality(self, ph, turbidity, dissolved_oxygen):
        """
        Check water quality and generate alerts if parameters are abnormal.
        
        Args:
            ph (float): Water pH level
            turbidity (float): Water turbidity (NTU)
            dissolved_oxygen (float): Dissolved oxygen (mg/L)
            
        Returns:
            list: Generated alerts
        """
        water_alerts = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check pH levels
        if ph < self.THRESHOLDS['ph']['low_danger']:
            alert = {
                'type': 'WATER',
                'severity': 'DANGER',
                'title': 'Low pH Alert',
                'message': f'Water pH is too acidic: {ph}',
                'timestamp': timestamp,
                'recommendations': [
                    'Investigate industrial discharge sources',
                    'Implement lime treatment at water facilities',
                    'Issue water usage advisory',
                    'Increase water treatment capacity'
                ]
            }
            water_alerts.append(alert)
        elif ph > self.THRESHOLDS['ph']['high_danger']:
            alert = {
                'type': 'WATER',
                'severity': 'DANGER',
                'title': 'High pH Alert',
                'message': f'Water pH is too alkaline: {ph}',
                'timestamp': timestamp,
                'recommendations': [
                    'Check for industrial contamination',
                    'Implement acid treatment at water facilities',
                    'Issue water quality advisory',
                    'Monitor downstream effects'
                ]
            }
            water_alerts.append(alert)
        
        # Check turbidity
        if turbidity >= self.THRESHOLDS['turbidity']['danger']:
            alert = {
                'type': 'WATER',
                'severity': 'DANGER',
                'title': 'High Turbidity Alert',
                'message': f'Water turbidity exceeds limits: {turbidity} NTU',
                'timestamp': timestamp,
                'recommendations': [
                    'Increase filtration at treatment plants',
                    'Check for sedimentation sources',
                    'Issue boil water advisory if necessary',
                    'Inspect industrial wastewater discharge'
                ]
            }
            water_alerts.append(alert)
        elif turbidity >= self.THRESHOLDS['turbidity']['warning']:
            alert = {
                'type': 'WATER',
                'severity': 'WARNING',
                'title': 'Elevated Turbidity Warning',
                'message': f'Water turbidity slightly elevated: {turbidity} NTU',
                'timestamp': timestamp,
                'recommendations': [
                    'Monitor turbidity trends',
                    'Prepare additional treatment chemicals',
                    'Check source water conditions'
                ]
            }
            water_alerts.append(alert)
        
        # Check dissolved oxygen
        if dissolved_oxygen < self.THRESHOLDS['dissolved_oxygen']['danger']:
            alert = {
                'type': 'WATER',
                'severity': 'DANGER',
                'title': 'Low Dissolved Oxygen Alert',
                'message': f'Dissolved oxygen critically low: {dissolved_oxygen} mg/L',
                'timestamp': timestamp,
                'recommendations': [
                    'Investigate organic pollution sources',
                    'Check for sewage discharge',
                    'Implement aeration treatment',
                    'Monitor aquatic life impact'
                ]
            }
            water_alerts.append(alert)
        
        self.alerts.extend(water_alerts)
        return water_alerts
    
    def check_noise_levels(self, sound_level, zone='General'):
        """
        Check noise levels and generate alerts based on zone.
        
        Args:
            sound_level (float): Sound level in dB
            zone (str): Zone type (Residential, Commercial, Industrial)
            
        Returns:
            list: Generated alerts
        """
        noise_alerts = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Zone-specific thresholds
        zone_limits = {
            'Residential': {'day': 55, 'night': 45},
            'Commercial': {'day': 65, 'night': 55},
            'Industrial': {'day': 75, 'night': 70}
        }
        
        zone_limit = zone_limits.get(zone, {'day': 70, 'night': 60})
        
        if sound_level >= self.THRESHOLDS['noise']['critical']:
            alert = {
                'type': 'NOISE',
                'severity': 'CRITICAL',
                'title': 'Critical Noise Level Alert',
                'message': f'Noise level extremely high: {sound_level} dB in {zone} zone',
                'timestamp': timestamp,
                'recommendations': [
                    'Immediately identify and shut down noise source',
                    'Issue health hazard warning',
                    'Require hearing protection in area',
                    'Fine violating establishments',
                    'Consider temporary zone closure'
                ]
            }
            noise_alerts.append(alert)
        elif sound_level >= self.THRESHOLDS['noise']['danger']:
            alert = {
                'type': 'NOISE',
                'severity': 'DANGER',
                'title': 'High Noise Level Alert',
                'message': f'Noise exceeds safe limits: {sound_level} dB in {zone} zone',
                'timestamp': timestamp,
                'recommendations': [
                    'Identify primary noise sources',
                    'Issue noise violation notices',
                    'Mandate sound barriers for construction',
                    'Restrict heavy vehicle movement'
                ]
            }
            noise_alerts.append(alert)
        elif sound_level >= self.THRESHOLDS['noise']['warning']:
            alert = {
                'type': 'NOISE',
                'severity': 'WARNING',
                'title': 'Elevated Noise Warning',
                'message': f'Noise levels moderately high: {sound_level} dB in {zone} zone',
                'timestamp': timestamp,
                'recommendations': [
                    'Monitor noise trends',
                    'Advise noise source operators',
                    'Consider time-based restrictions'
                ]
            }
            noise_alerts.append(alert)
        
        self.alerts.extend(noise_alerts)
        return noise_alerts
    
    def get_all_alerts(self):
        """
        Get all current alerts.
        
        Returns:
            list: All generated alerts
        """
        return self.alerts
    
    def get_alerts_by_severity(self, severity):
        """
        Get alerts filtered by severity level.
        
        Args:
            severity (str): Severity level (CRITICAL, DANGER, WARNING)
            
        Returns:
            list: Filtered alerts
        """
        return [a for a in self.alerts if a['severity'] == severity]
    
    def get_alerts_by_type(self, alert_type):
        """
        Get alerts filtered by pollution type.
        
        Args:
            alert_type (str): Alert type (AIR, WATER, NOISE)
            
        Returns:
            list: Filtered alerts
        """
        return [a for a in self.alerts if a['type'] == alert_type]
    
    def get_alert_summary(self):
        """
        Get a summary of all current alerts.
        
        Returns:
            dict: Summary with counts by severity and type
        """
        summary = {
            'total': len(self.alerts),
            'by_severity': {
                'CRITICAL': len(self.get_alerts_by_severity('CRITICAL')),
                'DANGER': len(self.get_alerts_by_severity('DANGER')),
                'WARNING': len(self.get_alerts_by_severity('WARNING'))
            },
            'by_type': {
                'AIR': len(self.get_alerts_by_type('AIR')),
                'WATER': len(self.get_alerts_by_type('WATER')),
                'NOISE': len(self.get_alerts_by_type('NOISE'))
            }
        }
        return summary


def generate_control_recommendations(pollution_type, severity, current_value):
    """
    Generate specific control recommendations based on pollution type and severity.
    
    Args:
        pollution_type (str): Type of pollution (air, water, noise)
        severity (str): Severity level
        current_value (float): Current pollution measurement
        
    Returns:
        list: List of control recommendations
    """
    recommendations = []
    
    if pollution_type == 'air':
        if severity in ['CRITICAL', 'DANGER']:
            recommendations = [
                {'action': 'Traffic Control', 'description': 'Implement immediate vehicle restrictions in affected areas', 'priority': 'HIGH'},
                {'action': 'Industrial Shutdown', 'description': 'Temporarily halt non-essential industrial operations', 'priority': 'HIGH'},
                {'action': 'Construction Ban', 'description': 'Stop all construction activities generating dust', 'priority': 'MEDIUM'},
                {'action': 'Public Advisory', 'description': 'Issue health advisory through emergency broadcast', 'priority': 'HIGH'}
            ]
        else:
            recommendations = [
                {'action': 'Traffic Monitoring', 'description': 'Increase traffic monitoring and encourage public transport', 'priority': 'MEDIUM'},
                {'action': 'Emission Checks', 'description': 'Conduct random vehicle emission checks', 'priority': 'LOW'}
            ]
    
    elif pollution_type == 'water':
        if severity in ['CRITICAL', 'DANGER']:
            recommendations = [
                {'action': 'Treatment Enhancement', 'description': 'Increase water treatment chemical dosing', 'priority': 'HIGH'},
                {'action': 'Source Investigation', 'description': 'Identify and stop pollution discharge sources', 'priority': 'HIGH'},
                {'action': 'Usage Advisory', 'description': 'Issue water usage advisory to public', 'priority': 'HIGH'},
                {'action': 'Alternative Supply', 'description': 'Arrange alternative water supply if needed', 'priority': 'MEDIUM'}
            ]
        else:
            recommendations = [
                {'action': 'Regular Monitoring', 'description': 'Increase water quality testing frequency', 'priority': 'MEDIUM'},
                {'action': 'System Check', 'description': 'Inspect treatment plant operations', 'priority': 'LOW'}
            ]
    
    elif pollution_type == 'noise':
        if severity in ['CRITICAL', 'DANGER']:
            recommendations = [
                {'action': 'Source Shutdown', 'description': 'Immediately stop the noise source', 'priority': 'HIGH'},
                {'action': 'Fine Issuance', 'description': 'Issue violation fines to responsible parties', 'priority': 'MEDIUM'},
                {'action': 'Barrier Installation', 'description': 'Install temporary sound barriers', 'priority': 'MEDIUM'},
                {'action': 'Zone Restriction', 'description': 'Implement time-based activity restrictions', 'priority': 'MEDIUM'}
            ]
        else:
            recommendations = [
                {'action': 'Noise Mapping', 'description': 'Update noise maps and monitor trends', 'priority': 'LOW'},
                {'action': 'Public Awareness', 'description': 'Conduct noise awareness campaigns', 'priority': 'LOW'}
            ]
    
    return recommendations
