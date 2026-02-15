"""
email_service.py - Email Notification Service
==============================================
Handles sending email alerts for pollution warnings.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

try:
    from config import config
except ImportError:
    config = None


class EmailService:
    """
    Email notification service for pollution alerts.
    """
    
    def __init__(self):
        """Initialize email service with configuration."""
        self.server = config.MAIL_SERVER if config else 'smtp.gmail.com'
        self.port = config.MAIL_PORT if config else 587
        self.use_tls = True
        self.username = config.MAIL_USERNAME if config else ''
        self.password = config.MAIL_PASSWORD if config else ''
        self.default_sender = config.MAIL_DEFAULT_SENDER if config else 'noreply@pollutionmonitor.com'
    
    def send_alert_email(self, to_email, alert_data):
        """
        Send pollution alert email.
        
        Args:
            to_email (str): Recipient email address
            alert_data (dict): Alert information
            
        Returns:
            bool: True if sent successfully
        """
        if not self.username or not self.password:
            print("Email not configured - alert logged instead")
            self._log_alert(alert_data)
            return False
        
        subject = f"🚨 {alert_data.get('severity', 'ALERT')}: {alert_data.get('title', 'Pollution Alert')}"
        
        html_content = self._generate_alert_html(alert_data)
        text_content = self._generate_alert_text(alert_data)
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.default_sender
            msg['To'] = to_email
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            print(f"Alert email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            self._log_alert(alert_data)
            return False
    
    def _generate_alert_html(self, alert_data):
        """Generate HTML email content for alert."""
        severity_colors = {
            'CRITICAL': '#dc2626',
            'DANGER': '#ef4444',
            'WARNING': '#f59e0b'
        }
        
        severity = alert_data.get('severity', 'WARNING')
        color = severity_colors.get(severity, '#6b7280')
        
        recommendations_html = ""
        for rec in alert_data.get('recommendations', []):
            recommendations_html += f"<li style='margin: 8px 0;'>{rec}</li>"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f3f4f6; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: {color}; color: white; padding: 24px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .content {{ padding: 24px; }}
                .alert-badge {{ display: inline-block; background: {color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-bottom: 16px; }}
                .message {{ font-size: 16px; line-height: 1.6; color: #374151; margin-bottom: 24px; }}
                .recommendations {{ background: #f9fafb; padding: 20px; border-radius: 8px; }}
                .recommendations h3 {{ margin: 0 0 12px 0; color: #1f2937; }}
                .recommendations ul {{ margin: 0; padding-left: 20px; color: #4b5563; }}
                .footer {{ background: #f9fafb; padding: 16px 24px; text-align: center; color: #6b7280; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌍 Pollution Monitoring System</h1>
                </div>
                <div class="content">
                    <span class="alert-badge">{severity}</span>
                    <h2 style="margin: 0 0 16px 0; color: #1f2937;">{alert_data.get('title', 'Alert')}</h2>
                    <p class="message">{alert_data.get('message', '')}</p>
                    
                    {f'''<div class="recommendations">
                        <h3>📋 Recommended Actions:</h3>
                        <ul>{recommendations_html}</ul>
                    </div>''' if recommendations_html else ''}
                </div>
                <div class="footer">
                    <p>Alert generated at {alert_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
                    <p>Integrated Pollution Monitoring and Control System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _generate_alert_text(self, alert_data):
        """Generate plain text email content."""
        text = f"""
POLLUTION MONITORING SYSTEM - {alert_data.get('severity', 'ALERT')}

{alert_data.get('title', 'Alert')}
{'='*50}

{alert_data.get('message', '')}

"""
        if alert_data.get('recommendations'):
            text += "Recommended Actions:\n"
            for i, rec in enumerate(alert_data['recommendations'], 1):
                text += f"  {i}. {rec}\n"
        
        text += f"\nAlert Time: {alert_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}"
        return text
    
    def _log_alert(self, alert_data):
        """Log alert when email is not configured."""
        print(f"[ALERT LOG] {alert_data.get('severity')}: {alert_data.get('title')}")
        print(f"  Message: {alert_data.get('message')}")
    
    def send_daily_report(self, to_email, report_data):
        """
        Send daily pollution summary report.
        
        Args:
            to_email (str): Recipient email
            report_data (dict): Daily statistics
            
        Returns:
            bool: Success status
        """
        subject = f"📊 Daily Pollution Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        html_content = self._generate_report_html(report_data)
        
        if not self.username or not self.password:
            print("Email not configured - report logged")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.default_sender
            msg['To'] = to_email
            
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Failed to send report: {e}")
            return False
    
    def _generate_report_html(self, report_data):
        """Generate HTML daily report."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f3f4f6; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; }}
                .header {{ background: linear-gradient(135deg, #4338ca, #6366f1); color: white; padding: 24px; text-align: center; }}
                .content {{ padding: 24px; }}
                .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
                .stat-card {{ background: #f9fafb; padding: 16px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 28px; font-weight: bold; color: #1f2937; }}
                .stat-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Pollution Report</h1>
                    <p>{datetime.now().strftime('%B %d, %Y')}</p>
                </div>
                <div class="content">
                    <div class="stat-grid">
                        <div class="stat-card">
                            <div class="stat-value">{report_data.get('avg_aqi', 'N/A')}</div>
                            <div class="stat-label">Avg AQI</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{report_data.get('water_quality', 'N/A')}</div>
                            <div class="stat-label">Water Quality</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{report_data.get('noise_level', 'N/A')} dB</div>
                            <div class="stat-label">Avg Noise</div>
                        </div>
                    </div>
                    <p>Total Alerts: {report_data.get('total_alerts', 0)}</p>
                </div>
            </div>
        </body>
        </html>
        """


# Singleton instance
email_service = EmailService()
