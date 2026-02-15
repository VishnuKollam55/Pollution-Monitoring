"""
report_generator.py - PDF Report Generation Service
====================================================
Generates downloadable PDF and CSV reports for pollution data.
"""

import io
import csv
from datetime import datetime, timedelta


class ReportGenerator:
    """
    Generates reports in various formats (PDF, CSV, Excel).
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.pdf_available = self._check_pdf_support()
    
    def _check_pdf_support(self):
        """Check if PDF generation libraries are available."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            return True
        except ImportError:
            print("ReportLab not installed - PDF generation will use HTML fallback")
            return False
    
    def generate_csv_report(self, data_type, data, city=None):
        """
        Generate CSV report from pollution data.
        
        Args:
            data_type (str): Type of data ('air', 'water', 'noise')
            data (list): List of data records
            city (str, optional): City filter
            
        Returns:
            str: CSV content as string
        """
        output = io.StringIO()
        
        if data_type == 'air':
            fieldnames = ['date', 'city', 'state', 'pm25', 'pm10', 'co2', 'aqi', 'level']
        elif data_type == 'water':
            fieldnames = ['date', 'city', 'state', 'ph', 'turbidity', 'dissolved_oxygen', 'quality']
        else:  # noise
            fieldnames = ['date', 'city', 'state', 'zone', 'sound_level', 'level']
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in data:
            # Convert sqlite3.Row to dict if needed
            if hasattr(row, 'keys'):
                row_dict = {k: row[k] for k in fieldnames if k in row.keys()}
            else:
                row_dict = row
            writer.writerow(row_dict)
        
        return output.getvalue()
    
    def generate_pdf_report(self, title, data, data_type, city=None, date_range=None):
        """
        Generate PDF report.
        
        Args:
            title (str): Report title
            data (list): Data records
            data_type (str): Type of pollution data
            city (str, optional): City filter
            date_range (tuple, optional): (start_date, end_date)
            
        Returns:
            bytes: PDF content or HTML fallback
        """
        if self.pdf_available:
            return self._generate_pdf_reportlab(title, data, data_type, city, date_range)
        else:
            return self._generate_html_report(title, data, data_type, city, date_range)
    
    def _generate_pdf_reportlab(self, title, data, data_type, city, date_range):
        """Generate PDF using ReportLab."""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#4338ca')
        )
        story.append(Paragraph(title, title_style))
        
        # Subtitle with date
        subtitle = f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}"
        if city:
            subtitle += f" | City: {city}"
        story.append(Paragraph(subtitle, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Summary statistics
        if data:
            summary = self._calculate_summary(data, data_type)
            summary_text = f"""
            <b>Summary Statistics:</b><br/>
            Total Records: {len(data)}<br/>
            {summary}
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Data table
        if data and len(data) > 0:
            table_data = self._prepare_table_data(data, data_type)
            if table_data:
                t = Table(table_data, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4338ca')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9fafb')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
                ]))
                story.append(t)
        
        # Footer
        story.append(Spacer(1, 30))
        footer = Paragraph(
            "Integrated Pollution Monitoring and Control System | MCA Project 2026",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)
        )
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _generate_html_report(self, title, data, data_type, city, date_range):
        """Generate HTML report as fallback."""
        table_rows = ""
        
        if data_type == 'air':
            headers = ['Date', 'City', 'PM2.5', 'PM10', 'CO2', 'AQI', 'Level']
            for row in data[:50]:  # Limit to 50 rows
                level_class = self._get_level_class(row.get('level', ''))
                table_rows += f"""
                <tr>
                    <td>{row.get('date', '')}</td>
                    <td>{row.get('city', '')}</td>
                    <td>{row.get('pm25', '')}</td>
                    <td>{row.get('pm10', '')}</td>
                    <td>{row.get('co2', '')}</td>
                    <td>{row.get('aqi', '')}</td>
                    <td><span class="badge {level_class}">{row.get('level', '')}</span></td>
                </tr>
                """
        elif data_type == 'water':
            headers = ['Date', 'City', 'pH', 'Turbidity', 'DO', 'Quality']
            for row in data[:50]:
                table_rows += f"""
                <tr>
                    <td>{row.get('date', '')}</td>
                    <td>{row.get('city', '')}</td>
                    <td>{row.get('ph', '')}</td>
                    <td>{row.get('turbidity', '')}</td>
                    <td>{row.get('dissolved_oxygen', '')}</td>
                    <td>{row.get('quality', '')}</td>
                </tr>
                """
        else:
            headers = ['Date', 'City', 'Zone', 'Sound Level', 'Level']
            for row in data[:50]:
                table_rows += f"""
                <tr>
                    <td>{row.get('date', '')}</td>
                    <td>{row.get('city', '')}</td>
                    <td>{row.get('zone', '')}</td>
                    <td>{row.get('sound_level', '')} dB</td>
                    <td>{row.get('level', '')}</td>
                </tr>
                """
        
        header_html = "".join([f"<th>{h}</th>" for h in headers])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                @media print {{
                    body {{ margin: 0; }}
                    .no-print {{ display: none; }}
                }}
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #fff; }}
                h1 {{ color: #4338ca; margin-bottom: 10px; }}
                .subtitle {{ color: #6b7280; margin-bottom: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ background: #4338ca; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; }}
                tr:nth-child(even) {{ background: #f9fafb; }}
                .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
                .good {{ background: #d1fae5; color: #065f46; }}
                .moderate {{ background: #fef3c7; color: #92400e; }}
                .poor {{ background: #fee2e2; color: #991b1b; }}
                .footer {{ margin-top: 40px; text-align: center; color: #9ca3af; font-size: 12px; }}
                .btn {{ background: #4338ca; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; margin: 10px 5px; }}
            </style>
        </head>
        <body>
            <div class="no-print">
                <button class="btn" onclick="window.print()">🖨️ Print Report</button>
                <button class="btn" onclick="window.close()">✖️ Close</button>
            </div>
            <h1>🌍 {title}</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}{f' | City: {city}' if city else ''}</p>
            
            <table>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
            
            <div class="footer">
                <p>Integrated Pollution Monitoring and Control System | MCA Project 2026</p>
            </div>
        </body>
        </html>
        """
        return html.encode('utf-8')
    
    def _calculate_summary(self, data, data_type):
        """Calculate summary statistics."""
        if not data:
            return ""
        
        if data_type == 'air':
            aqis = [r.get('aqi', 0) or 0 for r in data if r.get('aqi')]
            if aqis:
                return f"Average AQI: {sum(aqis)/len(aqis):.1f} | Max AQI: {max(aqis)} | Min AQI: {min(aqis)}"
        elif data_type == 'water':
            phs = [r.get('ph', 0) or 0 for r in data if r.get('ph')]
            if phs:
                return f"Average pH: {sum(phs)/len(phs):.2f}"
        elif data_type == 'noise':
            levels = [r.get('sound_level', 0) or 0 for r in data if r.get('sound_level')]
            if levels:
                return f"Average Sound Level: {sum(levels)/len(levels):.1f} dB"
        
        return ""
    
    def _prepare_table_data(self, data, data_type):
        """Prepare data for PDF table."""
        if data_type == 'air':
            headers = ['Date', 'City', 'PM2.5', 'PM10', 'CO2', 'AQI', 'Level']
            rows = [headers]
            for row in data[:30]:  # Limit rows for PDF
                rows.append([
                    str(row.get('date', '')),
                    str(row.get('city', '')),
                    str(row.get('pm25', '')),
                    str(row.get('pm10', '')),
                    str(row.get('co2', '')),
                    str(row.get('aqi', '')),
                    str(row.get('level', ''))
                ])
            return rows
        elif data_type == 'water':
            headers = ['Date', 'City', 'pH', 'Turbidity', 'DO', 'Quality']
            rows = [headers]
            for row in data[:30]:
                rows.append([
                    str(row.get('date', '')),
                    str(row.get('city', '')),
                    str(row.get('ph', '')),
                    str(row.get('turbidity', '')),
                    str(row.get('dissolved_oxygen', '')),
                    str(row.get('quality', ''))
                ])
            return rows
        else:
            headers = ['Date', 'City', 'Zone', 'Sound Level', 'Level']
            rows = [headers]
            for row in data[:30]:
                rows.append([
                    str(row.get('date', '')),
                    str(row.get('city', '')),
                    str(row.get('zone', '')),
                    f"{row.get('sound_level', '')} dB",
                    str(row.get('level', ''))
                ])
            return rows
    
    def _get_level_class(self, level):
        """Get CSS class for pollution level."""
        level_lower = str(level).lower()
        if level_lower in ['good', 'safe', 'low']:
            return 'good'
        elif level_lower in ['moderate', 'medium']:
            return 'moderate'
        else:
            return 'poor'


# Singleton instance
report_generator = ReportGenerator()
