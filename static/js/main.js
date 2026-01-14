/**
 * main.js - Pollution Monitoring System JavaScript
 * ==================================================
 * Handles chart rendering, API calls, and interactive features.
 */

// ==========================================
// CHART CONFIGURATION
// ==========================================

// Color palette for charts
const chartColors = {
    primary: '#283593',
    secondary: '#3949ab',
    good: '#28a745',
    moderate: '#ffc107',
    poor: '#fd7e14',
    veryPoor: '#dc3545',
    severe: '#721c24',
    air: '#3498db',
    water: '#2ecc71',
    noise: '#9b59b6',
    predicted: '#e74c3c',
    grid: 'rgba(0, 0, 0, 0.1)'
};

// Common chart options
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 20,
                font: {
                    family: "'Segoe UI', sans-serif",
                    size: 12
                }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleFont: { size: 14 },
            bodyFont: { size: 12 },
            padding: 12,
            cornerRadius: 8
        }
    },
    scales: {
        x: {
            grid: { color: chartColors.grid },
            ticks: { font: { size: 11 } }
        },
        y: {
            grid: { color: chartColors.grid },
            ticks: { font: { size: 11 } }
        }
    }
};

// ==========================================
// CHART RENDERING FUNCTIONS
// ==========================================

/**
 * Render AQI trend line chart
 */
function renderAQIChart(canvasId, historicalData, predictedData = []) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = historicalData.map(d => d.date);
    const aqiValues = historicalData.map(d => d.aqi);

    const datasets = [
        {
            label: 'Historical AQI',
            data: aqiValues,
            borderColor: chartColors.air,
            backgroundColor: 'rgba(52, 152, 219, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointHoverRadius: 6
        }
    ];

    // Add predicted data if available
    if (predictedData && predictedData.length > 0) {
        const predictedLabels = predictedData.map(d => d.date);
        const predictedValues = predictedData.map(d => d.aqi);

        // Extend labels
        labels.push(...predictedLabels);

        // Create predicted dataset with null values for historical period
        const paddedPredicted = new Array(aqiValues.length - 1).fill(null);
        paddedPredicted.push(aqiValues[aqiValues.length - 1]); // Connect to last historical point
        paddedPredicted.push(...predictedValues);

        datasets.push({
            label: 'Predicted AQI',
            data: paddedPredicted,
            borderColor: chartColors.predicted,
            backgroundColor: 'rgba(231, 76, 60, 0.1)',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    }

    new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            ...commonChartOptions,
            plugins: {
                ...commonChartOptions.plugins,
                annotation: {
                    annotations: {
                        goodLine: {
                            type: 'line',
                            yMin: 50,
                            yMax: 50,
                            borderColor: chartColors.good,
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: 'Good',
                                enabled: true,
                                position: 'start'
                            }
                        },
                        moderateLine: {
                            type: 'line',
                            yMin: 100,
                            yMax: 100,
                            borderColor: chartColors.moderate,
                            borderWidth: 1,
                            borderDash: [5, 5]
                        },
                        poorLine: {
                            type: 'line',
                            yMin: 150,
                            yMax: 150,
                            borderColor: chartColors.poor,
                            borderWidth: 1,
                            borderDash: [5, 5]
                        }
                    }
                }
            },
            scales: {
                ...commonChartOptions.scales,
                y: {
                    ...commonChartOptions.scales.y,
                    min: 0,
                    max: 300,
                    title: {
                        display: true,
                        text: 'AQI Value'
                    }
                }
            }
        }
    });
}

/**
 * Render PM levels bar chart
 */
function renderPMChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = data.map(d => d.date);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'PM2.5',
                    data: data.map(d => d.pm25),
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: chartColors.air,
                    borderWidth: 1
                },
                {
                    label: 'PM10',
                    data: data.map(d => d.pm10),
                    backgroundColor: 'rgba(155, 89, 182, 0.7)',
                    borderColor: chartColors.noise,
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...commonChartOptions,
            scales: {
                ...commonChartOptions.scales,
                y: {
                    ...commonChartOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Concentration (μg/m³)'
                    }
                }
            }
        }
    });
}

/**
 * Render water quality chart
 */
function renderWaterChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = data.map(d => d.date);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'pH',
                    data: data.map(d => d.ph),
                    borderColor: chartColors.water,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Turbidity (NTU)',
                    data: data.map(d => d.turbidity),
                    borderColor: chartColors.moderate,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Dissolved Oxygen (mg/L)',
                    data: data.map(d => d.dissolved_oxygen),
                    borderColor: chartColors.air,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            ...commonChartOptions,
            scales: {
                x: commonChartOptions.scales.x,
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'pH / DO'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Turbidity (NTU)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

/**
 * Render noise levels chart
 */
function renderNoiseChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = data.map(d => d.date);

    // Color bars based on level
    const backgroundColors = data.map(d => {
        if (d.sound_level < 55) return chartColors.good;
        if (d.sound_level < 75) return chartColors.moderate;
        return chartColors.veryPoor;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Sound Level (dB)',
                data: data.map(d => d.sound_level),
                backgroundColor: backgroundColors,
                borderColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            ...commonChartOptions,
            plugins: {
                ...commonChartOptions.plugins,
                annotation: {
                    annotations: {
                        dangerLine: {
                            type: 'line',
                            yMin: 85,
                            yMax: 85,
                            borderColor: chartColors.veryPoor,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                content: 'Danger: 85dB',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            },
            scales: {
                ...commonChartOptions.scales,
                y: {
                    ...commonChartOptions.scales.y,
                    min: 0,
                    max: 120,
                    title: {
                        display: true,
                        text: 'Sound Level (dB)'
                    }
                }
            }
        }
    });
}

/**
 * Render dashboard summary charts
 */
function renderDashboardCharts(airData, waterData, noiseData) {
    // Air mini chart
    const airCtx = document.getElementById('airMiniChart');
    if (airCtx && airData.length > 0) {
        new Chart(airCtx, {
            type: 'line',
            data: {
                labels: airData.map(d => d.date.substring(5)),
                datasets: [{
                    data: airData.map(d => d.aqi),
                    borderColor: chartColors.air,
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false }
                }
            }
        });
    }

    // Water mini chart
    const waterCtx = document.getElementById('waterMiniChart');
    if (waterCtx && waterData.length > 0) {
        new Chart(waterCtx, {
            type: 'line',
            data: {
                labels: waterData.map(d => d.date.substring(5)),
                datasets: [{
                    data: waterData.map(d => d.ph),
                    borderColor: chartColors.water,
                    backgroundColor: 'rgba(46, 204, 113, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false }
                }
            }
        });
    }

    // Noise mini chart
    const noiseCtx = document.getElementById('noiseMiniChart');
    if (noiseCtx && noiseData.length > 0) {
        new Chart(noiseCtx, {
            type: 'bar',
            data: {
                labels: noiseData.map(d => d.date.substring(5)),
                datasets: [{
                    data: noiseData.map(d => d.sound_level),
                    backgroundColor: chartColors.noise,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false }
                }
            }
        });
    }
}

// ==========================================
// API FUNCTIONS
// ==========================================

/**
 * Make prediction API call
 */
async function makePrediction(type, data) {
    try {
        const response = await fetch(`/api/predict/${type}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        return await response.json();
    } catch (error) {
        console.error('Prediction error:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Predict air quality
 */
async function predictAirQuality() {
    const pm25 = parseFloat(document.getElementById('pm25Input').value) || 0;
    const pm10 = parseFloat(document.getElementById('pm10Input').value) || 0;
    const co2 = parseFloat(document.getElementById('co2Input').value) || 0;

    const resultDiv = document.getElementById('airPredictionResult');
    resultDiv.innerHTML = '<div class="loading"><div class="loading-spinner"></div>Analyzing...</div>';

    const result = await makePrediction('air', { pm25, pm10, co2 });

    if (result.success) {
        resultDiv.innerHTML = `
            <div class="aqi-value-box" style="background-color: ${result.color}; margin: 0 auto;">
                <span class="aqi-number">${result.aqi}</span>
                <span class="aqi-label">${result.classification}</span>
            </div>
            <p class="mt-2 text-center">${result.description}</p>
            ${result.alerts && result.alerts.length > 0 ? `
                <div class="alert-recommendations mt-2">
                    <h5>⚠️ Alerts</h5>
                    <ul>
                        ${result.alerts.map(a => `<li>${a.message}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
    } else {
        resultDiv.innerHTML = '<p class="text-danger">Error making prediction</p>';
    }
}

/**
 * Predict water quality
 */
async function predictWaterQuality() {
    const ph = parseFloat(document.getElementById('phInput').value) || 7;
    const turbidity = parseFloat(document.getElementById('turbidityInput').value) || 0;
    const dissolved_oxygen = parseFloat(document.getElementById('doInput').value) || 0;

    const resultDiv = document.getElementById('waterPredictionResult');
    resultDiv.innerHTML = '<div class="loading"><div class="loading-spinner"></div>Analyzing...</div>';

    const result = await makePrediction('water', { ph, turbidity, dissolved_oxygen });

    if (result.success) {
        resultDiv.innerHTML = `
            <div class="status-badge ${result.quality.toLowerCase()}" style="font-size: 1.5rem; padding: 15px 30px;">
                ${result.quality}
            </div>
            ${result.probabilities ? `
                <div class="mt-2" style="font-size: 0.9rem; color: var(--text-secondary);">
                    Confidence: ${Object.entries(result.probabilities).map(([k, v]) => `${k}: ${v}%`).join(' | ')}
                </div>
            ` : ''}
            ${result.alerts && result.alerts.length > 0 ? `
                <div class="alert-recommendations mt-2">
                    <h5>⚠️ Alerts</h5>
                    <ul>
                        ${result.alerts.map(a => `<li>${a.message}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
    } else {
        resultDiv.innerHTML = '<p class="text-danger">Error making prediction</p>';
    }
}

/**
 * Predict noise level
 */
async function predictNoiseLevel() {
    const sound_level = parseFloat(document.getElementById('soundLevelInput').value) || 0;
    const zone = document.getElementById('zoneInput').value || 'Residential';

    const resultDiv = document.getElementById('noisePredictionResult');
    resultDiv.innerHTML = '<div class="loading"><div class="loading-spinner"></div>Analyzing...</div>';

    const result = await makePrediction('noise', { sound_level, zone });

    if (result.success) {
        resultDiv.innerHTML = `
            <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">
                ${result.sound_level} dB
            </div>
            <div class="status-badge ${result.level.toLowerCase()}" style="font-size: 1.2rem; padding: 10px 25px;">
                ${result.level}
            </div>
            <div class="mt-2" style="color: var(--text-secondary);">
                Zone: ${result.zone}
            </div>
            ${result.alerts && result.alerts.length > 0 ? `
                <div class="alert-recommendations mt-2">
                    <h5>⚠️ Alerts</h5>
                    <ul>
                        ${result.alerts.map(a => `<li>${a.message}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
    } else {
        resultDiv.innerHTML = '<p class="text-danger">Error making prediction</p>';
    }
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

/**
 * Update current time display
 */
function updateTime() {
    const timeElement = document.getElementById('currentTime');
    if (timeElement) {
        const now = new Date();
        timeElement.textContent = now.toLocaleString('en-IN', {
            dateStyle: 'medium',
            timeStyle: 'short'
        });
    }
}

/**
 * Get status badge class based on level
 */
function getStatusClass(level) {
    const levelLower = level.toLowerCase().replace(' ', '-');
    const classMap = {
        'good': 'good',
        'safe': 'safe',
        'low': 'low',
        'moderate': 'moderate',
        'medium': 'medium',
        'poor': 'poor',
        'very-poor': 'very-poor',
        'very poor': 'very-poor',
        'high': 'high',
        'severe': 'severe',
        'polluted': 'polluted'
    };
    return classMap[levelLower] || 'moderate';
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Animate counter
 */
function animateCounter(element, target, duration = 1000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = formatNumber(Math.round(target));
            clearInterval(timer);
        } else {
            element.textContent = formatNumber(Math.round(current));
        }
    }, 16);
}

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', function () {
    // Update time every minute
    updateTime();
    setInterval(updateTime, 60000);

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card, .summary-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });

    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    console.log('Pollution Monitoring System initialized');
});

// ==========================================
// PAGE-SPECIFIC CHART INITIALIZERS
// ==========================================

/**
 * Initialize charts on the Air Quality page
 */
function initAirCharts(historyData) {
    if (!historyData || historyData.length === 0) {
        console.log('No air history data available for charts');
        return;
    }

    // AQI Trend Chart
    const trendCtx = document.getElementById('aqi-trend-chart');
    if (trendCtx) {
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: historyData.map(d => d.date),
                datasets: [{
                    label: 'AQI',
                    data: historyData.map(d => d.aqi),
                    borderColor: chartColors.air,
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: historyData.map(d => {
                        if (d.aqi <= 50) return chartColors.good;
                        if (d.aqi <= 100) return chartColors.moderate;
                        if (d.aqi <= 150) return chartColors.poor;
                        if (d.aqi <= 200) return chartColors.veryPoor;
                        return chartColors.severe;
                    })
                }]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        min: 0,
                        max: 300,
                        title: { display: true, text: 'AQI Value' }
                    }
                }
            }
        });
    }

    // Pollutant Comparison Chart
    const pollutantCtx = document.getElementById('pollutant-chart');
    if (pollutantCtx) {
        const recentData = historyData.slice(-7);
        new Chart(pollutantCtx, {
            type: 'bar',
            data: {
                labels: recentData.map(d => d.date),
                datasets: [
                    {
                        label: 'PM2.5',
                        data: recentData.map(d => d.pm25),
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: chartColors.air,
                        borderWidth: 1
                    },
                    {
                        label: 'PM10',
                        data: recentData.map(d => d.pm10),
                        backgroundColor: 'rgba(155, 89, 182, 0.7)',
                        borderColor: chartColors.noise,
                        borderWidth: 1
                    }
                ]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        title: { display: true, text: 'Concentration (μg/m³)' }
                    }
                }
            }
        });
    }
}

/**
 * Initialize charts on the Water Quality page
 */
function initWaterCharts(historyData) {
    if (!historyData || historyData.length === 0) {
        console.log('No water history data available for charts');
        return;
    }

    // pH Trend Chart
    const phCtx = document.getElementById('ph-trend-chart');
    if (phCtx) {
        new Chart(phCtx, {
            type: 'line',
            data: {
                labels: historyData.map(d => d.date),
                datasets: [{
                    label: 'pH Level',
                    data: historyData.map(d => d.ph),
                    borderColor: chartColors.water,
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4
                }]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        min: 5,
                        max: 10,
                        title: { display: true, text: 'pH Level' }
                    }
                },
                plugins: {
                    ...commonChartOptions.plugins,
                    annotation: {
                        annotations: {
                            safeZone: {
                                type: 'box',
                                yMin: 6.5,
                                yMax: 8.5,
                                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                                borderColor: 'transparent',
                                label: { content: 'Safe Zone', enabled: true }
                            }
                        }
                    }
                }
            }
        });
    }

    // Water Parameters Chart
    const paramsCtx = document.getElementById('water-params-chart');
    if (paramsCtx) {
        const recentData = historyData.slice(-7);
        new Chart(paramsCtx, {
            type: 'bar',
            data: {
                labels: recentData.map(d => d.date),
                datasets: [
                    {
                        label: 'Turbidity (NTU)',
                        data: recentData.map(d => d.turbidity),
                        backgroundColor: 'rgba(241, 196, 15, 0.7)',
                        borderColor: chartColors.moderate,
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Dissolved O₂ (mg/L)',
                        data: recentData.map(d => d.dissolved_oxygen),
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: chartColors.air,
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    x: commonChartOptions.scales.x,
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'Turbidity (NTU)' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'DO (mg/L)' },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
    }
}

/**
 * Initialize charts on the Noise Pollution page
 */
function initNoiseCharts(historyData) {
    if (!historyData || historyData.length === 0) {
        console.log('No noise history data available for charts');
        return;
    }

    // Noise Level Trend Chart
    const trendCtx = document.getElementById('noise-trend-chart');
    if (trendCtx) {
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: historyData.map(d => d.date),
                datasets: [{
                    label: 'Sound Level (dB)',
                    data: historyData.map(d => d.decibel),
                    borderColor: chartColors.noise,
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: historyData.map(d => {
                        if (d.decibel < 55) return chartColors.good;
                        if (d.decibel < 75) return chartColors.moderate;
                        return chartColors.veryPoor;
                    })
                }]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        min: 0,
                        max: 120,
                        title: { display: true, text: 'Sound Level (dB)' }
                    }
                }
            }
        });
    }

    // Zone Comparison Chart
    const zoneCtx = document.getElementById('zone-comparison-chart');
    if (zoneCtx) {
        // Group data by zone
        const zones = ['Industrial', 'Commercial', 'Residential', 'Silence'];
        const zoneData = zones.map(zone => {
            const zoneRecords = historyData.filter(d => d.zone === zone);
            if (zoneRecords.length === 0) return 0;
            return zoneRecords.reduce((sum, d) => sum + d.decibel, 0) / zoneRecords.length;
        });
        const zoneLimits = [75, 65, 55, 50]; // Day limits

        new Chart(zoneCtx, {
            type: 'bar',
            data: {
                labels: zones,
                datasets: [
                    {
                        label: 'Average Level',
                        data: zoneData,
                        backgroundColor: 'rgba(155, 89, 182, 0.7)',
                        borderColor: chartColors.noise,
                        borderWidth: 1
                    },
                    {
                        label: 'Permissible Limit',
                        data: zoneLimits,
                        backgroundColor: 'rgba(46, 204, 113, 0.5)',
                        borderColor: chartColors.good,
                        borderWidth: 1
                    }
                ]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        min: 0,
                        max: 100,
                        title: { display: true, text: 'Sound Level (dB)' }
                    }
                }
            }
        });
    }
}

// Export functions for global access
window.renderAQIChart = renderAQIChart;
window.renderPMChart = renderPMChart;
window.renderWaterChart = renderWaterChart;
window.renderNoiseChart = renderNoiseChart;
window.renderDashboardCharts = renderDashboardCharts;
window.predictAirQuality = predictAirQuality;
window.predictWaterQuality = predictWaterQuality;
window.predictNoiseLevel = predictNoiseLevel;
window.initAirCharts = initAirCharts;
window.initWaterCharts = initWaterCharts;
window.initNoiseCharts = initNoiseCharts;
