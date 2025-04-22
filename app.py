from flask import Flask, render_template, jsonify, send_file
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)

# Configuration
WEATHER_STATION_API_KEY = os.getenv("WEATHER_STATION_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
STATION_ID = "ICANDA11"
CANDABA_LAT = 15.13
CANDABA_LON = 120.90

# Data storage
weather_data = []
openweather_data = []

def fetch_station_data():
    url = f"https://api.weather.com/v2/pws/observations/current?stationId={STATION_ID}&format=json&units=m&apiKey={WEATHER_STATION_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "observations" in data and data["observations"]:
            obs = data["observations"][0]
            metric = obs.get('metric', {})
            
            # Calculate dew point if not provided
            temp = metric.get('temp', 0)
            humidity = obs.get('humidity', 0)
            dew_point = temp - ((100 - humidity) / 5)  # Approximate calculation
            
            weather_data.append({
                'timestamp': datetime.now().isoformat(),
                'temperature': metric.get('temp', 0),
                'dew_point': dew_point,
                'humidity': obs.get('humidity', 0),
                'wind_speed': metric.get('windSpeed', 0),
                'wind_gust': metric.get('windGust', 0),
                'wind_direction': obs.get('winddir', 0),
                'pressure': metric.get('pressure', 0),
                'precip_rate': metric.get('precipRate', 0),
                'precip_total': metric.get('precipTotal', 0)
            })
            
            # Keep only last 24 hours of data
            while len(weather_data) > 288:  # 288 = 24 hours * 12 (5-minute intervals)
                weather_data.pop(0)
    except Exception as e:
        print(f"Error fetching station data: {e}")

def fetch_openweather_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={CANDABA_LAT}&lon={CANDABA_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        main_data = data.get('main', {})
        wind_data = data.get('wind', {})
        rain_data = data.get('rain', {})
        
        openweather_data.append({
            'timestamp': datetime.now().isoformat(),
            'temperature': main_data.get('temp', 0),
            'humidity': main_data.get('humidity', 0),
            'pressure': main_data.get('pressure', 0),
            'wind_speed': wind_data.get('speed', 0),
            'wind_gust': wind_data.get('gust', 0),
            'wind_direction': wind_data.get('deg', 0),
            'precip_rate': rain_data.get('1h', 0),  # mm/hour
            'precip_total': rain_data.get('3h', 0)  # mm/3hours
        })
        
        while len(openweather_data) > 288:
            openweather_data.pop(0)
    except Exception as e:
        print(f"Error fetching OpenWeather data: {e}")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_station_data, 'interval', minutes=5)
scheduler.add_job(fetch_openweather_data, 'interval', minutes=5)
scheduler.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/current')
def current_data():
    if weather_data:
        return jsonify({
            'station_data': weather_data[-1],
            'openweather_data': openweather_data[-1] if openweather_data else None
        })
    return jsonify({'error': 'No data available'})

@app.route('/api/history')
def history_data():
    return jsonify({
        'station_data': weather_data,
        'openweather_data': openweather_data
    })

@app.route('/download/csv')
def download_csv():
    df = pd.DataFrame(weather_data)
    csv_path = 'weather_data.csv'
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    # Fetch initial data
    fetch_station_data()
    fetch_openweather_data()
    app.run(debug=True) 