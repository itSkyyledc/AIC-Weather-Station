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

# Create data directory if it doesn't exist
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def save_to_csv(station_data, openweather_data):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(DATA_DIR, f"weather_data_{today}.csv")
    
    # Combine both data sources
    combined_data = {
        'timestamp': station_data['timestamp'],
        'station_temperature': station_data['temperature'],
        'station_dew_point': station_data['dew_point'],
        'station_humidity': station_data['humidity'],
        'station_wind_speed': station_data['wind_speed'],
        'station_wind_gust': station_data['wind_gust'],
        'station_wind_direction': station_data['wind_direction'],
        'station_pressure': station_data['pressure'],
        'station_precip_rate': station_data['precip_rate'],
        'station_precip_total': station_data['precip_total'],
        'openweather_temperature': openweather_data['temperature'],
        'openweather_humidity': openweather_data['humidity'],
        'openweather_pressure': openweather_data['pressure'],
        'openweather_wind_speed': openweather_data['wind_speed'],
        'openweather_wind_gust': openweather_data['wind_gust'],
        'openweather_wind_direction': openweather_data['wind_direction'],
        'openweather_precip_rate': openweather_data['precip_rate'],
        'openweather_precip_total': openweather_data['precip_total']
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([combined_data])
    
    # If file exists, append to it, otherwise create new file
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

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
            
            new_data = {
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
            }
            
            weather_data.append(new_data)
            
            # Keep only last 24 hours of data in memory
            while len(weather_data) > 288:  # 288 = 24 hours * 12 (5-minute intervals)
                weather_data.pop(0)
                
            return new_data
    except Exception as e:
        print(f"Error fetching station data: {e}")
        return None

def fetch_openweather_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={CANDABA_LAT}&lon={CANDABA_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        main_data = data.get('main', {})
        wind_data = data.get('wind', {})
        rain_data = data.get('rain', {})
        
        new_data = {
            'timestamp': datetime.now().isoformat(),
            'temperature': main_data.get('temp', 0),
            'humidity': main_data.get('humidity', 0),
            'pressure': main_data.get('pressure', 0),
            'wind_speed': wind_data.get('speed', 0),
            'wind_gust': wind_data.get('gust', 0),
            'wind_direction': wind_data.get('deg', 0),
            'precip_rate': rain_data.get('1h', 0),  # mm/hour
            'precip_total': rain_data.get('3h', 0)  # mm/3hours
        }
        
        openweather_data.append(new_data)
        
        while len(openweather_data) > 288:
            openweather_data.pop(0)
            
        return new_data
    except Exception as e:
        print(f"Error fetching OpenWeather data: {e}")
        return None

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
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(DATA_DIR, f"weather_data_{today}.csv")
    
    if os.path.exists(filename):
        return send_file(
            filename,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"weather_data_{today}.csv"
        )
    return "No data available for download", 404

@app.route('/api/refresh')
def refresh_data():
    try:
        station_data = fetch_station_data()
        openweather_data = fetch_openweather_data()
        
        if station_data and openweather_data:
            save_to_csv(station_data, openweather_data)
            return jsonify({'status': 'success', 'message': 'Data refreshed successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to fetch data from one or more sources'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Fetch initial data
    fetch_station_data()
    fetch_openweather_data()
    app.run(debug=True) 