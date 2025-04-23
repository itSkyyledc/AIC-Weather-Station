from flask import Flask, render_template, jsonify, send_file, request
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import cv2
import numpy as np
from urllib.parse import urlparse
import subprocess
import tempfile
import time
import io
import threading
from collections import defaultdict

load_dotenv()

app = Flask(__name__)

# Configuration
WEATHER_STATION_API_KEY = os.getenv("WEATHER_STATION_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
STATION_ID = "ICANDA12"
CANDABA_LAT = 15.13
CANDABA_LON = 120.90

# Camera configuration
CAMERAS = {
    'camera1': {
        'name': 'Camera 1',
        'rtsp_url': 'rtsp://TAPO941A:visibility@192.168.0.112:554/stream1',
        'location': 'Location 1'
    },
    'camera2': {
        'name': 'Camera 2',
        'rtsp_url': 'rtsp://TAPOC8EA:visibility@192.168.0.111:554/stream1',
        'location': 'Location 2'
    },
    'camera3': {
        'name': 'Camera 3',
        'rtsp_url': 'rtsp://buth:4ytkfe@192.168.0.100/live/ch00_1',
        'location': 'Location 3'
    }
}

# Data storage
weather_data = []
openweather_data = []

# Create directories if they don't exist
DATA_DIR = "data"
VISIBILITY_DATA_DIR = os.path.join(DATA_DIR, "visibility")
STATIC_IMAGES_DIR = "static/images"
ROI_CONFIG_FILE = os.path.join(DATA_DIR, "roi_config.json")

for directory in [DATA_DIR, VISIBILITY_DATA_DIR, STATIC_IMAGES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Camera snapshot cache and visibility data storage
camera_cache = {
    'camera1': {'image': None, 'timestamp': None},
    'camera2': {'image': None, 'timestamp': None},
    'camera3': {'image': None, 'timestamp': None}
}

# Modify the camera_rois and visibility_history structure to support multiple ROIs
camera_rois = defaultdict(dict)  # {camera_id: {roi_id: {coords: {...}, distance: int, label: str}}}
visibility_data = defaultdict(lambda: {
    'roi_data': {},
    'max_visibility': 0
})

def calculate_visibility_distance(edge_counts, threshold_percentage=5):
    """Calculate the maximum visibility distance based on ROI edge counts."""
    max_visibility = 0
    for roi_id, data in sorted(edge_counts.items(), key=lambda x: x[1]['distance']):
        current_edge_density = data['edge_density']
        distance = data['distance']
        visibility = data['visibility']
        
        # Update max visibility based on the highest visibility value
        max_visibility = max(max_visibility, visibility)
    
    return max_visibility

def calculate_edges(image, roi_coords):
    """Calculate edge density in the specified ROI."""
    try:
        # Extract ROI coordinates
        x1, y1, x2, y2 = roi_coords
        roi = image[y1:y2, x1:x2]
        
        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density (percentage of edge pixels)
        total_pixels = (y2 - y1) * (x2 - x1)
        edge_pixels = np.count_nonzero(edges)
        edge_density = (edge_pixels / total_pixels) * 100
        
        return edge_density
        
    except Exception as e:
        print(f"Error in calculate_edges: {str(e)}")
        return 0

def calculate_visibility(edge_density, distance):
    """Calculate visibility metric based on edge density and distance."""
    try:
        # Base visibility on edge density and distance
        # Higher edge density indicates better visibility
        # Scale the visibility based on the distance
        visibility = (edge_density / 100.0) * distance
        
        # Ensure visibility doesn't exceed the actual distance
        visibility = min(visibility, distance)
        
        return round(visibility, 2)
        
    except Exception as e:
        print(f"Error in calculate_visibility: {str(e)}")
        return 0

def get_camera_snapshot(rtsp_url):
    """Capture a snapshot from an RTSP stream using OpenCV."""
    try:
        # Create a VideoCapture object
        cap = cv2.VideoCapture(rtsp_url)
        
        # Set a timeout for the connection
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        
        # Try to read a frame
        ret, frame = cap.read()
        
        # Release the capture
        cap.release()
        
        if ret and frame is not None:
            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return buffer.tobytes()
        
        print(f"Failed to capture frame from {rtsp_url}")
        return None
    except Exception as e:
        print(f"Exception capturing snapshot: {str(e)}")
        return None

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

def update_camera_cache():
    """Update the camera cache with fresh snapshots"""
    while True:
        for camera_id, camera in CAMERAS.items():
            try:
                image_data = get_camera_snapshot(camera['rtsp_url'])
                if image_data:
                    camera_cache[camera_id] = {
                        'image': image_data,
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"Error updating camera {camera_id}: {e}")
        time.sleep(30)  # Update every 30 seconds

# Start the camera update thread
camera_thread = threading.Thread(target=update_camera_cache, daemon=True)
camera_thread.start()

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

@app.route('/visibility')
def visibility():
    return render_template('visibility.html')

@app.route('/api/camera/<camera_id>/snapshot')
def camera_snapshot(camera_id):
    if camera_id not in CAMERAS:
        return jsonify({'error': 'Camera not found'}), 404
    
    if camera_id not in camera_cache or not camera_cache[camera_id]['image']:
        # If no cached image, try to get a fresh one
        camera = CAMERAS[camera_id]
        image_data = get_camera_snapshot(camera['rtsp_url'])
        if image_data:
            camera_cache[camera_id] = {
                'image': image_data,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Return error image if snapshot failed
            return send_file(
                os.path.join(STATIC_IMAGES_DIR, 'error.png'),
                mimetype='image/png',
                as_attachment=False
            )
    
    return send_file(
        io.BytesIO(camera_cache[camera_id]['image']),
        mimetype='image/jpeg',
        as_attachment=False
    )

# Load ROI configuration from file
def load_roi_config():
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ROI config: {e}")
    return {}

def save_roi_config():
    try:
        with open(ROI_CONFIG_FILE, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            config_data = {camera_id: dict(rois) for camera_id, rois in camera_rois.items()}
            json.dump(config_data, f, indent=2)
    except Exception as e:
        print(f"Error saving ROI config: {e}")

# Initialize camera_rois from saved config
camera_rois = defaultdict(dict, load_roi_config())

@app.route('/api/camera/<camera_id>/roi', methods=['POST'])
def set_roi(camera_id):
    roi_data = request.json
    roi_id = f"roi_{len(camera_rois[camera_id])}"
    
    camera_rois[camera_id][roi_id] = {
        'coords': {
            'x': roi_data['x'],
            'y': roi_data['y'],
            'width': roi_data['width'],
            'height': roi_data['height']
        },
        'distance': roi_data['distance'],
        'label': roi_data.get('label', f'ROI {roi_data["distance"]}m')
    }
    
    # Save ROI configuration to file
    save_roi_config()
    
    return jsonify({'status': 'success', 'roi_id': roi_id})

@app.route('/api/camera/<camera_id>/rois', methods=['GET'])
def get_rois(camera_id):
    """Get all ROIs for a camera."""
    return jsonify({
        'status': 'success',
        'rois': camera_rois.get(camera_id, {})
    })

def save_visibility_to_csv(camera_id, visibility_data):
    """Save visibility data to a CSV file for the specific camera."""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(VISIBILITY_DATA_DIR, f"visibility_data_{camera_id}_{today}.csv")
    
    # Prepare data for CSV
    data_row = {
        'timestamp': visibility_data['timestamp'],
        'max_visibility': visibility_data['max_visibility']
    }
    
    # Add ROI-specific data
    for roi_id, roi_data in visibility_data['roi_data'].items():
        prefix = f"{roi_id}_"
        data_row.update({
            f"{prefix}edge_density": roi_data['edge_density'],
            f"{prefix}visibility": roi_data['visibility'],
            f"{prefix}change_percentage": roi_data['change_percentage'],
            f"{prefix}distance": roi_data['distance'],
            f"{prefix}label": roi_data['label']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame([data_row])
    
    # If file exists, append to it, otherwise create new file
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

@app.route('/refresh_camera', methods=['POST'])
def refresh_camera():
    """Process new camera frame and calculate visibility metrics for all ROIs."""
    try:
        # Get the frame data
        frame_data = request.json.get('frame')
        rois = request.json.get('rois', [])
        
        # Convert frame data to image
        frame_array = np.array(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Process each ROI
        roi_results = []
        max_visibility = 0
        
        for roi in rois:
            # Extract ROI coordinates and distance
            roi_coords = (roi['x'], roi['y'], 
                        roi['x'] + roi['width'], 
                        roi['y'] + roi['height'])
            distance = roi.get('distance', 1000)  # Default to 1000m if not specified
            
            # Calculate edge density for this ROI
            edge_density = calculate_edges(frame, roi_coords)
            
            # Calculate visibility for this ROI
            visibility = calculate_visibility(edge_density, distance)
            
            # Update max visibility if this ROI has better visibility
            max_visibility = max(max_visibility, visibility)
            
            # Store results for this ROI
            roi_results.append({
                'id': roi.get('id', ''),
                'label': roi.get('label', ''),
                'edge_density': round(edge_density, 2),
                'visibility': visibility,
                'distance': distance
            })
        
        # Prepare response data
        response_data = {
            'max_visibility': max_visibility,
            'roi_data': roi_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in refresh_camera: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/<camera_id>/download/csv')
def download_visibility_csv(camera_id):
    """Download visibility data CSV for a specific camera."""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(VISIBILITY_DATA_DIR, f"visibility_data_{camera_id}_{today}.csv")
    
    if os.path.exists(filename):
        return send_file(
            filename,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"visibility_data_{camera_id}_{today}.csv"
        )
    return jsonify({'error': 'No data available for download'}), 404

if __name__ == '__main__':
    # Fetch initial data
    fetch_station_data()
    fetch_openweather_data()
    app.run(debug=True, host='0.0.0.0', port=5000) 