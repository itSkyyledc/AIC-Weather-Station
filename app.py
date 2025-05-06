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
from pytz import utc

load_dotenv()

app = Flask(__name__)

# Configuration
WEATHER_STATION_API_KEY = os.getenv("WEATHER_STATION_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
STATION_ID = "IQUEZO39"
STATION_ID_2 = "ICANDA12"  # Add second station ID

CANDABA_LAT = 14.63
CANDABA_LON = 121.078
STATION2_LAT = 15.13  # Add second station coordinates
STATION2_LON = 120.90

# Directory configuration
DATA_DIR = "data"
VISIBILITY_DATA_DIR = os.path.join(DATA_DIR, "visibility")
STATIC_IMAGES_DIR = "static/images"
CONFIG_DIR = os.path.join(DATA_DIR, "config")
ROI_CONFIG_FILE = os.path.join(CONFIG_DIR, "roi_config.json")

# Create directories if they don't exist
for directory in [DATA_DIR, VISIBILITY_DATA_DIR, STATIC_IMAGES_DIR, CONFIG_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Camera configuration
CAMERAS = {
    'camera1': {
        'name': 'Camera 1',
        'rtsp_url': 'rtsp://TAPO941A:visibility@192.168.0.112:554/stream1',
        'location': 'Location 1',
        'username': 'TAPO941A',
        'password': 'visibility',
        'ip': '192.168.0.112',
        'port': 554,
        'stream': 'stream1'
    },
    'camera2': {
        'name': 'Camera 2',
        'rtsp_url': 'rtsp://TAPOC8EA:visibility@192.168.0.111:554/stream1',
        'location': 'Location 2',
        'username': 'TAPOC8EA',
        'password': 'visibility',
        'ip': '192.168.0.111',
        'port': 554,
        'stream': 'stream1'
    },
    'camera3': {
        'name': 'Camera 3',
        'rtsp_url': 'rtsp://buth:4ytkfe@192.168.0.100/live/ch00_1',
        'location': 'Location 3',
        'username': 'buth',
        'password': '4ytkfe',
        'ip': '192.168.0.100',
        'port': 554,
        'stream': 'live/ch00_1'
    }
}

# Data storage
weather_data = []
weather_data_2 = []  # Add storage for second station
openweather_data = []
openweather_data_2 = []  # Add storage for second station's OpenWeather data

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

# Add reference visibility storage with history
reference_visibility = defaultdict(lambda: {
    'current': {
        'roi_data': {},
        'timestamp': None,
        'max_visibility': 0
    },
    'history': [],  # List of historical midday references
    'best': {       # Best visibility reference (100% baseline)
        'roi_data': {},
        'timestamp': None,
        'max_visibility': 0
    }
})

def calculate_visibility_distance(edge_counts, threshold_percentage=5):
    """Calculate the maximum visibility distance based on ROI edge counts."""
    try:
        max_visibility = 0.0
        for roi_id, data in sorted(edge_counts.items(), key=lambda x: x[1]['distance']):
            current_edge_density = float(data['edge_density'])
            distance = float(data['distance'])
            visibility = float(data['visibility'])
            
            # Update max visibility based on the highest visibility value
            max_visibility = max(max_visibility, visibility)
        
        return float(max_visibility)
    except Exception as e:
        print(f"Error in calculate_visibility_distance: {str(e)}")
        return 0.0

def calculate_edges(image, roi_coords):
    """Calculate edge density in the specified ROI."""
    try:
        # Extract ROI coordinates
        x1, y1, x2, y2 = roi_coords
        roi = image[y1:y2, x1:x2]
        
        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density (percentage of edge pixels)
        total_pixels = (y2 - y1) * (x2 - x1)
        edge_pixels = np.count_nonzero(edges)
        edge_density = (edge_pixels / total_pixels) * 100
        
        return float(edge_density)  # Ensure we return a scalar value
        
    except Exception as e:
        print(f"Error in calculate_edges: {str(e)}")
        return 0.0

def calculate_visibility(edge_density, distance, scaling_factor=1.8):
    """Calculate visibility based on edge density and distance."""
    try:
        # Ensure inputs are scalar values
        edge_density = float(edge_density)
        distance = float(distance)

        # Adjust sigmoid function to better reflect visibility
        visibility_factor = 1 / (1 + np.exp(-(edge_density - 25) / 5))

        # Scale the visibility factor to the distance
        visibility = distance * visibility_factor * scaling_factor

        # Boost visibility for high edge density in clear conditions
        if edge_density > 70:
            visibility *= 1.2  # 20% boost for very clear conditions

        # Ensure visibility doesn't exceed the ROI's distance
        return float(min(visibility, distance))
    except Exception as e:
        print(f"Error in calculate_visibility: {str(e)}")
        return 0.0

def calculate_visibility_with_weather(edge_density, distance, weather_data):
    """Calculate visibility with weather adjustments."""
    visibility = calculate_visibility(edge_density, distance)
    if weather_data:
        humidity = weather_data.get('humidity', 0)
        precipitation = weather_data.get('precip_rate', 0)
        visibility = adjust_visibility_for_weather(visibility, humidity, precipitation)
    return visibility

def calculate_visibility_with_sampling(edge_density_samples, distance):
    """Calculate visibility by sampling edge density multiple times and averaging the middle values."""
    try:
        # Sort the edge density samples
        sorted_samples = sorted(edge_density_samples)

        # Remove the 3 lowest and 3 highest samples
        trimmed_samples = sorted_samples[3:-3] if len(sorted_samples) > 6 else sorted_samples

        # Calculate the average of the remaining samples
        average_edge_density = sum(trimmed_samples) / len(trimmed_samples) if trimmed_samples else 0

        # Use the average edge density to calculate visibility
        return calculate_visibility(average_edge_density, distance)
    except Exception as e:
        print(f"Error in calculate_visibility_with_sampling: {str(e)}")
        return 0.0

def get_camera_snapshot(rtsp_url):
    """Capture a snapshot from an RTSP stream."""
    try:
        # Create video capture object
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Failed to open RTSP stream: {rtsp_url}")
            return None
            
        # Set timeout
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        # Check if frame is valid
        if not ret or frame is None or frame.size == 0 or not isinstance(frame, np.ndarray):
            print("Failed to read frame from RTSP stream")
            return None
            
        # Ensure frame has valid dimensions
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Invalid frame dimensions")
            return None
            
        return frame
        
    except Exception as e:
        print(f"Error in get_camera_snapshot: {str(e)}")
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

def fetch_metar_data():
    """Fetch METAR data and include it in the weather station."""
    try:
        # Example METAR API URL (replace with actual API endpoint)
        url = f"https://api.metarservice.com/metar?station={STATION_ID}&format=json"
        response = requests.get(url)
        data = response.json()

        # Extract relevant METAR fields
        metar_data = {
            'timestamp': datetime.now().isoformat(),
            'temperature': data.get('temperature', 0),
            'dew_point': data.get('dew_point', 0),
            'humidity': data.get('humidity', 0),
            'wind_speed': data.get('wind_speed', 0),
            'wind_direction': data.get('wind_direction', 0),
            'pressure': data.get('pressure', 0),
            'visibility': data.get('visibility', 0),
            'cloud_cover': data.get('cloud_cover', 0),
            'weather_conditions': data.get('weather_conditions', '')
        }

        # Save METAR data to a CSV file
        today = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DATA_DIR, f"metar_data_{today}.csv")
        df = pd.DataFrame([metar_data])
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)

        return metar_data
    except Exception as e:
        print(f"Error fetching METAR data: {e}")
        return None

def process_visibility_data():
    """Process visibility data for all cameras."""
    for camera_id, camera in CAMERAS.items():
        try:
            # Get the latest snapshot for the camera
            snapshot = get_camera_snapshot(camera['rtsp_url'])
            if snapshot is None:
                continue

            # Get ROIs for the camera
            rois = get_camera_rois(camera_id)
            if not rois:
                continue

            # Use the latest weather data
            latest_weather_data = weather_data[-1] if weather_data else {}

            # Process the frame and calculate metrics
            metrics = process_frame_with_sampling(snapshot, rois, latest_weather_data)
            if metrics is None:
                continue

            # Save visibility data to CSV
            save_visibility_to_csv(camera_id, {
                'timestamp': datetime.now().isoformat(),
                'max_visibility': metrics['max_visibility'],
                'avg_visibility': metrics['avg_visibility'],
                'roi_data': metrics['roi_data']
            })
        except Exception as e:
            print(f"Error processing visibility data for camera {camera_id}: {str(e)}")

def validate_camera_connection(camera_id):
    """Validate camera connection and update RTSP URL if needed."""
    try:
        camera = CAMERAS.get(camera_id)
        if not camera:
            return False
            
        # Try different RTSP URL formats
        rtsp_formats = [
            f"rtsp://{camera['username']}:{camera['password']}@{camera['ip']}:{camera['port']}/{camera['stream']}",
            f"rtsp://{camera['username']}:{camera['password']}@{camera['ip']}:{camera['port']}/stream1",
            f"rtsp://{camera['username']}:{camera['password']}@{camera['ip']}:{camera['port']}/live/ch00_1",
            f"rtsp://{camera['username']}:{camera['password']}@{camera['ip']}:{camera['port']}/cam/realmonitor?channel=1&subtype=0"
        ]
        
        for rtsp_url in rtsp_formats:
            print(f"Trying RTSP URL: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    # Update the camera's RTSP URL
                    CAMERAS[camera_id]['rtsp_url'] = rtsp_url
                    print(f"Successfully connected to camera {camera_id} using URL: {rtsp_url}")
                    return True
                    
        print(f"Failed to connect to camera {camera_id} with any RTSP URL format")
        return False
        
    except Exception as e:
        print(f"Error validating camera {camera_id} connection: {str(e)}")
        return False

def update_camera_cache():
    """Update the camera cache with fresh snapshots"""
    while True:
        for camera_id, camera in CAMERAS.items():
            try:
                image_data = get_camera_snapshot(camera['rtsp_url'])
                if image_data is not None and isinstance(image_data, np.ndarray):
                    camera_cache[camera_id] = {
                        'image': image_data,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    print(f"Invalid image data for camera {camera_id}")
            except Exception as e:
                print(f"Error updating camera {camera_id}: {str(e)}")
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
    try:
        # Fetch historical data for all cameras
        historical_data = {}
        for camera_id in CAMERAS.keys():
            history = get_camera_history(camera_id)
            if history:
                historical_data[camera_id] = {
                    'timestamps': history.get('timestamps', []),
                    'max_visibility': history.get('max_visibility', []),
                    'avg_visibility': history.get('avg_visibility', []),
                    'roi_data': history.get('roi_data', []),
                    'reference': history.get('reference', None)  # Use get() with default None
                }

        # Pass historical data to the template
        return render_template('visibility.html', 
                             historical_data=json.dumps(historical_data),
                             cameras=CAMERAS)
    except Exception as e:
        print(f"Error in visibility route: {str(e)}")
        return render_template('visibility.html', 
                             error="Failed to load historical data",
                             cameras=CAMERAS)

@app.route('/api/camera/<camera_id>/snapshot')
def camera_snapshot(camera_id):
    """Get a snapshot from the specified camera."""
    try:
        if camera_id not in CAMERAS:
            return jsonify({'error': 'Camera not found'}), 404
            
        # Get camera configuration
        camera = CAMERAS[camera_id]
        
        # Get snapshot from RTSP stream
        image_data = get_camera_snapshot(camera['rtsp_url'])
        if image_data is None:
            # Return error image if snapshot failed
            return send_file(
                os.path.join(STATIC_IMAGES_DIR, 'error.png'),
                mimetype='image/png',
                as_attachment=False
            )
            
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', image_data)
        if not ret:
            return jsonify({'error': 'Failed to encode image'}), 500
            
        # Return image data
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"Error in camera_snapshot: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Load ROI configuration from file
def load_roi_config():
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ROI config: {e}")
    return {}

def save_roi_config(camera_id, rois):
    """Save ROI configuration for a camera."""
    try:
        config = {}
        if os.path.exists(ROI_CONFIG_FILE):
            with open(ROI_CONFIG_FILE, 'r') as f:
                config = json.load(f)
        
        config[camera_id] = {'rois': rois}
        
        with open(ROI_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
            
    except Exception as e:
        print(f"Error saving ROI config: {str(e)}")

# Initialize camera_rois from saved config
camera_rois = defaultdict(dict, load_roi_config())

@app.route('/api/camera/<camera_id>/roi', methods=['POST'])
def set_roi(camera_id):
    try:
        roi_data = request.json
        rois = get_camera_rois(camera_id)
        
        # Ensure 'coords' is initialized when setting ROI
        if 'coords' not in roi_data or not all(k in roi_data for k in ['x', 'y', 'width', 'height']):
            raise ValueError("Missing 'coords' or required keys (x, y, width, height) in ROI data.")
        
        # Generate a unique ROI ID
        roi_id = f"roi_{len(rois)}"
        
        # Add new ROI
        rois[roi_id] = {
            'coords': {
                'x': int(roi_data['x']),
                'y': int(roi_data['y']),
                'width': int(roi_data['width']),
                'height': int(roi_data['height'])
            },
            'distance': float(roi_data['distance']),
            'label': roi_data.get('label', f'ROI {roi_data["distance"]}m')
        }
        
        # Save updated configuration
        save_roi_config(camera_id, rois)
        
        return jsonify({'status': 'success', 'roi_id': roi_id})
        
    except Exception as e:
        print(f"Error setting ROI: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/<camera_id>/rois', methods=['GET'])
def get_rois(camera_id):
    """Get all ROIs for a camera."""
    try:
        rois = get_camera_rois(camera_id)
        return jsonify({
            'status': 'success',
            'rois': rois
        })
    except Exception as e:
        print(f"Error getting ROIs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/<camera_id>/roi/<roi_id>', methods=['DELETE'])
def delete_roi(camera_id, roi_id):
    """Delete a specific ROI for a camera."""
    try:
        if camera_id not in camera_rois:
            return jsonify({'status': 'error', 'message': 'Camera not found'}), 404
            
        if roi_id not in camera_rois[camera_id]:
            return jsonify({'status': 'error', 'message': 'ROI not found'}), 404
            
        # Remove the ROI
        del camera_rois[camera_id][roi_id]
        
        # Save the updated configuration
        save_roi_config(camera_id, camera_rois[camera_id])
        
        return jsonify({'status': 'success', 'message': 'ROI deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting ROI: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def update_reference_visibility(camera_id, visibility_data):
    """Update reference visibility during midday and maintain historical references."""
    current_time = datetime.now()
    if 11 <= current_time.hour <= 13:
        # Create reference data entry with focus on edge density
        ref_entry = {
            'roi_data': {roi['id']: {
                'edge_density': roi['edge_density'],
                'visibility': roi['visibility'],
                'distance': roi['distance'],
                'label': roi['label']
            } for roi in visibility_data['roi_data']},
            'timestamp': visibility_data['timestamp'],
            'max_visibility': visibility_data['max_visibility'],
            'avg_visibility': visibility_data['avg_visibility']
        }
        
        # Add to history
        reference_visibility[camera_id]['history'].append(ref_entry)
        
        # Update current reference
        reference_visibility[camera_id]['current'] = ref_entry
        
        # Keep only last 30 days of history
        max_history = 30 * 2  # 30 days * 2 measurements per day
        if len(reference_visibility[camera_id]['history']) > max_history:
            reference_visibility[camera_id]['history'] = reference_visibility[camera_id]['history'][-max_history:]
        
        # Update best reference based on average edge density
        current_best = reference_visibility[camera_id]['best']
        current_avg_edge_density = sum(roi['edge_density'] for roi in visibility_data['roi_data']) / len(visibility_data['roi_data'])
        best_avg_edge_density = 0
        
        if current_best['timestamp']:
            best_avg_edge_density = sum(roi['edge_density'] for roi in current_best['roi_data'].values()) / len(current_best['roi_data'])
        
        if not current_best['timestamp'] or current_avg_edge_density > best_avg_edge_density:
            reference_visibility[camera_id]['best'] = ref_entry
            print(f"New best visibility reference for camera {camera_id}:")
            print(f"Average Edge Density: {current_avg_edge_density:.2f}% at {ref_entry['timestamp']}")
        
        print(f"Updated reference visibility for camera {camera_id} at {current_time}")

def calculate_visibility_score(current_data, camera_id, roi_id):
    """Calculate visibility score relative to best reference data based on edge density."""
    best_reference = reference_visibility[camera_id]['best']
    if not best_reference or not best_reference['roi_data']:
        return current_data['visibility'], 100  # Return raw visibility if no reference
    
    # Get reference data for this ROI
    ref_data = best_reference['roi_data'].get(roi_id, {})
    if not ref_data:
        return current_data['visibility'], 100
    
    # Calculate visibility score based on the ratio of current visibility to maximum possible visibility
    visibility = current_data['visibility']
    distance = ref_data.get('distance', visibility)  # Use ROI distance as maximum possible
    
    if distance > 0:
        # Score is the percentage of maximum possible visibility achieved
        score = (visibility / distance) * 100
    else:
        score = 100 if visibility > 0 else 0
    
    return visibility, min(100, max(0, score))

@app.route('/api/camera/<camera_id>/references', methods=['GET'])
def get_visibility_references(camera_id):
    """Get visibility reference data for a camera."""
    try:
        if camera_id not in reference_visibility:
            return jsonify({'error': 'No reference data available'}), 404
        
        ref_data = reference_visibility[camera_id]
        return jsonify({
            'current': ref_data['current'],
            'best': ref_data['best'],
            'history': sorted(ref_data['history'], 
                            key=lambda x: x['timestamp'],
                            reverse=True)[:10]  # Return last 10 references
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def save_visibility_to_csv(camera_id, data):
    """Save visibility data to CSV file."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(VISIBILITY_DATA_DIR, f"visibility_data_{camera_id}_{today}.csv")
        
        # Prepare data for CSV
        row = {
            'timestamp': data['timestamp'],
            'max_visibility': data['max_visibility'],
            'avg_visibility': data['avg_visibility']
        }
        
        # Add ROI data
        for roi in data['roi_data']:
            # Calculate average edge density from samples if available
            if 'edge_density_samples' in roi:
                edge_density = sum(roi['edge_density_samples']) / len(roi['edge_density_samples'])
            elif 'edge_density' in roi:
                edge_density = roi['edge_density']
            else:
                print(f"Missing edge density data for ROI {roi['id']}")
                continue

            prefix = f"roi_{roi['id']}_"
            row.update({
                f"{prefix}id": roi['id'],
                f"{prefix}label": roi['label'],
                f"{prefix}distance": roi['distance'],
                f"{prefix}edge_density": edge_density,
                f"{prefix}visibility": roi['visibility'],
                f"{prefix}visibility_percentage": roi['visibility_percentage']
            })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Write to CSV
        df = pd.DataFrame([row])
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
            
    except Exception as e:
        print(f"Error saving visibility data: {str(e)}")

@app.route('/api/camera/<camera_id>/history/1hour', methods=['GET'])
def get_camera_history_api(camera_id):
    """API endpoint to get camera history."""
    history = get_camera_history(camera_id)
    if history is None:
        return jsonify({'error': 'Failed to fetch camera history'}), 500
    return jsonify(history)

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

def get_camera_history(camera_id):
    """Get historical visibility data for a camera."""
    try:
        # Get the camera's history from the global dictionary
        camera_history = visibility_data.get(camera_id, {})
        
        # Get reference data
        reference = get_reference_data(camera_id)
        
        # Process ROI data
        roi_data = []
        for roi_id, roi in camera_history.get('camera_rois', {}).items():
            roi_history = {
                'id': roi_id,
                'label': roi.get('label', f'ROI {roi_id}'),
                'distance': roi.get('distance', 0),
                'history': {
                    'visibility': roi.get('visibility_history', []),
                    'edge_density': roi.get('edge_density_history', [])
                }
            }
            roi_data.append(roi_history)
        
        # Return structured history data
        return {
            'timestamps': camera_history.get('timestamps', []),
            'max_visibility': camera_history.get('max_visibility', []),
            'avg_visibility': camera_history.get('avg_visibility', []),
            'roi_data': roi_data,
            'reference': reference
        }
    except Exception as e:
        print(f"Error getting camera history: {str(e)}")
        return {
            'timestamps': [],
            'max_visibility': [],
            'avg_visibility': [],
            'roi_data': [],
            'reference': None
        }

def get_reference_data(camera_id):
    """Get reference visibility data for a camera."""
    try:
        if camera_id not in reference_visibility:
            return None
            
        ref_data = reference_visibility[camera_id]
        if not ref_data['best']['timestamp']:
            return None
            
        # Calculate average edge density for best reference
        best_roi_data = ref_data['best']['roi_data']
        avg_edge_density = sum(roi['edge_density'] for roi in best_roi_data.values()) / len(best_roi_data) if best_roi_data else 0
        
        return {
            'best': {
                'timestamp': ref_data['best']['timestamp'],
                'max_visibility': ref_data['best']['max_visibility'],
                'avg_visibility': ref_data['best']['avg_visibility'],
                'avg_edge_density': avg_edge_density,
                'roi_data': ref_data['best']['roi_data']
            }
        }
    except Exception as e:
        print(f"Error getting reference data: {str(e)}")
        return None

def get_camera_rois(camera_id):
    """Get ROI definitions for a camera."""
    try:
        if not os.path.exists(ROI_CONFIG_FILE):
            return {}
            
        with open(ROI_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get(camera_id, {}).get('rois', {})
    except Exception as e:
        print(f"Error getting camera ROIs: {str(e)}")
        return {}

def process_frame_with_sampling(frame, rois, weather_data=None):
    """Process a frame and calculate visibility metrics for all ROIs using sampling."""
    try:
        roi_data = []
        max_visibility = 0.0
        total_visibility = 0.0

        for roi_id, roi in rois.items():
            try:
                # Extract ROI coordinates
                x = int(roi['coords']['x'])
                y = int(roi['coords']['y'])
                w = int(roi['coords']['width'])
                h = int(roi['coords']['height'])
                distance = float(roi['distance'])

                # Validate ROI dimensions
                if w <= 0 or h <= 0:
                    print(f"Invalid ROI dimensions for {roi_id}")
                    continue

                # Sample edge density 12 times
                edge_density_samples = []
                for _ in range(12):
                    edge_density = calculate_edges(frame, (x, y, x + w, y + h))
                    edge_density_samples.append(edge_density)

                # Calculate visibility using sampling
                visibility = calculate_visibility_with_sampling(edge_density_samples, distance)

                # Update max and total visibility
                max_visibility = max(max_visibility, visibility)
                total_visibility += visibility

                # Calculate visibility percentage relative to the ROI's distance
                visibility_percentage = (visibility / distance) * 100 if distance > 0 else 0

                # Create ROI data entry
                roi_data.append({
                    'id': roi_id,
                    'label': roi['label'],
                    'distance': distance,
                    'edge_density_samples': edge_density_samples,
                    'visibility': visibility,
                    'visibility_percentage': visibility_percentage
                })
            except Exception as e:
                print(f"Error processing ROI {roi_id}: {str(e)}")
                continue

        # Calculate average visibility
        avg_visibility = total_visibility / len(roi_data) if roi_data else 0.0

        return {
            'max_visibility': max_visibility,
            'avg_visibility': avg_visibility,
            'roi_data': roi_data
        }
    except Exception as e:
        print(f"Error in process_frame_with_sampling: {str(e)}")
        return None

def adjust_visibility_for_weather(visibility, humidity, precipitation):
    """Adjust visibility based on weather conditions."""
    try:
        # Reduce visibility based on humidity (e.g., 1% reduction per 10% humidity above 50%)
        if (humidity > 50):
            visibility *= (1 - (humidity - 50) / 100)
        
        # Further reduce visibility if precipitation is present
        if (precipitation > 0):
            visibility *= (1 - precipitation / 10)  # Example: 10% reduction per mm/hour
        
        return max(0, visibility)  # Ensure visibility is non-negative
    except Exception as e:
        print(f"Error in adjust_visibility_for_weather: {str(e)}")
        return visibility

@app.route('/refresh_camera', methods=['POST'])
def refresh_camera():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400

        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({'error': 'No camera ID provided'}), 400

        # Get current ROIs for this camera
        rois = get_camera_rois(camera_id)
        if not rois:
            return jsonify({'error': 'No ROIs defined'}), 400

        # Get frame data
        frame_data = data.get('frame')
        width = data.get('width')
        height = data.get('height')
        
        if not all([frame_data, width, height]):
            return jsonify({'error': 'Missing frame data'}), 400

        try:
            # Convert frame data to image
            frame_data = np.array(frame_data, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            print(f"Error processing frame data: {str(e)}")
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Use the latest weather data
        latest_weather_data = weather_data[-1] if weather_data else {}

        # Process frame and calculate metrics
        metrics = process_frame_with_sampling(frame, rois, latest_weather_data)
        if metrics is None:
            return jsonify({'error': 'Failed to process frame'}), 500
        
        # Get historical data
        history = get_camera_history(camera_id)
        if history is None:
            history = {
                'timestamps': [],
                'max_visibility': [],
                'avg_visibility': [],
                'roi_data': []
            }
        
        # Get reference data
        reference = get_reference_data(camera_id)
        
        # Save data to CSV
        save_visibility_to_csv(camera_id, {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'max_visibility': metrics['max_visibility'],
            'avg_visibility': metrics['avg_visibility'],
            'roi_data': metrics['roi_data']
        })
        
        # Return complete data package
        return jsonify({
            'max_visibility': metrics['max_visibility'],
            'avg_visibility': metrics['avg_visibility'],
            'roi_data': metrics['roi_data'],
            'reference': reference,
            'history': history,
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        })
        
    except Exception as e:
        print(f"Error in refresh_camera: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/weather_history')
def weather_history():
    try:
        # Collect today's weather data CSV file
        today = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DATA_DIR, f"weather_data_{today}.csv")

        if not os.path.exists(filename):
            return render_template('weather_history.html', error="No weather data available for today")

        # Read today's weather data
        weather_data = pd.read_csv(filename)

        # Filter data for the past 2-3 hours
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
        three_hours_ago = datetime.now() - timedelta(hours=3)
        recent_weather_data = weather_data[weather_data['timestamp'] >= three_hours_ago]

        # Pass recent weather data to the template
        return render_template('weather_history.html', weather_data=recent_weather_data.to_dict(orient='records'))
    except Exception as e:
        print(f"Error in weather_history route: {str(e)}")
        return render_template('weather_history.html', error="Failed to load weather history")

@app.route('/visibility_history')
def visibility_history():
    try:
        # Collect today's visibility data CSV files
        today = datetime.now().strftime("%Y-%m-%d")
        visibility_files = [
            os.path.join(VISIBILITY_DATA_DIR, f) for f in os.listdir(VISIBILITY_DATA_DIR)
            if f.endswith(f"_{today}.csv")
        ]

        if not visibility_files:
            return render_template('visibility_history.html', error="No visibility data available for today")

        # Read and combine today's visibility data
        visibility_data = pd.concat(
            [pd.read_csv(file) for file in visibility_files],
            ignore_index=True
        )

        # Filter data for the past 2-3 hours
        visibility_data['timestamp'] = pd.to_datetime(visibility_data['timestamp'])
        three_hours_ago = datetime.now() - timedelta(hours=3)
        recent_visibility_data = visibility_data[visibility_data['timestamp'] >= three_hours_ago]

        # Pass recent visibility data to the template
        return render_template('visibility_history.html', visibility_data=recent_visibility_data.to_dict(orient='records'))
    except Exception as e:
        print(f"Error in visibility_history route: {str(e)}")
        return render_template('visibility_history.html', error="Failed to load visibility history")

def fetch_station_data_2():
    """Fetch data from the second weather station."""
    url = f"https://api.weather.com/v2/pws/observations/current?stationId={STATION_ID_2}&format=json&units=m&apiKey={WEATHER_STATION_API_KEY}"
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
            
            weather_data_2.append(new_data)
            
            # Keep only last 24 hours of data in memory
            while len(weather_data_2) > 288:  # 288 = 24 hours * 12 (5-minute intervals)
                weather_data_2.pop(0)
                
            return new_data
    except Exception as e:
        print(f"Error fetching station 2 data: {e}")
        return None

def fetch_openweather_data_2():
    """Fetch OpenWeather data for the second station location."""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={STATION2_LAT}&lon={STATION2_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
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
        
        openweather_data_2.append(new_data)
        
        while len(openweather_data_2) > 288:
            openweather_data_2.pop(0)
            
        return new_data
    except Exception as e:
        print(f"Error fetching OpenWeather data for station 2: {e}")
        return None

def save_to_csv_2(station_data, openweather_data):
    """Save weather data for the second station to CSV."""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(DATA_DIR, f"weather_data_2_{today}.csv")
    
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

@app.route('/api/weather/station2')
def station2_data():
    """API endpoint for second weather station data."""
    if weather_data_2:
        return jsonify({
            'station_data': weather_data_2[-1],
            'openweather_data': openweather_data_2[-1] if openweather_data_2 else None
        })
    return jsonify({'error': 'No data available'})

@app.route('/api/weather/station2/history')
def station2_history():
    """API endpoint for second weather station historical data."""
    return jsonify({
        'station_data': weather_data_2,
        'openweather_data': openweather_data_2
    })

@app.route('/download/station2/csv')
def download_station2_csv():
    """Download CSV data for the second weather station."""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(DATA_DIR, f"weather_data_2_{today}.csv")
    
    if os.path.exists(filename):
        return send_file(
            filename,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"weather_data_2_{today}.csv"
        )
    return "No data available for download", 404

@app.route('/api/refresh/station2')
def refresh_station2_data():
    """Refresh data for the second weather station."""
    try:
        station_data = fetch_station_data_2()
        openweather_data = fetch_openweather_data_2()
        
        if station_data and openweather_data:
            save_to_csv_2(station_data, openweather_data)
            return jsonify({'status': 'success', 'message': 'Data refreshed successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to fetch data from one or more sources'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    try:
        # Fetch historical data for all cameras
        historical_data = {}
        for camera_id in CAMERAS.keys():
            history = get_camera_history(camera_id)
            if history:
                historical_data[camera_id] = {
                    'timestamps': history.get('timestamps', []),
                    'max_visibility': history.get('max_visibility', []),
                    'avg_visibility': history.get('avg_visibility', []),
                    'roi_data': history.get('roi_data', []),
                    'reference': history.get('reference', None)
                }

        # Pass historical data to the template
        return render_template('dashboard.html', 
                             historical_data=json.dumps(historical_data),
                             cameras=CAMERAS)
    except Exception as e:
        print(f"Error in dashboard route: {str(e)}")
        return render_template('dashboard.html', 
                             error="Failed to load dashboard data",
                             cameras=CAMERAS)

if __name__ == '__main__':
    try:
        # Initialize scheduler
        scheduler = BackgroundScheduler(timezone=utc)
        
        # Add jobs to scheduler
        scheduler.add_job(fetch_station_data, 'interval', minutes=5, id='fetch_station_data')
        scheduler.add_job(fetch_openweather_data, 'interval', minutes=5, id='fetch_openweather_data')
        scheduler.add_job(fetch_station_data_2, 'interval', minutes=5, id='fetch_station_data_2')
        scheduler.add_job(fetch_openweather_data_2, 'interval', minutes=5, id='fetch_openweather_data_2')
        scheduler.add_job(process_visibility_data, 'interval', minutes=10, id='process_visibility_data')
        
        # Start the scheduler
        scheduler.start()
        
        # Fetch initial data
        fetch_station_data()
        fetch_openweather_data()
        fetch_station_data_2()
        fetch_openweather_data_2()
        
        # Start the camera update thread
        camera_thread = threading.Thread(target=update_camera_cache, daemon=True)
        camera_thread.start()
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting application: {str(e)}")
    finally:
        # Ensure scheduler is properly shut down
        if scheduler.running:
            scheduler.shutdown(wait=False)