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

def calculate_visibility(edge_density, distance):
    """Calculate visibility based on edge density and distance."""
    try:
        # Ensure inputs are scalar values
        edge_density = float(edge_density)
        distance = float(distance)
        
        # Use a sigmoid-like function to model the relationship
        # between edge density and visibility
        visibility_factor = 1 / (1 + np.exp(-(edge_density - 50) / 10))
        
        # Scale the visibility factor to the distance
        visibility = distance * visibility_factor
        
        # Ensure visibility doesn't exceed the ROI's distance
        return float(min(visibility, distance))
    except Exception as e:
        print(f"Error in calculate_visibility: {str(e)}")
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
    return render_template('visibility.html')

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
            prefix = f"roi_{roi['id']}_"
            row.update({
                f"{prefix}id": roi['id'],
                f"{prefix}label": roi['label'],
                f"{prefix}distance": roi['distance'],
                f"{prefix}edge_density": roi['edge_density'],
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
    """Get the last hour of visibility data for a camera."""
    try:
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Read today's CSV file
        today = now.strftime("%Y-%m-%d")
        filename = os.path.join(VISIBILITY_DATA_DIR, f"visibility_data_{camera_id}_{today}.csv")
        
        if not os.path.exists(filename):
            return {
                'timestamps': [],
                'max_visibility': [],
                'avg_visibility': [],
                'roi_data': {}
            }
        
        # Read CSV with dynamic column detection
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter for last hour
        df = df[df['timestamp'] >= one_hour_ago]
        
        # Initialize ROI data structure
        roi_data = {}
        
        # Get all ROI columns
        roi_columns = [col for col in df.columns if col.startswith('roi_')]
        roi_ids = set()
        
        # Extract ROI IDs
        for col in roi_columns:
            if col.endswith('_id'):
                roi_ids.update(df[col].unique())
        
        # Collect data for each ROI
        for roi_id in roi_ids:
            roi_prefix = next(col[:-3] for col in roi_columns if col.endswith('_id') and roi_id in df[col].unique())
            roi_data[roi_id] = {
                'visibility': df[f'{roi_prefix}visibility'].tolist(),
                'edge_density': df[f'{roi_prefix}edge_density'].tolist(),
                'distance': df[f'{roi_prefix}distance'].iloc[0],
                'label': df[f'{roi_prefix}label'].iloc[0]
            }
        
        return {
            'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'max_visibility': df['max_visibility'].tolist(),
            'avg_visibility': df['avg_visibility'].tolist(),
            'roi_data': roi_data
        }
        
    except Exception as e:
        print(f"Error getting camera history: {str(e)}")
        return None

def get_reference_data(camera_id):
    """Get reference visibility data for a camera."""
    try:
        filename = os.path.join(VISIBILITY_DATA_DIR, f"reference_data_{camera_id}.json")
        if not os.path.exists(filename):
            return None
            
        with open(filename, 'r') as f:
            return json.load(f)
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

def process_frame(frame, rois):
    """Process a frame and calculate visibility metrics for all ROIs."""
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
                
                # Get ROI image
                roi_img = frame[y:y+h, x:x+w]
                if roi_img.size == 0:
                    print(f"Empty ROI image for {roi_id}")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Calculate edge density
                edges = cv2.Canny(blurred, 50, 150)
                edge_density = float(np.count_nonzero(edges)) / (w * h) * 100
                
                # Calculate visibility
                visibility = calculate_visibility(edge_density, distance)
                
                # Update max and total visibility
                max_visibility = max(max_visibility, visibility)
                total_visibility += visibility
                
                # Create ROI data entry
                roi_data.append({
                    'id': roi_id,
                    'label': roi['label'],
                    'distance': distance,
                    'edge_density': edge_density,
                    'visibility': visibility,
                    'visibility_percentage': (visibility / distance) * 100
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
        print(f"Error in process_frame: {str(e)}")
        return None

@app.route('/refresh_camera', methods=['POST'])
def refresh_camera():
    try:
        data = request.json
        camera_id = data['camera_id']
        
        # Get current ROIs for this camera
        rois = get_camera_rois(camera_id)
        if not rois:
            return jsonify({'error': 'No ROIs defined'})

        # Convert frame data to image
        frame_data = np.array(data['frame'], dtype=np.uint8)
        frame = frame_data.reshape((data['height'], data['width'], 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Process frame and calculate metrics
        metrics = process_frame(frame, rois)
        if metrics is None:
            return jsonify({'error': 'Failed to process frame'})
        
        # Get historical data
        history = get_camera_history(camera_id)
        
        # Get reference data
        reference = get_reference_data(camera_id)
        
        # Save data to CSV
        save_visibility_to_csv(camera_id, {
            'timestamp': data['timestamp'],
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
            'timestamp': data['timestamp']
        })
        
    except Exception as e:
        print(f"Error in refresh_camera: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Fetch initial data
    fetch_station_data()
    fetch_openweather_data()
    app.run(debug=True, host='0.0.0.0', port=5000) 