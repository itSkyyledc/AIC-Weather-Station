# Weather Station Dashboard

A real-time weather monitoring dashboard that displays data from your local weather station and compares it with OpenWeatherMap data.

## Features

- Real-time weather data display
- 5-minute automatic updates
- Historical data visualization with interactive graphs
- Cross-validation with OpenWeatherMap
- CSV data export
- Responsive design for all devices

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:
```
WEATHER_STATION_API_KEY=your_weather_station_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Data Sources

- Local Weather Station (ICANDA11)
- OpenWeatherMap API

## Data Storage

The dashboard stores the last 24 hours of data in memory, with measurements taken every 5 minutes. Data can be exported to CSV format for further analysis.

## Graphs and Visualizations

- Temperature history
- Humidity and pressure trends
- Wind speed and direction
- Precipitation data

## Contributing

Feel free to submit issues and enhancement requests! 