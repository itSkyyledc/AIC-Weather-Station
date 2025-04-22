import requests


API_KEY = "b8db167989b440559b167989b45055fe"
STATION_ID = "ICANDA11"


URL = f"https://api.weather.com/v2/pws/observations/current?stationId={STATION_ID}&format=json&units=m&apiKey={API_KEY}"

try:
    response = requests.get(URL)
    data = response.json()

    if "observations" in data and data["observations"]:
        obs = data["observations"][0]
        metric = obs["metric"]

        print(f"📡 Weather Data for {STATION_ID} ({obs['neighborhood']})")
        print(f"📍 Location: {obs['lat']}, {obs['lon']} - {obs['country']}")
        print(f"🕒 Time: {obs['obsTimeLocal']}")
        print(f"🌡️ Temperature: {metric['temp']}°C")
        print(f"💧 Humidity: {obs['humidity']}%")
        print(f"🌬️ Wind Speed: {metric['windSpeed']} km/h, Direction: {obs['winddir']}°")
        print(f"🌧️ Precipitation: {metric['precipRate']} mm (Total: {metric['precipTotal']} mm)")
        print(f"🔵 Pressure: {metric['pressure']} hPa")
        print(f"⛰️ Elevation: {metric['elev']} m")

    else:
        print("⚠️ No observation data available.")

except requests.exceptions.RequestException as e:
    print(f"❌ Error fetching data: {e}")
