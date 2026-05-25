import os
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
ISLAMABAD_LAT = 33.6844
ISLAMABAD_LON = 73.0479

def calculate_aqi_from_pm25(pm25: float) -> int:
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi))
    if pm25 > 500.4:
        return 500
    return 0

def fetch_live_aqi():
    if not OPENWEATHER_API_KEY:
        print("OPENWEATHERMAP_API_KEY not set!")
        return
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={ISLAMABAD_LAT}&lon={ISLAMABAD_LON}&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('list'):
            item = data['list'][0]
            main = item.get('main', {})
            components = item.get('components', {})
            pm25 = components.get('pm2_5', 0)
            aqi = calculate_aqi_from_pm25(pm25)
            print(f"Timestamp: {datetime.fromtimestamp(item.get('dt', 0))}")
            print(f"AQI: {aqi}")
            print(f"PM2.5: {pm25} µg/m³")
            print(f"PM10: {components.get('pm10', 0)} µg/m³")
            print(f"CO: {components.get('co', 0)} µg/m³")
            print(f"NO2: {components.get('no2', 0)} µg/m³")
            print(f"O3: {components.get('o3', 0)} µg/m³")
            print(f"SO2: {components.get('so2', 0)} µg/m³")
        else:
            print("No AQI data returned.")
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    fetch_live_aqi()
