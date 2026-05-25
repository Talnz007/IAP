"""
Fetch AQI and Weather data from OpenWeatherMap APIs.

APIs Used:
1. One Call 3.0 API (Historical Weather)
   - Provides: temperature, humidity, pressure, wind_speed, wind_direction, etc.
   - Endpoint: https://api.openweathermap.org/data/3.0/onecall/timemachine
   
2. Air Pollution API (Historical Pollution Data)
   - Provides: pm2_5, pm10, no2, so2, co, o3
   - Endpoint: http://api.openweathermap.org/data/2.5/air_pollution/history

Data is merged by timestamp to create complete feature set for ML model.
"""
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class AQIDataFetcher:
    """
    Fetches and merges weather + pollution data from OpenWeatherMap APIs.
    
    - Weather features from One Call 3.0 API
    - Pollutant features from Air Pollution API
    - Merged by timestamp for complete ML feature set
    """
    
    # API Endpoints
    ONECALL_BASE = "https://api.openweathermap.org/data/3.0/onecall"
    AIR_POLLUTION_BASE = "http://api.openweathermap.org/data/2.5/air_pollution"
    GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
    
    # City coordinates cache (Pakistan major cities)
    CITY_COORDS = {
        "karachi": {"lat": 24.8607, "lon": 67.0011},
        "lahore": {"lat": 31.5497, "lon": 74.3436},
        "islamabad": {"lat": 33.6844, "lon": 73.0479},
        "peshawar": {"lat": 34.0151, "lon": 71.5249},
        "quetta": {"lat": 30.1798, "lon": 66.9750},
        "faisalabad": {"lat": 31.4504, "lon": 73.1350},
        "multan": {"lat": 30.1575, "lon": 71.5249},
        "rawalpindi": {"lat": 33.5651, "lon": 73.0169},
    }
    
    # AQI conversion from OpenWeatherMap scale (1-5) to US EPA scale (0-500)
    AQI_SCALE_MAP = {
        1: 25,   # Good (0-50)
        2: 75,   # Fair (51-100)
        3: 125,  # Moderate (101-150)
        4: 175,  # Poor (151-200)
        5: 300,  # Very Poor (201-300+)
    }
    
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY not found in environment variables")
    
    def _get_coordinates(self, city: str, country: str = "Pakistan") -> tuple:
        """Get latitude and longitude for a city."""
        city_lower = city.lower()
        
        # Check cache first
        if city_lower in self.CITY_COORDS:
            coords = self.CITY_COORDS[city_lower]
            return coords["lat"], coords["lon"]
        
        # Otherwise, use Geocoding API
        geo_url = f"{self.GEO_URL}?q={city},{country}&limit=1&appid={self.api_key}"
        response = requests.get(geo_url, timeout=30)
        response.raise_for_status()
        geo_data = response.json()
        
        if not geo_data:
            raise ValueError(f"City not found: {city}, {country}")
        
        return geo_data[0]['lat'], geo_data[0]['lon']
    
    def _convert_aqi(self, owm_aqi: int) -> int:
        """Convert OpenWeatherMap AQI (1-5) to US EPA scale (0-500)."""
        return self.AQI_SCALE_MAP.get(owm_aqi, 50)
    
    # ==================== ONE CALL 3.0 API (Weather) ====================
    
    def fetch_weather_current(self, city: str, country: str = "Pakistan") -> Dict[str, Any]:
        """
        Fetch current weather data from One Call 3.0 API.
        
        Returns: temperature, humidity, pressure, wind_speed, wind_deg, clouds, etc.
        """
        lat, lon = self._get_coordinates(city, country)
        
        url = f"{self.ONECALL_BASE}?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            return {
                "city": city,
                "timestamp": datetime.fromtimestamp(current.get("dt", 0)),
                "unix_time": current.get("dt", 0),
                
                # Weather features for ML model
                "temp": current.get("temp"),
                "feels_like": current.get("feels_like"),
                "pressure": current.get("pressure"),
                "humidity": current.get("humidity"),
                "dew_point": current.get("dew_point"),
                "uvi": current.get("uvi"),
                "clouds": current.get("clouds"),
                "visibility": current.get("visibility"),
                "wind_speed": current.get("wind_speed"),
                "wind_deg": current.get("wind_deg"),
                "wind_gust": current.get("wind_gust", 0),
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {}
    
    def fetch_weather_historical(
        self, 
        city: str, 
        dt: datetime,
        country: str = "Pakistan"
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical weather data for a specific date from One Call 3.0 API.
        
        Args:
            city: City name
            dt: Date to fetch weather for
            country: Country name
            
        Returns:
            List of hourly weather records for that day
        """
        lat, lon = self._get_coordinates(city, country)
        unix_time = int(dt.timestamp())
        
        url = f"{self.ONECALL_BASE}/timemachine?lat={lat}&lon={lon}&dt={unix_time}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            records = []
            for item in data.get("data", []):
                records.append({
                    "city": city,
                    "timestamp": datetime.fromtimestamp(item.get("dt", 0)),
                    "unix_time": item.get("dt", 0),
                    
                    # Weather features
                    "temp": item.get("temp"),
                    "feels_like": item.get("feels_like"),
                    "pressure": item.get("pressure"),
                    "humidity": item.get("humidity"),
                    "dew_point": item.get("dew_point"),
                    "uvi": item.get("uvi", 0),
                    "clouds": item.get("clouds"),
                    "visibility": item.get("visibility", 10000),
                    "wind_speed": item.get("wind_speed"),
                    "wind_deg": item.get("wind_deg"),
                    "wind_gust": item.get("wind_gust", 0),
                })
            
            return records
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical weather: {e}")
            return []
    
    # ==================== AIR POLLUTION API ====================
    
    def fetch_pollution_current(self, city: str, country: str = "Pakistan") -> Dict[str, Any]:
        """
        Fetch current air pollution data from Air Pollution API.
        
        Returns: aqi, pm2_5, pm10, no2, so2, co, o3, etc.
        """
        lat, lon = self._get_coordinates(city, country)
        
        url = f"{self.AIR_POLLUTION_BASE}?lat={lat}&lon={lon}&appid={self.api_key}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('list'):
                return {}
            
            item = data['list'][0]
            main = item.get('main', {})
            components = item.get('components', {})
            owm_aqi = main.get('aqi', 1)
            
            return {
                "city": city,
                "timestamp": datetime.fromtimestamp(item.get("dt", 0)),
                "unix_time": item.get("dt", 0),
                
                # AQI
                "aqi": self._convert_aqi(owm_aqi),
                "aqi_raw": owm_aqi,
                
                # Pollutant features for ML model
                "co": components.get("co", 0),
                "no": components.get("no", 0),
                "no2": components.get("no2", 0),
                "o3": components.get("o3", 0),
                "so2": components.get("so2", 0),
                "pm2_5": components.get("pm2_5", 0),
                "pm10": components.get("pm10", 0),
                "nh3": components.get("nh3", 0),
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pollution data: {e}")
            return {}
    
    def fetch_pollution_historical(
        self, 
        city: str, 
        start_date: datetime, 
        end_date: datetime,
        country: str = "Pakistan"
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical air pollution data from Air Pollution API.
        
        Args:
            city: City name
            start_date: Start date
            end_date: End date
            country: Country name
            
        Returns:
            List of hourly pollution records
        """
        lat, lon = self._get_coordinates(city, country)
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        url = f"{self.AIR_POLLUTION_BASE}/history?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={self.api_key}"
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            records = []
            for item in data.get('list', []):
                main = item.get('main', {})
                components = item.get('components', {})
                owm_aqi = main.get('aqi', 1)
                
                records.append({
                    "city": city,
                    "timestamp": datetime.fromtimestamp(item.get("dt", 0)),
                    "unix_time": item.get("dt", 0),
                    
                    # AQI
                    "aqi": self._convert_aqi(owm_aqi),
                    "aqi_raw": owm_aqi,
                    
                    # Pollutant features
                    "co": components.get("co", 0),
                    "no": components.get("no", 0),
                    "no2": components.get("no2", 0),
                    "o3": components.get("o3", 0),
                    "so2": components.get("so2", 0),
                    "pm2_5": components.get("pm2_5", 0),
                    "pm10": components.get("pm10", 0),
                    "nh3": components.get("nh3", 0),
                })
            
            return records
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical pollution: {e}")
            return []
    
    def fetch_pollution_forecast(self, city: str, country: str = "Pakistan") -> List[Dict[str, Any]]:
        """Fetch pollution forecast (up to 5 days)."""
        lat, lon = self._get_coordinates(city, country)
        
        url = f"{self.AIR_POLLUTION_BASE}/forecast?lat={lat}&lon={lon}&appid={self.api_key}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            records = []
            for item in data.get('list', []):
                main = item.get('main', {})
                components = item.get('components', {})
                owm_aqi = main.get('aqi', 1)
                
                records.append({
                    "city": city,
                    "timestamp": datetime.fromtimestamp(item.get("dt", 0)),
                    "unix_time": item.get("dt", 0),
                    "aqi": self._convert_aqi(owm_aqi),
                    "pm2_5": components.get("pm2_5", 0),
                    "pm10": components.get("pm10", 0),
                })
            
            return records
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pollution forecast: {e}")
            return []
    
    # ==================== MERGED DATA (Weather + Pollution) ====================
    
    def fetch_current_merged(self, city: str, country: str = "Pakistan") -> Dict[str, Any]:
        """
        Fetch current data from BOTH APIs and merge.
        
        Returns complete feature set: weather + pollution data
        """
        weather = self.fetch_weather_current(city, country)
        pollution = self.fetch_pollution_current(city, country)
        
        if not weather or not pollution:
            return weather or pollution or {}
        
        # Merge weather and pollution data
        merged = {**weather}
        
        # Add pollution features (skip duplicate keys)
        for key, value in pollution.items():
            if key not in ['city', 'timestamp', 'unix_time']:
                merged[key] = value
        
        return merged
    
    def fetch_historical_merged(
        self, 
        city: str, 
        start_date: datetime, 
        end_date: datetime,
        country: str = "Pakistan",
        delay: float = 0.5
    ) -> pd.DataFrame:
        """
        Fetch historical data from BOTH APIs and merge by timestamp.
        
        This is the main method for building training dataset!
        
        Args:
            city: City name
            start_date: Start date for historical data
            end_date: End date for historical data
            country: Country name
            delay: Delay between API calls (to respect rate limits)
            
        Returns:
            DataFrame with merged weather + pollution features
        """
        print(f"Fetching historical data for {city}...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # 1. Fetch pollution data (single API call for date range)
        print("  Fetching pollution data...")
        pollution_records = self.fetch_pollution_historical(city, start_date, end_date, country)
        pollution_df = pd.DataFrame(pollution_records)
        
        if pollution_df.empty:
            print("  No pollution data available!")
            return pd.DataFrame()
        
        print(f"  Got {len(pollution_df)} pollution records")
        
        # 2. Fetch weather data (one API call per day)
        print("  Fetching weather data...")
        weather_records = []
        current_date = start_date
        
        while current_date <= end_date:
            records = self.fetch_weather_historical(city, current_date, country)
            weather_records.extend(records)
            current_date += timedelta(days=1)
            time.sleep(delay)  # Rate limiting
        
        weather_df = pd.DataFrame(weather_records)
        
        if weather_df.empty:
            print("  No weather data available!")
            return pollution_df
        
        print(f"  Got {len(weather_df)} weather records")
        
        # 3. Merge by unix_time (round to nearest hour for alignment)
        print("  Merging datasets...")
        
        # Round timestamps to nearest hour
        pollution_df['hour_ts'] = (pollution_df['unix_time'] // 3600) * 3600
        weather_df['hour_ts'] = (weather_df['unix_time'] // 3600) * 3600
        
        # Drop duplicate columns before merge
        weather_cols = ['hour_ts', 'temp', 'feels_like', 'pressure', 'humidity', 
                       'dew_point', 'uvi', 'clouds', 'visibility', 'wind_speed', 
                       'wind_deg', 'wind_gust']
        weather_df = weather_df[weather_cols].drop_duplicates(subset=['hour_ts'])
        
        # Merge on hour timestamp
        merged_df = pd.merge(
            pollution_df, 
            weather_df, 
            on='hour_ts', 
            how='left'
        )
        
        # Clean up
        merged_df = merged_df.drop(columns=['hour_ts'])
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Final merged dataset: {len(merged_df)} records")
        
        return merged_df
    
    def backfill_year(
        self, 
        city: str, 
        year: int = 2024,
        country: str = "Pakistan"
    ) -> pd.DataFrame:
        """
        Backfill a full year of data for training.
        
        Args:
            city: City name
            year: Year to backfill
            country: Country name
            
        Returns:
            DataFrame with full year of merged data
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        # Fetch in monthly chunks to avoid timeout
        all_data = []
        
        for month in range(1, 13):
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)
            
            print(f"\n=== {city} - {month_start.strftime('%B %Y')} ===")
            
            df = self.fetch_historical_merged(city, month_start, month_end, country)
            if not df.empty:
                all_data.append(df)
            
            time.sleep(1)  # Pause between months
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=['unix_time']).reset_index(drop=True)
            return final_df
        
        return pd.DataFrame()
    
    def fetch_multiple_cities(
        self, 
        cities: List[str], 
        country: str = "Pakistan"
    ) -> pd.DataFrame:
        """Fetch current merged data for multiple cities."""
        records = []
        
        for city in cities:
            try:
                data = self.fetch_current_merged(city, country)
                if data:
                    records.append(data)
            except Exception as e:
                print(f"Error fetching data for {city}: {e}")
                continue
        
        return pd.DataFrame(records)


if __name__ == "__main__":
    # Example usage
    fetcher = AQIDataFetcher()
    
    # 1. Fetch current merged data (weather + pollution)
    print("=" * 60)
    print("CURRENT DATA (Weather + Pollution)")
    print("=" * 60)
    
    data = fetcher.fetch_current_merged("Lahore")
    if data:
        print(f"\nCity: {data['city']}")
        print(f"Timestamp: {data['timestamp']}")
        print("\n--- Weather Features ---")
        print(f"Temperature: {data.get('temp')}°C")
        print(f"Humidity: {data.get('humidity')}%")
        print(f"Pressure: {data.get('pressure')} hPa")
        print(f"Wind Speed: {data.get('wind_speed')} m/s")
        print(f"Wind Direction: {data.get('wind_deg')}°")
        print(f"Clouds: {data.get('clouds')}%")
        print("\n--- Pollution Features ---")
        print(f"AQI: {data.get('aqi')}")
        print(f"PM2.5: {data.get('pm2_5')} μg/m³")
        print(f"PM10: {data.get('pm10')} μg/m³")
        print(f"NO2: {data.get('no2')} μg/m³")
        print(f"O3: {data.get('o3')} μg/m³")
        print(f"CO: {data.get('co')} μg/m³")
        print(f"SO2: {data.get('so2')} μg/m³")
    
    # 2. Fetch historical merged data (for training)
    print("\n" + "=" * 60)
    print("HISTORICAL DATA (Last 7 days)")
    print("=" * 60)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = fetcher.fetch_historical_merged("Lahore", start_date, end_date)
    
    if not df.empty:
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df[['timestamp', 'temp', 'humidity', 'wind_speed', 'aqi', 'pm2_5', 'pm10']].head(10))
