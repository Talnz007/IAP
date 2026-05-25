"""
Data Backfill Script with Checkpointing

This script fetches historical AQI and weather data with:
- Checkpoint support: Resume from where you left off if interrupted
- Progress tracking: See exactly how much data has been collected
- Rate limiting: Respect API limits (1000 calls/day for One Call 3.0)
- Local storage: Save raw data before feature engineering

Data Requirements (from PDF):
- Historical Data: 1 year of hourly data for training
- Features: Weather + Pollutants merged by timestamp
- Target: AQI prediction for next 3 days (72 hours)

API Limits:
- One Call 3.0: 1000 free calls/day
- Air Pollution: Unlimited (with valid API key)

Recommended Strategy:
- Fetch ~30 days at a time (uses ~30 One Call API calls per day per city)
- Full year = ~365 calls per city = 1-2 days of backfill per city
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


class CheckpointManager:
    """Manages checkpoints for resumable data fetching."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "fetch_checkpoint.json"
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save checkpoint to file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def get_last_date(self, city: str) -> Optional[datetime]:
        """Get the last successfully fetched date for a city."""
        checkpoint = self.load_checkpoint()
        if city in checkpoint and checkpoint[city].get('last_date'):
            return datetime.fromisoformat(checkpoint[city]['last_date'])
        return None
    
    def update_progress(self, city: str, date: datetime, records_count: int):
        """Update progress for a city."""
        checkpoint = self.load_checkpoint()
        
        if city not in checkpoint:
            checkpoint[city] = {
                'start_time': datetime.now().isoformat(),
                'total_records': 0,
                'days_completed': 0
            }
        
        checkpoint[city]['last_date'] = date.isoformat()
        checkpoint[city]['total_records'] += records_count
        checkpoint[city]['days_completed'] += 1
        checkpoint[city]['last_updated'] = datetime.now().isoformat()
        
        self.save_checkpoint(checkpoint)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of all progress."""
        return self.load_checkpoint()
    
    def reset_city(self, city: str):
        """Reset checkpoint for a specific city."""
        checkpoint = self.load_checkpoint()
        if city in checkpoint:
            del checkpoint[city]
            self.save_checkpoint(checkpoint)


class DataBackfiller:
    """
    Fetches and saves historical data with checkpoint support.
    """
    
    # API Endpoints
    ONECALL_TIMEMACHINE = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    AIR_POLLUTION_HISTORY = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    
    # City coordinates
    CITIES = {
        "islamabad": {"lat": 33.6844, "lon": 73.0479},
        "lahore": {"lat": 31.5497, "lon": 74.3436},
        "karachi": {"lat": 24.8607, "lon": 67.0011},
        "peshawar": {"lat": 34.0151, "lon": 71.5249},
        "quetta": {"lat": 30.1798, "lon": 66.9750},
        "faisalabad": {"lat": 31.4504, "lon": 73.1350},
        "multan": {"lat": 30.1575, "lon": 71.5249},
        "rawalpindi": {"lat": 33.5651, "lon": 73.0169},
    }
    
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY not found in .env file")
        
        # Setup directories
        self.data_dir = PROJECT_ROOT / "data"
        self.raw_dir = self.data_dir / "raw"
        self.checkpoint_dir = self.data_dir / "checkpoints"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint = CheckpointManager(self.checkpoint_dir)
        
        # Track API calls
        self.api_calls_today = 0
        self.max_calls_per_day = 950  # Leave buffer from 1000 limit
    
    def _get_coords(self, city: str) -> tuple:
        """Get coordinates for a city."""
        city_lower = city.lower()
        if city_lower in self.CITIES:
            return self.CITIES[city_lower]["lat"], self.CITIES[city_lower]["lon"]
        raise ValueError(f"Unknown city: {city}")
    
    def fetch_weather_for_date(self, city: str, date: datetime) -> List[Dict]:
        """
        Fetch weather data for a specific date.
        Uses Current/Forecast API with pollution data timestamps for merging.
        Since One Call 3.0 Time Machine returns only 1 record per call,
        we use pollution API as the primary time series and fill weather from 
        the single daily snapshot.
        """
        lat, lon = self._get_coords(city)
        # Use noon timestamp for best representative weather
        noon_date = date.replace(hour=12, minute=0, second=0)
        unix_ts = int(noon_date.timestamp())
        
        url = f"{self.ONECALL_TIMEMACHINE}"
        params = {
            "lat": lat,
            "lon": lon,
            "dt": unix_ts,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self.api_calls_today += 1
            
            if response.status_code == 200:
                data = response.json()
                records = []
                
                for item in data.get('data', []):
                    # Estimate visibility based on weather conditions
                    # Base visibility: 10000m (clear day)
                    # Reduce based on humidity and clouds
                    humidity = item.get('humidity', 50)
                    clouds = item.get('clouds', 0)
                    
                    # Visibility estimation formula:
                    # - High humidity (>80%) reduces visibility
                    # - High cloud cover reduces visibility
                    # - Base: 10km, min: 1km
                    base_visibility = 10000  # meters
                    humidity_factor = max(0.3, 1 - (humidity - 50) / 100) if humidity > 50 else 1.0
                    cloud_factor = max(0.5, 1 - clouds / 200)
                    estimated_visibility = int(base_visibility * humidity_factor * cloud_factor)
                    estimated_visibility = max(1000, min(10000, estimated_visibility))
                    
                    # Create a template for this day's weather
                    weather_template = {
                        'temp': item.get('temp'),
                        'feels_like': item.get('feels_like'),
                        'humidity': item.get('humidity'),
                        'pressure': item.get('pressure'),
                        'wind_speed': item.get('wind_speed'),
                        'wind_deg': item.get('wind_deg'),
                        'clouds': item.get('clouds'),
                        'visibility': item.get('visibility') or estimated_visibility,
                        'dew_point': item.get('dew_point'),
                        'uvi': item.get('uvi', 0),
                    }
                    
                    # Generate hourly records for the entire day
                    day_start = date.replace(hour=0, minute=0, second=0)
                    for hour in range(24):
                        hour_dt = day_start + timedelta(hours=hour)
                        record = weather_template.copy()
                        record['unix_time'] = int(hour_dt.timestamp())
                        records.append(record)
                
                return records
            
            elif response.status_code == 429:
                print(f"⚠️ Rate limit hit! Waiting 60 seconds...")
                time.sleep(60)
                return self.fetch_weather_for_date(city, date)  # Retry
            
            else:
                print(f"❌ Weather API Error {response.status_code}: {response.text[:100]}")
                return []
                
        except Exception as e:
            print(f"❌ Exception fetching weather: {e}")
            return []
    
    def fetch_pollution_range(self, city: str, start: datetime, end: datetime) -> List[Dict]:
        """
        Fetch pollution data for a date range from Air Pollution History API.
        """
        lat, lon = self._get_coords(city)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        url = f"{self.AIR_POLLUTION_HISTORY}"
        params = {
            "lat": lat,
            "lon": lon,
            "start": start_ts,
            "end": end_ts,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                records = []
                
                for item in data.get('list', []):
                    components = item.get('components', {})
                    main = item.get('main', {})
                    
                    records.append({
                        'unix_time': item.get('dt'),
                        'aqi': main.get('aqi'),
                        'pm2_5': components.get('pm2_5'),
                        'pm10': components.get('pm10'),
                        'no2': components.get('no2'),
                        'so2': components.get('so2'),
                        'co': components.get('co'),
                        'o3': components.get('o3'),
                        'nh3': components.get('nh3'),
                        'no': components.get('no'),
                    })
                
                return records
            else:
                print(f"❌ Pollution API Error {response.status_code}: {response.text[:100]}")
                return []
                
        except Exception as e:
            print(f"❌ Exception fetching pollution: {e}")
            return []
    
    def merge_data(self, weather_records: List[Dict], pollution_records: List[Dict], city: str) -> pd.DataFrame:
        """Merge weather and pollution data by timestamp (rounded to hour)."""
        if not weather_records or not pollution_records:
            return pd.DataFrame()
        
        weather_df = pd.DataFrame(weather_records)
        pollution_df = pd.DataFrame(pollution_records)
        
        # Round to nearest hour for merging
        weather_df['hour_ts'] = (weather_df['unix_time'] // 3600) * 3600
        pollution_df['hour_ts'] = (pollution_df['unix_time'] // 3600) * 3600
        
        # Merge on hour timestamp
        merged = pd.merge(
            weather_df,
            pollution_df,
            on='hour_ts',
            how='inner',
            suffixes=('_weather', '_pollution')
        )
        
        # Clean up
        merged['unix_time'] = merged['hour_ts']
        merged['timestamp'] = pd.to_datetime(merged['unix_time'], unit='s')
        merged['city'] = city
        
        # Drop duplicate columns
        cols_to_drop = [c for c in merged.columns if c.endswith('_weather') or c.endswith('_pollution')]
        merged = merged.drop(columns=cols_to_drop + ['hour_ts'], errors='ignore')
        
        return merged
    
    def backfill_city(
        self,
        city: str,
        start_date: datetime,
        end_date: datetime,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Backfill data for a single city with checkpoint support.
        
        Args:
            city: City name
            start_date: Start date for backfill
            end_date: End date for backfill
            resume: Whether to resume from last checkpoint
        """
        city_lower = city.lower()
        
        # Check for existing checkpoint
        if resume:
            last_date = self.checkpoint.get_last_date(city_lower)
            if last_date and last_date >= start_date:
                start_date = last_date + timedelta(days=1)
                print(f"📌 Resuming from checkpoint: {start_date.strftime('%Y-%m-%d')}")
        
        if start_date > end_date:
            print(f"✅ {city} already complete!")
            return self._load_existing_data(city_lower)
        
        total_days = (end_date - start_date).days + 1
        print(f"\n{'='*60}")
        print(f"📊 Backfilling {city}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"📅 Total days: {total_days}")
        print(f"🔑 API calls needed: ~{total_days} (One Call 3.0)")
        print(f"{'='*60}\n")
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Check API limit
            if self.api_calls_today >= self.max_calls_per_day:
                print(f"\n⚠️ Daily API limit reached ({self.api_calls_today} calls)")
                print(f"📌 Progress saved. Run again tomorrow to continue.")
                break
            
            day_str = current_date.strftime('%Y-%m-%d')
            progress = ((current_date - start_date).days + 1) / total_days * 100
            
            print(f"[{progress:5.1f}%] Fetching {day_str}...", end=" ")
            
            # Fetch weather for this day
            weather_data = self.fetch_weather_for_date(city_lower, current_date)
            
            if weather_data:
                # Fetch pollution for this day
                day_start = current_date.replace(hour=0, minute=0, second=0)
                day_end = current_date.replace(hour=23, minute=59, second=59)
                pollution_data = self.fetch_pollution_range(city_lower, day_start, day_end)
                
                # Merge
                merged = self.merge_data(weather_data, pollution_data, city)
                
                if not merged.empty:
                    all_data.append(merged)
                    records = len(merged)
                    print(f"✅ {records} records")
                    
                    # Update checkpoint
                    self.checkpoint.update_progress(city_lower, current_date, records)
                else:
                    print(f"⚠️ No merged data")
            else:
                print(f"❌ No weather data")
            
            current_date += timedelta(days=1)
            time.sleep(0.5)  # Small delay between requests
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=['unix_time']).reset_index(drop=True)
            
            # Save to file
            self._save_data(city_lower, final_df)
            
            return final_df
        
        return pd.DataFrame()
    
    def _save_data(self, city: str, df: pd.DataFrame):
        """Save data to parquet file, appending to existing data."""
        file_path = self.raw_dir / f"{city}_raw.parquet"
        
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=['unix_time']).reset_index(drop=True)
        
        df.to_parquet(file_path, index=False)
        print(f"\n💾 Saved {len(df)} total records to {file_path}")
    
    def _load_existing_data(self, city: str) -> pd.DataFrame:
        """Load existing data for a city."""
        file_path = self.raw_dir / f"{city}_raw.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return pd.DataFrame()
    
    def fix_visibility(self, city: str = None):
        """
        Fix NULL visibility values in existing data using probabilistic estimation.
        Uses humidity, clouds, and PM2.5 to estimate visibility.
        """
        cities_to_fix = [city.lower()] if city else list(self.CITIES.keys())
        
        for city_name in cities_to_fix:
            file_path = self.raw_dir / f"{city_name}_raw.parquet"
            if not file_path.exists():
                continue
            
            df = pd.read_parquet(file_path)
            
            if 'visibility' not in df.columns:
                df['visibility'] = None
            
            null_count = df['visibility'].isnull().sum()
            if null_count == 0:
                print(f"✅ {city_name}: No NULL visibility values")
                continue
            
            print(f"🔧 Fixing {null_count} NULL visibility values for {city_name}...")
            
            # Estimate visibility based on multiple factors
            for idx in df[df['visibility'].isnull()].index:
                humidity = df.loc[idx, 'humidity'] if 'humidity' in df.columns else 50
                clouds = df.loc[idx, 'clouds'] if 'clouds' in df.columns else 0
                pm25 = df.loc[idx, 'pm2_5'] if 'pm2_5' in df.columns else 50
                
                # Base visibility: 10km (clear conditions)
                base_visibility = 10000
                
                # Humidity factor: high humidity reduces visibility
                humidity_factor = max(0.3, 1 - (humidity - 50) / 100) if humidity > 50 else 1.0
                
                # Cloud factor: clouds reduce visibility slightly
                cloud_factor = max(0.5, 1 - clouds / 200)
                
                # PM2.5 factor: pollution significantly reduces visibility
                # PM2.5 > 150 = very poor visibility
                # PM2.5 > 300 = hazardous, very low visibility
                if pm25 > 300:
                    pm25_factor = 0.2
                elif pm25 > 150:
                    pm25_factor = 0.4
                elif pm25 > 75:
                    pm25_factor = 0.6
                elif pm25 > 35:
                    pm25_factor = 0.8
                else:
                    pm25_factor = 1.0
                
                # Calculate estimated visibility
                estimated = int(base_visibility * humidity_factor * cloud_factor * pm25_factor)
                estimated = max(500, min(10000, estimated))
                
                df.loc[idx, 'visibility'] = estimated
            
            # Save fixed data
            df.to_parquet(file_path, index=False)
            print(f"✅ {city_name}: Fixed {null_count} visibility values")
            
            # Show sample
            print(f"   Sample visibility values: {df['visibility'].head(5).tolist()}")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all backfill operations."""
        status = {
            'api_calls_today': self.api_calls_today,
            'max_calls_per_day': self.max_calls_per_day,
            'cities': {}
        }
        
        checkpoint = self.checkpoint.get_progress_summary()
        
        for city in self.CITIES.keys():
            file_path = self.raw_dir / f"{city}_raw.parquet"
            
            city_status = {
                'has_data': file_path.exists(),
                'records': 0,
                'checkpoint': checkpoint.get(city, {})
            }
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                city_status['records'] = len(df)
                if 'timestamp' in df.columns:
                    city_status['date_range'] = {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    }
            
            status['cities'][city] = city_status
        
        return status


def main():
    """Main entry point for data backfill."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill AQI data with checkpoint support")
    parser.add_argument("--city", type=str, default="islamabad", help="City to backfill")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backfill")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--status", action="store_true", help="Show backfill status")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint for city")
    parser.add_argument("--all-cities", action="store_true", help="Backfill all cities")
    parser.add_argument("--fix-visibility", action="store_true", help="Fix NULL visibility values")
    
    args = parser.parse_args()
    
    backfiller = DataBackfiller()
    
    if args.status:
        status = backfiller.get_status()
        print("\n📊 Backfill Status")
        print("="*60)
        print(f"API calls today: {status['api_calls_today']}/{status['max_calls_per_day']}")
        print("\nCity Status:")
        for city, info in status['cities'].items():
            print(f"\n  {city.title()}:")
            print(f"    Records: {info['records']}")
            if info.get('date_range'):
                print(f"    Date Range: {info['date_range']['start'][:10]} to {info['date_range']['end'][:10]}")
            if info.get('checkpoint', {}).get('days_completed'):
                print(f"    Days Completed: {info['checkpoint']['days_completed']}")
        return
    
    if args.reset:
        backfiller.checkpoint.reset_city(args.city.lower())
        print(f"✅ Checkpoint reset for {args.city}")
        return
    
    if args.fix_visibility:
        if args.all_cities:
            backfiller.fix_visibility()
        else:
            backfiller.fix_visibility(args.city)
        return
    
    # Set dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days - 1)
    
    # Backfill
    if args.all_cities:
        for city in backfiller.CITIES.keys():
            print(f"\n\n{'#'*60}")
            print(f"# Processing: {city.title()}")
            print(f"{'#'*60}")
            backfiller.backfill_city(city, start_date, end_date)
    else:
        backfiller.backfill_city(args.city, start_date, end_date)
    
    # Show final status
    print("\n" + "="*60)
    print("📊 Final Status")
    print("="*60)
    status = backfiller.get_status()
    for city, info in status['cities'].items():
        if info['records'] > 0:
            print(f"  {city.title()}: {info['records']} records")


if __name__ == "__main__":
    main()
