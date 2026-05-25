"""
Data Requirements Summary

Based on PDF specifications:

📊 DATA REQUIREMENTS
══════════════════════════════════════════════════════════════

1. HISTORICAL DATA VOLUME
   - Training Data: 1 YEAR (365 days) of hourly data
   - This provides ~8,760 hourly records per city
   - Required for proper time-series modeling
   
2. FEATURES NEEDED (18 total)
   
   Weather Features (10):
   ├── temp          - Temperature (°C)
   ├── feels_like    - Feels like temperature (°C)
   ├── humidity      - Humidity (%)
   ├── pressure      - Atmospheric pressure (hPa)
   ├── wind_speed    - Wind speed (m/s)
   ├── wind_deg      - Wind direction (degrees)
   ├── clouds        - Cloud cover (%)
   ├── visibility    - Visibility (meters)
   ├── dew_point     - Dew point (°C)
   └── uvi           - UV Index
   
   Pollutant Features (8):
   ├── aqi           - Air Quality Index (1-5)
   ├── pm2_5         - Fine particulate matter (µg/m³)
   ├── pm10          - Coarse particulate matter (µg/m³)
   ├── no2           - Nitrogen dioxide (µg/m³)
   ├── so2           - Sulphur dioxide (µg/m³)
   ├── co            - Carbon monoxide (µg/m³)
   ├── o3            - Ozone (µg/m³)
   └── nh3           - Ammonia (µg/m³)

3. CITIES TO COVER (5 primary)
   ├── Lahore     (31.5497°N, 74.3436°E)
   ├── Karachi    (24.8607°N, 67.0011°E)
   ├── Islamabad  (33.6844°N, 73.0479°E)
   ├── Peshawar   (34.0151°N, 71.5249°E)
   └── Quetta     (30.1798°N, 66.9750°E)

4. PREDICTION TARGETS
   - Forecast horizons: 1, 6, 12, 24, 48, 72 hours ahead
   - Main target: PM2.5 levels

📡 API STRATEGY
══════════════════════════════════════════════════════════════

One Call 3.0 API (Weather):
- Rate Limit: 1000 free calls/day
- Returns: 24 hourly records per call (1 day)
- Cost for 1 year: 365 calls per city
- Total for 5 cities: ~1,825 calls (≈2 days of API calls)

Air Pollution History API:
- Rate Limit: Essentially unlimited
- Returns: Hourly records for any range
- Can fetch entire year in one call

⏱️ ESTIMATED TIME TO BACKFILL
══════════════════════════════════════════════════════════════

Per City (1 year):
- One Call API: 365 calls needed
- At 1000 calls/day limit: ~1 day per 2-3 cities
- Total for 5 cities: 2-3 days of running

With Checkpoints:
- Script can be stopped and resumed any time
- Progress saved after each day fetched
- No duplicate API calls

⚡ QUICK START COMMANDS
══════════════════════════════════════════════════════════════

# Check current status:
python scripts/backfill_data.py --status

# Fetch last 30 days for Islamabad:
python scripts/backfill_data.py --city islamabad --days 30

# Fetch specific date range:
python scripts/backfill_data.py --city islamabad --start-date 2024-01-01 --end-date 2024-12-31

# Fetch 30 days for all cities:
python scripts/backfill_data.py --all-cities --days 30

# Reset checkpoint for a city:
python scripts/backfill_data.py --city islamabad --reset

══════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_requirements():
    """Print data requirements summary."""
    print(__doc__)

def check_current_data():
    """Check current data status."""
    from scripts.backfill_data import DataBackfiller
    
    backfiller = DataBackfiller()
    status = backfiller.get_status()
    
    print("\n📦 CURRENT DATA STATUS")
    print("="*60)
    
    total_records = 0
    for city, info in status['cities'].items():
        if info['records'] > 0:
            total_records += info['records']
            date_range = info.get('date_range', {})
            start = date_range.get('start', 'N/A')[:10]
            end = date_range.get('end', 'N/A')[:10]
            print(f"  ✅ {city.title()}: {info['records']:,} records ({start} to {end})")
        else:
            print(f"  ⏳ {city.title()}: No data yet")
    
    print(f"\n  Total Records: {total_records:,}")
    
    # Calculate coverage
    target_records_per_city = 8760  # 1 year hourly
    total_target = target_records_per_city * 5  # 5 cities
    coverage = (total_records / total_target) * 100 if total_target > 0 else 0
    
    print(f"  Target (1 year × 5 cities): {total_target:,} records")
    print(f"  Coverage: {coverage:.1f}%")
    print("="*60)


if __name__ == "__main__":
    print_requirements()
    try:
        check_current_data()
    except Exception as e:
        print(f"\n⚠️ Could not check current data: {e}")
        print("  Run: python scripts/backfill_data.py --status")
