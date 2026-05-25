"""
Data Quality Report and Engineering Recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_report(file_path: str):
    df = pd.read_parquet(file_path)
    
    print("=" * 70)
    print("                    DATA CONSISTENCY REPORT")
    print("=" * 70)
    
    # 1. Basic Metrics
    print("\n1. BASIC METRICS")
    print("-" * 70)
    print(f"   Total Records: {len(df):,}")
    print(f"   Total Columns: {len(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    ts_col = 'timestamp'
    print(f"\n   Date Range: {df[ts_col].min()} to {df[ts_col].max()}")
    days_covered = (df[ts_col].max() - df[ts_col].min()).days + 1
    print(f"   Days Covered: {days_covered}")
    expected_records = days_covered * 24
    print(f"   Expected Records (24/day): {expected_records:,}")
    print(f"   Actual Records: {len(df):,}")
    coverage = len(df) / expected_records * 100
    print(f"   Coverage: {coverage:.1f}%")
    
    # 2. Missing Values
    print("\n2. MISSING VALUES")
    print("-" * 70)
    missing = df.isnull().sum()
    for col in df.columns:
        count = missing[col]
        pct = count / len(df) * 100
        status = "✅" if count == 0 else "⚠️"
        print(f"   {status} {col:15s}: {count:5d} ({pct:.1f}%)")
    
    # 3. Duplicate Check
    print("\n3. DUPLICATE CHECK")
    print("-" * 70)
    duplicates = df.duplicated(subset=['unix_time']).sum()
    print(f"   Duplicate timestamps: {duplicates}")
    status = "✅" if duplicates == 0 else "⚠️"
    print(f"   Status: {status}")
    
    # 4. Data Types
    print("\n4. DATA TYPES")
    print("-" * 70)
    for col in df.columns:
        print(f"   {col:15s}: {df[col].dtype}")
    
    # 5. Statistical Summary
    print("\n5. STATISTICAL SUMMARY")
    print("-" * 70)
    numeric_cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'pm2_5', 'pm10', 'aqi', 'co', 'no2', 'visibility']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    stats = df[numeric_cols].describe()
    print(stats.round(2).to_string())
    
    # 6. Time Gaps Analysis
    print("\n6. TIME GAPS ANALYSIS")
    print("-" * 70)
    df_sorted = df.sort_values('unix_time')
    time_diffs = df_sorted['unix_time'].diff().dropna()
    expected_gap = 3600  # 1 hour in seconds
    
    gaps_1h = (time_diffs == expected_gap).sum()
    gaps_gt_1h = (time_diffs > expected_gap).sum()
    gaps_lt_1h = (time_diffs < expected_gap).sum()
    
    print(f"   Normal gaps (1 hour): {gaps_1h}")
    print(f"   Gaps > 1 hour: {gaps_gt_1h}")
    print(f"   Gaps < 1 hour: {gaps_lt_1h}")
    
    if gaps_gt_1h > 0:
        large_gaps = time_diffs[time_diffs > expected_gap]
        print(f"\n   Largest gap: {large_gaps.max() / 3600:.1f} hours")
        print(f"   Average gap when > 1h: {large_gaps.mean() / 3600:.1f} hours")
    
    # 7. Value Range Validation
    print("\n7. VALUE RANGE VALIDATION")
    print("-" * 70)
    validations = {
        'temp': (-50, 60, '°C'),
        'humidity': (0, 100, '%'),
        'pressure': (900, 1100, 'hPa'),
        'wind_speed': (0, 100, 'm/s'),
        'aqi': (1, 5, 'index'),
        'pm2_5': (0, 1000, 'µg/m³'),
        'pm10': (0, 1000, 'µg/m³'),
        'visibility': (0, 50000, 'm'),
    }
    
    for col, (min_val, max_val, unit) in validations.items():
        if col in df.columns:
            below = (df[col] < min_val).sum()
            above = (df[col] > max_val).sum()
            status = "✅" if below == 0 and above == 0 else "⚠️"
            print(f"   {status} {col:12s}: {df[col].min():.1f} to {df[col].max():.1f} {unit}")
            if below > 0 or above > 0:
                print(f"      ↳ {below} below {min_val}, {above} above {max_val}")
    
    # 8. Correlation Matrix (key features)
    print("\n8. CORRELATION WITH PM2.5 (Target)")
    print("-" * 70)
    key_features = ['temp', 'humidity', 'pressure', 'wind_speed', 'visibility', 'pm10', 'no2', 'co', 'o3']
    key_features = [c for c in key_features if c in df.columns]
    
    correlations = df[key_features + ['pm2_5']].corr()['pm2_5'].drop('pm2_5').sort_values(ascending=False)
    for feat, corr in correlations.items():
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"   {feat:12s}: {sign}{abs(corr):.3f} {bar}")
    
    # 9. Data Engineering Recommendations
    print("\n" + "=" * 70)
    print("              DATA ENGINEERING RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    # Check missing data
    total_missing = missing.sum()
    if total_missing > 0:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Missing Values',
            'action': 'Implement interpolation or forward-fill for missing values',
            'code': 'df.interpolate(method="linear") or df.fillna(method="ffill")'
        })
    
    # Check coverage
    if coverage < 95:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': f'Data Coverage is {coverage:.1f}%',
            'action': 'Some hours are missing. Consider interpolation or resampling.',
            'code': 'df.set_index("timestamp").resample("1H").interpolate()'
        })
    
    # Check duplicates
    if duplicates > 0:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'{duplicates} Duplicate Timestamps',
            'action': 'Remove duplicates, keeping the last value',
            'code': 'df.drop_duplicates(subset=["unix_time"], keep="last")'
        })
    
    # Time features recommendation
    recommendations.append({
        'priority': 'HIGH',
        'issue': 'Add Time-Based Features',
        'action': 'Extract cyclical time features for better seasonality capture',
        'code': '''
# Hour of day (cyclical)
df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)

# Day of week (cyclical)
df["dow_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)

# Month (cyclical)
df["month_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.month / 12)

# Is weekend
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
'''
    })
    
    # Lag features
    recommendations.append({
        'priority': 'HIGH',
        'issue': 'Add Lag Features',
        'action': 'Create lagged PM2.5 values for time-series prediction',
        'code': '''
# Lag features (past values)
for lag in [1, 6, 12, 24, 48, 72]:
    df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)
    
# Target: Future PM2.5 values
for horizon in [1, 6, 12, 24, 48, 72]:
    df[f"pm2_5_target_{horizon}h"] = df["pm2_5"].shift(-horizon)
'''
    })
    
    # Rolling statistics
    recommendations.append({
        'priority': 'HIGH',
        'issue': 'Add Rolling Statistics',
        'action': 'Compute rolling averages and standard deviations',
        'code': '''
# Rolling statistics
for window in [6, 12, 24]:
    df[f"pm2_5_rolling_mean_{window}h"] = df["pm2_5"].rolling(window).mean()
    df[f"pm2_5_rolling_std_{window}h"] = df["pm2_5"].rolling(window).std()
    df[f"temp_rolling_mean_{window}h"] = df["temp"].rolling(window).mean()
'''
    })
    
    # Change rates
    recommendations.append({
        'priority': 'MEDIUM',
        'issue': 'Add Change Rate Features',
        'action': 'Calculate rate of change for key features',
        'code': '''
# Percentage change
df["pm2_5_pct_change_1h"] = df["pm2_5"].pct_change(1)
df["pm2_5_pct_change_6h"] = df["pm2_5"].pct_change(6)
df["temp_change_1h"] = df["temp"].diff(1)
'''
    })
    
    # Normalization
    recommendations.append({
        'priority': 'MEDIUM',
        'issue': 'Feature Scaling',
        'action': 'Normalize features for neural network training',
        'code': '''
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For neural networks
scaler = StandardScaler()
numeric_features = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10"]
df[numeric_features] = scaler.fit_transform(df[numeric_features])
'''
    })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        print(f"\n{i}. [{priority_emoji.get(rec['priority'], '⚪')} {rec['priority']}] {rec['issue']}")
        print(f"   Action: {rec['action']}")
        if 'code' in rec:
            print(f"   Code snippet:")
            for line in rec['code'].strip().split('\n'):
                print(f"      {line}")
    
    print("\n" + "=" * 70)
    print("                         SUMMARY")
    print("=" * 70)
    print(f"""
   ✅ Records: {len(df):,} hourly observations
   ✅ Date Range: {days_covered} days
   ✅ Coverage: {coverage:.1f}%
   ✅ Missing Values: {total_missing}
   ✅ Duplicates: {duplicates}
   
   📊 Key Statistics:
      - PM2.5 Range: {df['pm2_5'].min():.1f} - {df['pm2_5'].max():.1f} µg/m³
      - Average PM2.5: {df['pm2_5'].mean():.1f} µg/m³
      - Average AQI: {df['aqi'].mean():.1f}
   
   🔧 Recommended Engineering Steps:
      1. Add time-based cyclical features
      2. Create lag features (1h, 6h, 12h, 24h, 48h, 72h)
      3. Add rolling statistics (mean, std)
      4. Calculate change rates
      5. Scale features for model training
    """)


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/islamabad_raw.parquet"
    generate_report(file_path)
