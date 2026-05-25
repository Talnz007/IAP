"""
Validate Data in Hopsworks Feature Store
=========================================
This script connects to Hopsworks and validates the uploaded feature data.
"""

import hopsworks
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


def main():
    print('='*70)
    print('     HOPSWORKS DATA VALIDATION')
    print('='*70)

    api_key = os.getenv('HOPSWORKS_API_KEY')
    project_name = os.getenv('HOPSWORKS_PROJECT_NAME')

    print(f'\nConnecting to project: {project_name}')

    # Connect
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    print('\n1. FEATURE GROUP INFO')
    print('-'*70)

    # Get feature group
    fg = fs.get_feature_group('islamabad_aqi_features', version=1)
    print(f'   Name: {fg.name}')
    print(f'   Version: {fg.version}')
    print(f'   Primary Key: {fg.primary_key}')
    print(f'   Event Time: {fg.event_time}')

    print('\n2. FETCHING DATA FROM HOPSWORKS')
    print('-'*70)

    # Read data
    df = fg.read()
    print(f'   Total Records: {len(df):,}')
    print(f'   Total Features: {len(df.columns)}')
    print(f'   Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB')

    print('\n3. DATA QUALITY CHECK')
    print('-'*70)

    # Missing values
    missing = df.isnull().sum().sum()
    missing_status = '✅' if missing == 0 else '⚠️'
    print(f'   {missing_status} Missing values: {missing}')

    # Duplicates
    if 'observation_id' in df.columns:
        duplicates = df.duplicated(subset=['observation_id']).sum()
    else:
        duplicates = df.duplicated().sum()
    dup_status = '✅' if duplicates == 0 else '⚠️'
    print(f'   {dup_status} Duplicate rows: {duplicates}')

    # Infinity check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    inf_status = '✅' if inf_count == 0 else '⚠️'
    print(f'   {inf_status} Infinite values: {inf_count}')

    # Date range
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        days = (max_date - min_date).days
        print(f'   📅 Date range: {min_date} to {max_date}')
        print(f'   📅 Days covered: {days}')

    print('\n4. FEATURE CATEGORIES')
    print('-'*70)

    # Categorize features
    target_cols = [c for c in df.columns if 'target' in c]
    lag_cols = [c for c in df.columns if 'lag' in c]
    rolling_cols = [c for c in df.columns if 'rolling' in c]
    change_cols = [c for c in df.columns if 'change' in c or 'diff' in c or 'pct' in c]
    time_cols = [c for c in df.columns if any(x in c for x in ['hour', 'dow', 'month', 'weekend', 'night', 'rush', 'season'])]
    interaction_cols = [c for c in df.columns if any(x in c for x in ['ratio', 'wind_u', 'wind_v', 'heat_index'])]

    print(f'   🎯 Target columns: {len(target_cols)}')
    print(f'      {target_cols}')
    print(f'   ⏮️  Lag features: {len(lag_cols)}')
    print(f'   📊 Rolling features: {len(rolling_cols)}')
    print(f'   📈 Change features: {len(change_cols)}')
    print(f'   🕐 Time features: {len(time_cols)}')
    print(f'   🔗 Interaction features: {len(interaction_cols)}')

    print('\n5. TARGET VARIABLE STATISTICS')
    print('-'*70)
    for col in sorted(target_cols):
        stats = df[col].describe()
        print(f'   {col}:')
        print(f'      Range: {stats["min"]:.1f} - {stats["max"]:.1f} µg/m³')
        print(f'      Mean: {stats["mean"]:.1f}, Std: {stats["std"]:.1f}')

    print('\n6. KEY FEATURE STATISTICS')
    print('-'*70)
    key_features = ['pm2_5', 'pm10', 'temp', 'humidity', 'aqi']
    for col in key_features:
        if col in df.columns:
            stats = df[col].describe()
            print(f'   {col}: min={stats["min"]:.1f}, max={stats["max"]:.1f}, mean={stats["mean"]:.1f}')

    print('\n7. SAMPLE DATA')
    print('-'*70)
    print(df[['timestamp', 'pm2_5', 'temp', 'humidity', 'target_1h', 'target_24h']].head(5).to_string())

    # Summary
    print('\n' + '='*70)
    print('   VALIDATION SUMMARY')
    print('='*70)
    
    all_good = missing == 0 and duplicates == 0 and inf_count == 0
    
    if all_good:
        print('''
   ✅ DATA QUALITY: EXCELLENT
   
   Ready for model training:
   - Records: {:,}
   - Features: {:,} (including {} target variables)
   - Date coverage: {} days
   - Missing/Duplicates/Infinities: None
   
   Next step: Train models using this feature store data!
'''.format(len(df), len(df.columns), len(target_cols), days))
    else:
        print(f'''
   ⚠️  DATA QUALITY: NEEDS ATTENTION
   
   Issues found:
   - Missing values: {missing}
   - Duplicates: {duplicates}
   - Infinite values: {inf_count}
''')

    return df


if __name__ == "__main__":
    df = main()
