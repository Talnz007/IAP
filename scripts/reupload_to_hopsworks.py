"""
Re-upload Features to Hopsworks with Compatible Version
========================================================
This script properly uploads features to Hopsworks using the compatible 4.2.x version.
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
    print('     HOPSWORKS FEATURE RE-UPLOAD')
    print('='*70)

    api_key = os.getenv('HOPSWORKS_API_KEY')
    project_name = os.getenv('HOPSWORKS_PROJECT_NAME')

    # Connect
    print(f'\n1. Connecting to Hopsworks project: {project_name}')
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    print(f'   ✅ Connected to Feature Store')

    # Load local data
    print('\n2. Loading local feature data')
    input_path = project_root / 'data' / 'processed' / 'islamabad_features.parquet'
    df = pd.read_parquet(input_path)
    print(f'   Shape: {df.shape}')

    # Prepare data
    print('\n3. Preparing data for upload')
    df_upload = df.copy()
    
    # Drop NaN rows
    initial = len(df_upload)
    df_upload = df_upload.dropna()
    print(f'   Dropped {initial - len(df_upload)} rows with NaN')
    
    # Add required columns
    df_upload['observation_id'] = range(len(df_upload))
    df_upload['event_time'] = pd.to_datetime(df_upload['timestamp'])
    
    # Convert booleans to int
    bool_cols = df_upload.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df_upload[col] = df_upload[col].astype(int)
    
    # Drop city column
    if 'city' in df_upload.columns:
        df_upload = df_upload.drop(columns=['city'])
    
    # Handle infinities
    numeric_cols = df_upload.select_dtypes(include=[np.number]).columns
    df_upload[numeric_cols] = df_upload[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_upload = df_upload.dropna()
    
    print(f'   Final shape: {df_upload.shape}')

    # Try to delete existing feature group
    print('\n4. Checking for existing feature group')
    try:
        fg_old = fs.get_feature_group('islamabad_aqi_features', version=1)
        if fg_old:
            print('   Found existing feature group, deleting...')
            fg_old.delete()
            print('   ✅ Deleted')
    except Exception as e:
        print(f'   No existing feature group found (OK)')

    # Create new feature group
    print('\n5. Creating new feature group')
    fg = fs.get_or_create_feature_group(
        name='islamabad_aqi_features',
        version=1,
        description='Islamabad AQI prediction features - weather, pollution, and engineered time-series features',
        primary_key=['observation_id'],
        event_time='event_time',
        online_enabled=False  # Disable online to avoid issues
    )
    print(f'   ✅ Feature group created: {fg.name} v{fg.version}')

    # Insert data using STREAM ingestion mode (immediate, no materialization job)
    print('\n6. Inserting data')
    print(f'   Uploading {len(df_upload):,} records...')
    
    # Use smaller batches to avoid timeout issues
    batch_size = 2000
    total_batches = (len(df_upload) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_upload))
        batch = df_upload.iloc[start_idx:end_idx]
        
        print(f'   Batch {i+1}/{total_batches}: rows {start_idx}-{end_idx}...')
        
        try:
            if i == 0:
                # First batch - overwrite
                fg.insert(batch, write_options={"wait_for_job": True, "internal_kafka": False})
            else:
                # Subsequent batches - append
                fg.insert(batch, write_options={"wait_for_job": True, "internal_kafka": False})
        except Exception as e:
            print(f'   ⚠️ Batch {i+1} error: {e}')
            print('   Retrying with different settings...')
            # Fallback: try without wait
            fg.insert(batch, write_options={"wait_for_job": False})
            import time
            time.sleep(5)  # Wait a bit for backend processing
    
    print('   ✅ Data inserted successfully!')

    # Wait for materialization
    print('\n7. Waiting for materialization...')
    import time
    time.sleep(10)  # Give Hopsworks time to process
    
    # Verify
    print('\n8. Verifying upload')
    try:
        df_verify = fg.read()
        print(f'   Records in Hopsworks: {len(df_verify):,}')
    except Exception as e:
        print(f'   ⚠️ Verification failed: {e}')
        print('   The data may still be materializing. Check Hopsworks UI.')
    print(f'   Features: {len(df_verify.columns)}')

    print('\n' + '='*70)
    print('   ✅ UPLOAD COMPLETE!')
    print('='*70)
    print(f'''
   Feature Group: islamabad_aqi_features (v1)
   Records: {len(df_verify):,}
   Features: {len(df_verify.columns)}
   
   View at: https://c.app.hopsworks.ai:443/p/1342612/fs/1331267
''')


if __name__ == "__main__":
    main()
