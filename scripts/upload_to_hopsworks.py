"""
Upload Engineered Features to Hopsworks Feature Store
======================================================
This script uploads the processed features to Hopsworks for:
- Centralized feature storage
- Version control
- Easy retrieval for training and inference
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")


def connect_to_hopsworks():
    """Connect to Hopsworks Feature Store."""
    import hopsworks
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
    
    if not api_key or api_key == "your_hopsworks_api_key_here":
        print("\n" + "="*70)
        print("🔑 HOPSWORKS API KEY REQUIRED")
        print("="*70)
        print("""
To get your Hopsworks API key:

1. Go to https://app.hopsworks.ai/ and create a FREE account
2. Create a new project (e.g., 'aqi_predictor')
3. Go to Account Settings → API Keys
4. Generate a new API key
5. Update your .env file:
   
   HOPSWORKS_API_KEY=your_actual_api_key
   HOPSWORKS_PROJECT_NAME=your_project_name
""")
        print("="*70)
        sys.exit(1)
    
    print(f"🔗 Connecting to Hopsworks project: {project_name}")
    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name
    )
    
    return project.get_feature_store()


def prepare_features_for_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for Hopsworks upload."""
    print("📦 Preparing features for upload...")
    
    # Create a copy
    df_upload = df.copy()
    
    # Drop rows with NaN (from lag features)
    initial_rows = len(df_upload)
    df_upload = df_upload.dropna()
    print(f"   Dropped {initial_rows - len(df_upload)} rows with NaN values")
    
    # Ensure timestamp is datetime
    if 'timestamp' in df_upload.columns:
        df_upload['timestamp'] = pd.to_datetime(df_upload['timestamp'])
    
    # Add event_time for Hopsworks (required for time-travel)
    df_upload['event_time'] = df_upload['timestamp']
    
    # Create a unique ID for each row
    df_upload['observation_id'] = range(len(df_upload))
    
    # Convert boolean columns to int
    bool_cols = df_upload.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_upload[col] = df_upload[col].astype(int)
    
    # Convert object columns (except timestamp) to string
    for col in df_upload.select_dtypes(include=['object']).columns:
        if col != 'timestamp':
            df_upload[col] = df_upload[col].astype(str)
    
    # Handle infinity values
    numeric_cols = df_upload.select_dtypes(include=[np.number]).columns
    df_upload[numeric_cols] = df_upload[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_upload = df_upload.dropna()
    
    print(f"   Final shape: {df_upload.shape}")
    
    return df_upload


def create_feature_group(fs, df: pd.DataFrame, name: str, description: str):
    """Create or get feature group and insert data."""
    print(f"\n📊 Creating/updating feature group: {name}")
    
    # Define primary key and event time
    primary_key = ['observation_id']
    event_time = 'event_time'
    
    # Get or create feature group
    fg = fs.get_or_create_feature_group(
        name=name,
        version=1,
        description=description,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=True,  # Enable online serving for real-time predictions
    )
    
    # Insert data
    print(f"   Inserting {len(df):,} records...")
    fg.insert(df, write_options={"wait_for_job": True})
    
    print(f"   ✅ Feature group '{name}' updated successfully!")
    
    return fg


def create_feature_view(fs, feature_group, name: str):
    """Create a feature view for training."""
    print(f"\n🔍 Creating feature view: {name}")
    
    # Get all features except metadata
    exclude_cols = ['observation_id', 'event_time', 'unix_time', 'city']
    
    # Select features for the view
    query = feature_group.select_except(exclude_cols)
    
    # Create or get feature view
    fv = fs.get_or_create_feature_view(
        name=name,
        version=1,
        description="AQI prediction features with targets for multiple horizons",
        query=query,
        labels=['target_1h', 'target_6h', 'target_12h', 'target_24h', 'target_48h', 'target_72h']
    )
    
    print(f"   ✅ Feature view '{name}' created!")
    
    return fv


def main():
    parser = argparse.ArgumentParser(description="Upload features to Hopsworks")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/islamabad_features.parquet",
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default="islamabad_aqi_features",
        help="Name for the feature group"
    )
    parser.add_argument(
        "--create-view",
        action="store_true",
        help="Also create a feature view for training"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("      HOPSWORKS FEATURE STORE UPLOAD")
    print("="*70)
    
    # Load data
    input_path = project_root / args.input
    print(f"\n📂 Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Shape: {df.shape}")
    
    # Prepare features
    df_prepared = prepare_features_for_upload(df)
    
    # Connect to Hopsworks
    fs = connect_to_hopsworks()
    
    # Create feature group
    fg = create_feature_group(
        fs=fs,
        df=df_prepared,
        name=args.feature_group_name,
        description=f"AQI prediction features for Islamabad. Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}. Contains weather, pollution, and engineered time-series features."
    )
    
    # Optionally create feature view
    if args.create_view:
        fv = create_feature_view(
            fs=fs,
            feature_group=fg,
            name=f"{args.feature_group_name}_view"
        )
    
    print("\n" + "="*70)
    print("   ✅ UPLOAD COMPLETE!")
    print("="*70)
    print(f"""
   Feature Group: {args.feature_group_name}
   Records: {len(df_prepared):,}
   Features: {len(df_prepared.columns)}
   
   Next steps:
   1. Go to https://app.hopsworks.ai/
   2. Navigate to your project → Feature Store
   3. View the feature group: {args.feature_group_name}
   4. Create training data using the feature view
""")


if __name__ == "__main__":
    main()
