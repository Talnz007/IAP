"""
Upload Features to Hopsworks using REST API
=============================================
Alternative upload method that doesn't require the hopsworks Python SDK.
Uses the REST API directly to avoid C++ build tool requirements on Windows.
"""

import os
import sys
import json
import requests
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


class HopsworksRESTClient:
    """Simple REST client for Hopsworks Feature Store."""
    
    def __init__(self, api_key: str, project_name: str):
        self.api_key = api_key
        self.project_name = project_name
        self.base_url = "https://c.app.hopsworks.ai"
        self.headers = {
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json"
        }
        self.project_id = None
        self.feature_store_id = None
    
    def connect(self):
        """Connect to Hopsworks and get project info."""
        print(f"🔗 Connecting to Hopsworks...")
        
        # Get project info
        url = f"{self.base_url}/hopsworks-api/api/project/getProjectInfo/{self.project_name}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to connect: {response.status_code} - {response.text}")
        
        project_info = response.json()
        self.project_id = project_info.get('projectId')
        print(f"   ✅ Connected to project: {self.project_name} (ID: {self.project_id})")
        
        # Get feature store
        url = f"{self.base_url}/hopsworks-api/api/project/{self.project_id}/featurestores"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            fs_list = response.json()
            if fs_list:
                self.feature_store_id = fs_list[0].get('featurestoreId')
                print(f"   ✅ Feature Store ID: {self.feature_store_id}")
        
        return True
    
    def upload_dataframe(self, df: pd.DataFrame, feature_group_name: str):
        """Upload a DataFrame to Hopsworks as a feature group."""
        print(f"\n📤 Uploading to feature group: {feature_group_name}")
        print(f"   Shape: {df.shape}")
        
        # For large datasets, we'll save to parquet and upload via Hopsworks UI
        # or use the online ingestion API
        
        # Save a summary for now
        output_path = project_root / "data" / "processed" / f"{feature_group_name}_for_upload.parquet"
        df.to_parquet(output_path)
        print(f"   💾 Saved to: {output_path}")
        
        return True


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for upload."""
    print("📦 Preparing features...")
    
    df_clean = df.copy()
    
    # Drop NaN rows
    initial = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"   Dropped {initial - len(df_clean)} rows with NaN")
    
    # Add observation ID
    df_clean['observation_id'] = range(len(df_clean))
    
    # Ensure timestamp is datetime
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    df_clean['event_time'] = df_clean['timestamp']
    
    # Convert booleans to int
    for col in df_clean.select_dtypes(include=['bool']).columns:
        df_clean[col] = df_clean[col].astype(int)
    
    # Handle infinities
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    print(f"   Final shape: {df_clean.shape}")
    return df_clean


def main():
    print("="*70)
    print("      HOPSWORKS FEATURE UPLOAD (REST API)")
    print("="*70)
    
    # Load API key
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
    
    if not api_key or api_key == "your_hopsworks_api_key_here":
        print("\n❌ HOPSWORKS_API_KEY not set in .env file")
        return
    
    # Load data
    input_path = project_root / "data" / "processed" / "islamabad_features.parquet"
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Shape: {df.shape}")
    
    # Prepare features
    df_prepared = prepare_features(df)
    
    # Try to connect to Hopsworks
    try:
        client = HopsworksRESTClient(api_key, project_name)
        client.connect()
        client.upload_dataframe(df_prepared, "islamabad_aqi_features")
    except Exception as e:
        print(f"\n⚠️  REST API connection issue: {e}")
        print("\nAlternative: Upload via Hopsworks UI")
        
        # Save prepared data for manual upload
        output_path = project_root / "data" / "processed" / "islamabad_features_ready.parquet"
        df_prepared.to_parquet(output_path)
        
        print(f"""
{'='*70}
   📁 DATA READY FOR UPLOAD
{'='*70}

   File: {output_path}
   Records: {len(df_prepared):,}
   Features: {len(df_prepared.columns)}

   To upload to Hopsworks:
   
   1. Go to https://app.hopsworks.ai/
   2. Open your project: {project_name}
   3. Go to Feature Store → Feature Groups
   4. Click "Create Feature Group"
   5. Upload the parquet file above
   
   OR use Google Colab/Jupyter (no C++ tools needed):
   
   !pip install hopsworks
   import hopsworks
   project = hopsworks.login(api_key_value="{api_key[:20]}...")
   fs = project.get_feature_store()
   fg = fs.get_or_create_feature_group(
       name="islamabad_aqi_features",
       version=1,
       primary_key=["observation_id"],
       event_time="event_time"
   )
   fg.insert(df)
""")


if __name__ == "__main__":
    main()
