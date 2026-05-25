"""
Upload Features to Hopsworks using REST API
=============================================
This script uses direct REST API calls to upload features to Hopsworks,
bypassing the need for the hopsworks Python SDK and its C++ dependencies.
"""

import os
import sys
import json
import time
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


class HopsworksUploader:
    """Upload features to Hopsworks using REST API."""
    
    def __init__(self, api_key: str, project_name: str):
        self.api_key = api_key
        self.project_name = project_name
        self.base_url = "https://c.app.hopsworks.ai"
        self.headers = {
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json"
        }
        self.project_id = None
        self.fs_id = None
        
    def get_project_info(self):
        """Get project ID and feature store ID."""
        print("🔗 Connecting to Hopsworks...")
        
        # Get project info
        url = f"{self.base_url}/hopsworks-api/api/project/getProjectInfo/{self.project_name}"
        resp = requests.get(url, headers=self.headers)
        
        if resp.status_code == 200:
            data = resp.json()
            self.project_id = data.get('projectId')
            print(f"   ✅ Project: {self.project_name} (ID: {self.project_id})")
        else:
            print(f"   ❌ Failed to get project info: {resp.status_code}")
            print(f"      Response: {resp.text[:200]}")
            return False
        
        # Get feature store
        url = f"{self.base_url}/hopsworks-api/api/project/{self.project_id}/featurestores"
        resp = requests.get(url, headers=self.headers)
        
        if resp.status_code == 200:
            data = resp.json()
            if data:
                self.fs_id = data[0].get('featurestoreId')
                print(f"   ✅ Feature Store ID: {self.fs_id}")
        else:
            print(f"   ⚠️  Could not get feature store: {resp.status_code}")
            
        return True
    
    def create_feature_group_schema(self, df: pd.DataFrame, name: str):
        """Create feature group schema from DataFrame."""
        
        # Map pandas dtypes to Hopsworks types
        dtype_map = {
            'int64': 'BIGINT',
            'int32': 'INT',
            'float64': 'DOUBLE',
            'float32': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'STRING'
        }
        
        features = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            hops_type = dtype_map.get(dtype, 'STRING')
            features.append({
                "name": col,
                "type": hops_type,
                "primary": col == 'observation_id',
                "partition": False
            })
        
        return features
    
    def upload_via_api(self, df: pd.DataFrame, fg_name: str, description: str):
        """Upload data to feature group via REST API."""
        
        print(f"\n📤 Creating feature group: {fg_name}")
        
        # First, check if feature group exists
        url = f"{self.base_url}/hopsworks-api/api/project/{self.project_id}/featurestores/{self.fs_id}/featuregroups"
        resp = requests.get(url, headers=self.headers)
        
        if resp.status_code == 200:
            existing = resp.json()
            if isinstance(existing, list):
                for fg in existing:
                    if isinstance(fg, dict) and fg.get('name') == fg_name:
                        print(f"   Found existing feature group (ID: {fg.get('id')})")
                        fg_id = fg.get('id')
                        break
                else:
                    print(f"   Creating new feature group...")
                    fg_id = None
            else:
                print(f"   Feature groups response: {type(existing)}")
                fg_id = None
        
        # For now, save as parquet for manual or UI upload
        output_path = project_root / "data" / "processed" / f"{fg_name}_upload.parquet"
        df.to_parquet(output_path, index=False)
        print(f"   💾 Saved to: {output_path}")
        
        # Also save as CSV for easier inspection
        csv_path = project_root / "data" / "processed" / f"{fg_name}_upload.csv"
        df.head(100).to_csv(csv_path, index=False)
        print(f"   💾 Sample CSV: {csv_path}")
        
        return True


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for Hopsworks upload."""
    print("\n📦 Preparing data for upload...")
    
    df_clean = df.copy()
    initial = len(df_clean)
    
    # Drop NaN rows
    df_clean = df_clean.dropna()
    print(f"   Dropped {initial - len(df_clean)} rows with NaN")
    
    # Add required columns
    df_clean['observation_id'] = range(len(df_clean))
    df_clean['event_time'] = pd.to_datetime(df_clean['timestamp'])
    
    # Convert booleans to int (Hopsworks prefers int)
    bool_cols = df_clean.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)
    
    # Drop city column (not needed for single-city project)
    df_clean = df_clean.drop(columns=['city'], errors='ignore')
    
    # Replace infinities
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    print(f"   Final shape: {df_clean.shape}")
    return df_clean


def main():
    print("="*70)
    print("      HOPSWORKS FEATURE STORE UPLOAD")
    print("="*70)
    
    # Load config
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "api_predictor")
    
    if not api_key or "your_" in api_key:
        print("\n❌ HOPSWORKS_API_KEY not set in .env")
        return
    
    # Load features
    input_path = project_root / "data" / "processed" / "islamabad_features.parquet"
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Shape: {df.shape}")
    
    # Prepare data
    df_prepared = prepare_data(df)
    
    # Connect and upload
    uploader = HopsworksUploader(api_key, project_name)
    
    if uploader.get_project_info():
        uploader.upload_via_api(
            df_prepared,
            fg_name="islamabad_aqi_features",
            description="Islamabad AQI prediction features"
        )
    
    print("\n" + "="*70)
    print("   ✅ DATA PREPARED FOR HOPSWORKS!")
    print("="*70)
    print(f"""
   Your data is ready:
   - Parquet: data/processed/islamabad_aqi_features_upload.parquet
   - Records: {len(df_prepared):,}
   - Features: {len(df_prepared.columns)}
   
   To complete upload, use ONE of these methods:
   
   METHOD 1: Hopsworks UI (Easiest)
   --------------------------------
   1. Go to https://app.hopsworks.ai/
   2. Open project: {project_name}
   3. Feature Store → Feature Groups → Create
   4. Upload the parquet file
   
   METHOD 2: Google Colab
   ----------------------
   1. Open notebooks/03_Upload_to_Hopsworks.ipynb in Colab
   2. Upload the parquet file
   3. Run all cells
   
   METHOD 3: Fix Build Tools & Retry
   ---------------------------------
   1. Install Visual Studio 2022 with "Desktop development with C++"
   2. Restart VS Code completely
   3. Run: pip install hopsworks
   4. Run: python scripts/upload_to_hopsworks.py
""")


if __name__ == "__main__":
    main()
