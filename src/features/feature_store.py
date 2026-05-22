"""
Feature Store integration for storing and retrieving features.
Supports Supabase (free tier) or local storage.
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class FeatureStore:
    """Abstract base class for feature stores."""
    
    def save_features(self, df: pd.DataFrame, feature_group_name: str):
        raise NotImplementedError
        
    def get_features(self, feature_group_name: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        raise NotImplementedError


class LocalFeatureStore(FeatureStore):
    """Local file-based feature store for development."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "data" / "processed"
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_features(self, df: pd.DataFrame, feature_group_name: str):
        """Save features to local parquet file."""
        file_path = self.base_path / f"{feature_group_name}.parquet"
        
        if file_path.exists():
            # Append to existing data
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp'], keep='last')
            
        df.to_parquet(file_path, index=False)
        print(f"Saved {len(df)} records to {file_path}")
        
    def get_features(
        self, 
        feature_group_name: str, 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Load features from local parquet file."""
        file_path = self.base_path / f"{feature_group_name}.parquet"
        
        if not file_path.exists():
            return pd.DataFrame()
            
        df = pd.read_parquet(file_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
                
        return df


class SupabaseFeatureStore(FeatureStore):
    """Supabase feature store integration."""
    
    def __init__(self):
        from supabase import create_client, Client
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        self.client: Client = create_client(self.url, self.key)
        
    def save_features(self, df: pd.DataFrame, feature_group_name: str = 'aqi_features', primary_key: list = None):
        """Save features to Supabase."""
        # Convert timestamp to ISO format string if necessary
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            
        # Standard columns in our schema
        standard_cols = ['timestamp', 'city', 'aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 'humidity', 'wind_speed', 'pressure']
        
        records = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            
            record = {}
            features = {}
            for k, v in row_dict.items():
                if pd.isna(v):
                    v = None
                if k in standard_cols:
                    record[k] = v
                else:
                    features[k] = v
                    
            record['features'] = features
            records.append(record)
            
        # Upsert in batches of 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                self.client.table(feature_group_name).upsert(batch, on_conflict='timestamp,city').execute()
            except Exception as e:
                print(f"Error saving to Supabase: {e}")
                raise
                
        print(f"Saved {len(df)} records to Supabase table: {feature_group_name}")
            
    def get_features(
        self, 
        feature_group_name: str = 'aqi_features', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Get features from Supabase."""
        query = self.client.table(feature_group_name).select('*')
        
        if start_date:
            query = query.gte('timestamp', start_date.isoformat())
        if end_date:
            query = query.lte('timestamp', end_date.isoformat())
            
        try:
            response = query.execute()
            data = response.data
            
            if not data:
                return pd.DataFrame()
                
            # Flatten the 'features' JSONB into columns
            flattened = []
            for row in data:
                flat_row = {k: v for k, v in row.items() if k != 'features' and k != 'id' and k != 'created_at'}
                features = row.get('features') or {}
                flat_row.update(features)
                flattened.append(flat_row)
                
            df = pd.DataFrame(flattened)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
            
        except Exception as e:
            print(f"Error reading from Supabase: {e}")
            return pd.DataFrame()


def get_feature_store(use_supabase: bool = None) -> FeatureStore:
    """
    Factory function to get the appropriate feature store.
    
    Args:
        use_supabase: If True, use Supabase. If False, use local. 
                      If None, auto-detect based on environment.
                      
    Returns:
        FeatureStore instance
    """
    if use_supabase is None:
        use_supabase = os.getenv("SUPABASE_URL") is not None
        
    if use_supabase:
        return SupabaseFeatureStore()
    else:
        return LocalFeatureStore()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'city': 'Karachi',
        'aqi': np.random.randint(50, 200, 24),
        'pm2_5': np.random.uniform(10, 100, 24),
    })
    
    # Use local feature store for testing
    fs = LocalFeatureStore()
    fs.save_features(sample_data, 'aqi_features')
    
    loaded_data = fs.get_features('aqi_features')
    print(f"Loaded {len(loaded_data)} records")
