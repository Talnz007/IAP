"""
Feature Store integration for storing and retrieving features.
Supports Hopsworks (free tier) or local storage.
"""
import os
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


class HopsworksFeatureStore(FeatureStore):
    """Hopsworks feature store integration."""
    
    def __init__(self):
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
        self._connection = None
        self._fs = None
        
    def _connect(self):
        """Establish connection to Hopsworks."""
        if self._connection is None:
            try:
                import hopsworks
                self._connection = hopsworks.login(
                    api_key_value=self.api_key,
                    project=self.project_name
                )
                self._fs = self._connection.get_feature_store()
            except Exception as e:
                print(f"Error connecting to Hopsworks: {e}")
                raise
                
    def save_features(self, df: pd.DataFrame, feature_group_name: str, primary_key: list = None):
        """Save features to Hopsworks feature group."""
        self._connect()
        
        if primary_key is None:
            primary_key = ['timestamp', 'city']
            
        try:
            # Get or create feature group
            fg = self._fs.get_or_create_feature_group(
                name=feature_group_name,
                version=1,
                primary_key=primary_key,
                event_time='timestamp',
                description=f"AQI features for {feature_group_name}"
            )
            
            # Insert data
            fg.insert(df)
            print(f"Saved {len(df)} records to Hopsworks feature group: {feature_group_name}")
            
        except Exception as e:
            print(f"Error saving to Hopsworks: {e}")
            raise
            
    def get_features(
        self, 
        feature_group_name: str, 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Get features from Hopsworks feature group."""
        self._connect()
        
        try:
            fg = self._fs.get_feature_group(name=feature_group_name, version=1)
            
            # Build query
            query = fg.select_all()
            
            # Apply time filters if provided
            if start_date or end_date:
                df = query.read()
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    if start_date:
                        df = df[df['timestamp'] >= start_date]
                    if end_date:
                        df = df[df['timestamp'] <= end_date]
                return df
            else:
                return query.read()
                
        except Exception as e:
            print(f"Error reading from Hopsworks: {e}")
            return pd.DataFrame()


def get_feature_store(use_hopsworks: bool = None) -> FeatureStore:
    """
    Factory function to get the appropriate feature store.
    
    Args:
        use_hopsworks: If True, use Hopsworks. If False, use local. 
                      If None, auto-detect based on environment.
                      
    Returns:
        FeatureStore instance
    """
    if use_hopsworks is None:
        use_hopsworks = os.getenv("HOPSWORKS_API_KEY") is not None
        
    if use_hopsworks:
        return HopsworksFeatureStore()
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
