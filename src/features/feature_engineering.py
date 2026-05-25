"""
Feature Engineering Pipeline for AQI Prediction

This module creates all engineered features required for model training:
1. Time-based cyclical features (hour, day, month)
2. Lag features (historical PM2.5 values)
3. Rolling statistics (mean, std, min, max)
4. Change rate features
5. Target variables (future PM2.5 for various horizons)

Models to be trained (from PDF requirements):
- Statistical: Ridge Regression, Elastic Net
- Ensemble: Random Forest, Gradient Boosting (XGBoost, LightGBM)
- Deep Learning: LSTM, GRU, Transformer-based models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


class FeatureEngineer:
    """
    Feature engineering pipeline for AQI prediction.
    Creates all necessary features for training ML models.
    """
    
    # Feature groups
    WEATHER_FEATURES = ['temp', 'feels_like', 'humidity', 'pressure', 
                        'wind_speed', 'wind_deg', 'clouds', 'visibility', 
                        'dew_point', 'uvi']
    
    POLLUTANT_FEATURES = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3', 'no', 'aqi']
    
    LAG_HOURS = [1, 3, 6, 12, 24, 48, 72]  # Hours to look back
    ROLLING_WINDOWS = [6, 12, 24, 48]  # Rolling window sizes
    FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]  # Hours to predict ahead
    
    def __init__(self, target_column: str = 'pm2_5'):
        """
        Initialize the feature engineer.
        
        Args:
            target_column: The column to predict (default: pm2_5)
        """
        self.target_column = target_column
        self.scaler = None
        self.feature_columns = []
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time-based features.
        Uses sin/cos encoding for cyclical nature of time.
        """
        df = df.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if 'unix_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['unix_time'], unit='s')
            else:
                raise ValueError("No timestamp column found")
        
        ts = df['timestamp']
        
        # Hour of day (0-23) - cyclical
        df['hour'] = ts.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6) - cyclical
        df['day_of_week'] = ts.dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month (1-31) - cyclical
        df['day_of_month'] = ts.dt.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Month (1-12) - cyclical
        df['month'] = ts.dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Week of year (1-52) - cyclical
        df['week_of_year'] = ts.dt.isocalendar().week
        df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Season (Northern Hemisphere - Pakistan)
        df['season'] = pd.cut(df['month'], 
                              bins=[0, 3, 6, 9, 12], 
                              labels=['winter', 'spring', 'summer', 'fall'])
        df = pd.get_dummies(df, columns=['season'], prefix='season')
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                         columns: List[str] = None,
                         lags: List[int] = None) -> pd.DataFrame:
        """
        Add lagged versions of features (past values).
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for (default: target + key pollutants)
            lags: Lag hours (default: self.LAG_HOURS)
        """
        df = df.copy()
        
        if columns is None:
            columns = [self.target_column, 'pm10', 'temp', 'humidity', 'wind_speed']
            columns = [c for c in columns if c in df.columns]
        
        if lags is None:
            lags = self.LAG_HOURS
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame,
                            columns: List[str] = None,
                            windows: List[int] = None) -> pd.DataFrame:
        """
        Add rolling statistics (mean, std, min, max).
        
        Args:
            df: Input DataFrame
            columns: Columns to compute rolling stats for
            windows: Window sizes in hours
        """
        df = df.copy()
        
        if columns is None:
            columns = [self.target_column, 'pm10', 'temp', 'humidity']
            columns = [c for c in columns if c in df.columns]
        
        if windows is None:
            windows = self.ROLLING_WINDOWS
        
        for col in columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(
                    window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(
                    window=window, min_periods=1).std()
                
                # Rolling min
                df[f'{col}_rolling_min_{window}h'] = df[col].rolling(
                    window=window, min_periods=1).min()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}h'] = df[col].rolling(
                    window=window, min_periods=1).max()
        
        return df
    
    def add_change_features(self, df: pd.DataFrame,
                           columns: List[str] = None) -> pd.DataFrame:
        """
        Add rate of change features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute changes for
        """
        df = df.copy()
        
        if columns is None:
            columns = [self.target_column, 'pm10', 'temp', 'humidity', 'pressure']
            columns = [c for c in columns if c in df.columns]
        
        change_periods = [1, 6, 12, 24]
        
        for col in columns:
            for period in change_periods:
                # Absolute change
                df[f'{col}_diff_{period}h'] = df[col].diff(period)
                
                # Percentage change
                df[f'{col}_pct_change_{period}h'] = df[col].pct_change(period)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between weather and pollution.
        """
        df = df.copy()
        
        # Temperature × Humidity (affects pollution dispersion)
        if 'temp' in df.columns and 'humidity' in df.columns:
            df['temp_humidity'] = df['temp'] * df['humidity']
        
        # Wind speed × Wind direction (vector component)
        if 'wind_speed' in df.columns and 'wind_deg' in df.columns:
            df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_deg']))
            df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_deg']))
        
        # Pressure gradient (change indicator)
        if 'pressure' in df.columns:
            df['pressure_gradient'] = df['pressure'].diff(3)
        
        # PM2.5 to PM10 ratio
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.001)
        
        # Visibility inverse (higher pollution = lower visibility)
        if 'visibility' in df.columns:
            df['visibility_inv'] = 10000 / (df['visibility'] + 1)
        
        return df
    
    def add_target_variables(self, df: pd.DataFrame,
                            horizons: List[int] = None) -> pd.DataFrame:
        """
        Add target variables (future PM2.5 values).
        
        Args:
            df: Input DataFrame
            horizons: Forecast horizons in hours
        """
        df = df.copy()
        
        if horizons is None:
            horizons = self.FORECAST_HORIZONS
        
        for horizon in horizons:
            # Target: PM2.5 value at t+horizon
            df[f'target_{horizon}h'] = df[self.target_column].shift(-horizon)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         add_targets: bool = True) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Raw input DataFrame
            add_targets: Whether to add target variables
            
        Returns:
            DataFrame with all engineered features
        """
        print("🔧 Starting Feature Engineering Pipeline...")
        print(f"   Input shape: {df.shape}")
        
        # Sort by time
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif 'unix_time' in df.columns:
            df = df.sort_values('unix_time').reset_index(drop=True)
        
        # Step 1: Time features
        print("   ├── Adding time features...")
        df = self.add_time_features(df)
        
        # Step 2: Lag features
        print("   ├── Adding lag features...")
        df = self.add_lag_features(df)
        
        # Step 3: Rolling features
        print("   ├── Adding rolling statistics...")
        df = self.add_rolling_features(df)
        
        # Step 4: Change features
        print("   ├── Adding change rate features...")
        df = self.add_change_features(df)
        
        # Step 5: Interaction features
        print("   ├── Adding interaction features...")
        df = self.add_interaction_features(df)
        
        # Step 6: Target variables
        if add_targets:
            print("   ├── Adding target variables...")
            df = self.add_target_variables(df)
        
        print(f"   └── Output shape: {df.shape}")
        print(f"✅ Feature engineering complete! Added {df.shape[1] - 22} new features.")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame,
                           exclude_targets: bool = True) -> List[str]:
        """
        Get list of feature columns (excluding metadata and targets).
        
        Args:
            df: DataFrame with engineered features
            exclude_targets: Whether to exclude target columns
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['timestamp', 'unix_time', 'city', 'hour_ts', 
                        'hour', 'day_of_week', 'day_of_month', 'month', 'week_of_year']
        
        if exclude_targets:
            exclude_cols += [c for c in df.columns if c.startswith('target_')]
        
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols 
                       and not c.endswith('_x') 
                       and not c.endswith('_y')]
        
        self.feature_columns = feature_cols
        return feature_cols
    
    def prepare_for_training(self, df: pd.DataFrame,
                            target_horizon: int = 24,
                            test_size: float = 0.2,
                            scale: bool = True) -> Dict:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with engineered features
            target_horizon: Which horizon to use as target (hours)
            test_size: Fraction of data for testing
            scale: Whether to scale features
            
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, feature_names, scaler
        """
        # Get feature columns
        feature_cols = self.get_feature_columns(df, exclude_targets=True)
        target_col = f'target_{target_horizon}h'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found. Run add_target_variables first.")
        
        # Remove rows with NaN in target or features
        df_clean = df.dropna(subset=[target_col] + feature_cols)
        
        print(f"📊 Preparing data for training...")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target: {target_col}")
        print(f"   Clean samples: {len(df_clean)}")
        
        # Split (time-based, not random)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Scale features
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': self.scaler,
            'train_timestamps': train_df['timestamp'].values if 'timestamp' in train_df.columns else None,
            'test_timestamps': test_df['timestamp'].values if 'timestamp' in test_df.columns else None,
        }
    
    def save_features(self, df: pd.DataFrame, output_path: str):
        """Save engineered features to parquet file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"💾 Features saved to: {output_path}")
    
    def save_scaler(self, output_path: str):
        """Save the fitted scaler."""
        if self.scaler is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, output_path)
            print(f"💾 Scaler saved to: {output_path}")


def main():
    """
    Main function to run feature engineering on raw data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--input", type=str, default="data/raw/islamabad_raw.parquet",
                       help="Input raw data file")
    parser.add_argument("--output", type=str, default="data/processed/islamabad_features.parquet",
                       help="Output features file")
    parser.add_argument("--target", type=str, default="pm2_5",
                       help="Target column to predict")
    
    args = parser.parse_args()
    
    # Load raw data
    print(f"\n📂 Loading data from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   Shape: {df.shape}")
    
    # Initialize engineer
    engineer = FeatureEngineer(target_column=args.target)
    
    # Run pipeline
    df_features = engineer.engineer_features(df, add_targets=True)
    
    # Save features
    engineer.save_features(df_features, args.output)
    
    # Print summary
    print(f"\n📊 Feature Summary:")
    print(f"   Total features: {len(engineer.get_feature_columns(df_features))}")
    print(f"   Time features: 16")
    print(f"   Lag features: {len(engineer.LAG_HOURS) * 5}")
    print(f"   Rolling features: {len(engineer.ROLLING_WINDOWS) * 4 * 4}")
    print(f"   Change features: {4 * 5 * 2}")
    print(f"   Interaction features: 6")
    print(f"   Target horizons: {engineer.FORECAST_HORIZONS}")


if __name__ == "__main__":
    main()
