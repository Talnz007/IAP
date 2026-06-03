"""
Compute features from raw AQI data.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_month'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18))).astype(int)
    
    df['season_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['season_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['season_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['season_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    return df

def create_targets(df: pd.DataFrame, target_col: str = 'aqi', horizons: list = None) -> pd.DataFrame:
    df = df.copy()
    if horizons is None:
        horizons = [1, 3, 6, 12, 24, 48, 72]
    for horizon in horizons:
        df[f'target_{horizon}h'] = df[target_col].shift(-horizon)
    return df

def compute_all_features(df: pd.DataFrame, target_col: str = 'aqi') -> Tuple[pd.DataFrame, list, list]:
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df = compute_time_features(df)
    
    # Generate lag, rolling, diff features for specific columns
    lags = [1, 3, 6, 12, 24, 48, 72]
    diff_periods = [1, 6, 12, 24]
    windows = [6, 12, 24, 48]
    
    cols_for_lag = ['pm2_5', 'pm10', 'temp', 'humidity', 'wind_speed']
    cols_for_diff = ['pm2_5', 'pm10', 'temp', 'humidity', 'pressure']
    
    # Ensure numerical cols are float to prevent NoneType errors in diff/rolling
    all_num_cols = list(set(cols_for_lag + cols_for_diff + [target_col]))
    for col in all_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in cols_for_lag:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                
    cols_for_rolling = ['pm2_5', 'pm10', 'temp', 'humidity']
    for col in cols_for_rolling:
        if col in df.columns:
            for w in windows:
                df[f'{col}_rolling_mean_{w}h'] = df[col].rolling(w).mean()
                df[f'{col}_rolling_std_{w}h'] = df[col].rolling(w).std()
                df[f'{col}_rolling_min_{w}h'] = df[col].rolling(w).min()
                df[f'{col}_rolling_max_{w}h'] = df[col].rolling(w).max()
                
    cols_for_diff = ['pm2_5', 'pm10', 'temp', 'humidity', 'pressure']
    for col in cols_for_diff:
        if col in df.columns:
            for period in diff_periods:
                df[f'{col}_diff_{period}h'] = df[col].diff(period)
                df[f'{col}_pct_change_{period}h'] = df[col].pct_change(period)
                
    # Derived features
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['temp_humidity'] = df['temp'] * df['humidity']
        
    if 'wind_speed' in df.columns and 'wind_deg' in df.columns:
        df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_deg']))
        df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_deg']))
        
    if 'pressure' in df.columns:
        df['pressure_gradient'] = df['pressure'].diff(1)
        
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-5)
        
    if 'visibility' in df.columns:
        df['visibility_inv'] = 1.0 / (df['visibility'] + 1e-5)
        
    df = create_targets(df, target_col)
    
    exclude_cols = ['timestamp', 'city', 'observation_id', 'event_time']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in exclude_cols + target_cols]
    
    return df, feature_cols, target_cols


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'city': 'Karachi',
        'aqi': np.random.randint(50, 200, 100),
        'pm2_5': np.random.uniform(10, 100, 100),
        'pm10': np.random.uniform(20, 150, 100),
    })
    
    processed_df, features, targets = compute_all_features(sample_data)
    print(f"Features: {len(features)}")
    print(f"Targets: {targets}")
    print(processed_df.head())
