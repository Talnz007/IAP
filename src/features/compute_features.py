"""
Compute features from raw AQI data.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp.
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def compute_lag_features(df: pd.DataFrame, target_col: str = 'aqi', lags: list = None) -> pd.DataFrame:
    """
    Create lagged features for time series prediction.
    
    Args:
        df: DataFrame with target column
        target_col: Name of the target column
        lags: List of lag values to create
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]  # Hours
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    return df


def compute_rolling_features(df: pd.DataFrame, target_col: str = 'aqi', windows: list = None) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Args:
        df: DataFrame with target column
        target_col: Name of the target column
        windows: List of window sizes for rolling calculations
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    if windows is None:
        windows = [3, 6, 12, 24]  # Hours
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window).std()
        df[f'{target_col}_rolling_min_{window}h'] = df[target_col].rolling(window).min()
        df[f'{target_col}_rolling_max_{window}h'] = df[target_col].rolling(window).max()
    
    return df


def compute_change_rate(df: pd.DataFrame, target_col: str = 'aqi', periods: list = None) -> pd.DataFrame:
    """
    Calculate AQI change rate over different periods.
    
    Args:
        df: DataFrame with target column
        target_col: Name of the target column
        periods: List of periods for change rate calculation
        
    Returns:
        DataFrame with change rate features
    """
    df = df.copy()
    
    if periods is None:
        periods = [1, 3, 6, 12, 24]  # Hours
    
    for period in periods:
        df[f'{target_col}_change_{period}h'] = df[target_col].diff(period)
        df[f'{target_col}_pct_change_{period}h'] = df[target_col].pct_change(period)
    
    return df


def create_targets(df: pd.DataFrame, target_col: str = 'aqi', horizons: list = None) -> pd.DataFrame:
    """
    Create target variables for different prediction horizons.
    
    Args:
        df: DataFrame with target column
        target_col: Name of the column to predict
        horizons: List of future horizons to predict (in hours)
        
    Returns:
        DataFrame with target columns
    """
    df = df.copy()
    
    if horizons is None:
        horizons = [1, 3, 6, 12, 24, 48, 72]  # Hours (up to 3 days)
    
    for horizon in horizons:
        df[f'{target_col}_target_{horizon}h'] = df[target_col].shift(-horizon)
    
    return df


def compute_all_features(df: pd.DataFrame, target_col: str = 'aqi') -> Tuple[pd.DataFrame, list, list]:
    """
    Compute all features and targets.
    
    Args:
        df: Raw DataFrame with AQI data
        target_col: Name of the target column
        
    Returns:
        Tuple of (processed DataFrame, feature columns, target columns)
    """
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Apply all feature engineering steps
    df = compute_time_features(df)
    df = compute_lag_features(df, target_col)
    df = compute_rolling_features(df, target_col)
    df = compute_change_rate(df, target_col)
    df = create_targets(df, target_col)
    
    # Define feature and target columns
    exclude_cols = ['timestamp', 'city']
    target_cols = [col for col in df.columns if '_target_' in col]
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
