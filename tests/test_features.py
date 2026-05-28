"""
Tests for feature engineering module.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.compute_features import (
    compute_time_features,
    compute_lag_features,
    compute_rolling_features,
    compute_change_rate,
    create_targets,
    compute_all_features
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    return pd.DataFrame({
        'timestamp': dates,
        'city': 'TestCity',
        'aqi': np.random.randint(50, 200, 100),
        'pm2_5': np.random.uniform(10, 100, 100),
        'pm10': np.random.uniform(20, 150, 100),
    })


class TestTimeFeatures:
    """Tests for time feature computation."""
    
    def test_compute_time_features(self, sample_data):
        """Test time feature extraction."""
        result = compute_time_features(sample_data)
        
        assert 'hour' in result.columns
        assert 'day' in result.columns
        assert 'month' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
    
    def test_hour_range(self, sample_data):
        """Test hour values are in valid range."""
        result = compute_time_features(sample_data)
        
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
    
    def test_cyclical_encoding(self, sample_data):
        """Test cyclical encoding values are in valid range."""
        result = compute_time_features(sample_data)
        
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1
        assert result['hour_cos'].min() >= -1
        assert result['hour_cos'].max() <= 1


class TestLagFeatures:
    """Tests for lag feature computation."""
    
    def test_compute_lag_features(self, sample_data):
        """Test lag feature creation."""
        result = compute_lag_features(sample_data, 'aqi', lags=[1, 2, 3])
        
        assert 'aqi_lag_1h' in result.columns
        assert 'aqi_lag_2h' in result.columns
        assert 'aqi_lag_3h' in result.columns
    
    def test_lag_values(self, sample_data):
        """Test lag values are correct."""
        result = compute_lag_features(sample_data, 'aqi', lags=[1])
        
        # Check that lag_1 equals previous value
        assert result['aqi_lag_1h'].iloc[1] == result['aqi'].iloc[0]


class TestRollingFeatures:
    """Tests for rolling feature computation."""
    
    def test_compute_rolling_features(self, sample_data):
        """Test rolling feature creation."""
        result = compute_rolling_features(sample_data, 'aqi', windows=[3, 6])
        
        assert 'aqi_rolling_mean_3h' in result.columns
        assert 'aqi_rolling_std_3h' in result.columns
        assert 'aqi_rolling_mean_6h' in result.columns


class TestChangeRate:
    """Tests for change rate computation."""
    
    def test_compute_change_rate(self, sample_data):
        """Test change rate calculation."""
        result = compute_change_rate(sample_data, 'aqi', periods=[1, 3])
        
        assert 'aqi_change_1h' in result.columns
        assert 'aqi_pct_change_1h' in result.columns
        assert 'aqi_change_3h' in result.columns


class TestTargets:
    """Tests for target creation."""
    
    def test_create_targets(self, sample_data):
        """Test target creation."""
        result = create_targets(sample_data, 'aqi', horizons=[1, 24])
        
        assert 'aqi_target_1h' in result.columns
        assert 'aqi_target_24h' in result.columns
    
    def test_target_values(self, sample_data):
        """Test target values are correct."""
        result = create_targets(sample_data, 'aqi', horizons=[1])
        
        # Target should be the next value
        assert result['aqi_target_1h'].iloc[0] == result['aqi'].iloc[1]


class TestAllFeatures:
    """Tests for complete feature computation."""
    
    def test_compute_all_features(self, sample_data):
        """Test complete feature computation."""
        result, features, targets = compute_all_features(sample_data)
        
        assert len(features) > 0
        assert len(targets) > 0
        assert len(result) == len(sample_data)
    
    def test_no_target_in_features(self, sample_data):
        """Test that targets are not in feature list."""
        result, features, targets = compute_all_features(sample_data)
        
        for target in targets:
            assert target not in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
