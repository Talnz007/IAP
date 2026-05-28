"""
Tests for training module.
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.models.sklearn_models import (
    get_random_forest,
    get_ridge_regression,
    get_gradient_boosting,
    get_model,
    get_all_models
)
from src.training.evaluate import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    evaluate_model,
    get_best_model
)


@pytest.fixture
def sample_regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    
    # Split
    split_idx = int(n_samples * 0.8)
    return {
        'X_train': X[:split_idx],
        'X_test': X[split_idx:],
        'y_train': y[:split_idx],
        'y_test': y[split_idx:]
    }


class TestSklearnModels:
    """Tests for sklearn models."""
    
    def test_random_forest_creation(self):
        """Test random forest model creation."""
        model = get_random_forest()
        assert model is not None
    
    def test_ridge_regression_creation(self):
        """Test ridge regression model creation."""
        model = get_ridge_regression()
        assert model is not None
    
    def test_gradient_boosting_creation(self):
        """Test gradient boosting model creation."""
        model = get_gradient_boosting()
        assert model is not None
    
    def test_get_model_factory(self):
        """Test model factory function."""
        model = get_model('random_forest')
        assert model is not None
        
        with pytest.raises(ValueError):
            get_model('unknown_model')
    
    def test_get_all_models(self):
        """Test getting all models."""
        models = get_all_models()
        
        assert 'random_forest' in models
        assert 'ridge' in models
        assert len(models) >= 2
    
    def test_model_training(self, sample_regression_data):
        """Test model can be trained."""
        model = get_random_forest()
        
        model.fit(
            sample_regression_data['X_train'],
            sample_regression_data['y_train']
        )
        
        predictions = model.predict(sample_regression_data['X_test'])
        assert len(predictions) == len(sample_regression_data['y_test'])


class TestEvaluation:
    """Tests for evaluation metrics."""
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        return y_true, y_pred
    
    def test_rmse(self, predictions):
        """Test RMSE calculation."""
        y_true, y_pred = predictions
        rmse = calculate_rmse(y_true, y_pred)
        
        assert rmse >= 0
        assert rmse < 1  # Should be small for similar values
    
    def test_mae(self, predictions):
        """Test MAE calculation."""
        y_true, y_pred = predictions
        mae = calculate_mae(y_true, y_pred)
        
        assert mae >= 0
        assert mae < 1
    
    def test_r2(self, predictions):
        """Test R² calculation."""
        y_true, y_pred = predictions
        r2 = calculate_r2(y_true, y_pred)
        
        assert r2 <= 1
        assert r2 > 0.9  # Should be high for similar values
    
    def test_evaluate_model(self, predictions):
        """Test complete model evaluation."""
        y_true, y_pred = predictions
        metrics = evaluate_model(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
    
    def test_get_best_model(self):
        """Test best model selection."""
        results = {
            'model_a': {'rmse': 0.5, 'r2': 0.9},
            'model_b': {'rmse': 0.3, 'r2': 0.95},
            'model_c': {'rmse': 0.7, 'r2': 0.85}
        }
        
        # Best by RMSE (lower is better)
        best = get_best_model(results, 'rmse')
        assert best == 'model_b'
        
        # Best by R² (higher is better)
        best = get_best_model(results, 'r2')
        assert best == 'model_b'


class TestEndToEnd:
    """End-to-end training tests."""
    
    def test_train_and_evaluate(self, sample_regression_data):
        """Test complete training and evaluation pipeline."""
        model = get_random_forest()
        
        # Train
        model.fit(
            sample_regression_data['X_train'],
            sample_regression_data['y_train']
        )
        
        # Predict
        predictions = model.predict(sample_regression_data['X_test'])
        
        # Evaluate
        metrics = evaluate_model(
            sample_regression_data['y_test'],
            predictions
        )
        
        assert metrics['rmse'] >= 0
        assert 'r2' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
