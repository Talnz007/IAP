"""
Model evaluation metrics and utilities.
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    return {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }


def compare_models(
    models_predictions: Dict[str, np.ndarray], 
    y_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        models_predictions: Dictionary of model name to predictions
        y_true: True values
        
    Returns:
        Dictionary of model name to metrics
    """
    results = {}
    
    for model_name, y_pred in models_predictions.items():
        results[model_name] = evaluate_model(y_true, y_pred)
    
    return results


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "Model"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*50}")
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"R²:    {metrics['r2']:.4f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")
    print(f"{'='*50}\n")


def get_best_model(results: Dict[str, Dict[str, float]], metric: str = 'rmse') -> str:
    """
    Get the best model based on a metric.
    
    Args:
        results: Dictionary of model metrics
        metric: Metric to use for comparison
        
    Returns:
        Name of the best model
    """
    if metric in ['rmse', 'mae', 'mape']:
        # Lower is better
        best_model = min(results, key=lambda x: results[x][metric])
    else:
        # Higher is better (r2)
        best_model = max(results, key=lambda x: results[x][metric])
    
    return best_model
