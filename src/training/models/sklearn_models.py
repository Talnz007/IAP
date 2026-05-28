"""
Scikit-learn based models for AQI prediction.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Gradient Boosting Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_random_forest(params: Dict[str, Any] = None) -> Pipeline:
    """
    Create a Random Forest regressor pipeline.
    
    Args:
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline with scaler and model
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**default_params))
    ])
    
    return pipeline


def get_ridge_regression(params: Dict[str, Any] = None) -> Pipeline:
    """
    Create a Ridge regression pipeline.
    
    Args:
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline with scaler and model
    """
    default_params = {
        'alpha': 1.0,
        'random_state': 42
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(**default_params))
    ])
    
    return pipeline


def get_gradient_boosting(params: Dict[str, Any] = None) -> Pipeline:
    """
    Create a Gradient Boosting regressor pipeline.
    
    Args:
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline with scaler and model
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(**default_params))
    ])
    
    return pipeline


def get_elastic_net(params: Dict[str, Any] = None) -> Pipeline:
    """
    Create an Elastic Net regression pipeline.
    
    Args:
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline with scaler and model
    """
    default_params = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'random_state': 42
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(**default_params))
    ])
    
    return pipeline


def get_xgboost(params: Dict[str, Any] = None) -> Pipeline:
    """
    Create an XGBoost regressor pipeline.
    
    Args:
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline with scaler and XGBoost model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    default_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(**default_params))
    ])
    
    return pipeline


def get_lightgbm(params: Dict[str, Any] = None) -> Pipeline:
    """
    Create a LightGBM regressor pipeline.
    
    Args:
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline with scaler and LightGBM model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    default_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    if params:
        default_params.update(params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMRegressor(**default_params))
    ])
    
    return pipeline


def get_model(model_name: str, params: Dict[str, Any] = None) -> Pipeline:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model
        params: Model hyperparameters
        
    Returns:
        Sklearn pipeline
    """
    models = {
        'random_forest': get_random_forest,
        'ridge': get_ridge_regression,
        'gradient_boosting': get_gradient_boosting,
        'elastic_net': get_elastic_net,
        'xgboost': get_xgboost,
        'lightgbm': get_lightgbm,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](params)


def get_all_models() -> Dict[str, Pipeline]:
    """
    Get all available models with default parameters.
    
    Returns:
        Dictionary of model name to pipeline
    """
    models = {
        'random_forest': get_random_forest(),
        'ridge': get_ridge_regression(),
        'gradient_boosting': get_gradient_boosting(),
        'elastic_net': get_elastic_net(),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = get_xgboost()
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = get_lightgbm()
    
    return models
