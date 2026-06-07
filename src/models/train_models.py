"""
=============================================================================
           ISLAMABAD AQI PREDICTION - MODEL TRAINING PIPELINE
=============================================================================

This module implements a comprehensive model training pipeline for PM2.5
prediction at multiple time horizons (1h, 6h, 12h, 24h, 48h, 72h).

MODELS IMPLEMENTED:
-------------------
1. BASELINE MODELS
   - Persistence (last value)
   - Moving Average (24h window)

2. LINEAR MODELS  
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization for feature selection)

3. TREE-BASED MODELS
   - Random Forest (robust, interpretable)
   - XGBoost (state-of-the-art for tabular data)
   - LightGBM (fast, memory efficient)

4. DEEP LEARNING MODELS
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)

5. ENSEMBLE
   - Stacking (combine predictions from multiple models)

RATIONALE:
----------
- High autocorrelation (0.978 at 1h) → Persistence baseline will be strong
- 8,556 samples → Sufficient for tree models, borderline for deep learning
- 186 features → Regularization needed (Ridge/Lasso)
- Skewed distribution (2.39) → Log transform or Huber loss
- Multiple horizons → Separate models per horizon

Author: AQI Prediction Pipeline
Date: January 2026
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelConfig:
    """Model training configuration."""
    
    # Target horizons (hours ahead to predict)
    HORIZONS = [1, 6, 12, 24, 48, 72]
    
    # Train/test split ratio (time-based)
    TEST_SIZE = 0.2
    
    # Cross-validation folds
    CV_FOLDS = 5
    
    # Random seed
    RANDOM_STATE = 42
    
    # Features to exclude from training
    EXCLUDE_COLS = ['timestamp', 'city', 'unix_time', 'observation_id', 'event_time']
    
    # Model hyperparameters
    MODELS = {
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1,
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'lightgbm': {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
        },
        'lstm': {
            'units': 64,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'sequence_length': 24,
        }
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_hopsworks() -> pd.DataFrame:
    """Load features from Hopsworks Feature Store."""
    import hopsworks
    from dotenv import load_dotenv
    
    load_dotenv(project_root / ".env")
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "api_predictor")
    
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    
    # Get feature view
    fv = fs.get_feature_view(
        name="islamabad_aqi_features_view",
        version=1
    )
    
    print("📥 Loading features from Hopsworks...")
    df = fv.get_batch_data()
    
    print(f"   Loaded {len(df):,} records with {len(df.columns)} features")
    
    return df


def load_data_from_local() -> pd.DataFrame:
    """Load features from local parquet file."""
    path = project_root / "data" / "processed" / "islamabad_features.parquet"
    print(f"📂 Loading from: {path}")
    df = pd.read_parquet(path)
    df = df.dropna()
    print(f"   Loaded {len(df):,} records with {len(df.columns)} features")
    return df


def prepare_features(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for a specific horizon."""
    
    target_col = f'target_{horizon}h'
    
    # Get feature columns (exclude targets and metadata)
    target_cols = [f'target_{h}h' for h in ModelConfig.HORIZONS]
    exclude = ModelConfig.EXCLUDE_COLS + target_cols
    
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining issues
    X = X.select_dtypes(include=[np.number])
    
    return X, y


def time_series_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split data maintaining temporal order."""
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# BASELINE MODELS
# =============================================================================

class PersistenceModel:
    """Baseline: predict last known value."""
    
    def __init__(self):
        self.name = "Persistence"
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        # Use pm2_5 from features (current value)
        if 'pm2_5' in X.columns:
            return X['pm2_5'].values
        else:
            # Fallback to first numeric column that looks like pm2_5
            for col in X.columns:
                if 'pm2_5' in col and 'lag' not in col and 'rolling' not in col:
                    return X[col].values
            return np.zeros(len(X))


class MovingAverageModel:
    """Baseline: predict 24h moving average."""
    
    def __init__(self, window: int = 24):
        self.window = window
        self.name = f"MA_{window}h"
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        if f'pm2_5_rolling_mean_{self.window}h' in X.columns:
            return X[f'pm2_5_rolling_mean_{self.window}h'].values
        elif 'pm2_5' in X.columns:
            return X['pm2_5'].values
        return np.zeros(len(X))


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_ridge(X_train, y_train, X_test, y_test) -> Dict:
    """Train Ridge Regression with cross-validation for alpha selection."""
    from sklearn.linear_model import RidgeCV
    
    alphas = ModelConfig.MODELS['ridge']['alpha']
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'name': 'Ridge',
        'best_alpha': model.alpha_,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }


def train_lasso(X_train, y_train, X_test, y_test) -> Dict:
    """Train Lasso Regression with cross-validation."""
    from sklearn.linear_model import LassoCV
    
    alphas = ModelConfig.MODELS['lasso']['alpha']
    model = LassoCV(alphas=alphas, cv=5, max_iter=5000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Count non-zero coefficients (selected features)
    n_features = np.sum(model.coef_ != 0)
    
    return {
        'model': model,
        'name': 'Lasso',
        'best_alpha': model.alpha_,
        'n_features': n_features,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> Dict:
    """Train Random Forest Regressor."""
    params = ModelConfig.MODELS['random_forest']
    
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        n_jobs=params['n_jobs'],
        random_state=ModelConfig.RANDOM_STATE
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'name': 'RandomForest',
        'feature_importance': importance,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }


def train_xgboost(X_train, y_train, X_test, y_test) -> Dict:
    """Train XGBoost Regressor."""
    try:
        import xgboost as xgb
    except ImportError:
        print("   ⚠️ XGBoost not installed. Skipping...")
        return None
    
    params = ModelConfig.MODELS['xgboost']
    
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=ModelConfig.RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'name': 'XGBoost',
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }


def train_lightgbm(X_train, y_train, X_test, y_test) -> Dict:
    """Train LightGBM Regressor."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("   ⚠️ LightGBM not installed. Skipping...")
        return None
    
    params = ModelConfig.MODELS['lightgbm']
    
    model = lgb.LGBMRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        subsample=params['subsample'],
        random_state=ModelConfig.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'name': 'LightGBM',
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }


def train_lstm(X_train, y_train, X_test, y_test) -> Dict:
    """Train LSTM Neural Network."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("   ⚠️ TensorFlow not installed. Skipping LSTM...")
        return None
    
    params = ModelConfig.MODELS['lstm']
    seq_len = params['sequence_length']
    
    # Prepare sequences
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, seq_len)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("   ⚠️ Not enough data for LSTM sequences. Skipping...")
        return None
    
    # Build model
    model = Sequential([
        LSTM(params['units'], input_shape=(seq_len, X_train.shape[1]), return_sequences=True),
        Dropout(params['dropout']),
        LSTM(params['units'] // 2),
        Dropout(params['dropout']),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='huber')
    
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_seq, y_train_seq,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_test_seq, y_test_seq),
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    return {
        'model': model,
        'name': 'LSTM',
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test_seq, y_pred)
    }


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate regression metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_all_models(df: pd.DataFrame, horizon: int, use_scaling: bool = True) -> Dict:
    """Train all models for a specific horizon."""
    
    print(f"\n{'='*70}")
    print(f"   TRAINING MODELS FOR {horizon}h HORIZON")
    print(f"{'='*70}")
    
    # Prepare data
    X, y = prepare_features(df, horizon)
    print(f"\n📊 Data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = None
    if use_scaling:
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    results = {}
    
    # 1. Baseline: Persistence
    print("\n🎯 Baseline: Persistence")
    persistence = PersistenceModel()
    persistence.fit(X_train, y_train)
    y_pred = persistence.predict(X_test)
    results['Persistence'] = {
        'model': persistence,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    print(f"   MAE: {results['Persistence']['metrics']['MAE']:.2f}")
    
    # 2. Baseline: Moving Average
    print("\n🎯 Baseline: Moving Average (24h)")
    ma = MovingAverageModel(24)
    ma.fit(X_train, y_train)
    y_pred = ma.predict(X_test)
    results['MA_24h'] = {
        'model': ma,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    print(f"   MAE: {results['MA_24h']['metrics']['MAE']:.2f}")
    
    # 3. Ridge Regression
    print("\n🔧 Training: Ridge Regression")
    result = train_ridge(X_train_scaled, y_train, X_test_scaled, y_test)
    results['Ridge'] = result
    print(f"   MAE: {result['metrics']['MAE']:.2f}, Alpha: {result['best_alpha']}")
    
    # 4. Lasso Regression
    print("\n🔧 Training: Lasso Regression")
    result = train_lasso(X_train_scaled, y_train, X_test_scaled, y_test)
    results['Lasso'] = result
    print(f"   MAE: {result['metrics']['MAE']:.2f}, Features: {result['n_features']}")
    
    # 5. Random Forest
    print("\n🌲 Training: Random Forest")
    result = train_random_forest(X_train, y_train, X_test, y_test)
    results['RandomForest'] = result
    print(f"   MAE: {result['metrics']['MAE']:.2f}")
    
    # 6. XGBoost
    print("\n🚀 Training: XGBoost")
    result = train_xgboost(X_train, y_train, X_test, y_test)
    if result:
        results['XGBoost'] = result
        print(f"   MAE: {result['metrics']['MAE']:.2f}")
    
    # 7. LightGBM
    print("\n⚡ Training: LightGBM")
    result = train_lightgbm(X_train, y_train, X_test, y_test)
    if result:
        results['LightGBM'] = result
        print(f"   MAE: {result['metrics']['MAE']:.2f}")
    
    # 8. LSTM (optional, slower)
    # print("\n🧠 Training: LSTM")
    # result = train_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
    # if result:
    #     results['LSTM'] = result
    #     print(f"   MAE: {result['metrics']['MAE']:.2f}")
    
    return {
        'horizon': horizon,
        'scaler': scaler,
        'feature_cols': list(X.columns),
        'results': results,
        'y_test': y_test.values,
    }


def compare_models(all_results: Dict) -> pd.DataFrame:
    """Create comparison table of all models across all horizons."""
    
    rows = []
    for horizon, data in all_results.items():
        for model_name, result in data['results'].items():
            metrics = result['metrics']
            rows.append({
                'Horizon': f'{horizon}h',
                'Model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R²': metrics['R2'],
                'MAPE': metrics['MAPE']
            })
    
    df = pd.DataFrame(rows)
    return df


def save_models(all_results: Dict, output_dir: Path):
    """Save trained models and scalers."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for horizon, data in all_results.items():
        horizon_dir = output_dir / f"horizon_{horizon}h"
        horizon_dir.mkdir(exist_ok=True)
        
        # Save scaler
        if data['scaler']:
            joblib.dump(data['scaler'], horizon_dir / 'scaler.joblib')
        
        # Save feature columns
        with open(horizon_dir / 'feature_cols.json', 'w') as f:
            json.dump(data['feature_cols'], f)
        
        # Save each model
        for model_name, result in data['results'].items():
            model = result['model']
            
            # Skip non-serializable models
            if model_name in ['Persistence', 'MA_24h']:
                continue
            
            if model_name == 'LSTM':
                model.save(horizon_dir / f'{model_name}.keras')
            else:
                joblib.dump(model, horizon_dir / f'{model_name}.joblib')
    
    print(f"\n💾 Models saved to: {output_dir}")


def main():
    """Main training pipeline."""
    
    print("="*70)
    print("   🚀 AQI PREDICTION MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Horizons: {ModelConfig.HORIZONS}")
    
    # Load data
    try:
        df = load_data_from_hopsworks()
    except Exception as e:
        print(f"   ⚠️ Hopsworks failed: {e}")
        print("   Falling back to local data...")
        df = load_data_from_local()
    
    # Train models for each horizon
    all_results = {}
    
    for horizon in ModelConfig.HORIZONS:
        results = train_all_models(df, horizon)
        all_results[horizon] = results
    
    # Compare all models
    print("\n" + "="*70)
    print("   📊 MODEL COMPARISON (All Horizons)")
    print("="*70)
    
    comparison = compare_models(all_results)
    
    # Pivot to show by horizon
    pivot = comparison.pivot_table(
        index='Model',
        columns='Horizon',
        values='MAE',
        aggfunc='first'
    )
    print("\nMAE by Model and Horizon:")
    print(pivot.round(2).to_string())
    
    # Best model per horizon
    print("\n🏆 BEST MODELS:")
    for horizon in ModelConfig.HORIZONS:
        horizon_data = comparison[comparison['Horizon'] == f'{horizon}h']
        best = horizon_data.loc[horizon_data['MAE'].idxmin()]
        print(f"   {horizon}h: {best['Model']} (MAE: {best['MAE']:.2f})")
    
    # Save models
    output_dir = project_root / "models" / "trained"
    save_models(all_results, output_dir)
    
    # Save comparison
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    
    print("\n" + "="*70)
    print("   ✅ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
