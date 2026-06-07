"""
Training Pipeline - Fetches features from Hopsworks and trains 3 models:
1. XGBoost (Gradient Boosting)
2. LightGBM (Gradient Boosting)
3. Neural Network (TensorFlow)

As per PDF requirements:
- Fetch historical (features, targets) from Feature Store
- Train and evaluate ML models
- Evaluate using RMSE, MAE, R²
- Store trained models in Model Registry
"""
import os
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "api_predictor")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1

MODEL_SAVE_PATH = Path(__file__).parent.parent / "models"
TARGET_COL = "target_24h"

# Features to exclude from training
EXCLUDE_COLS = [
    'timestamp', 'event_time', 'observation_id',
    'target_1h', 'target_6h', 'target_12h', 'target_24h', 'target_48h', 'target_72h'
]


# ============================================================
# STEP 1: FETCH FEATURES FROM HOPSWORKS
# ============================================================
def fetch_features_from_hopsworks() -> pd.DataFrame:
    """
    Fetch historical features and targets from Hopsworks Feature Store.
    
    Returns:
        DataFrame with features and targets
    """
    print("\n" + "=" * 60)
    print("STEP 1: FETCHING FEATURES FROM HOPSWORKS")
    print("=" * 60)
    
    try:
        import hopsworks
        
        print(f"Connecting to Hopsworks project: {HOPSWORKS_PROJECT}")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        
        fs = project.get_feature_store()
        print("Connected to Feature Store!")
        
        # Get feature group
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        print(f"Found feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
        
        # Read all features
        df = fg.read()
        print(f"Fetched {len(df)} records with {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        print(f"Error connecting to Hopsworks: {e}")
        print("Falling back to local data...")
        return fetch_features_from_local()


def fetch_features_from_local() -> pd.DataFrame:
    """
    Fallback: Fetch features from local parquet file.
    """
    local_path = Path(__file__).parent.parent / "data" / "processed" / "islamabad_aqi_features_upload.parquet"
    print(f"Loading from local: {local_path}")
    df = pd.read_parquet(local_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


# ============================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================
def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    """
    Prepare data for training - proper time series split.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "=" * 60)
    print("STEP 2: PREPARING DATA")
    print("=" * 60)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove rows with NaN in target
    df = df.dropna(subset=[TARGET_COL])
    print(f"Samples after removing NaN: {len(df)}")
    
    # Define feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"Number of features: {len(feature_cols)}")
    
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    
    # Time series split (no shuffling - use last 20% as test)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Train target mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    print(f"Test target mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols, scaler


# ============================================================
# STEP 3: MODEL DEFINITIONS
# ============================================================
def get_xgboost_model() -> xgb.XGBRegressor:
    """XGBoost model - powerful gradient boosting."""
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )


def get_lightgbm_model() -> lgb.LGBMRegressor:
    """LightGBM model."""
    return lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )


def build_neural_network(input_dim: int):
    """
    Build improved Neural Network model (TensorFlow/Keras).
    
    Architecture (Deeper with Residual-like connections):
    - Input layer
    - Dense(256) + BatchNorm + LeakyReLU + Dropout
    - Dense(128) + BatchNorm + LeakyReLU + Dropout
    - Dense(64) + BatchNorm + LeakyReLU + Dropout
    - Dense(32) + BatchNorm + LeakyReLU
    - Output layer (1 unit for regression)
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # First block - wider layer
            layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.1),
            layers.Dropout(0.4),
            
            # Second block
            layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.1),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.1),
            layers.Dropout(0.2),
            
            # Fourth block
            layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.1),
            
            # Output layer
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='huber',  # Huber loss is more robust to outliers
            metrics=['mae']
        )
        
        return model
        
    except ImportError:
        print("TensorFlow not installed. Skipping Neural Network.")
        return None


# ============================================================
# STEP 4: TRAINING AND EVALUATION
# ============================================================
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics: RMSE, MAE, R²."""
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict]:
    """Train XGBoost model."""
    print("\n--- Training XGBoost ---")
    
    model = get_xgboost_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    
    print(f"Train - RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.3f}")
    print(f"Test  - RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.3f}")
    
    return model, test_metrics


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict]:
    """Train LightGBM model."""
    print("\n--- Training LightGBM ---")
    
    model = get_lightgbm_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    
    print(f"Train - RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.3f}")
    print(f"Test  - RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.3f}")
    
    return model, test_metrics


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict]:
    """Train Neural Network model with improved training strategy."""
    print("\n--- Training Neural Network (Improved) ---")
    
    model = build_neural_network(X_train.shape[1])
    
    if model is None:
        return None, {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
    
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        import tempfile
        import os
        
        # Create temp file for best model
        temp_model_path = os.path.join(tempfile.gettempdir(), 'best_nn_model.keras')
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=25,  # More patience
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3,  # More aggressive reduction
                patience=8, 
                min_lr=1e-7,
                verbose=0
            ),
            ModelCheckpoint(
                temp_model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,  # More epochs with early stopping
            batch_size=32,  # Smaller batch for better generalization
            callbacks=callbacks,
            verbose=0
        )
        
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_test = model.predict(X_test, verbose=0).flatten()
        
        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)
        
        print(f"Train - RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.3f}")
        print(f"Test  - RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.3f}")
        print(f"Epochs trained: {len(history.history['loss'])}")
        
        # Clean up temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        return model, test_metrics
        
    except Exception as e:
        print(f"Error training Neural Network: {e}")
        return None, {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}


# ============================================================
# STEP 5: SAVE MODELS TO REGISTRY
# ============================================================
def save_model(model: Any, model_name: str, metrics: Dict, 
               feature_names: List[str], scaler: StandardScaler = None):
    """
    Save model to Model Registry.
    
    Creates:
    - models/{model_name}/{version}/model.joblib (or model.h5 for NN)
    - models/{model_name}/{version}/metadata.json
    - models/{model_name}/{version}/scaler.joblib (if provided)
    - models/{model_name}/latest.txt
    """
    print(f"\nSaving {model_name} to Model Registry...")
    
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODEL_SAVE_PATH / model_name / version
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if model_name == "neural_network":
        model_path = model_dir / "model.h5"
        model.save(model_path)
    else:
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
    
    # Save scaler (needed for NN)
    if scaler is not None:
        scaler_path = model_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'version': version,
        'created_at': datetime.now().isoformat(),
        'target': TARGET_COL,
        'metrics': metrics,
        'feature_count': len(feature_names),
        'feature_names': feature_names[:50],  # Save first 50 feature names
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest version pointer
    latest_path = MODEL_SAVE_PATH / model_name / "latest.txt"
    with open(latest_path, 'w') as f:
        f.write(version)
    
    print(f"  Saved locally to: {model_dir}")
    
    # Upload to Hopsworks Model Registry
    try:
        import hopsworks
        
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        mr = project.get_model_registry()
        
        # Create model in Hopsworks (simplified - no schema)
        hopsworks_model = mr.python.create_model(
            name=f"islamabad_aqi_{model_name}",
            metrics=metrics,
            description=f"AQI prediction model for Islamabad - {model_name}"
        )
        
        # Save model directory to Hopsworks
        hopsworks_model.save(str(model_dir))
        print(f"  ✅ Uploaded to Hopsworks Model Registry!")
        
    except Exception as e:
        print(f"  ⚠️ Could not upload to Hopsworks: {e}")
        print(f"  (Model saved locally - Hopsworks upload is optional)")
    
    return str(model_dir)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
def run_training_pipeline():
    """
    Main training pipeline:
    1. Fetch features from Hopsworks
    2. Prepare data
    3. Train 3 models: XGBoost, LightGBM, Neural Network
    4. Evaluate with RMSE, MAE, R²
    5. Save models to registry
    """
    print("\n" + "=" * 60)
    print("AQI PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Target: {TARGET_COL} (24-hour ahead PM2.5 prediction)")
    
    # Step 1: Fetch features
    df = fetch_features_from_hopsworks()
    
    # Step 2: Prepare data
    (X_train, X_test, y_train, y_test, 
     X_train_scaled, X_test_scaled, 
     feature_names, scaler) = prepare_data(df)
    
    # Step 3 & 4: Train and evaluate models
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MODELS")
    print("=" * 60)
    
    results = {}
    
    # Model 1: XGBoost (uses raw features)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics, 'uses_scaler': False}
    
    # Model 2: LightGBM (uses raw features)
    lgbm_model, lgbm_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    results['lightgbm'] = {'model': lgbm_model, 'metrics': lgbm_metrics, 'uses_scaler': False}
    
    # Model 3: Neural Network (uses scaled features)
    nn_model, nn_metrics = train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
    results['neural_network'] = {'model': nn_model, 'metrics': nn_metrics, 'uses_scaler': True}
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<20} {m['rmse']:<10.2f} {m['mae']:<10.2f} {m['r2']:<10.3f}")
    
    # Find best model
    best_model_name = min(results, key=lambda x: results[x]['metrics']['rmse'])
    print(f"\nBest Model: {best_model_name} (RMSE: {results[best_model_name]['metrics']['rmse']:.2f})")
    
    # Step 5: Save models to registry
    print("\n" + "=" * 60)
    print("STEP 4: SAVING MODELS TO REGISTRY")
    print("=" * 60)
    
    for name, data in results.items():
        if data['model'] is not None:
            save_scaler = scaler if data['uses_scaler'] else None
            save_model(data['model'], name, data['metrics'], feature_names, save_scaler)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED")
    print(f"Finished at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_training_pipeline()
