"""
Training script that fetches features from Hopsworks and trains XGBoost model.

This script:
1. Connects to Hopsworks Feature Store
2. Fetches the AQI features
3. Trains multiple models (XGBoost, LightGBM, Random Forest, Ridge)
4. Evaluates using RMSE, MAE, R²
5. Saves the best model to the model registry
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


class HopsworksTrainer:
    """
    Trainer that fetches data from Hopsworks and trains AQI prediction models.
    """
    
    def __init__(self, project_name: str = None, api_key: str = None):
        """
        Initialize the trainer with Hopsworks credentials.
        
        Args:
            project_name: Hopsworks project name
            api_key: Hopsworks API key
        """
        self.project_name = project_name or os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
        self.api_key = api_key or os.getenv("HOPSWORKS_API_KEY")
        
        self.connection = None
        self.fs = None
        self.df = None
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def connect_to_hopsworks(self) -> bool:
        """
        Connect to Hopsworks Feature Store.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import hopsworks
            
            print("=" * 60)
            print("CONNECTING TO HOPSWORKS")
            print("=" * 60)
            print(f"Project: {self.project_name}")
            
            self.connection = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.connection.get_feature_store()
            
            print("[SUCCESS] Connected to Hopsworks Feature Store")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to Hopsworks: {e}")
            print("[INFO] Falling back to local data...")
            return False
    
    def fetch_features_from_hopsworks(self, feature_group_name: str = "islamabad_aqi_features") -> pd.DataFrame:
        """
        Fetch features from Hopsworks Feature Store.
        
        Args:
            feature_group_name: Name of the feature group
            
        Returns:
            DataFrame with features
        """
        try:
            print(f"\nFetching feature group: {feature_group_name}")
            
            fg = self.fs.get_feature_group(name=feature_group_name, version=1)
            df = fg.read()
            
            print(f"[SUCCESS] Fetched {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch from Hopsworks: {e}")
            return pd.DataFrame()
    
    def load_local_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load data from local parquet file as fallback.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            DataFrame with features
        """
        if file_path is None:
            file_path = Path(__file__).parent.parent / "data" / "processed" / "islamabad_aqi_features_upload.parquet"
        
        print(f"\nLoading local data from: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"[SUCCESS] Loaded {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = "target_24h",
        test_size: float = 0.2,
        feature_selection: bool = True,
        top_n_features: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training with time-series aware split.
        
        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            test_size: Fraction of data for testing
            feature_selection: Whether to select top features
            top_n_features: Number of top features to select
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        # Define columns to exclude
        meta_cols = ['timestamp', 'event_time', 'observation_id', 'city']
        target_cols = ['target_1h', 'target_6h', 'target_12h', 'target_24h', 'target_48h', 'target_72h']
        
        # Get feature columns
        feature_cols = [c for c in df.columns if c not in meta_cols + target_cols]
        
        print(f"Target column: {target_col}")
        print(f"Initial features: {len(feature_cols)}")
        
        # Feature selection based on correlation with target
        if feature_selection and len(feature_cols) > top_n_features:
            print(f"\nSelecting top {top_n_features} features based on correlation...")
            correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col).abs()
            top_features = correlations.nlargest(top_n_features).index.tolist()
            feature_cols = top_features
            print(f"Selected features: {len(feature_cols)}")
        
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=[target_col])
        print(f"Samples after removing NaN targets: {len(df_clean)}")
        
        # Sort by timestamp for time-series split
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        elif 'event_time' in df_clean.columns:
            df_clean = df_clean.sort_values('event_time').reset_index(drop=True)
        
        # Extract features and target
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        # Handle any remaining NaN in features
        X = np.nan_to_num(X, nan=0.0)
        
        # Time-series split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(feature_cols)}")
        
        self.feature_names = feature_cols
        return X_train, X_test, y_train, y_test, feature_cols
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoiding division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def train_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        models_to_train: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and evaluate them.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            models_to_train: List of model names to train
            
        Returns:
            Dictionary with model results
        """
        from src.training.models.sklearn_models import get_model, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
        
        if models_to_train is None:
            models_to_train = ['xgboost', 'lightgbm', 'random_forest', 'ridge', 'gradient_boosting']
        
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        results = {}
        
        for model_name in models_to_train:
            try:
                print(f"\n[{model_name.upper()}]")
                print("-" * 40)
                
                # Skip if not available
                if model_name == 'xgboost' and not XGBOOST_AVAILABLE:
                    print("  Skipped - XGBoost not installed")
                    continue
                if model_name == 'lightgbm' and not LIGHTGBM_AVAILABLE:
                    print("  Skipped - LightGBM not installed")
                    continue
                
                # Get model
                model = get_model(model_name)
                
                # Train
                start_time = datetime.now()
                model.fit(X_train, y_train)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Evaluate
                train_metrics = self.evaluate_model(y_train, y_pred_train)
                test_metrics = self.evaluate_model(y_test, y_pred_test)
                
                # Print results
                print(f"  Training time: {train_time:.2f}s")
                print(f"  Train RMSE: {train_metrics['rmse']:.4f}")
                print(f"  Test RMSE:  {test_metrics['rmse']:.4f}")
                print(f"  Test MAE:   {test_metrics['mae']:.4f}")
                print(f"  Test R2:    {test_metrics['r2']:.4f}")
                print(f"  Test MAPE:  {test_metrics['mape']:.2f}%")
                
                results[model_name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'train_time': train_time,
                    'predictions': y_pred_test
                }
                
            except Exception as e:
                print(f"  [ERROR] Failed to train {model_name}: {e}")
        
        return results
    
    def get_feature_importance(self, model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model pipeline
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        try:
            # Get the actual model from pipeline
            actual_model = model.named_steps['model']
            
            if hasattr(actual_model, 'feature_importances_'):
                importance = actual_model.feature_importances_
            elif hasattr(actual_model, 'coef_'):
                importance = np.abs(actual_model.coef_)
            else:
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
            
        except Exception as e:
            print(f"Could not get feature importance: {e}")
            return pd.DataFrame()
    
    def save_best_model(
        self, 
        results: Dict[str, Dict[str, Any]], 
        feature_names: List[str],
        metric: str = 'rmse'
    ) -> str:
        """
        Save the best model to disk.
        
        Args:
            results: Dictionary with model results
            feature_names: List of feature names
            metric: Metric to use for comparison (lower is better for rmse/mae)
            
        Returns:
            Path to saved model
        """
        print("\n" + "=" * 60)
        print("SAVING BEST MODEL")
        print("=" * 60)
        
        # Find best model
        if metric in ['rmse', 'mae', 'mape']:
            best_model_name = min(results, key=lambda x: results[x]['test_metrics'][metric])
        else:
            best_model_name = max(results, key=lambda x: results[x]['test_metrics'][metric])
        
        best_result = results[best_model_name]
        best_model = best_result['model']
        best_metrics = best_result['test_metrics']
        
        print(f"Best model: {best_model_name}")
        print(f"Test RMSE: {best_metrics['rmse']:.4f}")
        print(f"Test R2: {best_metrics['r2']:.4f}")
        
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / best_model_name / timestamp
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'version': timestamp,
            'created_at': datetime.now().isoformat(),
            'metrics': best_metrics,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'target': 'target_24h'
        }
        
        import json
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest pointer
        latest_path = self.models_dir / best_model_name / "latest.txt"
        with open(latest_path, 'w') as f:
            f.write(timestamp)
        
        # Get and save feature importance
        importance_df = self.get_feature_importance(best_model, feature_names)
        if not importance_df.empty:
            importance_path = model_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"\nTop 10 Important Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nModel saved to: {model_path}")
        
        return str(model_path)
    
    def run(
        self, 
        use_hopsworks: bool = True,
        target_col: str = "target_24h",
        models_to_train: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline.
        
        Args:
            use_hopsworks: Whether to fetch from Hopsworks (falls back to local if fails)
            target_col: Target column for prediction
            models_to_train: List of models to train
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "=" * 60)
        print("AQI PREDICTION MODEL TRAINING")
        print(f"Started at: {datetime.now().isoformat()}")
        print("=" * 60)
        
        # 1. Load data
        df = None
        if use_hopsworks:
            if self.connect_to_hopsworks():
                df = self.fetch_features_from_hopsworks()
        
        if df is None or df.empty:
            df = self.load_local_data()
        
        self.df = df
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            df, 
            target_col=target_col,
            feature_selection=True,
            top_n_features=50
        )
        
        # 3. Train models
        results = self.train_models(X_train, y_train, X_test, y_test, models_to_train)
        
        # 4. Print comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (Test Set)")
        print("=" * 60)
        print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE':>10}")
        print("-" * 60)
        
        for model_name, result in sorted(results.items(), key=lambda x: x[1]['test_metrics']['rmse']):
            metrics = result['test_metrics']
            print(f"{model_name:<20} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f} {metrics['r2']:>10.4f} {metrics['mape']:>9.2f}%")
        
        # 5. Save best model
        model_path = self.save_best_model(results, feature_names)
        
        print("\n" + "=" * 60)
        print(f"Training completed at: {datetime.now().isoformat()}")
        print("=" * 60)
        
        return {
            'results': results,
            'model_path': model_path,
            'feature_names': feature_names
        }


if __name__ == "__main__":
    # Run training
    trainer = HopsworksTrainer()
    
    # Train with XGBoost, LightGBM, and other models
    results = trainer.run(
        use_hopsworks=True,  # Will fall back to local if Hopsworks fails
        target_col="target_24h",
        models_to_train=['xgboost', 'lightgbm', 'random_forest', 'ridge', 'gradient_boosting']
    )
