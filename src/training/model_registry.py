"""
Model Registry for saving and loading trained models.
"""
import os
import json
import joblib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class LocalModelRegistry:
    """Local file-based model registry."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "models"
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_model(
        self, 
        model: Any, 
        model_name: str, 
        metrics: Dict[str, float] = None,
        params: Dict[str, Any] = None,
        version: str = None,
        feature_names: list = None
    ) -> str:
        """
        Save a model to the registry.
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics or {},
            'params': params or {},
            'feature_names': feature_names or []
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest version
        latest_path = self.base_path / model_name / "latest.txt"
        with open(latest_path, 'w') as f:
            f.write(version)
        
        print(f"Model saved to: {model_path}")
        return str(model_path)
    
    def load_model(self, model_name: str, version: str = None) -> Any:
        """
        Load a model from the registry.
        """
        if version is None:
            latest_path = self.base_path / model_name / "latest.txt"
            if latest_path.exists():
                version = latest_path.read_text().strip()
            else:
                raise ValueError(f"No models found for: {model_name}")
        
        model_path = self.base_path / model_name / version / "model.joblib"
        
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        return joblib.load(model_path)
    
    def get_model_metadata(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """Get metadata for a model."""
        if version is None:
            latest_path = self.base_path / model_name / "latest.txt"
            if latest_path.exists():
                version = latest_path.read_text().strip()
            else:
                raise ValueError(f"No models found for: {model_name}")
        
        metadata_path = self.base_path / model_name / version / "metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> Dict[str, list]:
        """List all models and their versions."""
        models = {}
        
        for model_dir in self.base_path.iterdir():
            if model_dir.is_dir():
                versions = [
                    v.name for v in model_dir.iterdir() 
                    if v.is_dir() and (v / "model.joblib").exists()
                ]
                if versions:
                    models[model_dir.name] = sorted(versions, reverse=True)
        
        return models


class MLflowModelRegistry:
    """MLflow-based model registry."""
    
    def __init__(self, tracking_uri: str = None):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = "aqi_predictor"
        mlflow.set_experiment(self.experiment_name)
    
    def save_model(
        self, 
        model: Any, 
        model_name: str, 
        metrics: Dict[str, float] = None,
        params: Dict[str, Any] = None,
        feature_names: list = None,
        **kwargs
    ) -> str:
        """Save a model using MLflow."""
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Register model
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{model_name}"
            
            mlflow.register_model(model_uri, model_name)
            
            return model_uri
    
    def load_model(self, model_name: str, version: str = None) -> Any:
        """Load a model from MLflow."""
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        return mlflow.sklearn.load_model(model_uri)


class SupabaseModelRegistry(LocalModelRegistry):
    """Supabase-based model registry."""
    
    def __init__(self, base_path: str = None):
        super().__init__(base_path)
        from supabase import create_client, Client
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        self.client: Client = create_client(self.url, self.key)
        
    def save_model(
        self, 
        model: Any, 
        model_name: str, 
        metrics: Dict[str, float] = None,
        params: Dict[str, Any] = None,
        version: str = None,
        feature_names: list = None
    ) -> str:
        """
        Save a model to Supabase Model Registry.
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Call LocalModelRegistry's save_model to save joblib and metadata file
        model_path = super().save_model(model, model_name, metrics, params, version, feature_names)
        
        # Save metadata to Supabase table
        try:
            record = {
                'model_name': model_name,
                'version': version,
                'metrics': metrics or {},
                'params': params or {},
                'feature_names': feature_names or [],
                'description': f"AQI Predictor - {model_name}"
            }
            
            self.client.table('model_registry').upsert(record, on_conflict='model_name,version').execute()
            print(f"Model metadata saved to Supabase: {model_name} v{version}")
            
        except Exception as e:
            print(f"Error saving to Supabase Model Registry: {e}")
            raise
            
        return f"{model_name}_v{version}"
    
    def get_model_metadata(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """Get metadata for a model from Supabase."""
        query = self.client.table('model_registry').select('*').eq('model_name', model_name)
        if version:
            query = query.eq('version', version)
        else:
            query = query.order('version', desc=True).limit(1)
            
        try:
            response = query.execute()
            data = response.data
            if data:
                return data[0]
            return super().get_model_metadata(model_name, version)
        except Exception as e:
            print(f"Error getting metadata from Supabase: {e}")
            return super().get_model_metadata(model_name, version)
    
    def list_models(self) -> Dict[str, list]:
        """List all models in Supabase Model Registry."""
        try:
            response = self.client.table('model_registry').select('model_name', 'version').execute()
            data = response.data
            models = {}
            for item in data:
                name = item['model_name']
                if name not in models:
                    models[name] = []
                models[name].append(item['version'])
            return models
        except Exception as e:
            print(f"Error listing models from Supabase: {e}")
            return super().list_models()


def get_model_registry(use_mlflow: bool = None, use_supabase: bool = None) -> Union[LocalModelRegistry, MLflowModelRegistry, SupabaseModelRegistry]:
    """
    Factory function to get model registry.
    """
    if use_supabase is None:
        use_supabase = os.getenv("SUPABASE_URL") is not None
    
    if use_supabase:
        return SupabaseModelRegistry()
    
    if use_mlflow is None:
        use_mlflow = MLFLOW_AVAILABLE and os.getenv("MLFLOW_TRACKING_URI") is not None
    
    if use_mlflow:
        return MLflowModelRegistry()
    
    return LocalModelRegistry()
