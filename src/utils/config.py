"""
Configuration settings for the AQI Predictor.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration."""
    aqi_api_key: str = os.getenv("AQI_API_KEY", "")
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")


@dataclass
class FeatureStoreConfig:
    """Feature store configuration."""
    use_hopsworks: bool = os.getenv("HOPSWORKS_API_KEY") is not None
    hopsworks_api_key: str = os.getenv("HOPSWORKS_API_KEY", "")
    hopsworks_project: str = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
    local_path: Path = Path(__file__).parent.parent.parent / "data" / "processed"


@dataclass
class ModelConfig:
    """Model configuration."""
    default_model: str = "random_forest"
    target_horizon: int = 24  # Hours
    sequence_length: int = 24  # For LSTM models
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001


@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    default_city: str = os.getenv("DEFAULT_CITY", "Karachi")
    default_country: str = os.getenv("DEFAULT_COUNTRY", "Pakistan")


@dataclass
class Config:
    """Main configuration class."""
    api: APIConfig = None
    feature_store: FeatureStoreConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    app: AppConfig = None
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    
    def __post_init__(self):
        if self.api is None:
            self.api = APIConfig()
        if self.feature_store is None:
            self.feature_store = FeatureStoreConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.app is None:
            self.app = AppConfig()
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "backfill").mkdir(exist_ok=True)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration."""
    return config
