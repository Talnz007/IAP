"""
Prediction module for AQI forecasting.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_store import get_feature_store
from src.features.compute_features import compute_all_features
from src.training.model_registry import get_model_registry


class AQIPredictor:
    """AQI prediction class."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_name: Name of the model to load. If None, loads the best model.
        """
        self.registry = get_model_registry()
        self.feature_store = get_feature_store()
        self.model = None
        self.model_name = model_name
        
        self._load_model()
    
    def _load_model(self):
        """Load the model from registry."""
        models = self.registry.list_models()
        
        if not models:
            raise ValueError("No trained models found in registry")
        
        if self.model_name is None:
            # Load the first available model
            self.model_name = list(models.keys())[0]
        
        self.model = self.registry.load_model(self.model_name)
        print(f"Loaded model: {self.model_name}")
    
    def get_latest_features(self, city: str = "Islamabad") -> pd.DataFrame:
        """
        Get the latest features for prediction.
        
        Args:
            city: City name (default: Islamabad)
            
        Returns:
            DataFrame with latest features
        """
        # Get recent data from feature store
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Get last week's data
        
        df = self.feature_store.get_features(
            'aqi_features',
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise ValueError("No feature data available")
        
        # Filter by city if available
        if 'city' in df.columns:
            df = df[df['city'] == city]
        
        # Sort by timestamp and get latest
        df = df.sort_values('timestamp').tail(1)
        
        return df
    
    def predict(self, features: pd.DataFrame = None, city: str = "Islamabad") -> Dict[str, Any]:
        """
        Make AQI predictions.
        
        Args:
            features: Optional feature DataFrame. If None, fetches latest.
            city: City name for predictions (default: Islamabad)
            
        Returns:
            Dictionary with predictions
        """
        if features is None:
            features = self.get_latest_features(city)
        
        # Process features if needed
        if 'hour' not in features.columns:
            features, feature_cols, _ = compute_all_features(features)
        else:
            exclude_cols = ['timestamp', 'city']
            target_cols = [col for col in features.columns if '_target_' in col]
            feature_cols = [col for col in features.columns if col not in exclude_cols + target_cols]
        
        # Drop rows with NaN and select feature columns
        features = features.dropna()
        X = features[feature_cols].values
        
        if len(X) == 0:
            raise ValueError("No valid features available for prediction")
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Get current values
        current_aqi = features['aqi'].iloc[-1] if 'aqi' in features.columns else None
        
        return {
            'city': city,
            'prediction_time': datetime.now().isoformat(),
            'current_aqi': float(current_aqi) if current_aqi else None,
            'predicted_aqi_24h': float(prediction),
            'model_used': self.model_name,
            'aqi_category': self._get_aqi_category(prediction),
            'health_advisory': self._get_health_advisory(prediction)
        }
    
    def predict_next_3_days(self, city: str = "Karachi") -> List[Dict[str, Any]]:
        """
        Predict AQI for the next 3 days.
        
        Args:
            city: City name
            
        Returns:
            List of predictions for each day
        """
        predictions = []
        
        # This is a simplified approach
        # In practice, you'd have models for different horizons
        base_prediction = self.predict(city=city)
        
        for day in range(1, 4):
            # Simple extrapolation (in practice, use proper multi-step forecasting)
            variation = np.random.uniform(-10, 10)  # Add some variation
            predicted_aqi = base_prediction['predicted_aqi_24h'] + variation * (day - 1)
            
            predictions.append({
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_aqi': float(max(0, predicted_aqi)),
                'aqi_category': self._get_aqi_category(predicted_aqi),
                'health_advisory': self._get_health_advisory(predicted_aqi)
            })
        
        return predictions
    
    @staticmethod
    def _get_aqi_category(aqi: float) -> str:
        """Get AQI category based on value."""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    @staticmethod
    def _get_health_advisory(aqi: float) -> str:
        """Get health advisory based on AQI."""
        if aqi <= 50:
            return "Air quality is satisfactory. Enjoy outdoor activities."
        elif aqi <= 100:
            return "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion."
        elif aqi <= 150:
            return "Members of sensitive groups may experience health effects. General public less likely to be affected."
        elif aqi <= 200:
            return "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects."
        elif aqi <= 300:
            return "Health alert: everyone may experience more serious health effects. Avoid outdoor activities."
        else:
            return "Health warning of emergency conditions. Everyone should avoid outdoor activities."


def main():
    """Example usage of the predictor."""
    try:
        predictor = AQIPredictor()
        
        # Single prediction for Islamabad
        prediction = predictor.predict(city="Islamabad")
        print("\nSingle Prediction for Islamabad:")
        print(f"  Current AQI: {prediction['current_aqi']}")
        print(f"  Predicted AQI (24h): {prediction['predicted_aqi_24h']:.1f}")
        print(f"  Category: {prediction['aqi_category']}")
        print(f"  Advisory: {prediction['health_advisory']}")
        
        # 3-day forecast
        print("\n3-Day Forecast for Islamabad:")
        forecasts = predictor.predict_next_3_days(city="Islamabad")
        for forecast in forecasts:
            print(f"  Day {forecast['day']} ({forecast['date']}): AQI {forecast['predicted_aqi']:.1f} - {forecast['aqi_category']}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train a model first by running src/training/train.py")


if __name__ == "__main__":
    main()
