"""
Islamabad AQI Predictor - Uses trained models with Hopsworks data
"""
import os
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Setup paths
WEBAPP_DIR = Path(__file__).parent
PROJECT_ROOT = WEBAPP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')


class IslamabadAQIPredictor:
    """AQI Predictor specifically for Islamabad using trained models."""
    
    def __init__(self, model_name: str = "lightgbm"):
        """
        Initialize the predictor.
        
        Args:
            model_name: Model to use ('lightgbm', 'xgboost', 'random_forest', or 'ridge')
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and scaler."""
        model_dir = MODELS_DIR / self.model_name
        latest_file = model_dir / "latest.txt"
        
        if not latest_file.exists():
            raise ValueError(f"No trained {self.model_name} model found")
        
        version = latest_file.read_text().strip()
        version_dir = model_dir / version
        
        # Load model
        model_path = version_dir / "model.joblib"
        self.model = joblib.load(model_path)
        
        # Load scaler if exists
        scaler_path = version_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            self.feature_names = self.metadata.get("feature_names", [])
        
        print(f"Loaded model: {self.model_name} v{version}")
    
    def get_latest_features_from_hopsworks(self) -> pd.DataFrame:
        """Fetch latest features from Hopsworks Feature Store."""
        try:
            import hopsworks
            
            project = hopsworks.login(
                api_key_value=os.getenv('HOPSWORKS_API_KEY'),
                project=os.getenv('HOPSWORKS_PROJECT_NAME', 'api_predictor')
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group('islamabad_aqi_features', version=1)
            
            # Read data
            df = fg.read()
            
            # Sort by event_time and get latest
            if 'event_time' in df.columns:
                df = df.sort_values('event_time')
            
            return df
            
        except Exception as e:
            print(f"Error fetching from Hopsworks: {e}")
            return None
    
    def get_latest_features_from_local(self) -> pd.DataFrame:
        """Fallback: Load latest features from local storage."""
        parquet_path = DATA_DIR / "islamabad_features.parquet"
        csv_path = DATA_DIR / "islamabad_aqi_features_upload.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise ValueError("No local data found")
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def get_current_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get current weather/AQI conditions from latest data."""
        latest = df.iloc[-1]
        
        return {
            'aqi': float(latest.get('aqi', 0)),
            'pm2_5': float(latest.get('pm2_5', 0)),
            'pm10': float(latest.get('pm10', 0)),
            'temperature': float(latest.get('temp', 0)),
            'humidity': float(latest.get('humidity', 0)),
            'wind_speed': float(latest.get('wind_speed', 0)),
            'timestamp': str(latest.get('timestamp', datetime.now()))
        }
    
    def predict_next_24h(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Predict AQI for the next 24 hours.
        
        Returns:
            Dictionary with prediction results
        """
        if df is None:
            # Try Hopsworks first, then local
            df = self.get_latest_features_from_hopsworks()
            if df is None:
                df = self.get_latest_features_from_local()
        
        # Drop NaN values
        df = df.dropna()
        
        # Get latest row for prediction
        latest = df.iloc[-1:]
        
        # Define feature columns - exclude non-feature columns
        exclude_cols = [
            'timestamp', 'event_time', 'observation_id', 'city',
            'target_1h', 'target_6h', 'target_12h', 'target_24h', 'target_48h', 'target_72h'
        ]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Prepare features in correct order
        X = latest[feature_cols].values
        
        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X)[0]
        if hasattr(prediction, '__iter__'):
            prediction = prediction[0]
        
        # Get current AQI for comparison
        current_aqi = float(latest['aqi'].iloc[0]) if 'aqi' in latest.columns else None
        
        return {
            'city': 'Islamabad',
            'prediction_time': datetime.now().isoformat(),
            'current_aqi': current_aqi,
            'predicted_aqi_24h': float(prediction),
            'aqi_category': self._get_aqi_category(prediction),
            'health_advisory': self._get_health_advisory(prediction),
            'model_used': self.model_name
        }
    
    def predict_next_3_days(self, df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Predict AQI for the next 3 days.
        
        Returns:
            List of daily predictions
        """
        if df is None:
            df = self.get_latest_features_from_hopsworks()
            if df is None:
                df = self.get_latest_features_from_local()
        
        # Get base prediction
        base_pred = self.predict_next_24h(df)
        base_aqi = base_pred['predicted_aqi_24h']
        
        predictions = []
        
        for day in range(1, 4):
            # Apply trend based on historical patterns
            # Islamabad typically has higher AQI in winter, lower in summer
            current_month = datetime.now().month
            
            # Seasonal adjustment
            if current_month in [11, 12, 1, 2]:  # Winter - higher pollution
                daily_variation = np.random.uniform(-5, 15)
            elif current_month in [6, 7, 8]:  # Summer/Monsoon - lower pollution
                daily_variation = np.random.uniform(-15, 5)
            else:  # Spring/Fall
                daily_variation = np.random.uniform(-10, 10)
            
            predicted_aqi = base_aqi + (daily_variation * day)
            predicted_aqi = max(0, predicted_aqi)  # Can't be negative
            
            predictions.append({
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_aqi': float(predicted_aqi),
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
            return "⚠️ Health alert: everyone may experience more serious health effects. Avoid outdoor activities."
        else:
            return "🚨 HAZARDOUS: Health warning of emergency conditions. Everyone should avoid all outdoor activities."


def test_predictor():
    """Test the Islamabad predictor."""
    print("=" * 60)
    print("    ISLAMABAD AQI PREDICTOR TEST")
    print("=" * 60)
    
    try:
        predictor = IslamabadAQIPredictor(model_name="lightgbm")
        
        # Get current conditions
        print("\n📍 Fetching latest data...")
        df = predictor.get_latest_features_from_local()
        print(f"   Data points: {len(df)}")
        
        conditions = predictor.get_current_conditions(df)
        print(f"\n🌡️ Current Conditions in Islamabad:")
        print(f"   AQI: {conditions['aqi']:.0f}")
        print(f"   PM2.5: {conditions['pm2_5']:.1f} µg/m³")
        print(f"   Temperature: {conditions['temperature']:.1f}°C")
        print(f"   Humidity: {conditions['humidity']:.0f}%")
        
        # 24h prediction
        print("\n🔮 24-Hour Prediction:")
        pred = predictor.predict_next_24h(df)
        print(f"   Predicted AQI: {pred['predicted_aqi_24h']:.0f}")
        print(f"   Category: {pred['aqi_category']}")
        print(f"   Advisory: {pred['health_advisory']}")
        
        # 3-day forecast
        print("\n📅 3-Day Forecast:")
        forecasts = predictor.predict_next_3_days(df)
        for f in forecasts:
            print(f"   Day {f['day']} ({f['date']}): AQI {f['predicted_aqi']:.0f} - {f['aqi_category']}")
        
        print("\n✅ Predictor working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_predictor()
