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
        
        # Load the feature names this model was trained on
        try:
            meta = self.registry.get_model_metadata(self.model_name)
            self.expected_feature_names: List[str] = meta.get('feature_names', [])
        except Exception:
            self.expected_feature_names = []
        
        if self.expected_feature_names:
            print(f"Model expects {len(self.expected_feature_names)} features")
    
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
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
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
        
        # Better fallback values
        if 'visibility' not in features.columns:
            features['visibility'] = 10000.0
        if 'clouds' not in features.columns:
            features['clouds'] = 0.0
        if 'uvi' not in features.columns:
            features['uvi'] = 5.0
        if 'dew_point' not in features.columns:
            features['dew_point'] = features['temperature'] if 'temperature' in features.columns else 15.0
        if 'feels_like' not in features.columns:
            features['feels_like'] = features['temperature'] if 'temperature' in features.columns else 25.0
            
        # Handle column naming differences between DB and training data
        if 'temperature' in features.columns and 'temp' not in features.columns:
            features = features.rename(columns={'temperature': 'temp'})
                
        core_cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'feels_like', 'clouds', 'visibility', 'dew_point', 'uvi']
        for col in core_cols:
            if col not in features.columns:
                features[col] = 0.0
                
        # Fill missing values from DB (due to cron failing to fetch weather)
        features['temp'] = pd.to_numeric(features['temp'], errors='coerce').fillna(25.0)
        features['humidity'] = pd.to_numeric(features['humidity'], errors='coerce').fillna(50.0)
        features['pressure'] = pd.to_numeric(features['pressure'], errors='coerce').fillna(1013.0)
        features['wind_speed'] = pd.to_numeric(features['wind_speed'], errors='coerce').fillna(2.0)
        features['wind_deg'] = pd.to_numeric(features['wind_deg'], errors='coerce').fillna(0.0)
        
        for col in core_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
                
        # Always recompute features to get the full 187 set
        features, feature_cols, _ = compute_all_features(features)
            
        # We only want to predict for the latest row
        features = features.tail(1)
        
        # Fill NaN values with 0 — weather/lag columns may be null when
        # the OpenWeatherMap API is unavailable or no historical lag data exists.
        # The model can still produce a valid prediction from the pollution metrics.
        features = features.fillna(0)
        
        # ── Feature alignment ────────────────────────────────────────────────
        # If the model has saved feature_names, align the inference DataFrame
        # to match exactly: add missing columns as 0, drop extra ones.
        if self.expected_feature_names:
            missing_cols = [col for col in self.expected_feature_names if col not in features.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=features.index, columns=missing_cols)
                features = pd.concat([features, missing_df], axis=1)
            feature_cols = self.expected_feature_names
        # ────────────────────────────────────────────────────────────────────
        
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
            'health_advisory': self._get_health_advisory(prediction),
            # Weather & Pollutants
            'temperature': float(features['temp'].iloc[-1]) if 'temp' in features.columns else None,
            'humidity': float(features['humidity'].iloc[-1]) if 'humidity' in features.columns else None,
            'pm2_5': float(features['pm2_5'].iloc[-1]) if 'pm2_5' in features.columns else None,
            'pm10': float(features['pm10'].iloc[-1]) if 'pm10' in features.columns else None,
            'co': float(features['co'].iloc[-1]) if 'co' in features.columns else None,
            'o3': float(features['o3'].iloc[-1]) if 'o3' in features.columns else None,
            'no2': float(features['no2'].iloc[-1]) if 'no2' in features.columns else None,
            'so2': float(features['so2'].iloc[-1]) if 'so2' in features.columns else None,
            'wind_speed': float(features['wind_speed'].iloc[-1]) if 'wind_speed' in features.columns else None,
            'clouds': float(features['clouds'].iloc[-1]) if 'clouds' in features.columns else None,
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
    
    def log_prediction(self, city: str, predicted_aqi: float, target_date: str):
        """Save a prediction to Supabase predictions_log table."""
        try:
            if hasattr(self.feature_store, 'client'):
                client = self.feature_store.client
                record = {
                    "city": city,
                    "target_date": target_date,
                    "predicted_aqi": float(predicted_aqi),
                    "model_used": self.model_name
                }
                client.table('predictions_log').insert(record).execute()
        except Exception as e:
            print(f"Failed to log prediction: {e}")
            
    def explain_prediction(self, features: pd.DataFrame = None, city: str = "Islamabad") -> Dict[str, Any]:
        """Explain the current prediction using feature importances."""
        if features is None:
            features = self.get_latest_features(city)
            
        if 'visibility' not in features.columns:
            features['visibility'] = 10000.0
        if 'clouds' not in features.columns:
            features['clouds'] = 0.0
        if 'uvi' not in features.columns:
            features['uvi'] = 5.0
        if 'dew_point' not in features.columns:
            features['dew_point'] = features['temperature'] if 'temperature' in features.columns else 15.0
        if 'feels_like' not in features.columns:
            features['feels_like'] = features['temperature'] if 'temperature' in features.columns else 25.0
            
        if 'temperature' in features.columns and 'temp' not in features.columns:
            features = features.rename(columns={'temperature': 'temp'})
                
        core_cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'feels_like', 'clouds', 'visibility', 'dew_point', 'uvi']
        for col in core_cols:
            if col not in features.columns:
                features[col] = 0.0
                
        features['temp'] = pd.to_numeric(features['temp'], errors='coerce').fillna(25.0)
        features['humidity'] = pd.to_numeric(features['humidity'], errors='coerce').fillna(50.0)
        features['pressure'] = pd.to_numeric(features['pressure'], errors='coerce').fillna(1013.0)
        features['wind_speed'] = pd.to_numeric(features['wind_speed'], errors='coerce').fillna(2.0)
        features['wind_deg'] = pd.to_numeric(features['wind_deg'], errors='coerce').fillna(0.0)
        
        for col in core_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
                
        features, feature_cols, _ = compute_all_features(features)
            
        # We only want to explain the latest row
        features = features.tail(1)
            
        features = features.fillna(0)
        
        if self.expected_feature_names:
            missing_cols = [col for col in self.expected_feature_names if col not in features.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=features.index, columns=missing_cols)
                features = pd.concat([features, missing_df], axis=1)
            feature_cols = self.expected_feature_names
            
        # Extract base model from pipeline if necessary
        base_model = self.model.named_steps['model'] if hasattr(self.model, 'named_steps') else self.model

        importances = []
        if hasattr(base_model, 'feature_importances_'):
            fi = base_model.feature_importances_
        elif hasattr(base_model, 'coef_'):
            fi = np.abs(base_model.coef_)
        else:
            fi = []
            
        if len(fi) > 0:
            for i, col in enumerate(feature_cols):
                if i < len(fi):
                    val = features[col].iloc[-1]
                    # Filter out purely generated/unhelpful features like time sines for cleaner UI
                    if '_sin' not in col and '_cos' not in col:
                        importances.append({
                            "feature": col,
                            "importance": float(fi[i]),
                            "current_value": float(val) if pd.notnull(val) else 0.0
                        })
                    
            # Normalize importances so the UI bar renders nicely (0 to 1 range)
            max_imp = max((x["importance"] for x in importances), default=1.0)
            if max_imp > 0:
                for x in importances:
                    x["importance"] = x["importance"] / max_imp
                    
            importances.sort(key=lambda x: x["importance"], reverse=True)
            
        return {
            "city": city,
            "top_drivers": importances[:8]  # Return top 8 most important features
        }
        
    def get_accuracy_metrics(self, city: str = "Islamabad", days: int = 7) -> Dict[str, Any]:
        """Get historical predictions vs actual AQI for the last N days."""
        try:
            if not hasattr(self.feature_store, 'client'):
                return {"city": city, "error": "Supabase client not available", "history": []}
                
            client = self.feature_store.client
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Fetch predictions
            preds_res = client.table('predictions_log') \
                .select('*') \
                .eq('city', city) \
                .eq('model_used', self.model_name) \
                .gte('target_date', start_date) \
                .order('target_date') \
                .execute()
                
            # Fetch actuals
            start_datetime = (datetime.now() - timedelta(days=days)).isoformat()
            actuals_res = client.table('aqi_features') \
                .select('timestamp, aqi') \
                .eq('city', city) \
                .gte('timestamp', start_datetime) \
                .execute()
                
            predictions = preds_res.data
            actuals = actuals_res.data
            
            # Group predictions by date (take daily average)
            daily_preds = {}
            for row in predictions:
                if row.get('target_date') and row.get('predicted_aqi') is not None:
                    date = row['target_date'].split('T')[0]
                    if date not in daily_preds:
                        daily_preds[date] = []
                    daily_preds[date].append(row['predicted_aqi'])
                    
            for date in daily_preds:
                daily_preds[date] = sum(daily_preds[date]) / len(daily_preds[date])

            # Group actuals by date (take daily average)
            daily_actuals = {}
            for row in actuals:
                if row.get('timestamp') and row.get('aqi') is not None:
                    date = row['timestamp'].split('T')[0]
                    if date not in daily_actuals:
                        daily_actuals[date] = []
                    daily_actuals[date].append(row['aqi'])
                    
            for date in daily_actuals:
                daily_actuals[date] = sum(daily_actuals[date]) / len(daily_actuals[date])
                
            # Combine
            history = []
            errors = []
            for date, pred_val in daily_preds.items():
                actual = daily_actuals.get(date)
                
                history.append({
                    "date": date,
                    "predicted": pred_val,
                    "actual": actual
                })
                
                if actual is not None and actual > 0:
                    errors.append(abs(pred_val - actual) / actual) # MAPE
                    
            mape = sum(errors) / len(errors) if errors else 0
            accuracy = max(0, 100 - (mape * 100)) if errors else None
            
            return {
                "city": city,
                "accuracy_score": accuracy,
                "history": history
            }
        except Exception as e:
            print(f"Error getting accuracy metrics: {e}")
            return {"city": city, "error": str(e), "history": []}
    
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
