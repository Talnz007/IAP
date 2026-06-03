"""
FastAPI backend for AQI Predictor.
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(
    title="AQI Predictor API",
    description="API for Air Quality Index predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PredictionResponse(BaseModel):
    city: str
    prediction_time: str
    current_aqi: Optional[float]
    predicted_aqi_24h: float
    model_used: str
    aqi_category: str
    health_advisory: str
    # Secondary Pollutants & Weather
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    wind_speed: Optional[float] = None
    clouds: Optional[float] = None


class ForecastItem(BaseModel):
    day: int
    date: str
    predicted_aqi: float
    aqi_category: str
    health_advisory: str


class ForecastResponse(BaseModel):
    city: str
    forecasts: List[ForecastItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# Global predictors cache
predictors = {}


def get_predictor(model_name: Optional[str] = None):
    """Get or create the predictor instance."""
    if model_name is None:
        try:
            from src.training.model_registry import get_model_registry
            registry = get_model_registry()
            models = registry.list_models()
            if models:
                model_name = list(models.keys())[0]
            else:
                model_name = "random_forest"
        except Exception:
            model_name = "random_forest"
            
    if model_name not in predictors:
        try:
            from src.inference.predict import AQIPredictor
            predictors[model_name] = AQIPredictor(model_name=model_name)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' not available: {str(e)}"
            )
    return predictors[model_name]


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AQI Predictor API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pred = get_predictor()
        model_loaded = pred.model is not None
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/predict/{city}", response_model=PredictionResponse)
async def predict(city: str, model_name: Optional[str] = None):
    """
    Get AQI prediction for a city.
    
    Args:
        city: City name (e.g., Karachi, Lahore)
        model_name: Optional specific model to use
        
    Returns:
        Prediction response with current and predicted AQI
    """
    try:
        pred = get_predictor(model_name)
        result = pred.predict(city=city)
        
        return PredictionResponse(
            city=result['city'],
            prediction_time=result['prediction_time'],
            current_aqi=result['current_aqi'],
            predicted_aqi_24h=result['predicted_aqi_24h'],
            model_used=result['model_used'],
            aqi_category=result['aqi_category'],
            health_advisory=result['health_advisory'],
            temperature=result.get('temperature'),
            humidity=result.get('humidity'),
            pm2_5=result.get('pm2_5'),
            pm10=result.get('pm10'),
            co=result.get('co'),
            o3=result.get('o3'),
            no2=result.get('no2'),
            so2=result.get('so2'),
            wind_speed=result.get('wind_speed'),
            clouds=result.get('clouds')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{city}", response_model=ForecastResponse)
async def forecast(city: str):
    """
    Get 3-day AQI forecast for a city.
    
    Args:
        city: City name
        
    Returns:
        3-day forecast response
    """
    try:
        pred = get_predictor()
        forecasts = pred.predict_next_3_days(city=city)
        
        forecast_items = [
            ForecastItem(
                day=f['day'],
                date=f['date'],
                predicted_aqi=f['predicted_aqi'],
                aqi_category=f['aqi_category'],
                health_advisory=f['health_advisory']
            )
            for f in forecasts
        ]
        
        return ForecastResponse(
            city=city,
            forecasts=forecast_items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{city}")
async def explain(city: str, model_name: Optional[str] = None):
    """Get feature importances for the current prediction."""
    try:
        pred = get_predictor(model_name)
        return pred.explain_prediction(city=city)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accuracy/{city}")
async def accuracy(city: str, model_name: Optional[str] = None):
    """Get historical prediction accuracy metrics."""
    try:
        pred = get_predictor(model_name)
        return pred.get_accuracy_metrics(city=city, days=7)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities", response_model=List[str])
async def get_cities():
    """Get list of supported cities."""
    return ["Karachi", "Lahore", "Islamabad", "Faisalabad", "Rawalpindi"]


@app.get("/models")
async def list_available_models():
    """List available models and their metadata/metrics."""
    try:
        from src.training.model_registry import get_model_registry
        registry = get_model_registry()
        models = registry.list_models()
        
        allowed_models = {"random_forest", "xgboost", "lightgbm"}
        
        result = {}
        for m_name in models.keys():
            if m_name not in allowed_models:
                continue
                
            try:
                meta = registry.get_model_metadata(m_name)
                metrics = meta.get("metrics", {})
                result[m_name] = {
                    "model_name": m_name,
                    "version": meta.get("version"),
                    "created_at": meta.get("created_at"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                    "mae": metrics.get("mae"),
                    "mape": metrics.get("mape")
                }
            except Exception as e:
                result[m_name] = {
                    "model_name": m_name,
                    "error": str(e)
                }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
