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
sys.path.append(str(Path(__file__).parent.parent.parent))

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


# Global predictor instance
predictor = None


def get_predictor():
    """Get or create the predictor instance."""
    global predictor
    if predictor is None:
        try:
            from src.inference.predict import AQIPredictor
            predictor = AQIPredictor()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    return predictor


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
async def predict(city: str):
    """
    Get AQI prediction for a city.
    
    Args:
        city: City name (e.g., Karachi, Lahore)
        
    Returns:
        Prediction response with current and predicted AQI
    """
    try:
        pred = get_predictor()
        result = pred.predict(city=city)
        
        return PredictionResponse(
            city=result['city'],
            prediction_time=result['prediction_time'],
            current_aqi=result['current_aqi'],
            predicted_aqi_24h=result['predicted_aqi_24h'],
            model_used=result['model_used'],
            aqi_category=result['aqi_category'],
            health_advisory=result['health_advisory']
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


@app.get("/cities", response_model=List[str])
async def get_cities():
    """Get list of supported cities."""
    return ["Karachi", "Lahore", "Islamabad", "Faisalabad", "Rawalpindi"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
