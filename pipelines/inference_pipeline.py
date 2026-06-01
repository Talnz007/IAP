"""
Inference Pipeline - Scheduled script for generating predictions.
"""
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predict import AQIPredictor
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger("inference_pipeline")


def run_inference_pipeline(cities: list = None, output_file: str = None):
    """
    Run the inference pipeline to generate predictions.
    
    Args:
        cities: List of cities to predict for
        output_file: Optional path to save predictions
    """
    if cities is None:
        cities = ["Karachi", "Lahore", "Islamabad"]
    
    config = get_config()
    
    logger.info("="*50)
    logger.info("Starting Inference Pipeline")
    logger.info(f"Cities: {cities}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*50)
    
    try:
        predictor = AQIPredictor()
        all_predictions = []
        
        for city in cities:
            try:
                logger.info(f"Generating predictions for {city}...")
                
                # Get prediction
                prediction = predictor.predict(city=city)
                forecasts = predictor.predict_next_3_days(city=city)
                
                city_result = {
                    'city': city,
                    'current_prediction': prediction,
                    'forecasts': forecasts,
                    'generated_at': datetime.now().isoformat()
                }
                
                all_predictions.append(city_result)
                
                # Check for alerts
                max_aqi = max(f['predicted_aqi'] for f in forecasts)
                if max_aqi > 200:
                    logger.warning(f"  ⚠️ ALERT: High AQI predicted for {city} (max: {max_aqi:.0f})")
                elif max_aqi > 150:
                    logger.warning(f"  ⚠️ WARNING: Elevated AQI predicted for {city} (max: {max_aqi:.0f})")
                else:
                    logger.info(f"  ✓ Predictions generated for {city} (max AQI: {max_aqi:.0f})")
                
            except Exception as e:
                logger.error(f"  ✗ Error generating predictions for {city}: {e}")
        
        # Save predictions
        if output_file or all_predictions:
            if output_file is None:
                output_dir = config.data_dir / "predictions"
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            
            logger.info(f"Predictions saved to: {output_file}")
        
        logger.info("Inference pipeline completed")
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        raise
    
    logger.info("="*50)
    return all_predictions


if __name__ == "__main__":
    run_inference_pipeline()
