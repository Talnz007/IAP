"""
Tests for inference module.
"""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predict import AQIPredictor


class TestAQICategory:
    """Tests for AQI category classification."""
    
    def test_good_aqi(self):
        """Test good AQI category."""
        category = AQIPredictor._get_aqi_category(30)
        assert category == "Good"
    
    def test_moderate_aqi(self):
        """Test moderate AQI category."""
        category = AQIPredictor._get_aqi_category(75)
        assert category == "Moderate"
    
    def test_unhealthy_sensitive_aqi(self):
        """Test unhealthy for sensitive groups category."""
        category = AQIPredictor._get_aqi_category(125)
        assert category == "Unhealthy for Sensitive Groups"
    
    def test_unhealthy_aqi(self):
        """Test unhealthy AQI category."""
        category = AQIPredictor._get_aqi_category(175)
        assert category == "Unhealthy"
    
    def test_very_unhealthy_aqi(self):
        """Test very unhealthy AQI category."""
        category = AQIPredictor._get_aqi_category(250)
        assert category == "Very Unhealthy"
    
    def test_hazardous_aqi(self):
        """Test hazardous AQI category."""
        category = AQIPredictor._get_aqi_category(350)
        assert category == "Hazardous"


class TestHealthAdvisory:
    """Tests for health advisory messages."""
    
    def test_good_advisory(self):
        """Test advisory for good AQI."""
        advisory = AQIPredictor._get_health_advisory(30)
        assert "satisfactory" in advisory.lower() or "enjoy" in advisory.lower()
    
    def test_hazardous_advisory(self):
        """Test advisory for hazardous AQI."""
        advisory = AQIPredictor._get_health_advisory(350)
        assert "emergency" in advisory.lower() or "avoid" in advisory.lower()


class TestAQICategoryBoundaries:
    """Tests for AQI category boundaries."""
    
    @pytest.mark.parametrize("aqi,expected", [
        (0, "Good"),
        (50, "Good"),
        (51, "Moderate"),
        (100, "Moderate"),
        (101, "Unhealthy for Sensitive Groups"),
        (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"),
        (200, "Unhealthy"),
        (201, "Very Unhealthy"),
        (300, "Very Unhealthy"),
        (301, "Hazardous"),
        (500, "Hazardous"),
    ])
    def test_boundary_values(self, aqi, expected):
        """Test AQI category at boundary values."""
        category = AQIPredictor._get_aqi_category(aqi)
        assert category == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
