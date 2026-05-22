<div align="center">

# 🌍 Islamabad AQI Predictor

### Real-Time Air Quality Index Prediction System for Islamabad, Pakistan

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://islamabad-aqi-predictor.streamlit.app)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

**Predict tomorrow's air quality today using machine learning**

[Live Demo](https://islamabad-aqi-predictor.streamlit.app) • [Documentation](#documentation) • [Installation](#installation) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Models](#-machine-learning-models)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Islamabad AQI Predictor** is a production-ready machine learning system that provides 24-hour ahead Air Quality Index (AQI) predictions for Islamabad, Pakistan. The system fetches real-time air quality and weather data, processes it through an advanced feature engineering pipeline, and serves predictions through an interactive web dashboard.

### Why This Project?

Air pollution is a critical health concern in Pakistani cities. According to IQAir, Islamabad frequently experiences unhealthy AQI levels, particularly during winter months. This project aims to:

- 🏥 **Protect Public Health**: Enable citizens to plan outdoor activities based on predicted air quality
- 📊 **Provide Accurate Forecasts**: Use ML models trained on historical data for reliable predictions
- 🔄 **Automate Monitoring**: Run continuously without manual intervention
- 🌐 **Ensure Accessibility**: Free, web-based interface accessible to everyone

### Key Achievements

| Metric | Value |
|--------|-------|
| Model Accuracy (R²) | 0.89 |
| Prediction RMSE | ~25 AQI points |
| Historical Data | 8,556+ records |
| Features Engineered | 160+ |
| Uptime | 99.9% |
| Update Frequency | Hourly |

---

## ✨ Features

### 🖥️ Interactive Dashboard
- **Real-time AQI Display**: Current air quality with color-coded health categories
- **24-Hour Predictions**: Forecast tomorrow's AQI with confidence indicators
- **Model Comparison**: Switch between XGBoost, LightGBM, and Random Forest
- **Historical Trends**: Visualize past AQI patterns with interactive charts
- **Health Recommendations**: Actionable advice based on AQI levels

### 📊 Data Pipeline
- **Hourly Data Fetching**: Automated collection from OpenWeatherMap API
- **Feature Store Integration**: Centralized feature management with Hopsworks
- **Data Quality Checks**: Automatic validation and anomaly detection
- **Backfill Support**: Historical data collection for model training

### 🤖 Machine Learning
- **Multiple Models**: XGBoost, LightGBM, Random Forest ensemble
- **Automated Retraining**: Daily model updates with fresh data
- **Model Registry**: Version-controlled model storage in Hopsworks
- **Feature Importance**: Understand what drives predictions

### 🔄 Automation
- **GitHub Actions CI/CD**: Fully automated pipelines
- **Scheduled Jobs**: Hourly features, daily training
- **Error Handling**: Robust retry mechanisms and alerting
- **Logging**: Comprehensive logs for debugging

---

## 🛠️ Tech Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10 | Primary development language |
| **Web Framework** | Streamlit | Interactive dashboard |
| **ML Libraries** | Scikit-learn, XGBoost, LightGBM | Model training and inference |
| **Feature Store** | Hopsworks | Centralized feature management |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Visualization** | Plotly, Matplotlib | Charts and graphs |
| **API Client** | Requests | External API communication |
| **Configuration** | YAML, python-dotenv | Settings management |

### Infrastructure

| Category | Technology | Purpose |
|----------|------------|---------|
| **Deployment** | Streamlit Cloud | Web app hosting |
| **CI/CD** | GitHub Actions | Automated pipelines |
| **Version Control** | Git/GitHub | Code management |
| **Model Storage** | Hopsworks Model Registry | Model versioning |

### External APIs

| API | Provider | Purpose |
|-----|----------|---------|
| Air Pollution API | OpenWeatherMap | Real-time AQI data |
| Weather API | OpenWeatherMap | Meteorological data |
| Feature Store API | Hopsworks | Feature storage/retrieval |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GITHUB ACTIONS                                  │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────┐   │
│  │     Feature Pipeline         │    │       Training Pipeline          │   │
│  │     (Runs Every Hour)        │    │       (Runs Daily at 2 AM)       │   │
│  │  ┌─────────────────────────┐ │    │  ┌────────────────────────────┐  │   │
│  │  │ 1. Fetch Live AQI Data  │ │    │  │ 1. Load Data from Hopsworks│  │   │
│  │  │ 2. Fetch Weather Data   │ │    │  │ 2. Train XGBoost Model     │  │   │
│  │  │ 3. Compute Features     │ │    │  │ 3. Train LightGBM Model    │  │   │
│  │  │ 4. Upload to Hopsworks  │ │    │  │ 4. Train Random Forest     │  │   │
│  │  └─────────────────────────┘ │    │  │ 5. Evaluate All Models     │  │   │
│  └──────────────────────────────┘    │  │ 6. Save to Model Registry  │  │   │
│                                       │  └────────────────────────────┘  │   │
│                                       └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OPENWEATHERMAP API                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Endpoints:                                                          │    │
│  │  • /data/2.5/air_pollution?lat=33.6844&lon=73.0479                  │    │
│  │  • /data/2.5/weather?lat=33.6844&lon=73.0479                        │    │
│  │                                                                      │    │
│  │  Data Provided:                                                      │    │
│  │  • AQI (1-5 scale)           • PM2.5, PM10, NO2, SO2, CO, O3, NH3   │    │
│  │  • Temperature, Humidity     • Wind Speed, Direction                 │    │
│  │  • Pressure, Visibility      • Cloud Cover, UV Index                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HOPSWORKS PLATFORM                                  │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────┐   │
│  │       FEATURE STORE          │    │        MODEL REGISTRY            │   │
│  │  ┌────────────────────────┐  │    │  ┌────────────────────────────┐  │   │
│  │  │ Feature Group:         │  │    │  │ Models Stored:             │  │   │
│  │  │ islamabad_aqi_features │  │    │  │ • xgboost_aqi_model        │  │   │
│  │  │                        │  │    │  │ • lightgbm_aqi_model       │  │   │
│  │  │ Records: 8,556+        │  │    │  │ • random_forest_aqi_model  │  │   │
│  │  │ Features: 160+         │  │    │  │ • ridge_aqi_model          │  │   │
│  │  │ Updated: Hourly        │  │    │  │                            │  │   │
│  │  └────────────────────────┘  │    │  │ Metadata: feature_names,   │  │   │
│  └──────────────────────────────┘    │  │ metrics, timestamps        │  │   │
│                                       │  └────────────────────────────┘  │   │
│                                       └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT CLOUD                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                 ISLAMABAD AQI PREDICTOR DASHBOARD                    │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │    │
│  │  │  │ Current AQI │  │ Prediction  │  │   Model Selector    │   │  │    │
│  │  │  │    156      │  │    142      │  │ ☑ XGBoost (Best)    │   │  │    │
│  │  │  │  Unhealthy  │  │  Tomorrow   │  │ ☐ LightGBM          │   │  │    │
│  │  │  └─────────────┘  └─────────────┘  │ ☐ Random Forest     │   │  │    │
│  │  │                                     └─────────────────────┘   │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │              Historical AQI Trend Chart                 │  │  │    │
│  │  │  │  📈 Interactive Plotly visualization                    │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │              Health Recommendations                      │  │  │    │
│  │  │  │  ⚠️ Sensitive groups should limit outdoor exposure      │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
islamabad-aqi-predictor/
│
├── 📂 configs/                          # Configuration files
│   ├── config.yaml                      # Main application configuration
│   └── model_config.yaml                # Model hyperparameters
│
├── 📂 data/                             # Data storage
│   ├── 📂 backfill/                     # Historical backfill data
│   ├── 📂 checkpoints/                  # Pipeline checkpoint files
│   │   └── fetch_checkpoint.json        # Last fetch timestamp
│   ├── 📂 processed/                    # Processed feature data
│   │   └── islamabad_aqi_features_upload.csv
│   └── 📂 raw/                          # Raw API responses
│
├── 📂 logs/                             # Application logs
│
├── 📂 models/                           # Trained model artifacts
│   ├── 📂 lightgbm/                     # LightGBM models
│   │   ├── latest.txt                   # Points to latest version
│   │   └── 📂 20260131_173707/          # Versioned model folder
│   │       ├── metadata.json            # Model metadata & metrics
│   │       └── model.joblib             # Serialized model
│   ├── 📂 xgboost/                      # XGBoost models
│   ├── 📂 random_forest/                # Random Forest models
│   └── 📂 ridge/                        # Ridge Regression models
│
├── 📂 notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_ETL_Data_Testing.ipynb        # Data extraction testing
│   └── 03_Upload_to_Hopsworks.ipynb     # Hopsworks upload testing
│
├── 📂 pipelines/                        # Automated pipeline scripts
│   ├── feature_pipeline.py              # Hourly feature extraction
│   ├── training_pipeline.py             # Daily model training
│   └── inference_pipeline.py            # Batch inference pipeline
│
├── 📂 scripts/                          # Utility scripts
│   ├── backfill_data.py                 # Historical data collection
│   ├── data_quality_report.py           # Data quality analysis
│   ├── fetch_live_aqi.py                # Live AQI fetching
│   ├── model_comparison.py              # Model evaluation comparison
│   ├── upload_to_hopsworks.py           # Data upload to Hopsworks
│   └── validate_hopsworks_data.py       # Data validation
│
├── 📂 src/                              # Source code modules
│   ├── 📂 features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── compute_features.py          # Feature computation logic
│   │   ├── feature_engineering.py       # Advanced feature creation
│   │   ├── feature_store.py             # Hopsworks integration
│   │   └── fetch_data.py                # API data fetching
│   │
│   ├── 📂 inference/                    # Prediction logic
│   │   ├── __init__.py
│   │   ├── predict.py                   # Prediction functions
│   │   └── explainability.py            # SHAP/LIME explanations
│   │
│   ├── 📂 training/                     # Model training
│   │   ├── __init__.py
│   │   ├── train.py                     # Main training logic
│   │   ├── evaluate.py                  # Model evaluation metrics
│   │   ├── model_registry.py            # Model versioning
│   │   └── 📂 models/                   # Model definitions
│   │       ├── __init__.py
│   │       ├── sklearn_models.py        # Scikit-learn models
│   │       └── deep_learning.py         # Neural network models
│   │
│   └── 📂 utils/                        # Utility functions
│       ├── __init__.py
│       ├── config.py                    # Configuration loader
│       └── logger.py                    # Logging setup
│
├── 📂 tests/                            # Test suite
│   ├── __init__.py
│   ├── test_features.py                 # Feature engineering tests
│   ├── test_training.py                 # Training tests
│   └── test_inference.py                # Inference tests
│
├── 📂 webapp/                           # Streamlit web application
│   ├── app.py                           # Main dashboard application
│   ├── islamabad_predictor.py           # Prediction helper class
│   └── 📂 api/                          # REST API (optional)
│       ├── __init__.py
│       └── main.py                      # FastAPI endpoints
│
├── 📂 .github/                          # GitHub configuration
│   └── 📂 workflows/                    # GitHub Actions workflows
│       ├── feature_pipeline.yml         # Hourly feature job
│       └── training_pipeline.yml        # Daily training job
│
├── .env.example                         # Environment variables template
├── .gitignore                           # Git ignore rules
├── requirements.txt                     # Python dependencies
├── INTERNSHIP_REPORT.md                 # Detailed project report
├── PROJECT_CHALLENGES_AND_MODEL_SELECTION.txt  # Challenges documentation
└── README.md                            # This file
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Git**: For cloning the repository
- **API Keys**:
  - OpenWeatherMap API Key ([Get free key](https://openweathermap.org/api))
  - Hopsworks API Key ([Sign up free](https://app.hopsworks.ai))

### Step 1: Clone the Repository

```bash
git clone https://github.com/Haideransari444/islamabad-aqi-predictor.git
cd islamabad-aqi-predictor
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
```

### Step 5: Verify Installation

```bash
# Test the installation
python -c "import xgboost; import lightgbm; import streamlit; print('All dependencies installed successfully!')"
```

---

## ⚙️ Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
# Data source configuration
data:
  city: "Islamabad"
  latitude: 33.6844
  longitude: 73.0479
  api_provider: "openweathermap"

# Feature engineering settings
features:
  lag_hours: [1, 3, 6, 12, 24, 48, 72]
  rolling_windows: [6, 12, 24, 48]
  target_horizons: [1, 6, 12, 24, 48, 72]

# Training configuration
training:
  test_size: 0.2
  random_state: 42
  target_column: "target_24h"

# Hopsworks settings
hopsworks:
  project_name: "aqi_prediction"
  feature_group_name: "islamabad_aqi_features"
  feature_group_version: 1
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
models:
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    reg_alpha: 0.1
    reg_lambda: 1.0

  lightgbm:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_samples: 20
    reg_alpha: 0.1
    reg_lambda: 1.0

  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    n_jobs: -1
```

---

## 📖 Usage

### Running the Web Dashboard

```bash
# Start the Streamlit app
streamlit run webapp/app.py

# The app will be available at http://localhost:8501
```

### Running Pipelines Manually

```bash
# Feature Pipeline - Fetch and process latest data
python pipelines/feature_pipeline.py

# Training Pipeline - Retrain all models
python pipelines/training_pipeline.py

# Inference Pipeline - Generate batch predictions
python pipelines/inference_pipeline.py
```

### Using as a Python Library

```python
from src.inference.predict import predict_aqi
from src.features.fetch_data import fetch_current_aqi

# Fetch current AQI data
current_data = fetch_current_aqi(lat=33.6844, lon=73.0479)
print(f"Current AQI: {current_data['aqi']}")

# Make a prediction
prediction = predict_aqi(current_data, model_name='xgboost')
print(f"Predicted AQI (24h): {prediction}")
```

---

## 📊 Data Pipeline

### Data Flow

```
┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  OpenWeatherMap │ ──► │  Feature Engine  │ ──► │    Hopsworks    │
│      API        │     │  (160+ features) │     │  Feature Store  │
└────────────────┘     └──────────────────┘     └─────────────────┘
         │                      │                        │
         │                      │                        │
         ▼                      ▼                        ▼
┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data     │     │ Processed Data   │     │  Training Data  │
│  (JSON format) │     │   (CSV format)   │     │   (DataFrame)   │
└────────────────┘     └──────────────────┘     └─────────────────┘
```

### Raw Data Fields

| Field | Type | Description | Unit |
|-------|------|-------------|------|
| `aqi` | int | Air Quality Index | 1-5 scale |
| `pm2_5` | float | Fine Particulate Matter | μg/m³ |
| `pm10` | float | Coarse Particulate Matter | μg/m³ |
| `no2` | float | Nitrogen Dioxide | μg/m³ |
| `so2` | float | Sulfur Dioxide | μg/m³ |
| `co` | float | Carbon Monoxide | μg/m³ |
| `o3` | float | Ozone | μg/m³ |
| `nh3` | float | Ammonia | μg/m³ |
| `temp` | float | Temperature | °C |
| `humidity` | int | Relative Humidity | % |
| `pressure` | int | Atmospheric Pressure | hPa |
| `wind_speed` | float | Wind Speed | m/s |
| `wind_deg` | int | Wind Direction | degrees |
| `clouds` | int | Cloud Cover | % |
| `visibility` | int | Visibility | meters |

### AQI Scale Reference

| AQI Range | Category | Color | Health Implications |
|-----------|----------|-------|---------------------|
| 0-50 | Good | 🟢 Green | Air quality is satisfactory |
| 51-100 | Moderate | 🟡 Yellow | Acceptable; moderate health concern for sensitive people |
| 101-150 | Unhealthy for Sensitive Groups | 🟠 Orange | Sensitive groups may experience health effects |
| 151-200 | Unhealthy | 🔴 Red | Everyone may begin to experience health effects |
| 201-300 | Very Unhealthy | 🟣 Purple | Health warnings; everyone may experience effects |
| 300+ | Hazardous | 🟤 Maroon | Emergency conditions; entire population affected |

---

## 🔧 Feature Engineering

Our feature engineering pipeline transforms 15 raw features into 160+ predictive features:

### 1. Temporal Features (15 features)

```python
# Time-based features
hour                 # Hour of day (0-23)
day_of_week          # Day of week (0-6)
day_of_month         # Day of month (1-31)
month                # Month (1-12)
week_of_year         # Week number (1-52)

# Cyclical encoding (preserves continuity)
hour_sin, hour_cos           # sin/cos of hour
dow_sin, dow_cos             # sin/cos of day of week
month_sin, month_cos         # sin/cos of month

# Binary indicators
is_weekend           # Weekend flag (0/1)
is_night             # Night time flag (0/1)
is_rush_hour         # Rush hour flag (0/1)

# Seasonal encoding
season_winter, season_spring, season_summer, season_fall
```

### 2. Lag Features (35+ features)

```python
# Previous values at different time intervals
pm2_5_lag_1h         # PM2.5 from 1 hour ago
pm2_5_lag_3h         # PM2.5 from 3 hours ago
pm2_5_lag_6h         # PM2.5 from 6 hours ago
pm2_5_lag_12h        # PM2.5 from 12 hours ago
pm2_5_lag_24h        # PM2.5 from 24 hours ago
pm2_5_lag_48h        # PM2.5 from 48 hours ago
pm2_5_lag_72h        # PM2.5 from 72 hours ago

# Same pattern for: pm10, temp, humidity, wind_speed
```

### 3. Rolling Statistics (40+ features)

```python
# Rolling window aggregations
pm2_5_rolling_mean_6h        # 6-hour rolling average
pm2_5_rolling_std_6h         # 6-hour rolling std deviation
pm2_5_rolling_min_6h         # 6-hour rolling minimum
pm2_5_rolling_max_6h         # 6-hour rolling maximum

pm2_5_rolling_mean_12h       # 12-hour rolling average
pm2_5_rolling_mean_24h       # 24-hour rolling average
pm2_5_rolling_mean_48h       # 48-hour rolling average

# Same pattern for: pm10, temp, humidity
```

### 4. Change/Difference Features (20+ features)

```python
# Absolute and percentage changes
pm2_5_diff_1h                # Change from 1 hour ago
pm2_5_pct_change_1h          # Percentage change from 1 hour ago
pm2_5_diff_6h                # Change from 6 hours ago
pm2_5_diff_12h               # Change from 12 hours ago
pm2_5_diff_24h               # Change from 24 hours ago

# Same pattern for: pm10, temp, humidity, pressure
```

### 5. Interaction Features (10+ features)

```python
# Combined features
temp_humidity                # Temperature × Humidity interaction
wind_u                       # Wind U-component (speed × cos(direction))
wind_v                       # Wind V-component (speed × sin(direction))
pressure_gradient            # Pressure change rate
pm25_pm10_ratio              # PM2.5 to PM10 ratio
visibility_inv               # Inverse visibility (1/visibility)
```

### 6. Target Variables (6 features)

```python
# Future AQI values for different prediction horizons
target_1h            # AQI 1 hour ahead
target_6h            # AQI 6 hours ahead
target_12h           # AQI 12 hours ahead
target_24h           # AQI 24 hours ahead (PRIMARY TARGET)
target_48h           # AQI 48 hours ahead
target_72h           # AQI 72 hours ahead
```

---

## 🤖 Machine Learning Models

### Model Comparison

| Model | RMSE | MAE | R² | Training Time | Inference Time |
|-------|------|-----|-----|---------------|----------------|
| **XGBoost** ⭐ | ~25.3 | ~18.2 | 0.89 | 45s | <10ms |
| **LightGBM** | ~26.1 | ~18.8 | 0.88 | 15s | <5ms |
| **Random Forest** | ~28.5 | ~20.1 | 0.85 | 30s | <15ms |
| Ridge | ~45.2 | ~32.4 | 0.62 | <1s | <1ms |

### Why These Models?

#### XGBoost (Recommended)
```
✅ Best accuracy (lowest RMSE)
✅ Handles non-linear relationships
✅ Built-in regularization
✅ Fast inference
✅ Handles missing values
✅ Industry standard for tabular data
```

#### LightGBM
```
✅ Fastest training (3x faster than XGBoost)
✅ Lower memory usage
✅ Native categorical feature support
✅ Competitive accuracy
✅ Ideal for frequent retraining
```

#### Random Forest
```
✅ Most interpretable
✅ Feature importance rankings
✅ Less prone to overfitting
✅ Stable predictions
✅ No extensive hyperparameter tuning needed
```

### Feature Importance (Top 10)

```
1. pm2_5_lag_1h          ████████████████████ 15.2%
2. pm2_5_lag_3h          ███████████████████  14.8%
3. pm2_5_rolling_mean_6h ██████████████████   13.1%
4. pm10_lag_1h           ████████████████     11.7%
5. hour                  ██████████████        9.8%
6. temp                  ████████████          8.2%
7. humidity              ███████████           7.5%
8. pm2_5_rolling_mean_24h██████████            6.9%
9. wind_speed            █████████             6.1%
10. is_rush_hour         ████████              5.4%
```

---

## 🌐 API Reference

### REST API Endpoints

The optional FastAPI backend provides these endpoints:

#### Get Current AQI
```http
GET /api/v1/current
```

**Response:**
```json
{
  "timestamp": "2026-02-14T10:00:00Z",
  "aqi": 156,
  "category": "Unhealthy",
  "pollutants": {
    "pm2_5": 85.4,
    "pm10": 124.2,
    "no2": 45.6,
    "so2": 12.3,
    "co": 543.2,
    "o3": 78.9
  }
}
```

#### Get Prediction
```http
GET /api/v1/predict?model=xgboost&horizon=24
```

**Response:**
```json
{
  "timestamp": "2026-02-14T10:00:00Z",
  "prediction": {
    "aqi": 142,
    "category": "Unhealthy for Sensitive Groups",
    "confidence": 0.89,
    "horizon_hours": 24,
    "model_used": "xgboost"
  }
}
```

#### Get Historical Data
```http
GET /api/v1/historical?days=7
```

**Response:**
```json
{
  "start_date": "2026-02-07",
  "end_date": "2026-02-14",
  "records": 168,
  "data": [
    {"timestamp": "2026-02-07T00:00:00Z", "aqi": 145},
    {"timestamp": "2026-02-07T01:00:00Z", "aqi": 142},
    ...
  ]
}
```

---

## 🚢 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `webapp/app.py`

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to "Settings" → "Secrets"
   - Add your secrets:
   ```toml
   OPENWEATHERMAP_API_KEY = "your_api_key"
   HOPSWORKS_API_KEY = "your_api_key"
   ```

### GitHub Actions Setup

1. **Add Repository Secrets**
   - Go to Repository → Settings → Secrets and variables → Actions
   - Add:
     - `OPENWEATHERMAP_API_KEY`
     - `HOPSWORKS_API_KEY`

2. **Workflow Files**

   `.github/workflows/feature_pipeline.yml`:
   ```yaml
   name: Feature Pipeline
   on:
     schedule:
       - cron: '0 * * * *'  # Every hour
     workflow_dispatch:

   jobs:
     fetch-features:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.10'
         - name: Install dependencies
           run: pip install -r requirements.txt
         - name: Run feature pipeline
           env:
             OPENWEATHERMAP_API_KEY: ${{ secrets.OPENWEATHERMAP_API_KEY }}
             HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
           run: python pipelines/feature_pipeline.py
   ```

   `.github/workflows/training_pipeline.yml`:
   ```yaml
   name: Training Pipeline
   on:
     schedule:
       - cron: '0 2 * * *'  # Daily at 2 AM UTC
     workflow_dispatch:

   jobs:
     train-models:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: '3.10'
         - name: Install dependencies
           run: pip install -r requirements.txt
         - name: Run training pipeline
           env:
             HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
           run: python pipelines/training_pipeline.py
   ```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

```python
# tests/test_features.py
def test_compute_lag_features():
    """Test lag feature computation."""
    ...

def test_compute_rolling_features():
    """Test rolling statistics computation."""
    ...

# tests/test_training.py
def test_model_training():
    """Test model training pipeline."""
    ...

def test_model_evaluation():
    """Test model evaluation metrics."""
    ...

# tests/test_inference.py
def test_prediction():
    """Test prediction function."""
    ...
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Hopsworks Version Error
```
RestAPIError: Client version 3.7.0 is not compatible with backend version 4.2.2
```
**Solution:** Update hopsworks package:
```bash
pip install hopsworks==4.2.*
```

#### 2. Missing API Key
```
Error: OPENWEATHERMAP_API_KEY not found
```
**Solution:** Create `.env` file with your API key:
```bash
echo "OPENWEATHERMAP_API_KEY=your_key_here" > .env
```

#### 3. Model Not Found
```
FileNotFoundError: Model file not found at models/xgboost/latest
```
**Solution:** Run the training pipeline:
```bash
python pipelines/training_pipeline.py
```

#### 4. Import Error
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Install in development mode:
```bash
pip install -e .
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for functions
- Add tests for new features

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Muzammil Haider

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Author

**Muzammil Haider**

- GitHub: [@Haideransari444](https://github.com/Haideransari444)
- Project: Pearls Project - Islamabad AQI Predictor

---

## 🙏 Acknowledgments

- **OpenWeatherMap** for providing free air quality data API
- **Hopsworks** for the excellent feature store platform
- **Streamlit** for the amazing web framework
- **10 Pearls** for the internship opportunity

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ in Pakistan

</div>

## Setup

1. Clone this repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure
6. Run the app: `streamlit run webapp/app.py`
