<div align="center">

# 🌍 Urban Intel: Islamabad AQI Predictor

### Real-Time Air Quality Index Prediction & Geospatial Simulation

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Svelte](https://img.shields.io/badge/Svelte-5-ff3e00.svg)](https://svelte.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

**Predicting tomorrow's air quality with high-performance ML and 3D simulation.**

Created by **Talha Niazi**
</div>

---

## 🎯 Overview

**Urban Intel** is a production-grade machine learning system that provides real-time monitoring and 24-hour predictive intelligence for the Air Quality Index (AQI) in Islamabad, Pakistan. 

Upgraded from a legacy prototype, this application now features a state-of-the-art **Svelte 5** frontend with immersive **MapLibre GL** 3D simulations, powered by a robust **FastAPI** backend and **Supabase** feature store.

### Key Capabilities

- 🏥 **Real-time Monitoring**: Live AQI, PM2.5, PM10, CO, NO2, SO2, O3, and weather metrics.
- 📊 **Multi-Model Inference**: Switch seamlessly between XGBoost, LightGBM, and Random Forest models with side-by-side metric comparison.
- 🎨 **Immersive 3D Simulation**: High-performance WebGL/Canvas particle engine rendering real-time smog and cloud cover over 3D extruded cityscapes.
- 🤖 **Explainable AI (XAI)**: Understand the driving features behind every prediction.

---

## 🏗️ Architecture Specification

The system is built on a modern, decoupled architecture:

1. **Frontend (Svelte 5 + MapLibre GL)**
   - Ultra-fast reactivity using Svelte runes.
   - Interactive SVG radial gauges for precise AQI visualization.
   - 60 FPS GPU-bound particle simulation reflecting real-time PM2.5 density.
   
2. **Backend (FastAPI + Python ML)**
   - High-throughput asynchronous API serving predictions.
   - Intelligent predictor caching to minimize latency across multi-model requests.
   
3. **Data & MLOps Pipeline (Supabase + Scikit-learn)**
   - Automated fetching from OpenWeatherMap APIs.
   - Centralized feature store and model registry in Supabase.
   - Hourly feature engineering and daily model retraining workflows.

---

## 📂 Dependency Blueprint & Directory Structure

```text
islamabad-aqi-predictor/
├── frontend/               # Modern SvelteKit Application
│   ├── src/                # Svelte components, routes, and styles
│   │   ├── lib/            # Reusable components (e.g., AQIGauge.svelte)
│   │   └── routes/         # Application views (+page.svelte)
│   └── package.json        # Frontend dependencies (svelte, maplibre-gl, etc.)
├── backend/                # FastAPI Application
│   └── main.py             # API Router and Predictor Cache
├── src/                    # Core Python Modules
│   ├── inference/          # Prediction logic (predict.py)
│   ├── features/           # Data fetching and engineering
│   └── training/           # Model training and evaluation
├── models/                 # Serialized ML Models (Joblib)
│   ├── xgboost/            
│   ├── lightgbm/           
│   └── random_forest/      
├── data/                   # Processed Datasets
├── tests/                  # Unit & Integration Tests
├── requirements.txt        # Backend Python dependencies
└── README.md               # Project Documentation
```

---

## 🚀 Installation & Setup

### Backend Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Configure `.env` with your API keys (OpenWeatherMap, Supabase).
3. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Svelte development server:
   ```bash
   npm run dev
   ```

---

## 📜 License

This project is licensed under the MIT License.
