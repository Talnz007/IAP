"""
Model Comparison Script - Find the best model for AQI prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_parquet('data/processed/islamabad_aqi_features_upload.parquet')
df = df.sort_values('timestamp').reset_index(drop=True)

# Define features and target
exclude_cols = ['timestamp', 'event_time', 'observation_id', 
                'target_1h', 'target_6h', 'target_12h', 'target_24h', 'target_48h', 'target_72h']
feature_cols = [c for c in df.columns if c not in exclude_cols]

df_clean = df.dropna(subset=['target_24h'])
X = df_clean[feature_cols].values
y = df_clean['target_24h'].values

print(f"Samples: {len(X)}, Features: {len(feature_cols)}")

# Time Series Cross-Validation (3 folds for speed)
tscv = TimeSeriesSplit(n_splits=3)

# Models to test
models = {
    'Ridge': Ridge(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=1000),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, 
                                 reg_alpha=1, reg_lambda=1, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                                   reg_alpha=1, reg_lambda=1, random_state=42, verbose=-1),
}

results = {name: {'rmse': [], 'mae': [], 'r2': []} for name in models}
scaler = StandardScaler()

print("\nRunning 3-Fold Time Series Cross-Validation...")
print("-" * 60)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Fold {fold+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    for name, model in models.items():
        # Use scaled data for linear models
        if name in ['Ridge', 'ElasticNet']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name]['rmse'].append(rmse)
        results[name]['mae'].append(mae)
        results[name]['r2'].append(r2)

# Print results
print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(f"{'Model':<18} {'RMSE':<15} {'MAE':<10} {'R2':<10}")
print("-" * 60)

final_results = []
for name in models:
    avg_rmse = np.mean(results[name]['rmse'])
    std_rmse = np.std(results[name]['rmse'])
    avg_mae = np.mean(results[name]['mae'])
    avg_r2 = np.mean(results[name]['r2'])
    final_results.append((name, avg_rmse, avg_mae, avg_r2, std_rmse))
    print(f"{name:<18} {avg_rmse:.2f} +/- {std_rmse:.1f}   {avg_mae:.2f}     {avg_r2:.3f}")

# Sort by RMSE
final_results.sort(key=lambda x: x[1])

print("\n" + "=" * 60)
print("RANKING (by RMSE)")
print("=" * 60)
for i, (name, rmse, mae, r2, std) in enumerate(final_results):
    print(f"{i+1}. {name:<18} RMSE: {rmse:.2f}  R2: {r2:.3f}")

best = final_results[0]
print("\n" + "=" * 60)
print(f"BEST MODEL: {best[0]}")
print(f"  RMSE: {best[1]:.2f} +/- {best[4]:.1f}")
print(f"  MAE:  {best[2]:.2f}")
print(f"  R2:   {best[3]:.3f}")
print("=" * 60)
