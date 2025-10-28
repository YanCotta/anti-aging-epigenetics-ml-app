#!/usr/bin/env python3
"""
LiveMore MVP Model Training
Trains Random Forest on simplified dataset and saves artifacts for Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import shap
import json
from pathlib import Path

print("=" * 60)
print("LIVEMORE MVP MODEL TRAINING")
print("Training Random Forest for Streamlit Demo")
print("=" * 60)

# Paths
DATA_DIR = Path(__file__).parent / "data_generation" / "datasets_livemore_mvp"
MODEL_DIR = Path(__file__).parent.parent / "antiaging-mvp" / "streamlit_app" / "app_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
print("\n📁 Loading training data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
print(f"   Training samples: {len(train_df)}")
print(f"   Features: {train_df.columns.tolist()}")

# Prepare features and target
feature_cols = [col for col in train_df.columns if col != 'biological_age']
X_train = train_df[feature_cols]
y_train = train_df['biological_age']

# Handle categorical encoding (gender)
X_train_encoded = X_train.copy()
if 'gender' in X_train_encoded.columns:
    X_train_encoded['gender'] = X_train_encoded['gender'].map({'M': 0, 'F': 1})

print(f"   Features used: {feature_cols}")
print(f"   Target range: {y_train.min():.1f} - {y_train.max():.1f} years")

# Scale features
print("\n⚙️  Fitting scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Train Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate on training data
y_pred_train = rf_model.predict(X_train_scaled)
train_r2 = r2_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"   ✓ Training complete!")
print(f"   R² Score: {train_r2:.4f}")
print(f"   MAE: {train_mae:.2f} years")
print(f"   RMSE: {train_rmse:.2f} years")

# Create SHAP explainer
print("\n🔍 Creating SHAP explainer...")
# Use a sample for faster explainer creation
sample_size = min(500, len(X_train_scaled))
X_sample = X_train_scaled[:sample_size]
explainer = shap.TreeExplainer(rf_model)

print(f"   ✓ SHAP explainer ready (using {sample_size} background samples)")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top Feature Importances:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# Save artifacts
print("\n💾 Saving model artifacts...")
joblib.dump(rf_model, MODEL_DIR / "livemore_rf_v2.joblib")
print(f"   ✓ Model: {MODEL_DIR / 'livemore_rf_v2.joblib'}")

joblib.dump(scaler, MODEL_DIR / "livemore_scaler_v2.joblib")
print(f"   ✓ Scaler: {MODEL_DIR / 'livemore_scaler_v2.joblib'}")

joblib.dump(explainer, MODEL_DIR / "livemore_explainer_v2.pkl")
print(f"   ✓ SHAP Explainer: {MODEL_DIR / 'livemore_explainer_v2.pkl'}")

# Save metadata
metadata = {
    'model_version': 'v2_mvp',
    'training_date': pd.Timestamp.now().isoformat(),
    'n_samples': len(train_df),
    'features': feature_cols,
    'performance': {
        'train_r2': float(train_r2),
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse)
    },
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5
    }
}

with open(MODEL_DIR / "model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata: {MODEL_DIR / 'model_metadata.json'}")

# Test on test sets
print("\n🧪 Testing on test sets...")
test_files = list(DATA_DIR.glob("test_*.csv"))
for test_file in test_files:
    test_df = pd.read_csv(test_file)
    X_test = test_df[feature_cols].copy()
    
    if 'gender' in X_test.columns:
        X_test['gender'] = X_test['gender'].map({'M': 0, 'F': 1})
    
    X_test_scaled = scaler.transform(X_test)
    y_test = test_df['biological_age']
    y_pred_test = rf_model.predict(X_test_scaled)
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   {test_file.name:35s}: R²={test_r2:.4f}, MAE={test_mae:.2f} years")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"\nArtifacts saved to: {MODEL_DIR}")
print("\nReady for Streamlit MVP development!")
print("Next step: Build the Streamlit app using these artifacts")
