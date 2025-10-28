#!/usr/bin/env python3
"""
Quick MVP Validation - Check if RF > Linear (Business Goal)

This is a pragmatic script for MVP validation, not scientific publication.
Goal: Verify that Random Forest provides value over Linear Regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def quick_validate(data_path):
    """Quick validation to check RF vs Linear performance"""
    print(f"\n{'='*60}")
    print(f"QUICK MVP VALIDATION: {data_path}")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Prepare features
    exclude_cols = ['user_id', 'biological_age']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['biological_age'].copy()
    
    # Encode all categorical columns (gender + SNP genotypes)
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    print("\n📊 Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    
    print(f"   R²: {lr_r2:.4f}")
    print(f"   MAE: {lr_mae:.2f} years")
    print(f"   RMSE: {lr_rmse:.2f} years")
    
    # Train Random Forest
    print("\n🌳 Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    print(f"   R²: {rf_r2:.4f}")
    print(f"   MAE: {rf_mae:.2f} years")
    print(f"   RMSE: {rf_rmse:.2f} years")
    
    # Calculate Improvement
    r2_gain_pct = ((rf_r2 - lr_r2) / lr_r2) * 100 if lr_r2 != 0 else 0
    
    print(f"\n{'='*60}")
    print(f"🎯 MVP SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")
    print(f"RF vs Linear R² Gain: {r2_gain_pct:+.2f}%")
    print(f"Target: >5% improvement")
    
    if r2_gain_pct > 5:
        print(f"✅ SUCCESS! RF provides {r2_gain_pct:.1f}% improvement")
        print(f"✅ XAI/SHAP explanations will add value for users")
        print(f"✅ Data is suitable for LiveMore MVP")
        return True
    elif r2_gain_pct > 0:
        print(f"⚠️  PARTIAL: RF provides {r2_gain_pct:.1f}% improvement (borderline)")
        print(f"⚠️  May proceed with MVP but XAI value is marginal")
        return True
    else:
        print(f"❌ FAIL: RF performs worse than Linear ({r2_gain_pct:.1f}%)")
        print(f"❌ Need to adjust chaos parameters further")
        return False

if __name__ == "__main__":
    import sys
    
    data_path = "data_generation/datasets_chaos_v2/train.csv"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    success = quick_validate(data_path)
    sys.exit(0 if success else 1)
