#!/usr/bin/env python3
"""
Simple Direct Comparison: Baseline vs Advanced Features

Directly compares model performance with and without Issue #46 advanced features.
Uses proper train/test split to avoid data leakage.

Author: Anti-Aging ML Project
Date: October 2025
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add paths
sys.path.append('/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/ml')
from aging_features import AdvancedAgingFeatureEngineer

print("=" * 80)
print("DIRECT COMPARISON: BASELINE VS ADVANCED FEATURES (Issues #45 & #46)")
print("=" * 80)

# Load data
data_path = "/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/data/datasets/train.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {df.shape}")

# Use chronological age as target (real scientific question)
target = 'age'

# Split data BEFORE any processing
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=pd.cut(df[target], bins=5)
)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Prepare features (exclude target, IDs, and derived age variables)
exclude_cols = [target, 'biological_age', 'user_id']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Base features: {len(feature_cols)}")

# === BASELINE MODEL (No Advanced Features) ===
print("\n" + "=" * 80)
print("BASELINE: Without Advanced Features")
print("=" * 80)

# Handle categorical features
train_baseline = train_df[feature_cols].copy()
test_baseline = test_df[feature_cols].copy()

# Get all categorical/string columns
categorical_cols = train_baseline.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# One-hot encode all categorical features
if categorical_cols:
    train_baseline = pd.get_dummies(train_baseline, columns=categorical_cols, drop_first=True)
    test_baseline = pd.get_dummies(test_baseline, columns=categorical_cols, drop_first=True)

# Align columns (test might have different categories)
common_cols = train_baseline.columns.intersection(test_baseline.columns)
X_train_baseline = train_baseline[common_cols]
X_test_baseline = test_baseline[common_cols]

y_train = train_df[target].values
y_test = test_df[target].values

print(f"Features: {X_train_baseline.shape[1]}")

# Train baseline model
model_baseline = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model_baseline.fit(X_train_baseline, y_train)
y_pred_baseline = model_baseline.predict(X_test_baseline)

# Evaluate baseline
r2_baseline = r2_score(y_test, y_pred_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

print(f"Performance:")
print(f"  R² = {r2_baseline:.4f}")
print(f"  MAE = {mae_baseline:.2f} years")
print(f"  RMSE = {rmse_baseline:.2f} years")

# === ADVANCED MODEL (With Advanced Features) ===
print("\n" + "=" * 80)
print("ADVANCED: With Advanced Feature Engineering (Issue #46)")
print("=" * 80)

# Initialize feature engineer
engineer = AdvancedAgingFeatureEngineer(
    create_interactions=True,
    include_polynomial=False
)

# Apply advanced features to training data
train_advanced = train_df[feature_cols + [target]].copy()
train_advanced = engineer.engineer_features(train_advanced)

# Get feature report
feature_report = engineer.get_feature_report()
print(f"\nFeature Engineering Report:")
print(f"  Total features created: {feature_report['total_features_created']}")
for group, count in feature_report['feature_groups'].items():
    if count > 0:
        print(f"    {group}: {count}")

# Apply same transformations to test data
test_advanced = test_df[feature_cols + [target]].copy()
test_advanced = engineer.engineer_features(test_advanced)

# Prepare advanced features (exclude target and IDs)
advanced_exclude = [target, 'biological_age', 'user_id']
advanced_feature_cols = [col for col in train_advanced.columns if col not in advanced_exclude]

X_train_advanced = train_advanced[advanced_feature_cols].copy()
X_test_advanced = test_advanced[advanced_feature_cols].copy()

# Handle categorical features in advanced data
categorical_cols_advanced = X_train_advanced.select_dtypes(include=['object']).columns.tolist()
if categorical_cols_advanced:
    print(f"Encoding categorical columns in advanced features: {len(categorical_cols_advanced)}")
    X_train_advanced = pd.get_dummies(X_train_advanced, columns=categorical_cols_advanced, drop_first=True)
    X_test_advanced = pd.get_dummies(X_test_advanced, columns=categorical_cols_advanced, drop_first=True)

# Align columns (in case test has different columns)
common_advanced_cols = X_train_advanced.columns.intersection(X_test_advanced.columns)
X_train_advanced = X_train_advanced[common_advanced_cols]
X_test_advanced = X_test_advanced[common_advanced_cols]

print(f"\nFeatures: {X_train_advanced.shape[1]}")
print(f"Features added: {X_train_advanced.shape[1] - X_train_baseline.shape[1]}")

# Train advanced model
model_advanced = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model_advanced.fit(X_train_advanced, y_train)
y_pred_advanced = model_advanced.predict(X_test_advanced)

# Evaluate advanced
r2_advanced = r2_score(y_test, y_pred_advanced)
mae_advanced = mean_absolute_error(y_test, y_pred_advanced)
rmse_advanced = np.sqrt(mean_squared_error(y_test, y_pred_advanced))

print(f"\nPerformance:")
print(f"  R² = {r2_advanced:.4f}")
print(f"  MAE = {mae_advanced:.2f} years")
print(f"  RMSE = {rmse_advanced:.2f} years")

# === COMPARISON ===
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

r2_improvement = r2_advanced - r2_baseline
mae_improvement = mae_baseline - mae_advanced  # Positive = better
rmse_improvement = rmse_baseline - rmse_advanced

print(f"\nAbsolute Improvements:")
print(f"  R² change: {r2_improvement:+.4f}")
print(f"  MAE change: {mae_improvement:+.3f} years (lower is better)")
print(f"  RMSE change: {rmse_improvement:+.3f} years (lower is better)")

if mae_baseline > 0:
    mae_pct = (mae_improvement / mae_baseline) * 100
    print(f"\nRelative Improvements:")
    print(f"  MAE improvement: {mae_pct:+.1f}%")

# === LITERATURE COMPARISON ===
print("\n" + "=" * 80)
print("COMPARISON WITH PUBLISHED AGING CLOCKS")
print("=" * 80)

def categorize_performance(r2, mae):
    """Categorize performance based on literature standards."""
    if r2 >= 0.75 and mae <= 5:
        return "EXCELLENT"
    elif r2 >= 0.65 and mae <= 7:
        return "GOOD"
    elif r2 >= 0.55 and mae <= 9:
        return "ACCEPTABLE"
    else:
        return "NEEDS IMPROVEMENT"

baseline_category = categorize_performance(r2_baseline, mae_baseline)
advanced_category = categorize_performance(r2_advanced, mae_advanced)

print(f"\nBaseline: {baseline_category}")
print(f"  R² = {r2_baseline:.3f}, MAE = {mae_baseline:.2f}")

print(f"\nAdvanced: {advanced_category}")
print(f"  R² = {r2_advanced:.3f}, MAE = {mae_advanced:.2f}")

if advanced_category != baseline_category:
    print(f"\n✓ Category improved from {baseline_category} to {advanced_category}!")
else:
    print(f"\nCategory remained: {advanced_category}")

# === AGE-STRATIFIED PERFORMANCE ===
print("\n" + "=" * 80)
print("AGE-STRATIFIED PERFORMANCE (Advanced Model)")
print("=" * 80)

age_ranges = [
    (25, 40, "Young Adults"),
    (40, 55, "Middle Aged"),
    (55, 70, "Older Adults"),
    (70, 85, "Elderly")
]

for age_min, age_max, label in age_ranges:
    mask = (y_test >= age_min) & (y_test < age_max)
    if mask.sum() > 0:
        age_r2 = r2_score(y_test[mask], y_pred_advanced[mask])
        age_mae = mean_absolute_error(y_test[mask], y_pred_advanced[mask])
        print(f"{label:15} ({age_min}-{age_max}): R² = {age_r2:.3f}, MAE = {age_mae:.2f} (n={mask.sum()})")

# === TOP FEATURES ===
print("\n" + "=" * 80)
print("TOP 10 MOST IMPORTANT FEATURES (Advanced Model)")
print("=" * 80)

# Use actual columns from X_train_advanced used in fitting
actual_feature_cols = X_train_advanced.columns.tolist()
if len(actual_feature_cols) == len(model_advanced.feature_importances_):
    feature_importance = pd.DataFrame({
        'feature': actual_feature_cols,
        'importance': model_advanced.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:40} {row['importance']:.4f}")
else:
    print(f"Note: Feature importance display skipped due to column alignment issue")

# === FINAL SUMMARY ===
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nBaseline (No Advanced Features):")
print(f"  Features: {X_train_baseline.shape[1]}")
print(f"  R² = {r2_baseline:.4f}, MAE = {mae_baseline:.2f}, Category: {baseline_category}")

print(f"\nAdvanced (With Issue #46 Features):")
print(f"  Features: {X_train_advanced.shape[1]} (+{X_train_advanced.shape[1] - X_train_baseline.shape[1]})")
print(f"  R² = {r2_advanced:.4f}, MAE = {mae_advanced:.2f}, Category: {advanced_category}")

print(f"\nImpact of Advanced Features:")
print(f"  R² improvement: {r2_improvement:+.4f}")
print(f"  MAE improvement: {mae_improvement:+.2f} years ({mae_pct:+.1f}%)")

print("\n" + "=" * 80)
