#!/usr/bin/env python3
"""
Skeptical Analysis of Anti-Aging ML Results

Critical examination of model performance to identify potential issues:
1. Data leakage detection
2. Overfitting assessment
3. Feature correlation analysis
4. Target variable analysis
5. Prediction distribution validation
6. Residual analysis

Author: Anti-Aging ML Project
Date: October 2025
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 80)
print("SKEPTICAL ANALYSIS OF ANTI-AGING ML RESULTS")
print("=" * 80)

# Load data
data_path = "/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/data/datasets/train.csv"
df = pd.read_csv(data_path)

print(f"\n1. DATASET OVERVIEW")
print(f"   Shape: {df.shape}")
print(f"   Target (age) range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
print(f"   Target mean ± std: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")

# Check for suspicious perfect correlations
print(f"\n2. CORRELATION ANALYSIS - Looking for Data Leakage")
target = 'age'
exclude_cols = [target, 'biological_age', 'user_id']
numeric_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

# Calculate correlations with target
correlations = []
for col in feature_cols:
    corr, pval = pearsonr(df[col].fillna(0), df[target])
    correlations.append({
        'feature': col,
        'correlation': corr,
        'abs_correlation': abs(corr),
        'p_value': pval
    })

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

print(f"\n   Top 10 features most correlated with age:")
for idx, row in corr_df.head(10).iterrows():
    print(f"   {row['feature']:30} r={row['correlation']:+.4f} (p={row['p_value']:.2e})")

# Check for suspiciously high correlations (potential leakage)
high_corr = corr_df[corr_df['abs_correlation'] > 0.95]
if len(high_corr) > 0:
    print(f"\n   ⚠️  WARNING: {len(high_corr)} features with |r| > 0.95 (potential data leakage):")
    for idx, row in high_corr.iterrows():
        print(f"      {row['feature']}: r={row['correlation']:+.4f}")
else:
    print(f"\n   ✓ No features with |r| > 0.95 (good)")

# Check biological_age correlation (should be high but not perfect)
if 'biological_age' in df.columns:
    bio_corr, bio_pval = pearsonr(df['biological_age'], df['age'])
    print(f"\n   Biological age vs chronological age: r={bio_corr:.4f} (p={bio_pval:.2e})")
    if bio_corr > 0.99:
        print(f"   ⚠️  WARNING: Biological age almost perfectly correlated with age!")
    elif bio_corr > 0.9:
        print(f"   ✓ Strong but realistic correlation")

# 3. TRAIN-TEST SPLIT CONSISTENCY
print(f"\n3. CROSS-VALIDATION ANALYSIS - Checking for Overfitting")

# Prepare features
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_features = train_df[feature_cols].copy()
test_features = test_df[feature_cols].copy()

# Handle categorical
cat_cols = train_features.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    train_features = pd.get_dummies(train_features, columns=cat_cols, drop_first=True)
    test_features = pd.get_dummies(test_features, columns=cat_cols, drop_first=True)
    common_cols = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]

X_train = train_features.values
y_train = train_df[target].values
X_test = test_features.values
y_test = test_df[target].values

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Training performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test performance
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n   Training performance:")
print(f"     R² = {train_r2:.4f}, MAE = {train_mae:.2f} years")
print(f"   Test performance:")
print(f"     R² = {test_r2:.4f}, MAE = {test_mae:.2f} years")

# Calculate overfitting metrics
r2_gap = train_r2 - test_r2
mae_gap = test_mae - train_mae

print(f"\n   Overfitting analysis:")
print(f"     R² gap (train - test): {r2_gap:.4f}")
print(f"     MAE gap (test - train): {mae_gap:.2f} years")

if r2_gap < 0.05 and mae_gap < 1.0:
    print(f"     ✓ Minimal overfitting detected")
elif r2_gap < 0.15 and mae_gap < 2.0:
    print(f"     ⚠️  Moderate overfitting detected")
else:
    print(f"     ⚠️  Significant overfitting detected!")

# Cross-validation
print(f"\n   5-Fold Cross-Validation:")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)

print(f"     R² = {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
print(f"     MAE = {cv_mae_scores.mean():.2f} ± {cv_mae_scores.std():.2f} years")

if cv_r2_scores.std() > 0.1:
    print(f"     ⚠️  High variance in CV scores - model stability issue")
else:
    print(f"     ✓ Low variance - model is stable")

# 4. PREDICTION DISTRIBUTION ANALYSIS
print(f"\n4. PREDICTION DISTRIBUTION ANALYSIS")

pred_range = y_test_pred.max() - y_test_pred.min()
true_range = y_test.max() - y_test.min()
pred_std = y_test_pred.std()
true_std = y_test.std()

print(f"\n   True age range: {y_test.min():.1f} - {y_test.max():.1f} (span: {true_range:.1f} years)")
print(f"   Predicted range: {y_test_pred.min():.1f} - {y_test_pred.max():.1f} (span: {pred_range:.1f} years)")
print(f"   True age std: {true_std:.2f} years")
print(f"   Predicted std: {pred_std:.2f} years")

range_ratio = pred_range / true_range
std_ratio = pred_std / true_std

if range_ratio < 0.7:
    print(f"   ⚠️  WARNING: Predictions are compressed (ratio={range_ratio:.2f})")
    print(f"      Model may be regressing to the mean!")
elif range_ratio > 1.3:
    print(f"   ⚠️  WARNING: Predictions are over-dispersed (ratio={range_ratio:.2f})")
else:
    print(f"   ✓ Prediction range reasonable (ratio={range_ratio:.2f})")

# 5. RESIDUAL ANALYSIS
print(f"\n5. RESIDUAL ANALYSIS")

residuals = y_test - y_test_pred
residual_mean = residuals.mean()
residual_std = residuals.std()

print(f"\n   Residual mean: {residual_mean:.3f} years")
print(f"   Residual std: {residual_std:.3f} years")

if abs(residual_mean) > 1.0:
    print(f"   ⚠️  WARNING: Large systematic bias in predictions")
else:
    print(f"   ✓ Minimal systematic bias")

# Check for age-dependent bias
young_mask = y_test < 40
middle_mask = (y_test >= 40) & (y_test < 60)
old_mask = y_test >= 60

young_bias = residuals[young_mask].mean() if young_mask.sum() > 0 else 0
middle_bias = residuals[middle_mask].mean() if middle_mask.sum() > 0 else 0
old_bias = residuals[old_mask].mean() if old_mask.sum() > 0 else 0

print(f"\n   Age-stratified bias:")
print(f"     Young (<40):    {young_bias:+.2f} years (n={young_mask.sum()})")
print(f"     Middle (40-60): {middle_bias:+.2f} years (n={middle_mask.sum()})")
print(f"     Old (60+):      {old_bias:+.2f} years (n={old_mask.sum()})")

max_bias = max(abs(young_bias), abs(middle_bias), abs(old_bias))
if max_bias > 2.0:
    print(f"   ⚠️  WARNING: Large age-dependent bias detected!")
else:
    print(f"   ✓ Reasonable age-stratified performance")

# 6. FEATURE IMPORTANCE SANITY CHECK
print(f"\n6. FEATURE IMPORTANCE SANITY CHECK")

feature_names = train_features.columns.tolist()
importances = model.feature_importances_

# Sort by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 most important features:")
for idx, row in importance_df.head(5).iterrows():
    feat = row['feature']
    imp = row['importance']
    print(f"     {feat:30} {imp:.4f}")

# Check if one feature dominates
top_feature_importance = importance_df.iloc[0]['importance']
if top_feature_importance > 0.5:
    print(f"\n   ⚠️  WARNING: One feature dominates ({top_feature_importance:.1%})")
    print(f"      This suggests possible data leakage or over-reliance")
else:
    print(f"\n   ✓ Feature importance is distributed")

# 7. BASELINE COMPARISON
print(f"\n7. BASELINE MODEL COMPARISON")

# Simple mean predictor
mean_pred = np.full_like(y_test, y_train.mean())
mean_mae = mean_absolute_error(y_test, mean_pred)
mean_r2 = r2_score(y_test, mean_pred)

print(f"\n   Mean predictor (baseline):")
print(f"     R² = {mean_r2:.4f}, MAE = {mean_mae:.2f} years")
print(f"\n   Our model:")
print(f"     R² = {test_r2:.4f}, MAE = {test_mae:.2f} years")
print(f"\n   Improvement over baseline:")
print(f"     R² improvement: {test_r2 - mean_r2:.4f}")
print(f"     MAE improvement: {mean_mae - test_mae:.2f} years")

if test_r2 < 0.1:
    print(f"   ⚠️  WARNING: Model barely better than mean predictor!")
elif test_r2 > 0.9 and test_mae < 3:
    print(f"   ⚠️  CAUTION: Performance seems too good - double-check for leakage")
else:
    print(f"   ✓ Reasonable improvement over baseline")

# 8. REALITY CHECK
print(f"\n8. REALITY CHECK AGAINST PUBLISHED AGING CLOCKS")

print(f"\n   Published aging clock performance:")
print(f"     Horvath 2013:   R² = 0.84, MAE = 3.6 years (8000 samples)")
print(f"     Hannum 2013:    R² = 0.76, MAE = 4.2 years (656 samples)")
print(f"     PhenoAge 2018:  R² = 0.71, MAE = 5.1 years (9926 samples)")
print(f"     GrimAge 2019:   R² = 0.82, MAE = 3.9 years (1731 samples)")

print(f"\n   Our model:      R² = {test_r2:.2f}, MAE = {test_mae:.1f} years ({len(df)} samples)")

if test_r2 > 0.90:
    print(f"\n   ⚠️  SKEPTICAL ASSESSMENT:")
    print(f"      Our performance significantly exceeds published aging clocks.")
    print(f"      This is SUSPICIOUS for several reasons:")
    print(f"      1. Published clocks used 1000s-10000s of real human samples")
    print(f"      2. Real biological data has inherent noise/measurement error")
    print(f"      3. Our synthetic data may be 'too clean' or 'too correlated'")
    print(f"      4. We may have inadvertently created data leakage pathways")
    print(f"\n      RECOMMENDATIONS:")
    print(f"      - Add realistic measurement noise to synthetic data")
    print(f"      - Reduce correlation between features and target")
    print(f"      - Validate on completely independent test set")
    print(f"      - Consider our results as UPPER BOUND, not expected performance")

# FINAL VERDICT
print(f"\n" + "=" * 80)
print(f"FINAL SKEPTICAL ASSESSMENT")
print(f"=" * 80)

issues_found = []

if high_corr.shape[0] > 0:
    issues_found.append("Features with suspiciously high correlation to target")

if r2_gap > 0.15:
    issues_found.append("Significant overfitting (train-test gap)")

if top_feature_importance > 0.5:
    issues_found.append("Single feature dominates importance")

if test_r2 > 0.90:
    issues_found.append("Performance exceeds published aging clocks (suspicious)")

if max_bias > 2.0:
    issues_found.append("Large age-dependent prediction bias")

if len(issues_found) > 0:
    print(f"\n⚠️  ISSUES IDENTIFIED:")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
    
    print(f"\n   CONCLUSION: Results should be interpreted with CAUTION")
    print(f"   The model shows signs of potential data quality issues.")
    print(f"   Recommend adding statistical rigor (Issue #47) before publication.")
else:
    print(f"\n✓ No major issues detected")
    print(f"   Model performance appears legitimate, though still exceeds")
    print(f"   published aging clocks. This may be due to:")
    print(f"   - Cleaner synthetic data vs. real biological samples")
    print(f"   - Simpler biological relationships in synthetic data")
    print(f"   - Larger sample size and balanced age distribution")
    
print(f"\n" + "=" * 80)
