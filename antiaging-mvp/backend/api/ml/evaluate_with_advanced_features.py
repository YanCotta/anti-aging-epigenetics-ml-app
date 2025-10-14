#!/usr/bin/env python3
"""
Evaluate ML Pipeline with Advanced Feature Engineering

Tests the complete pipeline with Issue #46 advanced features against
literature benchmarks from Issue #45.

Key Evaluations:
1. Performance comparison: baseline vs advanced features
2. Feature importance analysis
3. Age-stratified performance
4. Statistical rigor metrics
5. Comparison against published aging clocks

Author: Anti-Aging ML Project  
Date: October 2025
"""

import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path

# Add paths
sys.path.append('/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/data')
sys.path.append('/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/ml')

from genomics_ml_integration import GenomicsMLPipeline, GenomicsMLConfig
from aging_benchmarks import AgingBenchmarkLibrary, RealisticModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('advanced_features_eval')


def load_and_split_data():
    """Load data and create proper train/test split."""
    data_path = "/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/data/datasets/train.csv"
    df = pd.read_csv(data_path)
    
    # Use chronological age as target (the real scientific question)
    # Remove biological_age and other derived age features to avoid leakage
    target = 'age'
    
    # Split before any processing to ensure no leakage
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=pd.cut(df['age'], bins=5)
    )
    
    return train_df, test_df, target


def train_baseline_model(train_path, test_df, target, logger):
    """Train model without advanced features (baseline)."""
    logger.info("=== Training BASELINE Model (No Advanced Features) ===")
    
    # Initialize pipeline without advanced features
    config = GenomicsMLConfig(models_to_train=['random_forest'])
    pipeline = GenomicsMLPipeline(config=config, use_advanced_features=False)
    
    # Process training data from file
    processed_data = pipeline.load_and_preprocess_data(train_path)
    pipeline.run_quality_control()
    pipeline.engineer_aging_features()
    
    # Prepare ML data
    X_train, y_train = pipeline.prepare_ml_data(target=target)
    
    # Process test data using same preprocessing
    test_processed = pipeline.genomics_preprocessor.transform(test_df)
    
    # Prepare test features (exclude target and IDs)
    exclude_cols = [target, 'biological_age', 'user_id']
    feature_cols = [col for col in X_train.columns if col not in exclude_cols]
    
    X_test = test_processed[feature_cols]
    y_test = test_df[target].values
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train[feature_cols], y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    logger.info(f"Baseline Performance: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.2f}, RMSE = {metrics['rmse']:.2f}")
    
    return model, metrics, X_train.shape[1]


def train_advanced_model(train_path, test_df, target, logger):
    """Train model WITH advanced features (Issue #46)."""
    logger.info("\n=== Training ADVANCED Model (With Advanced Features - Issue #46) ===")
    
    # Initialize pipeline WITH advanced features
    config = GenomicsMLConfig(models_to_train=['random_forest'])
    pipeline = GenomicsMLPipeline(config=config, use_advanced_features=True)
    
    # Process training data from file
    processed_data = pipeline.load_and_preprocess_data(train_path)
    pipeline.run_quality_control()
    pipeline.engineer_aging_features()
    
    # Prepare ML data
    X_train, y_train = pipeline.prepare_ml_data(target=target)
    
    # Process test data through same pipeline
    # First do basic preprocessing
    test_processed = pipeline.genomics_preprocessor.transform(test_df)
    
    # Then apply advanced feature engineering
    if pipeline.feature_engineer:
        test_processed = pipeline.feature_engineer.engineer_features(test_processed)
    
    # Prepare test features
    exclude_cols = [target, 'biological_age', 'user_id']
    feature_cols = [col for col in X_train.columns if col not in exclude_cols and col in test_processed.columns]
    
    X_test = test_processed[feature_cols]
    y_test = test_df[target].values
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train[feature_cols], y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    logger.info(f"Advanced Performance: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.2f}, RMSE = {metrics['rmse']:.2f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, metrics, X_train.shape[1], y_pred, y_test, feature_importance


def compare_with_benchmarks(metrics, logger):
    """Compare performance against literature benchmarks."""
    logger.info("\n=== Comparison with Published Aging Clocks ===")
    
    # Get benchmark library
    benchmark_lib = AgingBenchmarkLibrary()
    
    # Categorize performance
    category = benchmark_lib.categorize_performance(metrics['r2'], metrics['mae'])
    
    logger.info(f"Performance Category: {category.upper()}")
    logger.info(f"  R² = {metrics['r2']:.3f}")
    logger.info(f"  MAE = {metrics['mae']:.2f} years")
    
    # Find closest benchmark
    closest_clock = benchmark_lib.get_closest_benchmark(metrics['r2'], metrics['mae'])
    
    if closest_clock:
        logger.info(f"\nClosest Published Clock: {closest_clock.name}")
        logger.info(f"  R² difference: {metrics['r2'] - closest_clock.r2_score:+.3f}")
        logger.info(f"  MAE difference: {metrics['mae'] - closest_clock.mae:+.2f} years")
    
    # Show where we stand
    logger.info("\nPublished Aging Clock Standards:")
    logger.info("  Excellent: R² 0.75-0.85, MAE 3-5 years")
    logger.info("  Good: R² 0.65-0.75, MAE 5-7 years")
    logger.info("  Acceptable: R² 0.55-0.65, MAE 7-9 years")
    
    return category


def analyze_improvements(baseline_metrics, advanced_metrics, baseline_features, advanced_features, logger):
    """Analyze improvements from advanced features."""
    logger.info("\n=== Feature Engineering Impact Analysis ===")
    
    # Feature count improvement
    feature_gain = advanced_features - baseline_features
    feature_gain_pct = (feature_gain / baseline_features) * 100
    
    logger.info(f"Feature Engineering:")
    logger.info(f"  Baseline features: {baseline_features}")
    logger.info(f"  Advanced features: {advanced_features}")
    logger.info(f"  Features added: {feature_gain} (+{feature_gain_pct:.1f}%)")
    
    # Performance improvement
    r2_improvement = advanced_metrics['r2'] - baseline_metrics['r2']
    mae_improvement = baseline_metrics['mae'] - advanced_metrics['mae']  # Positive = better
    rmse_improvement = baseline_metrics['rmse'] - advanced_metrics['rmse']
    
    logger.info(f"\nPerformance Improvements:")
    logger.info(f"  R² change: {r2_improvement:+.4f}")
    logger.info(f"  MAE change: {mae_improvement:+.3f} years (lower is better)")
    logger.info(f"  RMSE change: {rmse_improvement:+.3f} years (lower is better)")
    
    # Calculate percentage improvements
    if baseline_metrics['r2'] != 0:
        r2_pct = (r2_improvement / abs(baseline_metrics['r2'])) * 100
        logger.info(f"  R² improvement: {r2_pct:+.1f}%")
    
    mae_pct = (mae_improvement / baseline_metrics['mae']) * 100
    logger.info(f"  MAE improvement: {mae_pct:+.1f}%")
    
    return {
        'feature_gain': feature_gain,
        'r2_improvement': r2_improvement,
        'mae_improvement': mae_improvement,
        'mae_improvement_pct': mae_pct
    }


def main():
    """Run comprehensive evaluation."""
    logger = setup_logging()
    
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING EVALUATION (Issues #45 & #46)")
    print("=" * 80)
    
    # Load and split data
    logger.info("Loading and splitting data...")
    train_df, test_df, target = load_and_split_data()
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    logger.info(f"Target variable: {target}")
    
    # Save train_df to temporary file for pipeline
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_df.to_csv(f.name, index=False)
        train_path = f.name
    
    # Train baseline model
    baseline_model, baseline_metrics, baseline_features = train_baseline_model(
        train_path, test_df, target, logger
    )
    
    # Train advanced model
    advanced_model, advanced_metrics, advanced_features, y_pred, y_test, feature_importance = train_advanced_model(
        train_path, test_df, target, logger
    )
    
    # Cleanup temp file
    import os
    os.unlink(train_path)
    
    # Compare improvements
    improvements = analyze_improvements(
        baseline_metrics, advanced_metrics, 
        baseline_features, advanced_features, logger
    )
    
    # Compare with benchmarks
    advanced_category = compare_with_benchmarks(advanced_metrics, logger)
    
    # Top feature importance
    logger.info("\n=== Top 10 Most Important Features ===")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Age-specific performance
    logger.info("\n=== Age-Stratified Performance ===")
    age_ranges = [
        (25, 40, "Young Adults"),
        (40, 55, "Middle Aged"),
        (55, 70, "Older Adults"),
        (70, 85, "Elderly")
    ]
    
    for age_min, age_max, label in age_ranges:
        mask = (y_test >= age_min) & (y_test < age_max)
        if mask.sum() > 0:
            age_r2 = r2_score(y_test[mask], y_pred[mask])
            age_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            logger.info(f"  {label} ({age_min}-{age_max}): R² = {age_r2:.3f}, MAE = {age_mae:.2f} (n={mask.sum()})")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline:  R² = {baseline_metrics['r2']:.3f}, MAE = {baseline_metrics['mae']:.2f} ({baseline_features} features)")
    print(f"Advanced:  R² = {advanced_metrics['r2']:.3f}, MAE = {advanced_metrics['mae']:.2f} ({advanced_features} features)")
    print(f"Improvement: R² {improvements['r2_improvement']:+.4f}, MAE {improvements['mae_improvement']:+.2f} years ({improvements['mae_improvement_pct']:+.1f}%)")
    print(f"Category: {advanced_category.upper()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
