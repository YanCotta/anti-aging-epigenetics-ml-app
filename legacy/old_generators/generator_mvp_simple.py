#!/usr/bin/env python3
"""
LiveMore MVP Data Generator - Business-Focused Approach

This creates data optimized for the MVP demo, not scientific publication.
Focus: Clear lifestyle → biological age relationships that users understand.
Goal: Show ROI of healthy choices with interpretable SHAP explanations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def generate_livemore_mvp_data(n_samples=5000, seed=42):
    """Generate business-oriented data with clear cause-effect relationships"""
    np.random.seed(seed)
    
    data = {}
    
    # Demographics
    data['age'] = np.random.randint(25, 80, n_samples)
    data['gender'] = np.random.choice(['M', 'F'], n_samples)
    
    # Lifestyle factors (key demo variables - user can control these!)
    data['exercise_hours_week'] = np.random.gamma(2, 2, n_samples).clip(0, 20)
    data['diet_quality_score'] = np.random.normal(6, 2, n_samples).clip(1, 10)
    data['sleep_hours'] = np.random.normal(7, 1.2, n_samples).clip(4, 10)
    data['stress_level'] = np.random.normal(5, 2, n_samples).clip(1, 10)
    data['smoking_pack_years'] = np.random.exponential(5, n_samples).clip(0, 50)
    data['alcohol_drinks_week'] = np.random.gamma(1.5, 3, n_samples).clip(0, 30)
    
    # Simple genetic risk score (combined effect, not editable by user)
    data['genetic_risk_score'] = np.random.normal(5, 1.5, n_samples).clip(1, 10)
    
    # CREATE BIOLOGICAL AGE WITH CLEAR NON-LINEAR EFFECTS
    # This is what makes RF better than Linear!
    bio_age = data['age'] * 0.7  # Base chronological component (reduced to allow more variation)
    
    # Exercise: STRONG NON-LINEAR benefit (exponential curve)
    exercise_benefit = -np.log1p(data['exercise_hours_week']) * 5
    bio_age += exercise_benefit
    
    # Diet: THRESHOLD + EXPONENTIAL
    diet_effect = np.where(
        data['diet_quality_score'] >= 8,
        -(data['diet_quality_score'] - 5) ** 2 * 0.5,  # Quadratic benefit for excellent diet
        -(data['diet_quality_score'] - 5) * 0.3   # Linear for poor/moderate diet
    )
    bio_age += diet_effect
    
    # Sleep: STRONG U-SHAPED curve (penalties for too little OR too much)
    sleep_deviation = np.abs(data['sleep_hours'] - 7.5)
    bio_age += sleep_deviation ** 2 * 0.8  # Quadratic penalty
    
    # Stress: EXPONENTIAL damage at high levels
    stress_effect = np.exp((data['stress_level'] - 5) / 3) - 1
    bio_age += stress_effect * 3
    
    # Smoking: VERY EXPONENTIAL damage
    bio_age += np.exp(data['smoking_pack_years'] / 15) - 1
    
    # Alcohol: STRONG THRESHOLD effect
    alcohol_effect = np.where(
        data['alcohol_drinks_week'] <= 7,
        -np.sqrt(data['alcohol_drinks_week']) * 0.5,  # Slight protective effect
        (data['alcohol_drinks_week'] - 7) ** 1.5 * 0.3  # Strong harm above 7
    )
    bio_age += alcohol_effect
    
    # STRONG INTERACTION EFFECTS (key for RF advantage!)
    # Smoking × Stress MULTIPLICATIVE amplification
    bio_age += (data['smoking_pack_years'] ** 1.2) * (data['stress_level'] ** 1.2) * 0.03
    
    # Exercise × Diet EXPONENTIAL synergy (increased effect)
    lifestyle_score = (data['exercise_hours_week'] / 20 + data['diet_quality_score'] / 10) / 2
    bio_age -= np.exp(lifestyle_score * 2.5) * 0.7
    
    # Smoking × Alcohol MULTIPLICATIVE harm (increased)
    bio_age += (data['smoking_pack_years'] / 50) * (data['alcohol_drinks_week'] / 30) * 20
    
    # Genetic risk × Smoking INTERACTION (increased)
    bio_age += data['genetic_risk_score'] * np.log1p(data['smoking_pack_years']) * 0.5
    
    # Age × Lifestyle interaction (impact varies by age)
    age_factor = data['age'] / 50
    bio_age += age_factor * (10 - data['diet_quality_score']) * 0.7
    bio_age -= age_factor * np.log1p(data['exercise_hours_week']) * 2.5
    
    # Stress × Sleep deprivation interaction
    sleep_stress_penalty = np.where(
        (data['stress_level'] > 7) & (data['sleep_hours'] < 6),
        (data['stress_level'] - 7) * (6 - data['sleep_hours']) * 1.5,
        0
    )
    bio_age += sleep_stress_penalty
    
    # Add realistic noise (age-dependent, but reduced to allow patterns to show)
    noise_scale = 1.5 + (data['age'] - 25) / 55 * 2.5  # Less noise than before
    bio_age += np.random.normal(0, 1, n_samples) * noise_scale
    
    # Ensure biological age is reasonable
    bio_age = bio_age.clip(18, 110)
    data['biological_age'] = bio_age
    
    return pd.DataFrame(data)

def create_test_sets(df_train):
    """Create specialized test sets for demo"""
    
    # Test set 1: Young healthy (best case scenario for demo)
    young_healthy_filtered = df_train[
        (df_train['age'] < 40) &
        (df_train['exercise_hours_week'] > 5) &
        (df_train['diet_quality_score'] > 7) &
        (df_train['smoking_pack_years'] < 2)
    ]
    young_healthy = young_healthy_filtered.sample(
        n=min(200, max(50, len(young_healthy_filtered))), 
        random_state=42, 
        replace=len(young_healthy_filtered) < 200
    )
    
    # Test set 2: Middle-age unhealthy (intervention opportunity)
    middle_unhealthy_filtered = df_train[
        (df_train['age'].between(40, 60)) &
        (df_train['exercise_hours_week'] < 3) &
        (df_train['diet_quality_score'] < 5) &
        (df_train['smoking_pack_years'] > 10)
    ]
    middle_unhealthy = middle_unhealthy_filtered.sample(
        n=min(200, max(50, len(middle_unhealthy_filtered))), 
        random_state=43,
        replace=len(middle_unhealthy_filtered) < 200
    )
    
    # Test set 3: General test
    test_general = df_train.sample(n=1000, random_state=44)
    
    return {
        'test_young_healthy': young_healthy,
        'test_middle_unhealthy': middle_unhealthy,
        'test_general': test_general
    }

if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("LIVEMORE MVP DATA GENERATOR")
    print("Business-Focused Approach for Demo")
    print("=" * 60)
    
    # Generate training data
    print("\nGenerating 5000 training samples...")
    df_train = generate_livemore_mvp_data(n_samples=5000, seed=42)
    
    # Create output directory
    output_dir = "datasets_livemore_mvp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_path = os.path.join(output_dir, "train.csv")
    df_train.to_csv(train_path, index=False)
    print(f"✓ Saved: {train_path}")
    
    # Create and save test sets
    print("\nGenerating test sets...")
    test_sets = create_test_sets(df_train)
    for name, df in test_sets.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"✓ Saved: {path} ({len(df)} samples)")
    
    # Quick validation
    print("\n" + "=" * 60)
    print("QUICK VALIDATION")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # Prepare data
    X = df_train.drop(['biological_age'], axis=1)
    X['gender'] = (X['gender'] == 'M').astype(int)
    y = df_train['biological_age']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    # Results
    print(f"\nLinear Regression: R²={lr_r2:.4f}, MAE={lr_mae:.2f} years")
    print(f"Random Forest:     R²={rf_r2:.4f}, MAE={rf_mae:.2f} years")
    
    r2_gain = ((rf_r2 - lr_r2) / lr_r2) * 100
    print(f"\n🎯 RF vs Linear R² Gain: {r2_gain:+.2f}%")
    
    if r2_gain > 5:
        print("✅ SUCCESS! Data is suitable for LiveMore MVP")
    elif r2_gain > 0:
        print("⚠️ PARTIAL: RF provides marginal improvement")
    else:
        print("❌ FAIL: RF does not outperform Linear")
    
    print(f"\nDatasets saved in: {output_dir}/")
    print("Ready for Streamlit MVP development!")
