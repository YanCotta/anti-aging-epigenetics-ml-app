#!/usr/bin/env python3
"""
LiveMore V3 Data Generator - Biologically Realistic with Genetic Features

Key Improvements:
- 5 real SNPs from literature (APOE, FOXO3, TP53, SIRT1, TERT)
- Reduced age dominance (40% → lifestyle/genetics 60%)
- Fixed smoking to ALWAYS harm (negative SHAP)
- Stronger lifestyle impact for better demo
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_livemore_v3_data(n_samples=5000, seed=42):
    """Generate data with realistic SNP features and balanced factor importance"""
    np.random.seed(seed)
    
    data = {}
    
    # ==========================================
    # DEMOGRAPHICS
    # ==========================================
    data['age'] = np.random.randint(25, 80, n_samples)
    data['gender'] = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male
    
    # ==========================================
    # GENETIC FEATURES (5 KEY SNPs)
    # ==========================================
    # Each SNP has 3 genotypes: 0=protective, 1=neutral, 2=risk
    # Based on Hardy-Weinberg equilibrium with realistic allele frequencies
    
    # APOE ε4 (rs429358) - Alzheimer's risk, affects aging
    # ε4 allele frequency ~15%, so ε4/ε4 ~2%, ε3/ε4 ~25%, ε3/ε3 ~60%, ε2 carriers ~13%
    apoe_probs = [0.60, 0.28, 0.12]  # 0=protective (ε2/ε3), 1=neutral (ε3/ε3), 2=risk (ε4 carrier)
    data['APOE_rs429358'] = np.random.choice([0, 1, 2], n_samples, p=apoe_probs)
    
    # FOXO3 (rs2802292) - Longevity gene
    # T allele associated with longevity, MAF ~40%
    foxo3_probs = [0.16, 0.48, 0.36]  # 0=TT (protective), 1=GT, 2=GG (risk)
    data['FOXO3_rs2802292'] = np.random.choice([0, 1, 2], n_samples, p=foxo3_probs)
    
    # TP53 (rs1042522) - DNA repair, aging
    # Pro72 allele, MAF ~30%
    tp53_probs = [0.09, 0.42, 0.49]  # 0=Pro/Pro (protective), 1=Pro/Arg, 2=Arg/Arg
    data['TP53_rs1042522'] = np.random.choice([0, 1, 2], n_samples, p=tp53_probs)
    
    # SIRT1 (rs7069102) - Sirtuins, metabolic regulation
    # C allele, MAF ~20%
    sirt1_probs = [0.04, 0.32, 0.64]  # 0=CC (protective), 1=CG, 2=GG
    data['SIRT1_rs7069102'] = np.random.choice([0, 1, 2], n_samples, p=sirt1_probs)
    
    # TERT (rs2736100) - Telomere length
    # A allele, MAF ~48%
    tert_probs = [0.27, 0.50, 0.23]  # 0=AA (protective), 1=AC, 2=CC (shorter telomeres)
    data['TERT_rs2736100'] = np.random.choice([0, 1, 2], n_samples, p=tert_probs)
    
    # ==========================================
    # LIFESTYLE FACTORS
    # ==========================================
    data['exercise_hours_week'] = np.random.gamma(2, 2, n_samples).clip(0, 20)
    data['diet_quality_score'] = np.random.normal(6, 2, n_samples).clip(1, 10)
    data['sleep_hours'] = np.random.normal(7, 1.2, n_samples).clip(4, 10)
    data['stress_level'] = np.random.normal(5, 2, n_samples).clip(1, 10)
    data['smoking_pack_years'] = np.random.exponential(5, n_samples).clip(0, 50)
    data['alcohol_drinks_week'] = np.random.gamma(1.5, 3, n_samples).clip(0, 30)
    
    # ==========================================
    # BIOLOGICAL AGE CALCULATION
    # ==========================================
    
    # REDUCED age component (was 0.7, now 0.4 - reduces age dominance from 58% to ~35%)
    bio_age = data['age'] * 0.4 + 20  # Base component
    
    # ==========================================
    # GENETIC CONTRIBUTIONS (15-20% total importance)
    # ==========================================
    
    # APOE: Strong effect on aging (3-5 years impact)
    apoe_effect = np.where(data['APOE_rs429358'] == 0, -3,  # ε2 carriers live longer
                  np.where(data['APOE_rs429358'] == 1, 0,   # ε3/ε3 neutral
                           4))  # ε4 carriers age faster
    bio_age += apoe_effect
    
    # FOXO3: Longevity gene (2-4 years impact)
    foxo3_effect = (2 - data['FOXO3_rs2802292']) * 2  # 0→+4, 1→+2, 2→0
    bio_age -= foxo3_effect
    
    # TP53: DNA repair (1-3 years impact)
    tp53_effect = data['TP53_rs1042522'] * 1.5
    bio_age += tp53_effect
    
    # SIRT1: Metabolic regulation (1-2 years impact)
    sirt1_effect = (2 - data['SIRT1_rs7069102']) * 1
    bio_age -= sirt1_effect
    
    # TERT: Telomere length (2-3 years impact)
    tert_effect = data['TERT_rs2736100'] * 1.5
    bio_age += tert_effect
    
    # ==========================================
    # LIFESTYLE CONTRIBUTIONS (Strong non-linear patterns)
    # ==========================================
    
    # Exercise: STRONG logarithmic benefit (5-8 years range)
    exercise_benefit = -np.log1p(data['exercise_hours_week']) * 6.5
    bio_age += exercise_benefit
    
    # Diet: STRONG threshold effect (6-10 years range)
    diet_effect = np.where(
        data['diet_quality_score'] >= 8,
        -(data['diet_quality_score'] - 5) ** 2 * 0.8,  # Quadratic for excellent
        -(data['diet_quality_score'] - 5) * 0.5   # Linear otherwise
    )
    bio_age += diet_effect
    
    # Sleep: U-SHAPED with strong penalties (3-6 years range)
    sleep_deviation = np.abs(data['sleep_hours'] - 7.5)
    bio_age += sleep_deviation ** 2 * 1.2
    
    # Stress: EXPONENTIAL at high levels (4-8 years range)
    stress_effect = np.exp((data['stress_level'] - 5) / 2.5) - 1
    bio_age += stress_effect * 5
    
    # Smoking: ALWAYS HARMFUL with exponential curve (0-15 years added)
    # Fixed: This ensures smoking ALWAYS increases biological age
    smoking_harm = np.exp(data['smoking_pack_years'] / 12) - 1
    bio_age += smoking_harm * 2  # Scale to 0-15 years range
    
    # Alcohol: J-SHAPED curve (slight benefit ≤7, harm >7) (3-8 years range)
    alcohol_effect = np.where(
        data['alcohol_drinks_week'] <= 7,
        -np.sqrt(data['alcohol_drinks_week']) * 0.7,  # Slight benefit
        (data['alcohol_drinks_week'] - 7) ** 1.5 * 0.5  # Strong harm
    )
    bio_age += alcohol_effect
    
    # ==========================================
    # STRONG INTERACTIONS (Key for RF advantage)
    # ==========================================
    
    # Smoking × Stress: MULTIPLICATIVE damage
    bio_age += (data['smoking_pack_years'] / 50) * (data['stress_level'] / 10) * 12
    
    # Exercise × Diet: SYNERGISTIC benefit
    lifestyle_score = (data['exercise_hours_week'] / 20 + data['diet_quality_score'] / 10) / 2
    bio_age -= np.exp(lifestyle_score * 3) * 0.5
    
    # Smoking × APOE: Gene-environment interaction
    # APOE ε4 carriers more vulnerable to smoking
    smoking_apoe_interaction = np.where(
        data['APOE_rs429358'] == 2,  # ε4 carriers
        data['smoking_pack_years'] * 0.3,  # Extra damage
        0
    )
    bio_age += smoking_apoe_interaction
    
    # Age × Lifestyle: Effect varies by age
    age_factor = (data['age'] - 25) / 55  # 0 at age 25, 1 at age 80
    bio_age += age_factor * (10 - data['diet_quality_score']) * 1.2
    bio_age -= age_factor * np.log1p(data['exercise_hours_week']) * 3
    
    # Sleep × Stress: Compounding effect
    sleep_stress_penalty = np.where(
        (data['stress_level'] > 7) & (data['sleep_hours'] < 6),
        (data['stress_level'] - 7) * (6 - data['sleep_hours']) * 2.5,
        0
    )
    bio_age += sleep_stress_penalty
    
    # Gender × Lifestyle: Women more sensitive to sleep/stress
    gender_lifestyle = np.where(
        data['gender'] == 0,  # Female
        (data['stress_level'] / 10) * 3 + sleep_deviation * 1.5,
        0
    )
    bio_age += gender_lifestyle
    
    # ==========================================
    # REALISTIC NOISE (age-dependent but moderate)
    # ==========================================
    noise_scale = 2.0 + (data['age'] - 25) / 55 * 3.0
    bio_age += np.random.normal(0, 1, n_samples) * noise_scale
    
    # ==========================================
    # CONSTRAINTS
    # ==========================================
    bio_age = bio_age.clip(18, 90)  # Realistic bounds
    
    data['biological_age'] = bio_age
    
    return pd.DataFrame(data)


def generate_test_scenarios():
    """Generate specific test cases for validation"""
    
    # Scenario 1: Young Healthy (optimal genetics + lifestyle)
    young_healthy = {
        'age': [30, 32, 35, 28, 33],
        'gender': [1, 0, 1, 0, 1],
        'APOE_rs429358': [0, 0, 1, 0, 1],  # Mostly protective
        'FOXO3_rs2802292': [0, 0, 0, 1, 0],  # Longevity alleles
        'TP53_rs1042522': [0, 1, 0, 0, 1],
        'SIRT1_rs7069102': [0, 0, 1, 0, 0],
        'TERT_rs2736100': [0, 0, 0, 1, 0],
        'exercise_hours_week': [10, 12, 8, 15, 9],
        'diet_quality_score': [8, 9, 8, 9, 8],
        'sleep_hours': [7.5, 7, 8, 7.5, 7],
        'stress_level': [3, 2, 4, 3, 3],
        'smoking_pack_years': [0, 0, 0, 0, 0],
        'alcohol_drinks_week': [5, 3, 7, 2, 6]
    }
    
    # Scenario 2: Middle-aged Unhealthy (risk genetics + poor lifestyle)
    middle_unhealthy = {
        'age': [50, 52, 48, 55, 51],
        'gender': [1, 0, 1, 0, 1],
        'APOE_rs429358': [2, 2, 1, 2, 2],  # Risk alleles
        'FOXO3_rs2802292': [2, 2, 2, 1, 2],
        'TP53_rs1042522': [2, 2, 1, 2, 2],
        'SIRT1_rs7069102': [2, 2, 2, 2, 1],
        'TERT_rs2736100': [2, 2, 2, 1, 2],
        'exercise_hours_week': [1, 0, 2, 1, 0],
        'diet_quality_score': [4, 3, 5, 4, 3],
        'sleep_hours': [5, 5.5, 6, 5, 5.5],
        'stress_level': [8, 9, 8, 9, 8],
        'smoking_pack_years': [20, 25, 15, 30, 20],
        'alcohol_drinks_week': [15, 20, 12, 18, 16]
    }
    
    return young_healthy, middle_unhealthy


if __name__ == "__main__":
    print("=" * 60)
    print("LIVEMORE V3 DATA GENERATOR")
    print("With Real SNPs + Balanced Feature Importance")
    print("=" * 60)
    
    # Generate main dataset
    print("\n📊 Generating training data (5000 samples)...")
    train_df = generate_livemore_v3_data(n_samples=5000, seed=42)
    
    print(f"✓ Training data shape: {train_df.shape}")
    print(f"✓ Features: {list(train_df.columns)}")
    print(f"\n📈 Biological age stats:")
    print(f"   Mean: {train_df['biological_age'].mean():.1f} years")
    print(f"   Std: {train_df['biological_age'].std():.1f} years")
    print(f"   Range: {train_df['biological_age'].min():.1f} - {train_df['biological_age'].max():.1f} years")
    
    # Generate test scenarios
    print("\n📋 Generating test scenarios...")
    young_healthy, middle_unhealthy = generate_test_scenarios()
    
    young_df = pd.DataFrame(young_healthy)
    middle_df = pd.DataFrame(middle_unhealthy)
    
    # Calculate biological ages for test scenarios
    from generator_livemore_v3 import generate_livemore_v3_data
    # (Calculate using same logic - for demo, just add to DataFrame)
    
    # Save datasets
    output_dir = Path(__file__).parent / "datasets_livemore_v3"
    output_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    print(f"\n💾 Saved: {output_dir / 'train.csv'}")
    
    young_df.to_csv(output_dir / "test_young_healthy.csv", index=False)
    print(f"💾 Saved: {output_dir / 'test_young_healthy.csv'}")
    
    middle_df.to_csv(output_dir / "test_middle_unhealthy.csv", index=False)
    print(f"💾 Saved: {output_dir / 'test_middle_unhealthy.csv'}")
    
    # Generate general test set
    test_general = generate_livemore_v3_data(n_samples=1000, seed=999)
    test_general.to_csv(output_dir / "test_general.csv", index=False)
    print(f"💾 Saved: {output_dir / 'test_general.csv'}")
    
    print("\n✅ DATASET GENERATION COMPLETE")
    print(f"📁 Output directory: {output_dir}")
