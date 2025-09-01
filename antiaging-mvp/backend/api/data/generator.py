#PLACEHOLDER CODE #1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_synthetic_data(n_samples=1000, output_path=None):
    """Generate synthetic anti-aging dataset"""
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Basic demographics
        age = np.random.randint(25, 80)
        gender = random.choice(['M', 'F'])
        height = np.random.normal(170 if gender == 'M' else 165, 10)
        weight = np.random.normal(75 if gender == 'M' else 65, 15)
        bmi = weight / ((height / 100) ** 2)
        
        # Lifestyle factors
        exercise_frequency = np.random.randint(0, 8)  # days per week
        sleep_hours = np.random.normal(7.5, 1.5)
        stress_level = np.random.randint(1, 11)
        diet_quality = np.random.randint(1, 11)
        smoking = random.choice([0, 1])
        alcohol_consumption = np.random.randint(0, 20)  # drinks per week
        
        # Genetic/Epigenetic markers (simplified)
        methylation_score = np.random.uniform(0.2, 0.9)
        telomere_length = np.random.uniform(4000, 16000)
        dna_damage_score = np.random.uniform(0.1, 0.7)
        gene_expression_aging = np.random.uniform(0.1, 1.0)
        epigenetic_clock_value = np.random.uniform(0.3, 1.3)
        
        # Environmental factors
        pollution_exposure = np.random.uniform(0.1, 0.8)
        sun_exposure = np.random.uniform(0.2, 0.9)
        occupation_stress = np.random.uniform(0.1, 0.9)
        
        # Health markers
        systolic_bp = np.random.normal(120, 20)
        diastolic_bp = np.random.normal(80, 15)
        cholesterol = np.random.normal(200, 50)
        glucose = np.random.normal(90, 20)
        
        # Calculate biological age with realistic relationships
        biological_age = _calculate_biological_age(
            age, gender, bmi, exercise_frequency, sleep_hours, 
            stress_level, diet_quality, smoking, alcohol_consumption,
            methylation_score, telomere_length, dna_damage_score,
            gene_expression_aging, epigenetic_clock_value,
            pollution_exposure, systolic_bp, cholesterol
        )
        
        # Add some noise
        biological_age += np.random.normal(0, 2)
        biological_age = max(18, biological_age)  # Minimum age
        
        sample = {
            'user_id': f'user_{i:04d}',
            'age': age,
            'gender': gender,
            'height': round(height, 1),
            'weight': round(weight, 1),
            'bmi': round(bmi, 2),
            'exercise_frequency': exercise_frequency,
            'sleep_hours': round(max(3, min(12, sleep_hours)), 1),
            'stress_level': stress_level,
            'diet_quality': diet_quality,
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption,
            'methylation_score': round(methylation_score, 4),
            'telomere_length': round(telomere_length, 0),
            'dna_damage_score': round(dna_damage_score, 4),
            'gene_expression_aging': round(gene_expression_aging, 4),
            'epigenetic_clock_value': round(epigenetic_clock_value, 4),
            'pollution_exposure': round(pollution_exposure, 4),
            'sun_exposure': round(sun_exposure, 4),
            'occupation_stress': round(occupation_stress, 4),
            'systolic_bp': round(max(80, systolic_bp), 0),
            'diastolic_bp': round(max(50, diastolic_bp), 0),
            'cholesterol': round(max(100, cholesterol), 0),
            'glucose': round(max(60, glucose), 0),
            'biological_age': round(biological_age, 2)
        }
        
        data.append(sample)
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Generated {n_samples} samples and saved to {output_path}")
    
    return df


def _calculate_biological_age(age, gender, bmi, exercise_freq, sleep_hours,
                             stress_level, diet_quality, smoking, alcohol,
                             methylation, telomere_length, dna_damage,
                             gene_expression, epigenetic_clock, pollution,
                             systolic_bp, cholesterol):
    """Calculate biological age based on various factors"""
    
    # Start with chronological age
    bio_age = age
    
    # BMI effect
    if bmi < 18.5 or bmi > 30:
        bio_age += 2
    elif 18.5 <= bmi <= 25:
        bio_age -= 1
    
    # Exercise effect (more exercise = younger)
    exercise_effect = (exercise_freq - 3) * -0.5
    bio_age += exercise_effect
    
    # Sleep effect (optimal around 7-8 hours)
    sleep_deviation = abs(sleep_hours - 7.5)
    bio_age += sleep_deviation * 0.5
    
    # Stress effect
    bio_age += (stress_level - 5) * 0.3
    
    # Diet effect
    bio_age -= (diet_quality - 5) * 0.4
    
    # Smoking effect
    if smoking:
        bio_age += 5
    
    # Alcohol effect (moderate drinking might be protective)
    if alcohol > 14:  # Heavy drinking
        bio_age += (alcohol - 14) * 0.2
    elif alcohol > 7:  # Moderate drinking
        bio_age -= 0.5
    
    # Genetic/Epigenetic factors
    bio_age += (1 - methylation) * 10  # Lower methylation = older
    bio_age += (1 - telomere_length / 10000) * 8  # Shorter telomeres = older
    bio_age += dna_damage * 10  # More damage = older
    bio_age += gene_expression * 5  # Higher expression = older
    bio_age += (epigenetic_clock - 1) * 8  # Higher clock = older
    
    # Environmental factors
    bio_age += pollution * 3
    
    # Health markers
    if systolic_bp > 140:
        bio_age += 2
    if cholesterol > 240:
        bio_age += 1.5
    
    # Gender differences (women tend to live longer)
    if gender == 'F':
        bio_age -= 1
    
    return bio_age


def generate_test_datasets():
    """Generate multiple test datasets"""
    
    datasets_dir = 'datasets'
    
    # Training dataset
    print("Generating training dataset...")
    train_data = generate_synthetic_data(
        n_samples=5000, 
        output_path=f'{datasets_dir}/training.csv'
    )
    
    # Test datasets
    test_scenarios = [
        ('test_young.csv', {'age_range': (25, 40), 'n_samples': 500}),
        ('test_middle.csv', {'age_range': (40, 60), 'n_samples': 500}),
        ('test_elderly.csv', {'age_range': (60, 80), 'n_samples': 500}),
        ('test_healthy.csv', {'lifestyle_bias': 'healthy', 'n_samples': 300}),
        ('test_unhealthy.csv', {'lifestyle_bias': 'unhealthy', 'n_samples': 300}),
    ]
    
    for filename, params in test_scenarios:
        print(f"Generating {filename}...")
        test_data = generate_specialized_dataset(**params)
        test_data.to_csv(f'{datasets_dir}/{filename}', index=False)
    
    print("All datasets generated successfully!")


def generate_specialized_dataset(age_range=None, lifestyle_bias=None, n_samples=500):
    """Generate specialized test datasets"""
    
    # Start with base synthetic data
    data = generate_synthetic_data(n_samples * 2)  # Generate more, then filter
    
    if age_range:
        min_age, max_age = age_range
        data = data[(data['age'] >= min_age) & (data['age'] <= max_age)]
    
    if lifestyle_bias == 'healthy':
        # Bias towards healthier lifestyle choices
        data = data[
            (data['exercise_frequency'] >= 4) & 
            (data['stress_level'] <= 6) & 
            (data['diet_quality'] >= 7) & 
            (data['smoking'] == 0)
        ]
    elif lifestyle_bias == 'unhealthy':
        # Bias towards unhealthier lifestyle choices
        data = data[
            (data['exercise_frequency'] <= 2) | 
            (data['stress_level'] >= 7) | 
            (data['diet_quality'] <= 4) | 
            (data['smoking'] == 1)
        ]
    
    # Take only the requested number of samples
    return data.head(n_samples).reset_index(drop=True)


if __name__ == "__main__":
    # Generate basic training data
    data = generate_synthetic_data(1000, 'training_data.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Biological age stats: {data['biological_age'].describe()}")
    
    # Generate test datasets
    generate_test_datasets()


    #PLACEHOLDER CODE #2
    """
from Bio.Seq import Seq
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from faker import Faker
import random

fake = Faker()

# Expanded to 10 SNPs
snps = {
    'SIRT1_rs7896005': {'alleles': ['A', 'G'], 'freq': [0.7, 0.3]},
    'FOXO3_rs2802292': {'alleles': ['C', 'T'], 'freq': [0.6, 0.4]},
    'APOE_rs429358': {'alleles': ['C', 'T'], 'freq': [0.85, 0.15]},
    'KL_rs9536314': {'alleles': ['T', 'G'], 'freq': [0.8, 0.2]},
    'IGF1_rs6220': {'alleles': ['C', 'T'], 'freq': [0.5, 0.5]},
    'TP53_rs1042522': {'alleles': ['G', 'C'], 'freq': [0.65, 0.35]},
    'CETP_rs5882': {'alleles': ['A', 'G'], 'freq': [0.4, 0.6]},
    'SOD2_rs4880': {'alleles': ['C', 'T'], 'freq': [0.55, 0.45]},
    'MTOR_rs2295080': {'alleles': ['T', 'G'], 'freq': [0.7, 0.3]},
    'PPARG_rs1801282': {'alleles': ['C', 'G'], 'freq': [0.9, 0.1]},
}

def generate_genotype(info):
    alleles = np.random.choice(info['alleles'], 2, p=info['freq'])
    return '/'.join(sorted(alleles))

def generate_habits():
    return {
        'exercises_per_week': max(0, min(7, int(np.random.normal(3.5, 1.5)))),
        'daily_calories': random.randint(1500, 3000),
        'alcohol_doses_per_week': random.uniform(0, 14),
        'years_smoking': random.randint(0, 20),
        'hours_of_sleep': random.uniform(4, 10),
        'stress_level': random.randint(1, 10)
    }

def generate_demographics():
    return {
        'age': random.randint(20, 80),
        'gender': random.choice(['M', 'F', 'Other'])
    }

def generate_dataset(n):
    data = []
    for _ in range(n):
        genotype = {snp: generate_genotype(info) for snp, info in snps.items()}
        habits = generate_habits()
        demo = generate_demographics()
        # Simulate risk with rules + noise (multi-class)
        base_risk = 0
        if habits['alcohol_doses_per_week'] > 7: base_risk += 2
        if habits['stress_level'] > 7: base_risk += 1
        if genotype.get('APOE_rs429358') in ['T/C', 'T/T']: base_risk += 1.5
        risk_score = base_risk + np.random.normal(0, 0.5)
        if risk_score < 1.5:
            risk = 'low'
        elif risk_score < 3:
            risk = 'medium'
        else:
            risk = 'high'
        data.append({**genotype, **habits, **demo, 'risk': risk})
    return pd.DataFrame(data)

def validate(df):
    for snp in snps:
        observed = df[snp].value_counts(normalize=True).sort_index()
        p, q = snps[snp]['freq']
        expected = np.array([p**2, 2*p*q, q**2])
        if len(observed) == len(expected):
            stat, p_val = chisquare(observed, expected)
            if p_val < 0.05:
                print(f"Warning: Unrealistic frequencies for {snp} (p={p_val})")

# Generate datasets
df_train = generate_dataset(5000)
validate(df_train)
df_train.to_csv('api/data/datasets/training.csv', index=False)

df_test1 = generate_dataset(10)
df_test1.to_csv('api/data/datasets/test1.csv', index=False)

df_test2 = generate_dataset(1)
df_test2.to_csv('api/data/datasets/test2.csv', index=False)

print("Datasets generated and validated.")

    """