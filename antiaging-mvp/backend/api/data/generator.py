import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from pathlib import Path
import warnings
from typing import Dict, List, Tuple


# Aging-related genes and their known variants
AGING_RELATED_SNPS = {
    'APOE_rs429358': {'alleles': ['C', 'T'], 'risk_allele': 'T', 'effect': 3.0},  # APOE4
    'APOE_rs7412': {'alleles': ['C', 'T'], 'risk_allele': 'T', 'effect': 1.5},   # APOE2
    'FOXO3_rs2802292': {'alleles': ['G', 'T'], 'risk_allele': 'G', 'effect': -1.2},  # Longevity
    'SIRT1_rs7069102': {'alleles': ['C', 'G'], 'risk_allele': 'G', 'effect': 1.8},   # Aging
    'TP53_rs1042522': {'alleles': ['C', 'G'], 'risk_allele': 'G', 'effect': 1.0},    # DNA repair
    'CDKN2A_rs10757278': {'alleles': ['A', 'G'], 'risk_allele': 'G', 'effect': 2.2}, # Cellular aging
    'TERT_rs2736100': {'alleles': ['A', 'C'], 'risk_allele': 'C', 'effect': 1.5},    # Telomerase
    'TERC_rs12696304': {'alleles': ['C', 'G'], 'risk_allele': 'G', 'effect': 1.3},   # Telomerase RNA
    'IGF1_rs35767': {'alleles': ['C', 'T'], 'risk_allele': 'T', 'effect': 1.1},      # Growth hormone
    'KLOTHO_rs9536314': {'alleles': ['C', 'T'], 'risk_allele': 'T', 'effect': -0.8}, # Anti-aging
}

# CpG sites used in aging clocks (simplified subset)
AGING_CPG_SITES = [
    'cg09809672', 'cg02228185', 'cg16867657', 'cg25809905', 'cg17861230',
    'cg06493994', 'cg19761273', 'cg09809672', 'cg17760862', 'cg23696862',
    'cg01620164', 'cg25410668', 'cg15611023', 'cg27405400', 'cg16419235',
    'cg00339382', 'cg06126421', 'cg18473521', 'cg21572722', 'cg25138706'
]


def generate_genetic_markers(age: int, gender: str, seed: int) -> Dict:
    """
    Generate realistic genetic markers including SNPs and CpG methylation.
    
    Args:
        age (int): Individual's chronological age
        gender (str): Gender ('M' or 'F')
        seed (int): Random seed for reproducibility
    
    Returns:
        Dict: Dictionary containing genetic markers
    """
    np.random.seed(seed)
    genetic_data = {}
    
    # Generate SNP genotypes
    aging_score = 0
    for snp_id, snp_info in AGING_RELATED_SNPS.items():
        # Generate genotype (two alleles)
        allele1 = np.random.choice(snp_info['alleles'])
        allele2 = np.random.choice(snp_info['alleles'])
        genotype = f"{allele1}{allele2}"
        
        # Calculate aging effect
        risk_count = [allele1, allele2].count(snp_info['risk_allele'])
        aging_score += risk_count * snp_info['effect']
        
        genetic_data[snp_id] = genotype
    
    # Generate CpG methylation values (0-1, age-dependent)
        for cpg_site in AGING_CPG_SITES:
            # Base methylation with age-related drift (reduced to avoid overly strong linearity)
            base_methylation = np.random.uniform(0.3, 0.7)
            age_effect = (age - 25) * 0.0015  # Gradual change with age (reduced)
            noise = np.random.normal(0, 0.05)

            methylation = np.clip(base_methylation + age_effect + noise, 0, 1)
            genetic_data[f"{cpg_site}_methylation"] = round(methylation, 4)
    
    # Calculate overall genetic aging score
    genetic_data['genetic_aging_score'] = round(aging_score, 2)
    
    # Add some derived genetic metrics
    genetic_data['longevity_alleles'] = sum(1 for snp in ['FOXO3_rs2802292', 'KLOTHO_rs9536314'] 
                                           if genetic_data[snp].count('T') > 0)
    genetic_data['risk_alleles'] = sum(1 for snp in ['APOE_rs429358', 'CDKN2A_rs10757278'] 
                                      if snp in genetic_data and 
                                      genetic_data[snp].count(AGING_RELATED_SNPS[snp]['risk_allele']) > 0)
    
    return genetic_data


def generate_synthetic_data(n_samples=5000, output_path=None, random_seed=42):
    """
    Generate synthetic anti-aging dataset with enhanced validation and quality checks.
    
    Args:
        n_samples (int): Number of samples to generate (default: 5000)
        output_path (str): Path to save the CSV file (optional)
        random_seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        pd.DataFrame: Generated synthetic dataset
    """
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print(f"Generating {n_samples} synthetic samples with seed {random_seed}...")
    
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
        
        # Generate realistic genetic data
        genetic_data = generate_genetic_markers(age, gender, random_seed + i)
        
        # Telomere length (realistic range)
        telomere_length = np.random.normal(8000, 2000)  # Base pairs
        telomere_length = max(4000, min(15000, telomere_length))  # Clamp to realistic range
        
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
            genetic_data, telomere_length, pollution_exposure, 
            systolic_bp, cholesterol
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
            'telomere_length': round(telomere_length, 0),
            'pollution_exposure': round(pollution_exposure, 4),
            'sun_exposure': round(sun_exposure, 4),
            'occupation_stress': round(occupation_stress, 4),
            'systolic_bp': round(max(80, systolic_bp), 0),
            'diastolic_bp': round(max(50, diastolic_bp), 0),
            'cholesterol': round(max(100, cholesterol), 0),
            'glucose': round(max(60, glucose), 0),
            'biological_age': round(biological_age, 2)
        }
        
        # Add genetic markers to sample
        sample.update(genetic_data)
        
        data.append(sample)
    
    df = pd.DataFrame(data)
    
    # Perform data validation
    validation_report = validate_dataset(df)
    print(f"\nData Validation Report:")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate records: {df.duplicated().sum()}")
    print(f"Biological age range: {df['biological_age'].min():.2f} - {df['biological_age'].max():.2f}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()}")
    
    if not validation_report['valid']:
        warnings.warn(f"Data validation issues detected: {validation_report['issues']}")
    
    if output_path:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {n_samples} samples and saved to {output_path}")
    
    return df


def validate_dataset(df):
    """
    Comprehensive data validation function.
    
    Args:
        df (pd.DataFrame): Dataset to validate
    
    Returns:
        dict: Validation report with issues and recommendations
    """
    issues = []
    valid = True
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        issues.append(f"Found {missing_values} missing values")
        valid = False
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate records")
        valid = False
    
    # Validate age distributions
    age_stats = df['age'].describe()
    if age_stats['min'] < 18 or age_stats['max'] > 100:
        issues.append(f"Age range unusual: {age_stats['min']}-{age_stats['max']}")
        valid = False
    
    # Validate BMI ranges
    bmi_outliers = df[(df['bmi'] < 15) | (df['bmi'] > 50)].shape[0]
    if bmi_outliers > df.shape[0] * 0.05:  # More than 5% outliers
        issues.append(f"Excessive BMI outliers: {bmi_outliers} records")
        valid = False
    
    # Validate biological age relationship
    bio_age_correlation = df['age'].corr(df['biological_age'])
    if bio_age_correlation < 0.5:  # Should be reasonably correlated
        issues.append(f"Poor age-biological_age correlation: {bio_age_correlation:.3f}")
        valid = False
    
    # Check gender distribution
    gender_dist = df['gender'].value_counts(normalize=True)
    if any(gender_dist < 0.3) or any(gender_dist > 0.7):
        issues.append("Gender distribution imbalanced")
        valid = False
    
    # Validate lifestyle factor ranges
    if df['exercise_frequency'].min() < 0 or df['exercise_frequency'].max() > 7:
        issues.append("Exercise frequency out of valid range (0-7)")
        valid = False
    
    if df['stress_level'].min() < 1 or df['stress_level'].max() > 10:
        issues.append("Stress level out of valid range (1-10)")
        valid = False
    
    if df['diet_quality'].min() < 1 or df['diet_quality'].max() > 10:
        issues.append("Diet quality out of valid range (1-10)")
        valid = False
    
    # Validate genetic markers
    if 'genetic_aging_score' in df.columns:
        genetic_score_range = df['genetic_aging_score'].max() - df['genetic_aging_score'].min()
        if genetic_score_range < 5:  # Should have reasonable variation
            issues.append("Genetic aging score lacks variation")
            valid = False
    
    if df['telomere_length'].min() < 2000 or df['telomere_length'].max() > 20000:
        issues.append("Telomere length out of realistic range")
        valid = False
    
    # Validate CpG methylation values
    methylation_cols = [col for col in df.columns if col.endswith('_methylation')]
    for col in methylation_cols:
        if df[col].min() < 0 or df[col].max() > 1:
            issues.append(f"CpG methylation {col} out of valid range (0-1)")
            valid = False
    
    # Validate SNP genotypes
    snp_cols = [col for col in df.columns if any(gene in col for gene in ['APOE', 'FOXO3', 'SIRT1', 'TP53', 'CDKN2A', 'TERT', 'TERC', 'IGF1', 'KLOTHO'])]
    for col in snp_cols:
        # Check that all genotypes are 2 characters and contain only valid nucleotides
        genotypes = df[col].unique()
        valid_nucleotides = {'A', 'T', 'G', 'C'}
        for genotype in genotypes:
            if len(genotype) != 2 or not all(nt in valid_nucleotides for nt in genotype):
                issues.append(f"Invalid genotype format in {col}: {genotype}")
                valid = False
                break
    
    return {
        'valid': valid,
        'issues': issues,
        'summary': {
            'n_records': len(df),
            'n_features': len(df.columns),
            'missing_values': missing_values,
            'duplicates': duplicates,
            'age_range': (df['age'].min(), df['age'].max()),
            'bio_age_range': (df['biological_age'].min(), df['biological_age'].max()),
            'age_correlation': bio_age_correlation
        }
    }


def _calculate_biological_age(age, gender, bmi, exercise_freq, sleep_hours,
                             stress_level, diet_quality, smoking, alcohol,
                             genetic_data, telomere_length, pollution,
                             systolic_bp, cholesterol):
    """Calculate biological age based on various factors"""
    
    # Start with chronological age
    bio_age = age * 0.85 + 4
    
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
    bio_age += genetic_data.get('genetic_aging_score', 0) * 0.5
    bio_age += (1 - telomere_length / 10000) * 8  # Shorter telomeres = older
    
    # CpG methylation effect (average of aging-sensitive sites)
    cpg_values = [v for k, v in genetic_data.items() if k.endswith('_methylation')]
    if cpg_values:
        avg_methylation = np.mean(cpg_values)
        bio_age += (avg_methylation - 0.5) * 10  # Deviation from baseline
    
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


def generate_test_datasets(datasets_dir='datasets'):
    """
    Generate multiple test datasets for different scenarios.
    
    Args:
        datasets_dir (str): Directory to save datasets (relative to current script)
    """
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    datasets_path = script_dir / datasets_dir
    datasets_path.mkdir(exist_ok=True)
    
    print(f"Generating datasets in: {datasets_path}")
    
    # Training dataset (main dataset)
    print("Generating training dataset...")
    train_data = generate_synthetic_data(
        n_samples=5000, 
        output_path=datasets_path / 'train.csv'
    )
    
    # Test datasets for different scenarios
    test_scenarios = [
        ('test_small.csv', {'n_samples': 100}),  # Small test set
        ('test_young.csv', {'age_range': (25, 40), 'n_samples': 200}),
        ('test_middle.csv', {'age_range': (40, 60), 'n_samples': 200}), 
        ('test_elderly.csv', {'age_range': (60, 80), 'n_samples': 200}),
        ('test_healthy.csv', {'lifestyle_bias': 'healthy', 'n_samples': 150}),
        ('test_unhealthy.csv', {'lifestyle_bias': 'unhealthy', 'n_samples': 150}),
    ]
    
    for filename, params in test_scenarios:
        print(f"Generating {filename}...")
        test_data = generate_specialized_dataset(**params)
        output_path = datasets_path / filename
        test_data.to_csv(output_path, index=False)
        print(f"Saved {len(test_data)} samples to {output_path}")
    
    # Generate summary report
    generate_dataset_summary(datasets_path)
    
    print("All datasets generated successfully!")
    return datasets_path


def generate_specialized_dataset(age_range=None, lifestyle_bias=None, n_samples=500, random_seed=42):
    """
    Generate specialized test datasets with specific characteristics.
    
    Args:
        age_range (tuple): Min and max age for filtering (optional)
        lifestyle_bias (str): 'healthy' or 'unhealthy' bias (optional)
        n_samples (int): Target number of samples
        random_seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Filtered specialized dataset
    """
    
    # Generate more samples initially to allow for filtering
    initial_samples = min(n_samples * 3, 10000)  
    data = generate_synthetic_data(initial_samples, random_seed=random_seed)
    
    if age_range:
        min_age, max_age = age_range
        data = data[(data['age'] >= min_age) & (data['age'] <= max_age)]
    
    if lifestyle_bias == 'healthy':
        # Bias towards healthier lifestyle choices
        data = data[
            (data['exercise_frequency'] >= 4) & 
            (data['stress_level'] <= 6) & 
            (data['diet_quality'] >= 7) & 
            (data['smoking'] == 0) &
            (data['alcohol_consumption'] <= 7)
        ]
    elif lifestyle_bias == 'unhealthy':
        # Bias towards unhealthier lifestyle choices
        data = data[
            (data['exercise_frequency'] <= 2) | 
            (data['stress_level'] >= 7) | 
                biological_age += np.random.normal(0, 3)
            (data['smoking'] == 1) |
            (data['alcohol_consumption'] >= 14)
        ]
    
    # Take only the requested number of samples (or all if fewer available)
    final_data = data.head(n_samples).reset_index(drop=True)
    
    if len(final_data) < n_samples * 0.8:  # Less than 80% of requested samples
        warnings.warn(f"Only generated {len(final_data)} samples, requested {n_samples}")
    
    return final_data


def generate_dataset_summary(datasets_path):
    """
    Generate a summary report of all datasets in the specified directory.
    
    Args:
        datasets_path (Path): Path to the datasets directory
    """
    
    summary_path = datasets_path / 'dataset_summary.md'
    
    with open(summary_path, 'w') as f:
        f.write("# Dataset Generation Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # List all CSV files and their basic stats
        csv_files = list(datasets_path.glob('*.csv'))
        
        f.write("## Dataset Overview\n\n")
        f.write("| Dataset | Samples | Features | Age Range | Bio Age Range |\n")
        f.write("|---------|---------|----------|-----------|---------------|\n")
        
        total_samples = 0
        
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                total_samples += len(df)
                
                age_min, age_max = df['age'].min(), df['age'].max()
                bio_age_min, bio_age_max = df['biological_age'].min(), df['biological_age'].max()
                
                f.write(f"| {csv_file.name} | {len(df)} | {len(df.columns)} | "
                       f"{age_min}-{age_max} | {bio_age_min:.1f}-{bio_age_max:.1f} |\n")
                
            except Exception as e:
                f.write(f"| {csv_file.name} | ERROR | - | - | - |\n")
                print(f"Error reading {csv_file}: {e}")
        
        f.write(f"\n**Total Samples:** {total_samples}\n")
        f.write(f"**Total Datasets:** {len(csv_files)}\n\n")
        
        # Add validation summary for main training dataset
        train_file = datasets_path / 'train.csv'
        if train_file.exists():
            f.write("## Training Dataset Validation\n\n")
            try:
                train_df = pd.read_csv(train_file)
                validation_report = validate_dataset(train_df)
                
                f.write(f"**Validation Status:** {'✅ PASSED' if validation_report['valid'] else '❌ FAILED'}\n\n")
                
                if validation_report['issues']:
                    f.write("### Issues Found:\n")
                    for issue in validation_report['issues']:
                        f.write(f"- {issue}\n")
                    f.write("\n")
                
                f.write("### Summary Statistics:\n")
                summary = validation_report['summary']
                f.write(f"- Records: {summary['n_records']}\n")
                f.write(f"- Features: {summary['n_features']}\n")
                f.write(f"- Missing Values: {summary['missing_values']}\n")
                f.write(f"- Duplicates: {summary['duplicates']}\n")
                f.write(f"- Age-BioAge Correlation: {summary['age_correlation']:.3f}\n")
                
            except Exception as e:
                f.write(f"Error validating training dataset: {e}\n")
        
        f.write("\n## Data Generation Process\n\n")
        f.write("This synthetic dataset was generated using enhanced algorithms that:\n")
        f.write("- Maintain realistic demographic distributions\n")
        f.write("- Include comprehensive lifestyle and health factors\n")
        f.write("- Simulate genetic/epigenetic markers\n")
        f.write("- Apply biologically-informed aging calculations\n")
        f.write("- Include data validation and quality checks\n")
    
    print(f"Dataset summary saved to: {summary_path}")


if __name__ == "__main__":
    # Generate enhanced training data with 5000 samples
    print("=== Anti-Aging ML Dataset Generation ===")
    
    # Generate individual training dataset for quick testing
    data = generate_synthetic_data(5000, 'train_data.csv')
    print(f"Training dataset shape: {data.shape}")
    print(f"Biological age stats: {data['biological_age'].describe()}")
    
    # Generate complete test suite
    print("\n=== Generating Complete Dataset Suite ===")
    datasets_dir = generate_test_datasets()
    
    print(f"\n✅ All datasets generated successfully in: {datasets_dir}")
    print("📊 Check dataset_summary.md for detailed statistics")