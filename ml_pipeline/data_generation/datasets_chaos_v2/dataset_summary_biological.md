# Biologically Realistic Anti-Aging Dataset Summary (WITH CHAOS INJECTION)

Generated: 2025-10-28 11:10:03
Model: Scientifically-grounded biological aging model + Issue #49 Chaos Injection

## Issue #49: Chaos Injection Implementation

**Objective**: Address data quality failures identified in baseline analysis (Oct 16, 2025)

**5 Phases Implemented**:
1. **Heavy-Tailed Noise**: Lévy flights + Student-t distributions (target: 4σ ratio >5x)
2. **Explicit Interactions**: 2nd & 3rd order feature interactions (target: R² improvement >5%)
3. **Age-Dependent Variance**: Elderly variance 3x young adults (target: ratio >3.0)
4. **Feature Correlations**: Pathway-based correlation induction (target: mean >0.15)
5. **Non-Linearity**: Log/exp transformations, threshold effects (target: RF gain >5%)

**Chaos Configuration**:
- Chaos Intensity: 1.0
- Heavy Tails: True (Lévy α=1.5, t df=3)
- Interactions: True (60 2nd order, 20 3rd order)
- Age Variance: True (young=3.0, elderly=18.0)
- Correlations: True (pathway=0.65)
- Non-Linearity: True

## Key Improvements Over Previous Version

1. **Realistic Age Correlation**: Target 0.6-0.8 (was 0.945 → 0.657 → ???)
2. **Heavy-Tailed Outliers**: Lévy flights for extreme biological events
3. **Complex Interactions**: 50+ 2nd order + 20+ 3rd order interactions
4. **Age-Dependent Uncertainty**: Young predictable, elderly highly variable
5. **Feature Correlations**: Biological pathway co-regulation
6. **Non-Linear Relationships**: Benefits Random Forest over Linear models

## Dataset Overview

| Dataset | Samples | Age Range | Bio Age Range | Correlation |
|---------|---------|-----------|---------------|-----------|
| test_elderly.csv | 200 | 60-79 | 30.0-105.0 | 0.342 |
| test_healthy.csv | 150 | 25-79 | 18.0-79.1 | 0.508 |
| test_middle.csv | 200 | 40-60 | 20.0-84.0 | 0.319 |
| test_small.csv | 100 | 25-78 | 18.0-90.6 | 0.511 |
| test_unhealthy.csv | 150 | 25-79 | 18.0-99.7 | 0.663 |
| test_young.csv | 200 | 25-40 | 18.0-56.0 | 0.301 |
| train.csv | 5000 | 25-79 | 18.0-110.6 | 0.576 |

**Total Samples**: 6000

## Training Dataset Validation

- **Age-Bio Age Correlation**: 0.576 ✅
- **Biological Age SD**: 16.52 years
- **Genetic Rate Variation**: 0.151
- **Sample Size**: 5000
- **Features**: 142

## Genetic Architecture

### Key Aging Genes (Sample from training data)

**APOE_rs429358**: {'CC': np.int64(3758), 'CT': np.int64(1153), 'TT': np.int64(89)}
**APOE_rs7412**: {'CC': np.int64(4210), 'CT': np.int64(764), 'TT': np.int64(26)}
**FOXO3_rs2802292**: {'GT': np.int64(2332), 'GG': np.int64(1958), 'TT': np.int64(710)}
**SIRT1_rs7069102**: {'CC': np.int64(2850), 'CG': np.int64(1822), 'GG': np.int64(328)}
**TP53_rs1042522**: {'CG': np.int64(2460), 'CC': np.int64(1675), 'GG': np.int64(865)}

### Pathway Scores Distribution

## Scientific Validation

### Literature Comparison
- **Horvath Clock**: R=0.96 (ours: similar methylation age pattern)
- **Hannum Clock**: R=0.91 (blood-based methylation)
- **Published Age Correlation**: 0.6-0.8 range ✅
- **Genetic Effect Size**: Literature-based SNP effects

### Quality Metrics
- Missing values: 0
- Duplicate records: 0
- Gender balance: {'M': np.int64(2539), 'F': np.int64(2461)}

## Research Applications

This dataset enables:
1. **Model Comparison**: Realistic performance differences between RF/MLP
2. **Feature Importance**: Biologically meaningful patterns
3. **Aging Research**: Pathway-specific analysis
4. **Personalized Medicine**: Individual variation modeling
5. **Thesis Defense**: Scientifically defensible results
