# Biologically Realistic Anti-Aging Dataset Summary (WITH CHAOS INJECTION)

Generated: 2025-10-21 14:15:23
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
- Age Variance: True (young=2.0, elderly=6.0)
- Correlations: True (pathway=0.4)
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
| test_elderly.csv | 200 | 60-79 | 30.1-88.7 | 0.304 |
| test_healthy.csv | 150 | 25-79 | 18.0-78.8 | 0.663 |
| test_middle.csv | 200 | 40-60 | 20.5-72.7 | 0.407 |
| test_small.csv | 100 | 25-79 | 18.0-100.8 | 0.580 |
| test_unhealthy.csv | 150 | 25-79 | 18.0-100.8 | 0.616 |
| test_young.csv | 200 | 25-40 | 18.0-53.2 | 0.251 |
| train.csv | 5000 | 25-79 | 18.0-109.2 | 0.612 |

**Total Samples**: 6000

## Training Dataset Validation

- **Age-Bio Age Correlation**: 0.612 ✅
- **Biological Age SD**: 14.65 years
- **Genetic Rate Variation**: 0.155
- **Sample Size**: 5000
- **Features**: 142

## Genetic Architecture

### Key Aging Genes (Sample from training data)

**APOE_rs429358**: {'CC': np.int64(3663), 'CT': np.int64(1252), 'TT': np.int64(85)}
**APOE_rs7412**: {'CC': np.int64(4234), 'CT': np.int64(734), 'TT': np.int64(32)}
**FOXO3_rs2802292**: {'GT': np.int64(2338), 'GG': np.int64(1972), 'TT': np.int64(690)}
**SIRT1_rs7069102**: {'CC': np.int64(2808), 'CG': np.int64(1879), 'GG': np.int64(313)}
**TP53_rs1042522**: {'CG': np.int64(2460), 'CC': np.int64(1681), 'GG': np.int64(859)}

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
- Gender balance: {'M': np.int64(2508), 'F': np.int64(2492)}

## Research Applications

This dataset enables:
1. **Model Comparison**: Realistic performance differences between RF/MLP
2. **Feature Importance**: Biologically meaningful patterns
3. **Aging Research**: Pathway-specific analysis
4. **Personalized Medicine**: Individual variation modeling
5. **Thesis Defense**: Scientifically defensible results
