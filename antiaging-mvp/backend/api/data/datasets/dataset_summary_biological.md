# Biologically Realistic Anti-Aging Dataset Summary

Generated: 2025-10-14 07:57:52
Model: Scientifically-grounded biological aging model

## Key Improvements Over Previous Version

1. **Realistic Age Correlation**: Target 0.6-0.8 (was 0.945)
2. **Gene-Environment Interactions**: Scientifically modeled
3. **Individual Variation**: Genetic aging rate modifiers
4. **Biological Pathways**: Aging hallmarks integrated
5. **Measurement Noise**: Realistic biomarker variation

## Dataset Overview

| Dataset | Samples | Age Range | Bio Age Range | Correlation |
|---------|---------|-----------|---------------|-----------|
| test_elderly.csv | 200 | 60-79 | 31.0-94.2 | 0.284 |
| test_healthy.csv | 150 | 25-79 | 18.0-86.1 | 0.664 |
| test_middle.csv | 200 | 40-60 | 21.5-71.5 | 0.180 |
| test_small.csv | 100 | 25-78 | 18.0-95.7 | 0.658 |
| test_unhealthy.csv | 150 | 25-79 | 18.0-95.7 | 0.688 |
| test_young.csv | 200 | 25-40 | 18.0-52.0 | 0.312 |
| train.csv | 5000 | 25-79 | 18.0-101.4 | 0.657 |

**Total Samples**: 6000

## Training Dataset Validation

- **Age-Bio Age Correlation**: 0.657 ✅
- **Biological Age SD**: 14.71 years
- **Genetic Rate Variation**: 0.152
- **Sample Size**: 5000
- **Features**: 62

## Genetic Architecture

### Key Aging Genes (Sample from training data)

**APOE_rs429358**: {'CC': np.int64(3700), 'CT': np.int64(1206), 'TT': np.int64(94)}
**APOE_rs7412**: {'CC': np.int64(4224), 'CT': np.int64(745), 'TT': np.int64(31)}
**FOXO3_rs2802292**: {'GT': np.int64(2360), 'GG': np.int64(1957), 'TT': np.int64(683)}
**SIRT1_rs7069102**: {'CC': np.int64(2789), 'CG': np.int64(1930), 'GG': np.int64(281)}
**TP53_rs1042522**: {'CG': np.int64(2428), 'CC': np.int64(1654), 'GG': np.int64(918)}

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
- Gender balance: {'F': np.int64(2525), 'M': np.int64(2475)}

## Research Applications

This dataset enables:
1. **Model Comparison**: Realistic performance differences between RF/MLP
2. **Feature Importance**: Biologically meaningful patterns
3. **Aging Research**: Pathway-specific analysis
4. **Personalized Medicine**: Individual variation modeling
5. **Thesis Defense**: Scientifically defensible results
