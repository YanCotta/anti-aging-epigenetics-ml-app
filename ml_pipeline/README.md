# ML Pipeline

Core machine learning pipeline for the Anti-Aging Epigenetics ML Application.

## Structure

```
ml_pipeline/
├── data_generation/     # Synthetic data generation (Issue #49: Chaos Injection)
│   ├── generator_v2_biological.py    # Main data generator with chaos injection
│   ├── genomics_preprocessing.py     # Genomic data preprocessing
│   ├── genetic_qc.py                 # Quality control for genetic data
│   └── validation.py                 # Data validation utilities
│
├── models/              # ML model training and evaluation
│   ├── train_linear.py              # Linear regression baseline
│   ├── aging_benchmarks.py          # Random Forest, MLP benchmarks
│   ├── aging_features.py            # Feature engineering
│   ├── preprocessor.py              # Data preprocessing
│   ├── predict.py                   # Model prediction utilities
│   ├── statistical_rigor.py         # Statistical validation
│   ├── skeptical_analysis.py        # Critical analysis tools
│   └── publication_ready_evaluation.py  # Publication-ready metrics
│
└── evaluation/          # Model evaluation and comparison
```

## Quick Start

### Generate Chaos-Injected Datasets (Issue #49)

```bash
cd ml_pipeline/data_generation

# Full chaos injection (default)
python generator_v2_biological.py --output-dir datasets_chaos_v1

# Baseline comparison (no chaos)
python generator_v2_biological.py --no-chaos --output-dir datasets_baseline_v2

# Custom chaos intensity
python generator_v2_biological.py --chaos-intensity 0.5 --output-dir datasets_test
```

### Train Models

```bash
cd ml_pipeline/models

# Linear regression baseline
python train_linear.py --data-path ../data_generation/datasets_chaos_v1/train.csv

# Random Forest + MLP benchmarks
python aging_benchmarks.py --data-path ../data_generation/datasets_chaos_v1/train.csv
```

## Issue #49: Multi-Layer Chaos Injection

The data generator now implements 5 phases of chaos injection to address data quality issues:

1. **Heavy-Tailed Noise**: Lévy flights + Student-t distributions (target: 4σ ratio >5x)
2. **Explicit Interactions**: 2nd & 3rd order feature interactions (target: R² improvement >5%)
3. **Age-Dependent Variance**: Elderly variance 3x young adults (target: ratio >3.0)
4. **Feature Correlations**: Pathway-based correlations (target: mean >0.15)
5. **Non-Linearity**: Log/exp transformations (target: RF gain >5%)

See `data_generation/generator_v2_biological.py` for implementation details.

## Documentation

- Full documentation: `/docs/`
- Project status: `/docs/PROJECT_STATUS_OCT_2025.md`
- Changelog: `/docs/CHANGELOG.md`
- Roadmap: `/docs/ROADMAP.md`
