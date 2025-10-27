# Data Generation Engine (Chaos Engine)

This module is the core **"IP" (Intellectual Property)** of the project. It generates synthetic biological aging data with controlled complexity and non-linearity.

## Overview

The data generation engine creates realistic synthetic datasets that simulate the complex relationships between genetic factors, epigenetic markers, lifestyle choices, and biological aging.

## Key Components

- **`generator_v2_biological.py`**: Main script for synthetic data generation with chaos injection
- **`datasets_baseline_v2/`**: Clean baseline datasets (no chaos injection) - used for comparison
- **`datasets_chaos_v1/`**: **MOVED TO `/legacy/datasets_chaos_v1_invalid/`** ❌

## Status: Datasets Chaos v1 - INVALID

The `datasets_chaos_v1/` was archived because it **failed non-linearity validation**, as documented in `docs/PROJECT_STATUS_OCT_2025.md`.

**Key Failure Metrics:**
- **RF vs Linear Gain**: -1.82% (Random Forest performed WORSE than Linear Regression)
- **Target**: >5% improvement to justify advanced ML/XAI
- **Root Cause**: Insufficient interaction strength, age-dependent variance, and pathway correlations

**Validation Results:**
```
Linear Regression: R² = 0.5106, MAE = 8.42 years
Random Forest:     R² = 0.5012, MAE = 8.49 years (UNDERPERFORMS!)
```

## Next Steps: Issue #50

The **priority task** is implementing Issue #50 to generate `datasets_chaos_v2/` with:

1. **Increased `interaction_strength`**: 1.0 → 2.5-3.0
2. **Boosted `elderly_noise_scale`**: 6.0 → 12.0-15.0
3. **Strengthened `pathway_correlation`**: 0.4 → 0.6-0.7
4. **Added non-linear interaction terms**: exp(), log(), thresholds (not just multiplication)

**Target Metrics for v2:**
- RF vs Linear Gain: **>5%**
- Age-variance ratio: **>3.0**
- Mean correlation: **>0.15**

## Why This Matters

Without datasets that demonstrate non-linear complexity, the Streamlit MVP cannot showcase the value of:
- Advanced ML models (Random Forest)
- Explainable AI (SHAP)
- The "LiveMore" product hypothesis

**datasets_chaos_v2 is a prerequisite** for the entire MVP demonstration.

## References

- **Validation Report**: `docs/PROJECT_STATUS_OCT_2025.md`
- **Strategic Plan**: `docs/PIVOT.md`
- **Invalid Data Archive**: `/legacy/datasets_chaos_v1_invalid/`
