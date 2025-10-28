# ML Pipeline - LiveMore MVP Development

**Purpose:** Machine learning development pipeline for biological age prediction and explainable AI

**Current Status:** Week 1 Complete - Models Trained | Week 2 Starting - Streamlit Integration

---

## 📂 Directory Structure

```
ml_pipeline/
├── data_generation/
│   ├── generator_mvp_simple.py          # ✅ ACTIVE: Business-focused data generator
│   ├── generator_v2_biological.py       # Scientific chaos engine (research)
│   ├── datasets_livemore_mvp/           # ✅ ACTIVE: Current MVP dataset
│   │   ├── train.csv (5000 samples, 10 features)
│   │   ├── test_young_healthy.csv (51)
│   │   ├── test_middle_unhealthy.csv (50)
│   │   └── test_general.csv (1000)
│   └── datasets_chaos_v2/               # Scientific attempt (archived)
│
├── train_model_mvp.py                   # ✅ Model training script
├── quick_validation_mvp.py              # ✅ RF vs Linear validation
│
├── models/                              # Model utilities & evaluation
│   ├── aging_benchmarks.py
│   ├── aging_features.py
│   ├── multivariate_analysis.py
│   └── publication_ready_evaluation.py
│
└── evaluation/                          # Performance analysis tools
```

---

## 🎯 Key Scripts

### 1. `train_model_mvp.py` - Model Training Pipeline

**Purpose:** Train Random Forest on MVP dataset and save artifacts for Streamlit app

**Usage:**
```bash
cd ml_pipeline
python3 train_model_mvp.py
```

**What it does:**
1. Loads `datasets_livemore_mvp/train.csv`
2. Trains Random Forest (n_estimators=200, max_depth=15)
3. Creates SHAP TreeExplainer for interpretability
4. Saves artifacts to `../antiaging-mvp/streamlit_app/app_model/`

**Performance:** Training R²=0.95, MAE=2.02 years

---

### 2. `quick_validation_mvp.py` - Rapid Model Comparison

**Purpose:** Quick RF vs Linear validation

**Usage:**
```bash
cd ml_pipeline
python3 quick_validation_mvp.py
```

---

### 3. Data Generation

#### `generator_mvp_simple.py` ✅ ACTIVE

Business-focused approach with 9 features and explicit non-linear patterns.

**Usage:**
```bash
cd ml_pipeline/data_generation
python3 generator_mvp_simple.py
```

#### `generator_v2_biological.py` - Research Version

Complex 142-feature chaos injection engine (Issues #49-50). Generated `datasets_chaos_v2/` but RF underperformed. Archived as research foundation.

---

## 📚 Related Documentation

- [Main README](../README.md) - Project overview  
- [Streamlit App README](../antiaging-mvp/streamlit_app/README.md) - App setup  
- [PIVOT.md](../docs/PIVOT.md) - Strategic pivot rationale  
- [PROJECT_STATUS_OCT_2025.md](../docs/PROJECT_STATUS_OCT_2025.md) - Detailed status

---

**Last Updated:** October 28, 2025  
**Status:** Week 1 Complete ✅ | Week 2 Starting 🏗️
