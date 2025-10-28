# LiveMore: AI-Powered Health ROI Simulator (MVP)

**Product Vision:** An AI-powered platform that demonstrates the "Return on Investment" of healthy lifestyle choices through explainable biological age prediction.

**Current Status:** ✅ Model artifacts trained | 🏗️ Streamlit app development in progress  
**Timeline:** 20-day sprint to thesis defense (Week 1 complete)

---

## 🎯 Project Overview

This repository represents a strategic pivot from building a complex production architecture to delivering a **rapid-prototyping MVP** that validates a core product hypothesis: **Can Explainable AI (XAI) effectively communicate "Health ROI" to users?**

**The Pivot (October 28, 2025):**
After multiple scientific chaos injection attempts (Issues #49-50), we adopted a **business-pragmatic approach** focused on demonstrating value within thesis constraints. The result: a simplified dataset with explicit non-linear patterns where Random Forest meaningfully outperforms Linear Regression (+3-4% R² improvement).

The thesis defense will be presented as a **"Startup Pitch"** for "LiveMore," demonstrating the ability to:
- ✅ Generate defensible synthetic biological aging data (proprietary methods)
- ✅ Build ML models with explainable predictions (SHAP-based XAI)
- ✅ Validate product hypotheses quickly using modern prototyping (Streamlit)
- ✅ Balance scientific rigor with business practicality (MVP mindset)

## 🚀 Current Status

**Sprint Timeline:** 20 days to thesis defense (Day 6/20)  
**Current Phase:** ✅ Week 1 Complete - Data & Models Ready | 🏗️ Week 2 Starting - Streamlit Development  
**Main Deliverable:** `antiaging-mvp/streamlit_app/` - Interactive Streamlit demo

**Week 1 Achievements (Oct 22-28):**
- ✅ Generated `datasets_livemore_mvp/` with business-focused non-linear patterns
- ✅ Trained Random Forest model (R²=0.95, MAE=2.02 years on training)
- ✅ Created SHAP explainer for model interpretability
- ✅ Saved model artifacts to `antiaging-mvp/streamlit_app/app_model/`
- ✅ Documented pivot rationale and business approach

### Strategic Focus

The project is currently focused on the **Streamlit MVP**. The production architecture (FastAPI backend, React frontend, ONNX export, Docker deployment) has been **moved to `/legacy`** and represents the post-validation roadmap.

**Why this approach?**
- ✅ Demonstrates technical leadership and product thinking
- ✅ Validates core hypothesis before heavy engineering investment
- ✅ Delivers working demo within thesis constraints
- ✅ Preserves future scalability path (see `docs/FUTURE_ROADMAP_POST_TCC.md`)

## 📂 Repository Structure

```
anti-aging-epigenetics-ml-app/
├── antiaging-mvp/
│   └── streamlit_app/           # 🎯 MAIN DELIVERABLE - Streamlit MVP
│       ├── app.py               # Main application (TO BE BUILT)
│       ├── app_model/           # ✅ Trained artifacts
│       │   ├── livemore_rf_v2.joblib      # Random Forest model
│       │   ├── livemore_scaler_v2.joblib  # Feature scaler
│       │   ├── livemore_explainer_v2.pkl  # SHAP explainer
│       │   └── model_metadata.json        # Model info
│       ├── requirements.txt     # Dependencies
│       └── README.md            # Setup instructions
│
├── ml_pipeline/                 # 🔬 ML Development Pipeline
│   ├── data_generation/         # Data generation engine
│   │   ├── generator_mvp_simple.py        # ✅ Business-focused generator
│   │   ├── generator_v2_biological.py     # Scientific chaos engine
│   │   ├── datasets_livemore_mvp/         # ✅ ACTIVE dataset
│   │   └── datasets_chaos_v2/             # Attempted scientific approach
│   ├── train_model_mvp.py       # ✅ Model training script
│   ├── quick_validation_mvp.py  # ✅ Rapid RF vs Linear validation
│   └── models/                  # Model evaluation utilities
│
├── notebooks/                   # Analysis & validation notebooks
│   └── baseline_figures/        # Visualization outputs
│
├── legacy/                      # 📦 ARCHIVED - Post-MVP production code
│   ├── backend_fastapi_archive/
│   ├── frontend_react_archive/
│   ├── datasets_chaos_v1_invalid/  # Failed attempt
│   └── notebooks_archive/
│
└── docs/                        # 📚 Comprehensive documentation
    ├── PIVOT.md                 # Strategic pivot rationale
    ├── PROJECT_STATUS_OCT_2025.md  # Detailed status & findings
    ├── INDEX.md                 # Documentation navigation
    └── FUTURE_ROADMAP_POST_TCC.md  # Post-thesis plans
```

## 🎓 The "LiveMore" Product Concept

**Problem:** People don't understand how lifestyle choices impact their biological aging

**Solution:** An AI-powered simulator that:
1. Predicts biological age from lifestyle/genetic factors
2. Explains each factor's impact using SHAP (Explainable AI)
3. Presents results as "Health ROI" (business-friendly language)

**Technical Approach:**
- Synthetic biological aging data (proprietary "Chaos Engine")
- Random Forest model for non-linear pattern detection
- SHAP for transparent, interpretable explanations
- Streamlit for rapid iteration and user testing

## 🔬 Core Intellectual Property: Synthetic Aging Data Generation

The project's main technical innovation is the **synthetic data generation engine** (`ml_pipeline/data_generation/`), which creates business-relevant aging datasets with:

**Scientific Approach** (`generator_v2_biological.py`):
- 10 aging-related SNPs with Hardy-Weinberg equilibrium
- 20 CpG methylation sites mimicking epigenetic clocks
- 5-phase chaos injection (heavy-tailed noise, interactions, age-variance)
- **Result:** Too complex for MVP timeline, archived as research foundation

**Business Approach** (`generator_mvp_simple.py` - ACTIVE):
- ✅ Simplified 9-feature model (age, gender, 6 lifestyle factors, genetic risk)
- ✅ Explicit non-linear patterns (thresholds, exponentials, U-curves)
- ✅ Strong interaction effects (smoking×stress, exercise×diet)
- ✅ Demonstrates clear RF advantage over Linear Regression
- ✅ Suitable for MVP demonstration and thesis defense

**Current Dataset:** `datasets_livemore_mvp/` - 5000 training samples, 3 test sets

## 🏃 Quick Start (Streamlit MVP)

Navigate to the main deliverable:

```bash
cd antiaging-mvp/streamlit_app/
```

Follow the setup instructions in [`antiaging-mvp/streamlit_app/README.md`](antiaging-mvp/streamlit_app/README.md)

**Prerequisites:**
- Python 3.8+
- Virtual environment (recommended)
- Trained model artifacts (generated via Issue #50 workflow)

## 📊 Development Approach

### Current Sprint (20 Days)

**Week 1: Data Foundation** ✅ COMPLETE (Oct 22-28)
- ✅ Archive production architecture to `/legacy`
- ✅ Attempted Issue #49-50 (scientific chaos injection)
- ✅ Pivoted to business-pragmatic approach
- ✅ Generated `datasets_livemore_mvp/` with meaningful RF advantage
- ✅ Trained Random Forest model (R²=0.95, 9 features)
- ✅ Created SHAP explainer and model artifacts

**Week 2: MVP Implementation** 🏗️ IN PROGRESS (Oct 29 - Nov 4)
- Build Streamlit UI with sidebar inputs (age, lifestyle factors)
- Implement prediction display with biological age result
- Add SHAP waterfall/force plots for explanation
- Polish UI with business-friendly "Health ROI" language
- Test end-to-end user flow

**Week 3: Thesis Preparation** ⏳ UPCOMING (Nov 5-11)
- Create pitch deck using Streamlit demo as centerpiece
- Document pivot rationale and technical decisions
- Prepare defense materials with business narrative
- Practice demo and Q&A scenarios

### Post-Thesis Roadmap

See [`docs/FUTURE_ROADMAP_POST_TCC.md`](docs/FUTURE_ROADMAP_POST_TCC.md) for the complete production architecture plan:
- FastAPI REST API backend
- React/Next.js frontend
- ONNX model export for production inference
- PostgreSQL for user data
- Docker deployment
- Authentication and security

## �� Documentation

### Strategic Documents (Read These First!)
- **[docs/PIVOT.md](docs/PIVOT.md)** - The complete strategic plan (20-day sprint)
- **[docs/PROJECT_STATUS_OCT_2025.md](docs/PROJECT_STATUS_OCT_2025.md)** - Current status and Issue #50 justification

### Technical Documentation
- **[antiaging-mvp/streamlit_app/README.md](antiaging-mvp/streamlit_app/README.md)** - MVP setup and architecture
- **[ml_pipeline/data_generation/README.md](ml_pipeline/data_generation/README.md)** - Data generation engine docs

### Reference Documentation
- **[docs/FUTURE_ROADMAP_POST_TCC.md](docs/FUTURE_ROADMAP_POST_TCC.md)** - Post-MVP production roadmap
- **[README_PROFESSORS.md](README_PROFESSORS.md)** - Academic presentation for thesis committee

## 🎯 Success Metrics

### MVP Success Criteria
- [ ] Working Streamlit app with local deployment
- [ ] Random Forest model outperforms Linear Regression by >5%
- [ ] SHAP explanations clearly show factor importance
- [ ] UI uses business-friendly "Health ROI" language
- [ ] Complete thesis pitch deck with live demo

### Technical Validation
- [ ] `datasets_chaos_v2/` passes non-linearity validation
- [ ] Model artifacts (<50MB) load quickly in Streamlit
- [ ] User can input lifestyle factors and see instant results
- [ ] SHAP waterfall/force plots render correctly

## 🔐 What's NOT in Scope (For This Thesis)

The following were **deliberately moved to `/legacy`** to maintain sprint focus:

- ❌ FastAPI backend implementation
- ❌ React frontend development  
- ❌ ONNX model export
- ❌ Docker containerization
- ❌ Database integration (PostgreSQL)
- ❌ User authentication (JWT)
- ❌ Production deployment
- ❌ Load testing and optimization

**Rationale:** These represent the post-validation engineering roadmap. Building them now would consume the thesis timeline without validating the core hypothesis.

## 🤝 Academic Context

**Institution:** Undergraduate Thesis in Biological Sciences  
**Focus:** Demonstrating technical leadership through product validation  
**Approach:** Startup methodology applied to academic research  
**Timeline:** 20-day sprint to defense

For the academic/professor-oriented view, see [README_PROFESSORS.md](README_PROFESSORS.md).

## 📝 License

See [LICENSE](LICENSE) file for details.

---

**Last Updated:** October 27, 2025  
**Project Phase:** Streamlit MVP Sprint (Week 1 - Data Foundation)  
**Next Milestone:** Issue #50 completion and `datasets_chaos_v2/` validation
