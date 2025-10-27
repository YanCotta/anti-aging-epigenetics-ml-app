# LiveMore: A Startup MVP for Health ROI Simulation

**Product Vision:** An AI-powered platform that demonstrates the "Return on Investment" of healthy lifestyle choices through explainable biological age prediction.

**Current Focus:** Streamlit MVP for thesis defense (20-day sprint)

---

## 🎯 Project Overview

This repository represents a strategic pivot from building a complex production architecture to delivering a **rapid-prototyping MVP** that validates a core product hypothesis: **Can Explainable AI (XAI) effectively communicate "Health ROI" to users?**

The thesis defense will be presented as a **"Startup Pitch"** for "LiveMore," demonstrating the ability to:
- Validate product hypotheses quickly using modern prototyping tools (Streamlit)
- Build defensible intellectual property (synthetic data generation)
- Use advanced ML/XAI techniques to create user value (SHAP explanations)

## 🚀 Current Status

**Sprint Timeline:** 20 days to thesis defense  
**Current Phase:** Week 1 - Data Foundation (Issue #50)  
**Main Deliverable:** `antiaging-mvp/streamlit_app/` - Interactive Streamlit demo

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
│       ├── app.py               # Main application
│       ├── requirements.txt     # Clean dependencies (no ONNX/FastAPI)
│       └── README.md            # Local setup instructions
│
├── ml_pipeline/
│   └── data_generation/         # 🔬 CORE IP - Synthetic data engine
│       ├── generator_v2_biological.py
│       └── datasets_baseline_v2/
│
├── notebooks/                   # Validation notebooks
│
├── legacy/                      # 📦 ARCHIVED - Post-MVP production code
│   ├── backend_fastapi_archive/
│   ├── frontend_react_archive/
│   ├── nginx_archive/
│   ├── datasets_chaos_v1_invalid/
│   └── notebooks_archive/
│
└── docs/
    ├── PIVOT.md                 # 📋 STRATEGIC PLAN (Source of Truth)
    ├── PROJECT_STATUS_OCT_2025.md
    └── FUTURE_ROADMAP_POST_TCC.md
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

## 🔬 Core Intellectual Property: The Chaos Engine

The project's main technical innovation is the **synthetic data generation engine** (`ml_pipeline/data_generation/`), which creates biologically realistic datasets with:

- Genetic variation (10 aging-related SNPs)
- Epigenetic markers (20 CpG methylation sites)
- Lifestyle factors (exercise, diet, smoking, etc.)
- Controlled non-linear interactions
- Age-dependent variance modeling

**Current Status:** Implementing Issue #50 to generate `datasets_chaos_v2/` with sufficient non-linearity to demonstrate ML/XAI value.

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

**Week 1:** Data Foundation
- ✅ Archive production architecture to `/legacy`
- ⏳ Implement Issue #50 (strengthen data non-linearity)
- ⏳ Generate `datasets_chaos_v2/` passing validation criteria

**Week 2:** MVP Implementation
- Train Random Forest model on chaos_v2 data
- Build Streamlit UI (input sliders, prediction, SHAP viz)
- Polish product narrative and business language

**Week 3:** Thesis Preparation
- Create pitch deck using Streamlit demo as centerpiece
- Document technical decisions and future roadmap
- Prepare defense materials

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
