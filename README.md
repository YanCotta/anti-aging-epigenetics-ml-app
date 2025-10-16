# Anti-Aging Epigenetics ML Application

Product of my undergraduate thesis in Biological Sciences (BS) - A scientifically rigorous anti-aging epigenetics machine learning application capable of making personalized health recommendations and predictions based on individual genetic data and environmental factors.

## 🔄 Current Status: Strategic Pivot to Uncertainty Integration

We are executing a **professor-driven pivot** to inject explicit uncertainty, chaos, and stochastic interactions across the entire synthetic genomics pipeline. The goal is to simulate the unknown relationships between methylation sites, SNPs, lifestyle factors, environmental variables, and cohort assignments so that our models learn to cope with realistic ambiguity rather than perfectly clean data.

### Guidance Recap (October 16, 2025)
- Randomize and perturb interactions between every feature family (methylation ↔ SNP ↔ lifestyle ↔ biomarkers) to represent unknown regulatory mechanisms.
- Introduce cross-age uncertainty so outliers (e.g., unhealthy 20-year-olds who age well, resilient elderly cohorts) emerge naturally.
- Expand statistical diagnostics beyond point metrics: covariance drift, variance envelopes, residual distributions, entropy of feature importance, and cohort fairness summaries.
- Re-baseline all performance claims after Monte Carlo evaluations; treat previous EXCELLENT metrics as historical upper bounds only.

See [CHANGELOG.md](docs/CHANGELOG.md) for the full pivot log and [ROADMAP.md](docs/ROADMAP.md) for the updated execution plan.

> ℹ️ **Historical Baseline:** Sections below capture the publication-ready pipeline delivered on October 14, 2025. Treat all metrics as pre-chaos references until Monte Carlo uncertainty runs establish new confidence bands.

## 📋 Quick Links

- **🎯 Latest Achievements**: [CHANGELOG.md](docs/CHANGELOG.md)
- **📍 Roadmap & Next Steps**: [ROADMAP.md](docs/ROADMAP.md)
- **📋 Detailed Issues**: [DETAILED_ISSUES.md](docs/DETAILED_ISSUES.md)
- **🎓 Academic Overview**: [README_PROFESSORS.md](README_PROFESSORS.md)

## 🔬 Core Python Pipeline Components

### **Data Generation & Preprocessing**

#### **`generator_v2_biological.py`** - Scientifically Realistic Data Generation
**Purpose**: Generate biologically realistic synthetic aging data with proper gene-environment interactions
**Key Features**:
- Individual genetic aging rates (0.5-2.0x baseline)
- Gene-environment interactions (Exercise × FOXO3, etc.)
- Multi-pathway aging model (senescence, DNA repair, telomeres, metabolism)
- Hardy-Weinberg equilibrium compliance
- Realistic age-biological age correlation (0.657)

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python generator_v2_biological.py
# Generates 5,000 realistic samples + 6 specialized test sets
```

#### **`genomics_preprocessing.py`** - GWAS-Standard Preprocessing
**Purpose**: Comprehensive genomics preprocessing following established best practices
**Key Features**:
- Hardy-Weinberg equilibrium testing
- Population structure analysis (ancestry PCs)
- Feature-type aware scaling and encoding
- SNP quality control (call rate, MAF, HWE)
- Methylation probe validation

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python genomics_preprocessing.py
# Processes genetic data with 8 feature groups
```

#### **`genetic_qc.py`** - Genetic Quality Control
**Purpose**: Comprehensive genetic quality control following GWAS standards
**Key Features**:
- Sample-level QC (call rates, heterozygosity outliers)
- SNP-level QC (MAF filtering, HWE testing)
- Genetic relationship matrix calculation
- Population outlier detection
- Comprehensive QC reporting

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python genetic_qc.py
# Generates detailed QC report with recommendations
```

### **Advanced ML & Statistical Pipeline**

#### **`aging_benchmarks.py`** - Aging Clock Benchmarks
**Purpose**: Compare model performance against published aging clocks
**Key Features**:
- 5 published aging clocks (Horvath, Hannum, PhenoAge, GrimAge, Skin-Blood)
- Performance categorization (POOR to WORLD-CLASS)
- Literature-based comparison framework
- Realistic performance targets

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python aging_benchmarks.py
# Compares your model against published aging clocks
```

#### **`aging_features.py`** - Advanced Feature Engineering
**Purpose**: Biologically-informed feature engineering for aging prediction
**Key Features**:
- 8 feature engineering categories (pathway-based, polygenic risk, interactions)
- 19 new biologically-informed features
- Data leakage prevention
- Biological pathway database (Hallmarks of Aging)

**Usage**:
```python
from aging_features import AdvancedAgingFeatureEngineer
engineer = AdvancedAgingFeatureEngineer()
X_advanced = engineer.create_all_features(X_baseline, y_age)
```

#### **`statistical_rigor.py`** - Statistical Testing Framework
**Purpose**: Publication-ready statistical validation
**Key Features**:
- Bootstrap confidence intervals (n=2000)
- Permutation tests for feature importance (n=1000)
- Multiple testing correction (FDR, Bonferroni, Holm)
- Stratified cross-validation (5-fold)
- Effect size calculations (Cohen's d, Cliff's delta)

**Usage**:
```python
from statistical_rigor import StatisticalRigor
sr = StatisticalRigor(random_state=42)
results = sr.bootstrap_metric(y_true, y_pred, metric='r2', n_bootstrap=2000)
```

#### **`publication_ready_evaluation.py`** - Complete Evaluation Pipeline
**Purpose**: Run full statistical validation with all rigor
**Key Features**:
- Bootstrap CIs for all metrics
- Cross-validation with age stratification
- Permutation-tested feature importance with FDR correction
- Age-stratified performance analysis

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python publication_ready_evaluation.py
# Generates comprehensive publication-ready results
```

#### **`skeptical_analysis.py`** - Critical Results Examination
**Purpose**: Critical examination of model results for data quality issues
**Key Features**:
- 8-point skeptical analysis
- Feature dominance detection
- Overfitting assessment
- Performance vs literature comparison

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python skeptical_analysis.py
# Critically examines results for potential issues
```

#### **`genomics_ml_integration.py`** - End-to-End Pipeline
**Purpose**: Complete genomics-to-ML pipeline for aging prediction
**Key Features**:
- Integrates all preprocessing components
- Advanced aging-specific feature engineering
- Multiple ML model training (Linear, Random Forest, MLP)
- Genetic risk score calculation
- Methylation clock features
- Gene-environment interaction terms

**Usage**:
```bash
cd antiaging-mvp/backend/api/data
python genomics_ml_integration.py
# Runs complete pipeline: preprocessing → QC → feature engineering → ML training
```

**Current Performance** (with 95% Bootstrap CI):
- **Test R²**: 0.963 [0.960, 0.967], SE=0.0018
- **Test MAE**: 2.41 [2.29, 2.53] years
- **CV R²**: 0.960 ± 0.002 (5-fold stratified)
- **Performance Category**: EXCELLENT (exceeds published aging clocks)

## 📊 Realistic Synthetic Datasets

**6,000 total scientifically realistic samples** across 7 specialized datasets:

| Dataset | Samples | Purpose | Age Range | Features |
|---------|---------|---------|-----------|----------|
| `train.csv` | 5,000 | Main training dataset | 25-79 | 62 features |
| `test_small.csv` | 100 | Quick validation | 25-79 | 62 features |
| `test_young.csv` | 200 | Young adult testing | 25-40 | 62 features |
| `test_middle.csv` | 200 | Middle-age testing | 40-60 | 62 features |
| `test_elderly.csv` | 200 | Elderly testing | 60-79 | 62 features |
| `test_healthy.csv` | 200 | Healthy lifestyle bias | 25-79 | 62 features |
| `test_unhealthy.csv` | 200 | Risk factor analysis | 25-79 | 62 features |

### 🧬 Genetic Features (Scientifically Validated)
- **10 Aging-Related SNPs**: APOE, FOXO3, SIRT1, TP53, CDKN2A, TERT, TERC, IGF1, KLOTHO
- **20 CpG Methylation Sites**: Based on Horvath and Hannum aging clock research
- **Realistic Allele Frequencies**: 1000 Genomes Project-based population genetics
- **Age-Correlation**: 0.657 correlation (scientifically realistic range: 0.60-0.85)

## 🔬 Scientific Foundation

### **Biological Aging Model**
- **Multi-pathway approach**: Cellular senescence (25%), DNA repair (20%), Telomere maintenance (20%), Insulin/IGF signaling (35%)
- **Individual variation**: Realistic 6.0 SD biological aging differences
- **Measurement noise**: 5.0 SD biomarker measurement error
- **Gene-environment interactions**: 15+ scientifically based interaction terms

### **Population Genetics Compliance**
- **Hardy-Weinberg equilibrium**: Maintained across all SNPs
- **Linkage disequilibrium**: Realistic LD patterns
- **Population structure**: Principal components explain 31.6% variance
- **Quality control**: 100% SNPs and methylation probes pass QC thresholds

## 📈 Pipeline Execution Order

1. **Data Generation**: Run `generator_v2_biological.py` to create realistic datasets
2. **Preprocessing**: Use `genomics_preprocessing.py` for GWAS-standard processing
3. **Quality Control**: Execute `genetic_qc.py` for comprehensive QC analysis
4. **ML Pipeline**: Run `genomics_ml_integration.py` for complete training pipeline

## 🔐 Authentication System (FastAPI Backend)

**Fully implemented JWT-based authentication:**
- **User Registration**: `/signup` with password strength validation
- **Login**: `/token` OAuth2-compatible endpoint  
- **Protected Routes**: JWT token validation for all user endpoints
- **User Context**: `/me` endpoint for current user information
- **Security**: bcrypt password hashing, configurable JWT secrets

## Development Plan

See the comprehensive development plan in **[docs/ROADMAP.md](docs/ROADMAP.md)**.

**🚀 Implementation Ready:**
- **[docs/DETAILED_ISSUES.md](docs/DETAILED_ISSUES.md)** - Complete issue descriptions with acceptance criteria
- **[docs/ROADMAP.md](docs/ROADMAP.md)** - Architecture, timeline, and development strategy

### Quick Start

**Current Implementation Status:**
1. ✅ **Phase 1 Complete**: Synthetic datasets generated and validated
2. ✅ **Authentication**: JWT system fully implemented and tested
3. 🛠 **Phase 2 In Progress**: Critical scientific validity issues being addressed

**To Continue Development:**
1. Review current status and critical issues in [docs/ROADMAP.md](docs/ROADMAP.md)
2. Check detailed implementation progress in [docs/CHANGELOG.md](docs/CHANGELOG.md)
3. Address critical Issues #43-47 before resuming development
4. Follow comprehensive task specifications in [docs/DETAILED_ISSUES.md](docs/DETAILED_ISSUES.md)

**📖 Complete Documentation:**
- **[docs/ROADMAP.md](docs/ROADMAP.md)** - Comprehensive development plan, architecture, and current status
- **[docs/DETAILED_ISSUES.md](docs/DETAILED_ISSUES.md)** - All technical tasks and implementation specifications
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Complete implementation history and session logs
- **[docs/DOCUMENTATION_ORGANIZATION.md](docs/DOCUMENTATION_ORGANIZATION.md)** - Navigation guide and structure overview

## Architecture (implemented)

- **Backend**: `FastAPI` + `SQLAlchemy` + JWT (`python-jose`, `passlib`), served by `uvicorn` ✅
- **Authentication**: JWT token system with password hashing and OAuth2 flow ✅
- **Data**: PostgreSQL (Dockerized) for users/profiles/habits + synthetic datasets ✅
- **ML**: Random Forest (scikit-learn) and MLP (PyTorch) planned; preprocessing with sklearn
- **Explainability**: ONNX/SHAP planned for model explanations
- **Tracking**: `MLflow` integration planned for experiment management
- **Frontend**: Streamlit MVP for thesis defense; Next.js/React migration planned
- **Ops**: Docker Compose with services for `db`, `fastapi`, `streamlit`, `nginx`, and `mlflow`

## Services (docker-compose)

- `db`: Postgres database with health check.
- `fastapi`: API at `http://localhost:8001` (Swagger at `/docs`), mounted from `backend/`.
- `streamlit`: UI at `http://localhost:8501`, calls the FastAPI service.
- `mlflow`: Tracking UI at `http://localhost:5000` (artifacts volume-mounted).
- `nginx`: Proxy (kept from legacy; routing updates to FastAPI pending).

## Frontend Strategy

- Streamlit-first for rapid iteration and defense demo.
- Post-defense: migrate/expand to Next.js/React reusing the same API contracts.

## Roadmap Progress (from CHANGELOG.md)

### Phase 1: Setup + Data ✅ COMPLETE

- [x] ✅ Scale synthetic dataset to 5,000+ samples with validation
- [x] ✅ Generate 7 specialized datasets (5,851 total samples)
- [x] ✅ Implement 10 aging-related SNPs and 20 CpG methylation sites
- [x] ✅ Create automated validation pipeline with quality reports
- [x] ✅ Place curated datasets under `backend/api/data/datasets/`

### Phase 2: Backend + ML 🛠 IN PROGRESS

- [x] ✅ Scaffold FastAPI app and core endpoints (`/health`, `/signup`, `/token`, `/me`)
- [x] ✅ Implement JWT authentication system with password policies
- [x] ✅ Add comprehensive auth testing with SQLite test database
- [x] ✅ Add CORS configuration for frontend integration
- [x] ✅ Add MLflow tracking service to docker-compose
- [x] ✅ Add Streamlit MVP scaffold and service to docker-compose
- [x] ✅ Add FastAPI/SQLAlchemy/JWT/MLflow/Torch to backend requirements
- [⏳] 🛠 Create genetic data upload endpoints with schema validation (Issue #4)
- [ ] ⏳ Implement preprocessing pipeline alignment train/predict (Issue #5)
- [ ] ⏳ Train Random Forest baseline; export ONNX; add SHAP explanations (Issue #6)
- [ ] ⏳ Add MLP (PyTorch) and log both models to MLflow (Issue #7)
- [ ] ⏳ Wire prediction endpoint to load artifacts and return explanations (Issue #8)

### Phase 3: Frontend + Integration ⏳ PLANNED

- [⏳] Streamlit MVP integrated end-to-end (auth, upload, habits, predict)
- [x] ✅ Document Next.js/React migration plan in `docs/ROADMAP.md`

### Phase 4: Docker, Testing, Validation ⏳ PLANNED

- [x] ✅ Include MLflow in compose (service running on :5000)
- [ ] ⏳ Add health checks for app services and finalize NGINX routing to FastAPI
- [ ] ⏳ Pytest for ML and API (≥70% coverage) and basic load test for predict

### Phase 5: Thesis + Demo ⏳ PLANNED

- [ ] ⏳ MLflow screenshots, model comparison (RF vs MLP), ethics/limitations
- [ ] ⏳ Record demo video and prepare slides

### Backlog / Infrastructure ⏳ PLANNED

- [ ] ⏳ Migrate persistence fully from Django ORM to SQLAlchemy models
- [ ] ⏳ Replace Django container command with Uvicorn

**Legend**: ✅ complete, 🛠 in progress, ⏳ planned, ❌ blocked
