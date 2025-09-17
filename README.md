# anti-aging-epigenetics-ml-app

Product of my undergrad thesis in Biological Sciences (BS) - An anti-aging epigenetics machine learning based software capable of making personalized health recommendations and predictions based on individual genetic data and environmental exposure.

## 🚀 Current Implementation Status

**Phase 1: ✅ COMPLETE** - Synthetic Dataset Generation & Validation
- **5,851 synthetic samples** across 7 datasets with realistic genetic markers
- **53 features** including 10 aging-related SNPs and 20 CpG methylation sites  
- **Comprehensive validation pipeline** with automated quality reports
- **Datasets:** `train.csv` (5,000 samples) + 6 specialized test sets

**Phase 2: 🛠 IN PROGRESS** - Backend & Authentication
- **✅ FastAPI Authentication System** - JWT tokens, password hashing, OAuth2 flow
- **✅ Core Endpoints** - `/signup`, `/token`, `/me`, `/health` with full testing
- **✅ Security Features** - Password policies, CORS configuration, token validation
- **⏳ Data Upload** - Endpoints exist, schema validation in progress

See [CHANGELOG.md](docs/CHANGELOG.md) for detailed progress tracking and [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) for complete issue breakdown.

## 📊 Synthetic Datasets Generated

**5,851 total synthetic samples** across 7 specialized datasets with realistic genetic and lifestyle features:

| Dataset | Samples | Purpose | Age Range | Features |
|---------|---------|---------|-----------|----------|
| `train.csv` | 5,000 | Main training dataset | 25-79 | 53 features |
| `test_small.csv` | 100 | Quick validation | 25-79 | 53 features |
| `test_young.csv` | 188 | Young adult testing | 25-40 | 53 features |
| `test_middle.csv` | 200 | Middle-age testing | 40-60 | 53 features |
| `test_elderly.csv` | 200 | Elderly testing | 60-79 | 53 features |
| `test_healthy.csv` | 13 | Healthy lifestyle bias | 41-75 | 53 features |
| `test_unhealthy.csv` | 150 | Risk factor analysis | 25-79 | 53 features |

### 🧬 Genetic Features
- **10 Aging-Related SNPs**: APOE, FOXO3, SIRT1, TP53, CDKN2A, TERT/TERC, IGF1, KLOTHO
- **20 CpG Methylation Sites**: Based on Horvath and Hannum aging clocks
- **Realistic Allele Frequencies**: Population genetics compliance
- **Age-Correlation**: 0.958 correlation between age and biological age

## 🔐 Authentication System

**Fully implemented JWT-based authentication:**
- **User Registration**: `/signup` with password strength validation
- **Login**: `/token` OAuth2-compatible endpoint
- **Protected Routes**: JWT token validation for all user endpoints
- **User Context**: `/me` endpoint for current user information
- **Security**: bcrypt password hashing, configurable JWT secrets

## Development Plan

See the finalized roadmap in `DEV_PLAN.md`.

**🚀 Ready for Implementation:**
- **`GITHUB_ISSUES.md`** - 20 actionable GitHub issues with implementation strategy
- **`DETAILED_ISSUES.md`** - Complete issue descriptions with acceptance criteria  
- **`github_issues.json`** - Import-ready format for GitHub API/manual creation
- **`DEV_PLAN.md`** - Original comprehensive development plan and context

### Quick Start

**Current Implementation Status:**
1. ✅ **Phase 1 Complete**: Synthetic datasets generated and validated
2. ✅ **Authentication**: JWT system fully implemented and tested
3. 🛠 **Phase 2 In Progress**: Data upload endpoints being enhanced

**To Continue Development:**
1. Review implementation progress in [CHANGELOG.md](docs/CHANGELOG.md)
2. Check issue status in [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)
3. Import remaining GitHub issues from [DETAILED_ISSUES.md](docs/DETAILED_ISSUES.md)
4. Continue with Phase 2 data validation (Issue #4) and ML pipeline (Issues #5-8)
5. Follow the 5-phase timeline for systematic MVP development

**Documentation Structure:**
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Detailed progress tracking with evidence
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - 20 GitHub issues breakdown  
- **[DETAILED_ISSUES.md](docs/DETAILED_ISSUES.md)** - Complete issue descriptions with acceptance criteria
- **[DEV_PLAN.md](docs/DEV_PLAN.md)** - Original comprehensive development plan

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
- [x] ✅ Document Next.js/React migration plan in `DEV_PLAN.md`

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