
# Final Development Plan: Anti-Aging ML App (FastAPI + RF/NN + MLFlow)

This is the single, authoritative roadmap for building the MVP and preparing thesis materials. It consolidates project decisions: switch to FastAPI for the API layer, add a neural network alongside the Random Forest, and integrate MLFlow for experiment tracking. We keep what still serves the MVP (React/Next.js later, Postgres, Docker, ONNX/SHAP, synthetic data). Timeline remains ~6 weeks with buffer.

## 🎯 Actionable GitHub Issues

**This development plan has been converted into 20 specific, actionable GitHub issues.**

- **Quick Start:** See `GITHUB_ISSUES.md` for an organized overview and implementation strategy
- **Detailed Issues:** See `DETAILED_ISSUES.md` for complete issue descriptions with acceptance criteria
- **Import Ready:** Use `github_issues.json` to import issues directly into GitHub via API
- **Labels & Milestones:** Pre-defined label system and milestone structure for project organization

Each issue includes:
- Clear acceptance criteria and definition of done
- Implementation notes and technical guidance  
- File modification lists and dependencies
- Priority levels and phase assignments
- Milestone alignment with timeline below

## Goals

- Deliver a secure, explainable ML MVP where users upload genetics + submit habits and receive a biological-age estimate and recommendations with explanations.
- Implement two ML baselines: Random Forest and a simple MLP. Track experiments with MLFlow.
- Serve predictions through a FastAPI backend (Django code remains during transition but will be replaced).

## Architecture

- Backend: FastAPI + SQLAlchemy + JWT (python-jose + passlib). Uvicorn as ASGI server.
- ML: scikit-learn RF + PyTorch MLP, preprocessing with sklearn; ONNX for inference; SHAP for explanations; MLFlow for tracking.
- Frontend: React (existing app) via Axios; endpoints aligned to the FastAPI routes.
- Data: Postgres for users/profiles/habits, persisted in Docker.
- Ops: Docker Compose; add MLFlow tracking server as a service; NGINX as proxy.

Frontend strategy

- Streamlit-first: For the thesis MVP and rapid iteration, ship a Streamlit app that calls the FastAPI endpoints for auth, upload, habits, and predict. This accelerates demo/validation and simplifies visualization.
- Scale after defense: Migrate/expand the UI to a production-ready Next.js/React frontend, keeping the same API surface. React code in this repo remains a starting point for that migration.

Text diagram

User → React → FastAPI → ML Inference (RF/NN, ONNX/SHAP) → Postgres
                                 ↘ MLFlow (experiments/artifacts)

## Directory (target)

antiaging-mvp/
- backend/
  - api/ … legacy Django app (kept during transition)
  - fastapi_app/
    - main.py, db.py, auth.py (JWT helpers), schemas.py (Pydantic)
    - ml/ train.py, predict.py, preprocessor.py (reuse where useful)
  - requirements.txt
- frontend/ … React
- nginx/
- docker-compose.yml

## Phases and Milestones

Phase 1: Setup + Data (Sep 1 – Sep 7)

- Scale synthetic dataset to N≈5000 for model viability (train.csv) + tiny test CSVs. Keep demographics + habits.
- Validate distributions and simple rules; keep the current generator and improve later if needed.

Deliverables: datasets ready under backend/api/data/datasets/.

Phase 2: Backend + ML (Sep 8 – Sep 18)

- Scaffold FastAPI app with endpoints: /health, /signup, /token, /upload-genetic, /submit-habits, /predict?model_type=rf|nn.
- Implement preprocessing pipeline; train RF baseline; export ONNX; SHAP explanations.
- Add MLP (PyTorch) small architecture; log both models to MLFlow with metrics (F1 macro or regression metrics depending on label), params, and artifacts.

Deliverables: working FastAPI endpoints locally; MLFlow runs visible; RF+NN artifacts stored.

Phase 3: Frontend + Integration (Sep 19 – Sep 29)

- Build Streamlit MVP that integrates with FastAPI (upload CSV, habits form, predict with explanations). Use this for the defense demo.
- Prepare Next.js/React integration plan and stabilize API contracts so the React app (post-defense) consumes the same endpoints.

Deliverables: end-to-end flow in local Docker.

Phase 4: Docker, Testing, Validation (Sep 30 – Oct 6)

- Add MLFlow service to compose; ensure health checks.
- Pytest for ML and API; aim ≥70% coverage; basic load test for predict endpoint.

Deliverables: compose up runs db, backend (FastAPI), frontend, nginx, mlflow; tests green.

Phase 5: Thesis + Demo (Oct 7 – Oct 15 + buffer)

- Document model comparison (RF vs MLP) using MLFlow screenshots; ethics; limitations; future work.
- Record demo video; prepare slides.

## Key Decisions and Rationale

- FastAPI: lean, async, Pydantic validation, ideal for ML serving.
- Dual models: RF (strong baseline, interpretable) + MLP (captures interactions); compare in thesis.
- MLFlow: professional experiment tracking; supports model registry later.

## Implementation Details (Actionable)

**✅ All implementation details have been converted to GitHub Issues**

The sections below provide context, but for actionable tasks, see:
- `GITHUB_ISSUES.md` - Organized overview and quick reference  
- `DETAILED_ISSUES.md` - Complete issue descriptions
- `github_issues.json` - Import-ready GitHub API format

### Issue Summary by Phase

**Phase 1 (Sep 1-7): Setup + Data**
- Issue #1: Scale synthetic dataset to 5000 records  
- Issue #2: Validate data distributions and quality checks

**Phase 2 (Sep 8-18): Backend + ML**  
- Issue #3: FastAPI authentication with JWT
- Issue #4: Data upload and habits endpoints
- Issue #5: ML preprocessing pipeline  
- Issue #6: Random Forest with ONNX & SHAP
- Issue #7: MLP neural network with PyTorch
- Issue #8: Prediction endpoint with model selection

**Phase 3 (Sep 19-29): Frontend + Integration**
- Issue #9: Streamlit MVP with end-to-end integration
- Issue #10: React migration planning and API stabilization  
- Issue #11: End-to-end integration testing

**Phase 4 (Sep 30-Oct 6): Docker, Testing, Validation**
- Issue #12: Docker infrastructure and health checks
- Issue #13: Comprehensive testing suite (≥70% coverage)
- Issue #14: Performance optimization and load testing

**Phase 5 (Oct 7-15): Thesis + Demo**  
- Issue #15: MLFlow model comparison analysis
- Issue #16: Ethics and limitations documentation
- Issue #17: Demo preparation and presentation materials

**Backlog Items**
- Issue #18: Migrate Django ORM to SQLAlchemy
- Issue #19: Advanced CSV schema validation  
- Issue #20: Enhanced model explanations

Backend APIs (FastAPI)

- POST /signup: create user, return bearer token
- POST /token: OAuth2PasswordRequestForm (username/password) → token
- POST /upload-genetic: CSV upload; persist latest profile per user
- POST /submit-habits: JSON body; persist
- GET /predict?model_type=rf|nn: fetch latest genetic + habits, run inference, return prediction + explanations

Data/Models

- Keep current regression target (biological_age) for continuity; optionally add multi-class risk later. Ensure preprocessing consistent across train/predict; export RF to ONNX; keep PyTorch model for inference directly or via TorchScript/ONNX.

MLFlow

- Start tracking server in Docker compose at [http://mlflow:5000](http://mlflow:5000) with named volume. Use tracking URI in train scripts; log metrics, params, and artifacts (model files).

Security

- JWT via python-jose; password hashing via passlib[bcrypt]. CORS configured for frontend.

Testing

- Unit tests for preprocessing, training, prediction; API tests for auth, upload, submit, predict happy/edge paths.

## Backlog/Next Work Items

**✅ Converted to GitHub Issues #18-20**

See `GITHUB_ISSUES.md` and `DETAILED_ISSUES.md` for complete backlog items:

- **Issue #18:** Migrate persistence from Django ORM to SQLAlchemy models
- **Issue #19:** Implement true CSV schema validation for upload-genetic  
- **Issue #20:** Enhanced SHAP explanations and advanced interpretability features

All backlog items include detailed acceptance criteria, implementation notes, and priority levels.

## Runbook (high level)

- Install deps; run training to produce initial artifacts; start compose; hit /docs for FastAPI Swagger.
- MLFlow UI runs at [http://localhost:5000](http://localhost:5000) when compose service is enabled.

## Ethics Note

Predictions are for research and educational purposes; not medical advice. Synthetic data used during development.
