<<<<<<< HEAD
# anti-aging-epigenetics-ml-app
Product of my undergrad thesis in Biological Sciences (BS) - An anti-aging epigenetics machine learning based software capable of making personalized health recommendations and predictions based on individual genetic data and environmental exposure.

## Development Plan

See the finalized roadmap in `DEV_PLAN.md`.

## Architecture (current)

- Backend: `FastAPI` + `SQLAlchemy` + JWT (`python-jose`, `passlib`), served by `uvicorn`.
- ML: Random Forest (scikit-learn) and MLP (PyTorch), preprocessing with sklearn; ONNX/SHAP planned; experiments via `MLflow`.
- Frontend: Streamlit MVP for the thesis defense; Next.js/React after defense.
- Data: PostgreSQL (Dockerized) for users/profiles/habits.
- Ops: Docker Compose with services for `db`, `fastapi`, `streamlit`, `nginx`, and `mlflow`.

## Services (docker-compose)

- `db`: Postgres database with health check.
- `fastapi`: API at `http://localhost:8001` (Swagger at `/docs`), mounted from `backend/`.
- `streamlit`: UI at `http://localhost:8501`, calls the FastAPI service.
- `mlflow`: Tracking UI at `http://localhost:5000` (artifacts volume-mounted).
- `nginx`: Proxy (kept from legacy; routing updates to FastAPI pending).

## Frontend Strategy

- Streamlit-first for rapid iteration and defense demo.
- Post-defense: migrate/expand to Next.js/React reusing the same API contracts.

## Roadmap Checklist (from DEV_PLAN)

Phase 1: Setup + Data

- [ ] Scale synthetic dataset to ~5000 and validate distributions
- [ ] Place curated datasets under `backend/api/data/datasets/`

Phase 2: Backend + ML

- [x] Scaffold FastAPI app and core endpoints skeleton (`/health`, `/signup`, `/token`, `/upload-genetic`, `/submit-habits`, `/predict`)
- [x] Add MLflow tracking service to docker-compose
- [x] Add Streamlit MVP scaffold and service to docker-compose
- [x] Add FastAPI/SQLAlchemy/JWT/MLflow/Torch to backend requirements (core)
- [ ] Implement preprocessing pipeline alignment train/predict
- [ ] Train Random Forest baseline; export ONNX; add SHAP explanations
- [ ] Add MLP (PyTorch) and log both models to MLflow with metrics/params/artifacts
- [ ] Wire prediction to load artifacts/MLflow and return explanations
- [ ] Harden auth: decode JWT to current user; add proper dependencies
- [ ] Strict CSV schema validation for `/upload-genetic`

Phase 3: Frontend + Integration

- [~] Streamlit MVP integrated end-to-end (auth, upload, habits, predict)
- [x] Document Next.js/React migration plan in `DEV_PLAN.md`

Phase 4: Docker, Testing, Validation

- [x] Include MLflow in compose (service running on :5000)
- [ ] Add health checks for app services and finalize NGINX routing to FastAPI
- [ ] Pytest for ML and API (≥70% coverage) and basic load test for predict

Phase 5: Thesis + Demo

- [ ] MLflow screenshots, model comparison (RF vs MLP), ethics/limitations
- [ ] Record demo video and prepare slides

Backlog / Infra

- [ ] Migrate persistence fully from Django ORM to SQLAlchemy models
- [ ] Replace Django container command with Uvicorn or proxy both via NGINX during transition

Legend: [x] done, [ ] todo, [~] in progress