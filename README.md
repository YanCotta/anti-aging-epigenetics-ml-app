# Personalized Anti-Aging Recommendation System Based on Genetics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg)](https://reactjs.org/)

This repository contains the source code for a thesis project developing a **Personalized Anti-Aging Recommendation System**. The system analyzes an individual's genetic predispositions (via Single Nucleotide Polymorphisms, or SNPs) and lifestyle habits (e.g., nutrition, exercise, sleep, stress) to generate personalized risk assessments and actionable recommendations for healthy aging. It leverages machine learning for cross-analysis, with a focus on explainability, security, and scalability.

The MVP demonstrates a data-driven approach using synthetic datasets, model comparison (Random Forest vs. Neural Network), and professional MLOps practices. Post-thesis, it can be extended to real genetic data integrations.

## Features
- **Synthetic Data Generation**: Realistic genetic (SNPs) and lifestyle datasets created with BioPython, validated statistically (e.g., chi-squared for allelic frequencies).
- **User Authentication**: Secure signup/login with JWT tokens.
- **Data Upload and Input**: Upload genetic CSV files and submit lifestyle habits via an interactive UI.
- **ML-Driven Analysis**: 
  - Cross-analysis of genetics and habits using Random Forest (scikit-learn) and Feedforward Neural Network (PyTorch).
  - Model optimization with ONNX for fast inference.
  - Explainable AI via SHAP to show feature contributions (e.g., "This SNP contributes 30% to your risk").
- **Model Comparison**: Tracked with MLFlow for experiments, metrics (e.g., F1-score), and artifact logging.
- **Visualizations**: Risk dashboards with charts (Chart.js) and SHAP explanations.
- **Recommendations**: Actionable advice based on predictions (e.g., "Increase exercise to mitigate genetic risk").
- **Containerization**: Fully dockerized with Docker Compose, NGINX proxy, and health checks for easy deployment.
- **Ethics and Security**: Synthetic data for privacy; UI disclaimers; input validation and error handling.

## Planned End State
The MVP will be a robust, deployable prototype ready for thesis defense by mid-October 2025. Key milestones:
- Phase 1: Setup and synthetic data generation (completed by Sep 7).
- Phase 2: Backend with FastAPI, ML models (RF + NN), and MLFlow integration (by Sep 21).
- Phase 3: React frontend and API integration (by Sep 29).
- Phase 4: Dockerization and testing (by Oct 6).
- Phase 5: Documentation and demo video (by Oct 15).

Future extensions: Real data APIs (e.g., 23andMe), advanced epigenetics, mobile app, and cloud deployment (e.g., AWS).

## Technology Stack
- **Backend**: FastAPI (API framework), SQLAlchemy (ORM for PostgreSQL), Uvicorn (ASGI server).
- **Frontend**: React.js (UI), Bootstrap (styling), Chart.js (visualizations), Axios (API calls), React Hook Form (validation).
- **Machine Learning**: 
  - Scikit-learn (Random Forest, preprocessing, tuning).
  - PyTorch (Feedforward Neural Network).
  - ONNX (model optimization), SHAP (explainability).
  - MLFlow (experiment tracking and model registry).
- **Data Handling**: Pandas/NumPy (analysis), BioPython (synthetic genetics).
- **Database**: PostgreSQL (relational storage for users, profiles, habits).
- **DevOps**: Docker & Docker Compose (containerization), NGINX (reverse proxy), Git (versioning).
- **Security**: JWT (authentication), Passlib (password hashing), Pydantic (input validation).
- **Testing**: Pytest (backend/ML), Jest (frontend), with 70%+ coverage goal.

## Installation and Setup
### Prerequisites
- Python 3.12+
- Node.js 20+
- Docker & Docker Compose
- Git

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/anti-aging-recommender.git
   cd anti-aging-recommender
   ```

2. Set up environment variables: Copy `.env.example` to `.env` and fill in secrets (e.g., `SECRET_KEY`, DB creds).

3. Install dependencies:
   - Backend: `cd backend; pip install -r requirements.txt`
   - Frontend: `cd frontend; npm install`

4. Generate synthetic data: `python backend/api/data/generator.py` (creates CSVs in `backend/api/data/datasets/`).

5. Train ML models: `python backend/ml/train.py` (logs to MLFlow; access UI at http://localhost:5000 after docker up).

6. Run locally (without Docker):
   - Backend: `uvicorn backend.main:app --reload`
   - Frontend: `cd frontend; npm start`
   - MLFlow: `mlflow ui --host 0.0.0.0`

7. Run with Docker: `docker-compose up --build` (access app at http://localhost:80, API at /api, MLFlow at :5000).

## Usage
1. **Signup/Login**: Use the frontend forms or POST to `/signup` or `/token` with username/password.
2. **Upload Genetic Data**: Submit a CSV (from synthetic test files) via the upload form or POST to `/upload-genetic`.
3. **Submit Habits**: Fill the tabbed form (nutrition, exercise, etc.) or POST to `/submit-habits`.
4. **Get Predictions**: View dashboard or GET `/predict?model_type=rf` (or `nn`)—returns risk, explanations, and model used.
5. **Compare Models**: In MLFlow UI, view runs for RF vs NN metrics.

Example API Call (with token):
```
curl -H "Authorization: Bearer <token>" -F "file=@test2.csv" http://localhost:8000/upload-genetic
```

## Contributing
Contributions are welcome! Fork the repo, create a feature branch, and submit a PR. Focus on:
- Bug fixes in ML integration.
- Additional SNPs or habits.
- Enhanced visualizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thesis advisor and peers for feedback.
- Open-source tools: FastAPI, PyTorch, MLFlow.
- Inspired by genetic longevity research (e.g., APOE and FOXO3 SNPs).

For questions, open an issue.
```
