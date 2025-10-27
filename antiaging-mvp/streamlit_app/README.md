# LiveMore MVP - Streamlit Demo

This is the **core deliverable** for the thesis defense: a functional MVP demonstrating the "LiveMore" product concept using Streamlit for rapid prototyping.

## Strategic Context

This Streamlit app is the result of a strategic pivot (see `docs/PIVOT.md`) from a complex production architecture (FastAPI/React/ONNX) to a focused MVP approach. The goal is to validate the product hypothesis: **using Explainable AI (XAI) to demonstrate "Health ROI" to users**.

## Setup (Local)

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Pre-requisite) Train the model** (if artifacts don't exist):
   ```bash
   # This uses data from 'datasets_chaos_v2/' (to be generated via Issue #50)
   # python ../../ml_pipeline/train_model.py
   # NOTE: Model training script will be created as part of Issue #50 implementation
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Current Status

- **MVP Status:** In development (Week 2 of 20-day sprint)
- **Data Dependency:** Waiting for `datasets_chaos_v2/` generation (Issue #50)
- **Model Artifacts:** Will load `livemore_rf_v2.joblib` and `livemore_explainer_v2.pkl` from `app_model/` directory

## Architecture

The Streamlit app will:
1. Load trained Random Forest model and SHAP explainer
2. Provide interactive UI for lifestyle factor inputs (exercise, diet, smoking, etc.)
3. Predict biological age based on user inputs
4. Display SHAP explanations showing how each factor impacts the prediction
5. Present results using business-friendly language ("Health ROI")

## Post-MVP Roadmap

After successful thesis validation, the production architecture (FastAPI backend + React frontend) archived in `/legacy` will be implemented. See `docs/FUTURE_ROADMAP_POST_TCC.md` for details.
