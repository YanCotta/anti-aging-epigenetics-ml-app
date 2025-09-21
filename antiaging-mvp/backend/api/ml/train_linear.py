"""
Issue #21: Linear Regression Baseline

Trains a scikit-learn LinearRegression model on the primary dataset
`backend/api/data/datasets/train.csv`, logs metrics and artifacts to MLflow,
and saves the serialized model to disk.

Logged artifacts/metrics:
- Params: model type and core hyperparameters
- Metrics: RMSE, R2, MAE (on test split)
- Artifacts: trained model (joblib .pkl) and fitted preprocessor (joblib)

Usage (optional):
    python -m backend.api.ml.train_linear \
        --data "backend/api/data/datasets/train.csv" \
        --model-dir "backend/api/ml/models" \
        --experiment "linear_regression_baseline"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Support running both as a module and as a standalone script
try:
    # When executed as part of a package (recommended)
    from .preprocessor import DataPreprocessor
except Exception:
    # When executed directly: add current directory to sys.path and import
    import sys
    from pathlib import Path as _Path
    _this_dir = _Path(__file__).resolve().parent
    if str(_this_dir) not in sys.path:
        sys.path.append(str(_this_dir))
    from preprocessor import DataPreprocessor


TARGET_COL = "biological_age"
DROP_FEATURES = ["user_id"]  # Non-predictive identifiers


def resolve_default_paths() -> Dict[str, Path]:
    here = Path(__file__).resolve()
    api_dir = here.parents[1]  # .../backend/api
    data_csv = api_dir / "data" / "datasets" / "train.csv"
    model_dir = here.parent / "models"
    return {"data_csv": data_csv, "model_dir": model_dir}


def train_linear_regression(
    data_path: Path | str | None = None,
    model_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    experiment_name: str = "linear_regression_baseline",
) -> Dict[str, Any]:
    """Train LinearRegression on the primary dataset and log to MLflow."""

    # Resolve paths
    defaults = resolve_default_paths()
    data_csv = Path(data_path) if data_path else defaults["data_csv"]
    out_dir = Path(model_dir) if model_dir else defaults["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not data_csv.exists():
        raise FileNotFoundError(f"Training dataset not found at {data_csv}")
    df = pd.read_csv(data_csv)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    # Split features/target
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_FEATURES if c in df.columns])
    y = df[TARGET_COL]

    # CRITICAL FIX: Split data FIRST, then preprocess to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Preprocess (fit ONLY on training data to prevent leakage)
    pre = DataPreprocessor()
    X_train_processed = pre.fit_transform(X_train)
    X_test_processed = pre.transform(X_test)  # Only transform, don't fit!

    # Train model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Evaluate
    y_pred = model.predict(X_test_processed)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Prepare artifacts
    model_path = out_dir / "linear_regression_model.pkl"
    preproc_path = out_dir / "preprocessor_linear.pkl"
    joblib.dump(model, model_path)
    # Use project-conventional save method for preprocessor
    try:
        pre.save(str(preproc_path))
    except Exception:
        # Fallback in case save signature changes
        joblib.dump({
            "scalers": pre.scalers,
            "encoders": pre.encoders,
            "imputers": pre.imputers,
            "feature_columns": pre.feature_columns,
        }, preproc_path)

    # MLflow logging
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="linear_regression"):
        mlflow.log_params({
            "model": "LinearRegression",
            "fit_intercept": getattr(model, "fit_intercept", True),
            "copy_X": getattr(model, "copy_X", True),
            "positive": getattr(model, "positive", False),
            "n_features": X_train_processed.shape[1],
            "test_size": test_size,
            "random_state": random_state,
        })
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Artifacts
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(preproc_path))

    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "model_path": str(model_path),
        "preprocessor_path": str(preproc_path),
        "n_features": X_train_processed.shape[1],
    }


def main():
    import argparse

    defaults = resolve_default_paths()
    parser = argparse.ArgumentParser(description="Train LinearRegression baseline and log to MLflow")
    parser.add_argument("--data", type=str, default=str(defaults["data_csv"]))
    parser.add_argument("--model-dir", type=str, default=str(defaults["model_dir"]))
    parser.add_argument("--experiment", type=str, default="linear_regression_baseline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_linear_regression(
        data_path=Path(args.data),
        model_dir=Path(args.model_dir),
        test_size=args.test_size,
        random_state=args.seed,
        experiment_name=args.experiment,
    )
    print("Training complete:", result)


if __name__ == "__main__":
    main()
