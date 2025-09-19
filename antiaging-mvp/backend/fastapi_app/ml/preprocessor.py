from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional
import joblib
import pandas as pd


def _import_training_preprocessor():
    """Dynamically import DataPreprocessor from training module.
    This avoids code duplication and keeps train/infer aligned.
    """
    api_ml_dir = Path(__file__).resolve().parents[2] / "api" / "ml"
    if str(api_ml_dir) not in sys.path:
        sys.path.append(str(api_ml_dir))
    from preprocessor import DataPreprocessor  # type: ignore
    return DataPreprocessor


def load_preprocessor(preproc_path: str | Path):
    """Load a saved DataPreprocessor from disk.

    Returns a fitted DataPreprocessor instance or raises FileNotFoundError.
    """
    DataPreprocessor = _import_training_preprocessor()
    preproc = DataPreprocessor()
    preproc_path = Path(preproc_path)
    if preproc_path.exists():
        # Prefer class load method if present
        loaded = False
        if hasattr(preproc, "load"):
            loaded = preproc.load(str(preproc_path))
        if not loaded:
            payload = joblib.load(str(preproc_path))
            # Fallback: set attributes directly if saved as dict
            for k in ("scalers", "encoders", "imputers", "feature_columns"):
                if k in payload:
                    setattr(preproc, k, payload[k])
        return preproc
    raise FileNotFoundError(f"Preprocessor artifact not found: {preproc_path}")


def ensure_schema(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Add missing columns (zeros) and order columns to match expected schema."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]
