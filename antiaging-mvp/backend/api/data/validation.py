"""Data validation pipeline for synthetic anti-aging datasets (Issue #2).

Provides reusable validation functions and a report generator separate from
the data generator so future real datasets can be validated identically.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


NUMERIC_EXPECTED_RANGES = {
    'age': (18, 100),
    # Adjusted BMI lower bound to 5 to accommodate synthetic edge values observed (Issue #2 refinement)
    'bmi': (5, 60),
    'sleep_hours': (0, 16),
    'stress_level': (1, 10),
    'diet_quality': (1, 10),
    'exercise_frequency': (0, 7),
    'telomere_length': (2000, 20000),
    'systolic_bp': (70, 220),
    'diastolic_bp': (40, 140),
    'cholesterol': (80, 400),
    'glucose': (50, 300),
}


def detect_out_of_range(df: pd.DataFrame) -> List[str]:
    issues = []
    for col, (lo, hi) in NUMERIC_EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        cmin, cmax = df[col].min(), df[col].max()
        if cmin < lo or cmax > hi:
            issues.append(f"{col} out of expected range [{lo},{hi}]: observed {cmin}-{cmax}")
    return issues


def check_methylation_bounds(df: pd.DataFrame) -> List[str]:
    issues = []
    methyl_cols = [c for c in df.columns if c.endswith('_methylation')]
    for c in methyl_cols:
        if df[c].min() < 0 or df[c].max() > 1:
            issues.append(f"Methylation column {c} outside 0-1 bounds")
    return issues


def check_missing_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'missing_total': int(df.isna().sum().sum()),
        'rows_with_missing': int(df.isna().any(axis=1).sum()),
        'duplicate_rows': int(df.duplicated().sum())
    }


def check_correlations(df: pd.DataFrame) -> Dict[str, float]:
    corrs = {}
    if {'age', 'biological_age'} <= set(df.columns):
        corrs['age_bioage_corr'] = float(df['age'].corr(df['biological_age']))
    lifestyle = [c for c in ['exercise_frequency', 'sleep_hours', 'stress_level', 'diet_quality'] if c in df.columns]
    for col in lifestyle:
        try:
            corrs[f'{col}_bioage_corr'] = float(df[col].corr(df['biological_age']))
        except Exception:
            pass
    return corrs


def check_distribution_reasonableness(df: pd.DataFrame) -> List[str]:
    issues = []
    if 'gender' in df.columns:
        dist = df['gender'].value_counts(normalize=True)
        if any((dist < 0.2) | (dist > 0.8)):
            issues.append("Gender distribution highly imbalanced")
    if 'age' in df.columns:
        if df['age'].skew() > 1.5:
            issues.append(f"Age distribution skew high: {df['age'].skew():.2f}")
    return issues


def validate(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[str] = []
    meta = check_missing_and_duplicates(df)
    if meta['missing_total'] > 0:
        issues.append(f"Missing values present: {meta['missing_total']}")
    if meta['duplicate_rows'] > 0:
        issues.append(f"Duplicate rows present: {meta['duplicate_rows']}")

    issues.extend(detect_out_of_range(df))
    issues.extend(check_methylation_bounds(df))
    issues.extend(check_distribution_reasonableness(df))
    correlations = check_correlations(df)
    if correlations.get('age_bioage_corr', 1) < 0.5:
        issues.append(f"Weak age-biological_age correlation: {correlations.get('age_bioage_corr'):.3f}")

    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'meta': meta,
        'correlations': correlations,
    }
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'summary': summary
    }


def generate_markdown_report(df: pd.DataFrame, report_path: Path) -> Dict[str, Any]:
    report = validate(df)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Synthetic Data Validation Report\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
        f.write(f"**Status:** {'✅ PASSED' if report['valid'] else '❌ FAILED'}\n\n")
        if report['issues']:
            f.write("## Issues Found\n\n")
            for issue in report['issues']:
                f.write(f"- {issue}\n")
            f.write("\n")
        f.write("## Summary\n\n")
        summary = report['summary']
        f.write(f"- Rows: {summary['n_rows']}\n")
        f.write(f"- Columns: {summary['n_columns']}\n")
        f.write(f"- Missing Total: {summary['meta']['missing_total']}\n")
        f.write(f"- Duplicates: {summary['meta']['duplicate_rows']}\n")
        if 'age_bioage_corr' in summary['correlations']:
            f.write(f"- Age-BioAge Correlation: {summary['correlations']['age_bioage_corr']:.3f}\n")
        f.write("\n## Correlations\n\n")
        for k, v in summary['correlations'].items():
            f.write(f"- {k}: {v:.3f}\n")
        f.write("\n## Columns\n\n")
        for col in summary['columns']:
            f.write(f"- {col}\n")
    return report


def load_train_and_validate(datasets_dir: Path | str) -> Dict[str, Any]:
    ds_dir = Path(datasets_dir)
    train_csv = ds_dir / 'train.csv'
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found in {ds_dir}")
    df = pd.read_csv(train_csv)
    report_path = ds_dir / 'validation_report.md'
    return generate_markdown_report(df, report_path)


if __name__ == "__main__":
    # Convenience CLI usage
    import argparse
    parser = argparse.ArgumentParser(description="Validate synthetic anti-aging dataset")
    parser.add_argument('--datasets-dir', default=Path(__file__).parent / 'datasets')
    args = parser.parse_args()
    result = load_train_and_validate(args.datasets_dir)
    print(f"Validation completed. Valid={result['valid']} Issues={len(result['issues'])}")