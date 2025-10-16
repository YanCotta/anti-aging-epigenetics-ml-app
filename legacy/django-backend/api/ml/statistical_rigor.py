#!/usr/bin/env python3
"""
Statistical Rigor Module for Anti-Aging ML

Implements comprehensive statistical testing and validation:
1. Bootstrap confidence intervals for all metrics
2. Permutation tests for feature importance validation
3. Multiple testing correction (FDR, Bonferroni)
4. Cross-validation with proper stratification
5. Statistical power analysis
6. Model comparison tests

Scientific Foundation:
- Efron & Tibshirani bootstrap methods
- Benjamini-Hochberg FDR correction
- DeLong test for ROC comparison
- Proper biological stratification

Author: Anti-Aging ML Project
Date: October 2025
Issue: #47 - Statistical Rigor and Multiple Testing Correction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
import logging
from tqdm import tqdm


@dataclass
class BootstrapResult:
    """Container for bootstrap analysis results."""
    
    statistic: float
    ci_lower: float
    ci_upper: float
    std_error: float
    bootstrap_distribution: np.ndarray
    confidence_level: float = 0.95


@dataclass
class PermutationTestResult:
    """Container for permutation test results."""
    
    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    n_permutations: int
    significant: bool


class StatisticalRigor:
    """
    Comprehensive statistical testing and validation.
    
    Implements state-of-the-art statistical methods for ensuring
    robust, reproducible, and publication-quality results.
    """
    
    def __init__(self, random_state: int = 42, n_bootstrap: int = 1000,
                 n_permutations: int = 1000, confidence_level: float = 0.95):
        """
        Initialize statistical testing framework.
        
        Args:
            random_state: Random seed for reproducibility
            n_bootstrap: Number of bootstrap iterations
            n_permutations: Number of permutation test iterations
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.confidence_level = confidence_level
        
        np.random.seed(random_state)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('statistical_rigor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred: np.ndarray,
                        metric_func: Callable,
                        stratify: Optional[np.ndarray] = None) -> BootstrapResult:
        """
        Calculate bootstrap confidence intervals for a metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_func: Metric function (e.g., r2_score, mean_absolute_error)
            stratify: Optional stratification variable for sampling
            
        Returns:
            BootstrapResult with CI and distribution
        """
        n_samples = len(y_true)
        bootstrap_stats = []
        
        # Calculate observed statistic
        observed_stat = metric_func(y_true, y_pred)
        
        # Bootstrap resampling
        for i in range(self.n_bootstrap):
            # Stratified sampling if requested
            if stratify is not None:
                # Sample within strata
                unique_strata = np.unique(stratify)
                indices = []
                for stratum in unique_strata:
                    stratum_indices = np.where(stratify == stratum)[0]
                    sampled = np.random.choice(
                        stratum_indices, 
                        size=len(stratum_indices), 
                        replace=True
                    )
                    indices.extend(sampled)
                indices = np.array(indices)
            else:
                # Simple bootstrap
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Calculate metric on bootstrap sample
            boot_stat = metric_func(y_true[indices], y_pred[indices])
            bootstrap_stats.append(boot_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence intervals (percentile method)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return BootstrapResult(
            statistic=observed_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=np.std(bootstrap_stats),
            bootstrap_distribution=bootstrap_stats,
            confidence_level=self.confidence_level
        )
    
    def comprehensive_metrics_with_ci(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     stratify: Optional[np.ndarray] = None) -> Dict[str, BootstrapResult]:
        """
        Calculate comprehensive metrics with bootstrap CIs.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            stratify: Optional stratification for sampling
            
        Returns:
            Dictionary of metric name to BootstrapResult
        """
        self.logger.info(f"Calculating metrics with bootstrap CIs (n={self.n_bootstrap})...")
        
        metrics = {
            'R²': r2_score,
            'MAE': mean_absolute_error,
            'RMSE': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
            'Median_AE': lambda yt, yp: np.median(np.abs(yt - yp))
        }
        
        results = {}
        for metric_name, metric_func in metrics.items():
            result = self.bootstrap_metric(y_true, y_pred, metric_func, stratify)
            results[metric_name] = result
            self.logger.info(
                f"{metric_name}: {result.statistic:.4f} "
                f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
            )
        
        return results
    
    def permutation_test_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                          model, feature_idx: int,
                                          metric_func: Callable = r2_score) -> PermutationTestResult:
        """
        Permutation test for feature importance.
        
        Tests null hypothesis that feature has no predictive power by
        randomly permuting its values and measuring performance drop.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model
            feature_idx: Index of feature to test
            metric_func: Performance metric
            
        Returns:
            PermutationTestResult
        """
        # Original performance
        y_pred = model.predict(X)
        observed_stat = metric_func(y, y_pred)
        
        # Permutation distribution
        null_distribution = []
        X_permuted = X.copy()
        
        for _ in range(self.n_permutations):
            # Permute feature
            X_permuted[:, feature_idx] = np.random.permutation(X[:, feature_idx])
            
            # Calculate performance
            y_pred_permuted = model.predict(X_permuted)
            null_stat = metric_func(y, y_pred_permuted)
            null_distribution.append(null_stat)
        
        null_distribution = np.array(null_distribution)
        
        # Calculate p-value (one-sided: is observed better than null?)
        if metric_func == r2_score:  # Higher is better
            p_value = np.mean(null_distribution >= observed_stat)
        else:  # Lower is better (MAE, RMSE)
            p_value = np.mean(null_distribution <= observed_stat)
        
        return PermutationTestResult(
            observed_statistic=observed_stat,
            null_distribution=null_distribution,
            p_value=p_value,
            n_permutations=self.n_permutations,
            significant=(p_value < 0.05)
        )
    
    def multiple_testing_correction(self, p_values: np.ndarray,
                                   method: str = 'fdr_bh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple testing correction.
        
        Args:
            p_values: Array of p-values
            method: Correction method ('fdr_bh', 'bonferroni', 'holm')
            
        Returns:
            Tuple of (corrected p-values, rejection decisions)
        """
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            corrected_pvals = p_values * n_tests
            corrected_pvals = np.minimum(corrected_pvals, 1.0)
            reject = corrected_pvals < 0.05
            
        elif method == 'holm':
            # Holm-Bonferroni
            sorted_indices = np.argsort(p_values)
            sorted_pvals = p_values[sorted_indices]
            
            corrected_pvals = np.zeros_like(p_values)
            reject = np.zeros(n_tests, dtype=bool)
            
            for i, idx in enumerate(sorted_indices):
                corrected_pvals[idx] = sorted_pvals[i] * (n_tests - i)
                if corrected_pvals[idx] < 0.05 and i == 0:
                    reject[idx] = True
                elif corrected_pvals[idx] < 0.05 and reject[sorted_indices[i-1]]:
                    reject[idx] = True
                    
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            sorted_pvals = p_values[sorted_indices]
            
            # Calculate corrected p-values
            corrected_pvals = np.zeros_like(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_pvals[idx] = sorted_pvals[i] * n_tests / (i + 1)
            
            corrected_pvals = np.minimum(corrected_pvals, 1.0)
            
            # Find rejections
            reject = corrected_pvals < 0.05
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.logger.info(
            f"Multiple testing correction ({method}): "
            f"{np.sum(reject)}/{n_tests} significant after correction"
        )
        
        return corrected_pvals, reject
    
    def cross_validation_with_ci(self, X: np.ndarray, y: np.ndarray,
                                model_class, model_params: dict,
                                n_folds: int = 5,
                                stratify: Optional[np.ndarray] = None) -> Dict:
        """
        Cross-validation with confidence intervals.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_class: Model class to instantiate
            model_params: Model parameters
            n_folds: Number of CV folds
            stratify: Optional stratification variable
            
        Returns:
            Dictionary with CV results and CIs
        """
        self.logger.info(f"Running {n_folds}-fold cross-validation...")
        
        # Setup cross-validation
        if stratify is not None:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            cv_splits = cv.split(X, stratify)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            cv_splits = cv.split(X)
        
        # Collect results
        fold_results = {
            'r2': [],
            'mae': [],
            'rmse': [],
            'y_true': [],
            'y_pred': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            fold_results['r2'].append(r2_score(y_val, y_pred))
            fold_results['mae'].append(mean_absolute_error(y_val, y_pred))
            fold_results['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            fold_results['y_true'].extend(y_val)
            fold_results['y_pred'].extend(y_pred)
        
        # Concatenate predictions
        y_true_all = np.array(fold_results['y_true'])
        y_pred_all = np.array(fold_results['y_pred'])
        
        # Calculate overall metrics with bootstrap CI
        metrics_with_ci = self.comprehensive_metrics_with_ci(y_true_all, y_pred_all)
        
        # Add fold-level statistics
        results = {
            'metrics_with_ci': metrics_with_ci,
            'fold_r2_mean': np.mean(fold_results['r2']),
            'fold_r2_std': np.std(fold_results['r2']),
            'fold_mae_mean': np.mean(fold_results['mae']),
            'fold_mae_std': np.std(fold_results['mae']),
            'n_folds': n_folds,
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }
        
        return results
    
    def model_comparison_test(self, y_true: np.ndarray,
                            y_pred_model1: np.ndarray,
                            y_pred_model2: np.ndarray,
                            paired: bool = True) -> Dict:
        """
        Statistical test for comparing two models.
        
        Args:
            y_true: True values
            y_pred_model1: Predictions from model 1
            y_pred_model2: Predictions from model 2
            paired: Whether to use paired test
            
        Returns:
            Dictionary with test results
        """
        # Calculate absolute errors
        errors1 = np.abs(y_true - y_pred_model1)
        errors2 = np.abs(y_true - y_pred_model2)
        
        if paired:
            # Wilcoxon signed-rank test for paired samples
            statistic, p_value = wilcoxon(errors1, errors2)
            test_name = "Wilcoxon signed-rank"
        else:
            # Mann-Whitney U test for independent samples
            statistic, p_value = mannwhitneyu(errors1, errors2, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        # Effect size (Cliff's Delta for non-parametric)
        n1, n2 = len(errors1), len(errors2)
        delta = 0
        for e1 in errors1:
            for e2 in errors2:
                if e1 < e2:
                    delta += 1
                elif e1 == e2:
                    delta += 0.5
        
        cliffs_delta = (2 * delta / (n1 * n2)) - 1
        
        self.logger.info(
            f"{test_name} test: statistic={statistic:.2f}, "
            f"p-value={p_value:.4f}, Cliff's delta={cliffs_delta:.3f}"
        )
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'cliffs_delta': cliffs_delta,
            'significant': p_value < 0.05,
            'model1_mean_error': np.mean(errors1),
            'model2_mean_error': np.mean(errors2)
        }
    
    def power_analysis(self, effect_size: float, alpha: float = 0.05,
                      power: float = 0.8) -> int:
        """
        Calculate required sample size for desired statistical power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired statistical power
            
        Returns:
            Required sample size per group
        """
        # Simplified power calculation for t-test
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        n = int(np.ceil(n))
        
        self.logger.info(
            f"Power analysis: n={n} per group for "
            f"effect_size={effect_size}, alpha={alpha}, power={power}"
        )
        
        return n


def demonstrate_statistical_rigor():
    """Demonstrate statistical rigor methods."""
    print("=== Statistical Rigor Framework Demo ===\n")
    
    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.uniform(25, 75, n_samples)
    y_pred = y_true + np.random.normal(0, 3, n_samples)
    
    # Initialize framework
    stats = StatisticalRigor(n_bootstrap=1000, n_permutations=500)
    
    # 1. Bootstrap CIs
    print("1. Bootstrap Confidence Intervals:")
    metrics = stats.comprehensive_metrics_with_ci(y_true, y_pred)
    
    # 2. Multiple testing correction
    print("\n2. Multiple Testing Correction:")
    p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.08, 0.15, 0.25])
    corrected, reject = stats.multiple_testing_correction(p_values, method='fdr_bh')
    
    print(f"Original p-values: {p_values}")
    print(f"FDR-corrected:    {corrected}")
    print(f"Significant:       {reject}")
    
    # 3. Model comparison
    print("\n3. Model Comparison Test:")
    y_pred2 = y_true + np.random.normal(0, 4, n_samples)
    comparison = stats.model_comparison_test(y_true, y_pred, y_pred2)
    
    print(f"Test: {comparison['test_name']}")
    print(f"P-value: {comparison['p_value']:.4f}")
    print(f"Significant: {comparison['significant']}")
    
    print("\n✓ Statistical rigor framework initialized successfully")


if __name__ == "__main__":
    demonstrate_statistical_rigor()
