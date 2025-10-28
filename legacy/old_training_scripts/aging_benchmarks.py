#!/usr/bin/env python3
"""
Aging Research Benchmarks and Literature-Based Performance Standards

This module provides realistic performance targets and comparison with published
aging clocks (Horvath, Hannum, PhenoAge, GrimAge) to ensure our models meet
scientific standards expected in aging research.

Key Features:
1. Literature-based performance benchmarks
2. Comparison with published aging clocks
3. Age-stratified performance analysis
4. Sex-specific model evaluation
5. Confidence interval calculations
6. Statistical significance testing

Scientific References:
- Horvath S. (2013) Genome Biology. DNA methylation age of human tissues
- Hannum G. et al. (2013) Mol Cell. Genome-wide methylation profiles
- Levine M.E. et al. (2018) Aging. An epigenetic biomarker of aging (PhenoAge)
- Lu A.T. et al. (2019) Aging. DNA methylation GrimAge

Author: Anti-Aging ML Project
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AgingClockBenchmark:
    """Container for aging clock performance benchmarks."""
    
    name: str
    r2_score: float
    mae_years: float
    rmse_years: float
    correlation: float
    age_range: Tuple[int, int]
    sample_size: int
    tissue_type: str
    reference: str
    year_published: int


class AgingBenchmarkLibrary:
    """
    Library of published aging clock performance benchmarks.
    
    Provides realistic performance targets based on peer-reviewed
    aging research for comparison with our models.
    """
    
    # Published aging clock benchmarks from literature
    PUBLISHED_CLOCKS = {
        'horvath_2013': AgingClockBenchmark(
            name='Horvath Clock (2013)',
            r2_score=0.84,
            mae_years=3.6,
            rmse_years=4.9,
            correlation=0.92,
            age_range=(0, 101),
            sample_size=8000,
            tissue_type='Multi-tissue',
            reference='Horvath S. (2013) Genome Biol 14:R115',
            year_published=2013
        ),
        'hannum_2013': AgingClockBenchmark(
            name='Hannum Clock (2013)',
            r2_score=0.76,
            mae_years=4.2,
            rmse_years=5.8,
            correlation=0.87,
            age_range=(19, 101),
            sample_size=656,
            tissue_type='Blood',
            reference='Hannum G. et al. (2013) Mol Cell 49:359-367',
            year_published=2013
        ),
        'phenoage_2018': AgingClockBenchmark(
            name='PhenoAge (2018)',
            r2_score=0.71,
            mae_years=5.1,
            rmse_years=6.7,
            correlation=0.84,
            age_range=(20, 90),
            sample_size=9926,
            tissue_type='Blood',
            reference='Levine M.E. et al. (2018) Aging 10:573-591',
            year_published=2018
        ),
        'grimage_2019': AgingClockBenchmark(
            name='GrimAge (2019)',
            r2_score=0.82,
            mae_years=3.9,
            rmse_years=5.2,
            correlation=0.90,
            age_range=(18, 90),
            sample_size=1731,
            tissue_type='Blood',
            reference='Lu A.T. et al. (2019) Aging 11:303-327',
            year_published=2019
        ),
        'skinblood_2018': AgingClockBenchmark(
            name='Skin-Blood Clock (2018)',
            r2_score=0.78,
            mae_years=4.5,
            rmse_years=6.0,
            correlation=0.88,
            age_range=(20, 85),
            sample_size=4000,
            tissue_type='Skin/Blood',
            reference='Horvath S. et al. (2018) Aging 10:1758-1775',
            year_published=2018
        )
    }
    
    # Performance categories based on literature
    PERFORMANCE_CATEGORIES = {
        'excellent': {
            'r2_range': (0.75, 0.85),
            'mae_range': (3.0, 5.0),
            'description': 'State-of-the-art aging prediction (top-tier published clocks)'
        },
        'good': {
            'r2_range': (0.65, 0.75),
            'mae_range': (5.0, 7.0),
            'description': 'Strong aging prediction (good published clocks)'
        },
        'acceptable': {
            'r2_range': (0.55, 0.65),
            'mae_range': (7.0, 9.0),
            'description': 'Acceptable aging prediction (baseline research-grade)'
        },
        'needs_improvement': {
            'r2_range': (0.40, 0.55),
            'mae_range': (9.0, 12.0),
            'description': 'Below research standards (requires improvement)'
        }
    }
    
    @classmethod
    def get_benchmark(cls, clock_name: str) -> Optional[AgingClockBenchmark]:
        """Get benchmark for specific aging clock."""
        return cls.PUBLISHED_CLOCKS.get(clock_name)
    
    @classmethod
    def get_all_benchmarks(cls) -> Dict[str, AgingClockBenchmark]:
        """Get all published aging clock benchmarks."""
        return cls.PUBLISHED_CLOCKS.copy()
    
    @classmethod
    def categorize_performance(cls, r2: float, mae: float) -> Tuple[str, str]:
        """
        Categorize model performance based on literature standards.
        
        Args:
            r2: R-squared score
            mae: Mean absolute error in years
            
        Returns:
            Tuple of (category, description)
        """
        for category, criteria in cls.PERFORMANCE_CATEGORIES.items():
            r2_min, r2_max = criteria['r2_range']
            mae_min, mae_max = criteria['mae_range']
            
            if r2_min <= r2 <= r2_max and mae_min <= mae <= mae_max:
                return category, criteria['description']
        
        # If outside ranges, determine which side
        if r2 > 0.85 and mae < 3.0:
            return 'exceptional', 'Exceptional performance (exceeds published benchmarks)'
        else:
            return 'poor', 'Poor performance (significantly below research standards)'
    
    @classmethod
    def compare_to_literature(cls, r2: float, mae: float, rmse: float) -> Dict:
        """
        Compare model performance to published aging clocks.
        
        Args:
            r2: Model R-squared score
            mae: Model mean absolute error
            rmse: Model root mean squared error
            
        Returns:
            Dictionary with comparison results
        """
        category, description = cls.categorize_performance(r2, mae)
        
        # Find closest published clock
        closest_clock = None
        min_distance = float('inf')
        
        for clock_name, benchmark in cls.PUBLISHED_CLOCKS.items():
            # Euclidean distance in (R², MAE) space (normalized)
            distance = np.sqrt(
                ((r2 - benchmark.r2_score) / 0.1) ** 2 + 
                ((mae - benchmark.mae_years) / 2.0) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_clock = (clock_name, benchmark)
        
        comparison = {
            'performance_category': category,
            'category_description': description,
            'model_metrics': {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            },
            'closest_published_clock': {
                'name': closest_clock[1].name,
                'reference': closest_clock[1].reference,
                'r2_diff': r2 - closest_clock[1].r2_score,
                'mae_diff': mae - closest_clock[1].mae_years
            },
            'literature_context': cls._generate_literature_context(r2, mae)
        }
        
        return comparison
    
    @classmethod
    def _generate_literature_context(cls, r2: float, mae: float) -> str:
        """Generate interpretive context based on literature."""
        contexts = []
        
        # R² context
        if r2 >= 0.80:
            contexts.append("R² comparable to best published clocks (Horvath, GrimAge)")
        elif r2 >= 0.70:
            contexts.append("R² comparable to good published clocks (PhenoAge)")
        elif r2 >= 0.55:
            contexts.append("R² within acceptable range for aging prediction research")
        else:
            contexts.append("R² below typical aging clock performance")
        
        # MAE context
        if mae <= 4.0:
            contexts.append("MAE comparable to state-of-the-art aging clocks")
        elif mae <= 6.0:
            contexts.append("MAE within typical range for published aging clocks")
        elif mae <= 8.0:
            contexts.append("MAE acceptable for research-grade aging prediction")
        else:
            contexts.append("MAE higher than typical published aging clocks")
        
        return "; ".join(contexts)


class RealisticModelEvaluator:
    """
    Comprehensive model evaluation following aging research standards.
    
    Implements age-stratified analysis, sex-specific evaluation, confidence
    intervals, and comparison with published benchmarks.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.benchmark_library = AgingBenchmarkLibrary()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('aging_evaluator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_comprehensive(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              age: np.ndarray,
                              sex: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive evaluation with literature comparison.
        
        Args:
            y_true: True biological ages
            y_pred: Predicted biological ages
            age: Chronological ages
            sex: Optional sex labels for sex-specific analysis
            
        Returns:
            Comprehensive evaluation report
        """
        self.logger.info("Starting comprehensive aging model evaluation")
        
        report = {
            'overall_performance': self._calculate_overall_metrics(y_true, y_pred),
            'age_stratified': self._age_stratified_analysis(y_true, y_pred, age),
            'literature_comparison': None,
            'confidence_intervals': self._bootstrap_confidence_intervals(y_true, y_pred),
            'residual_analysis': self._analyze_residuals(y_true, y_pred, age)
        }
        
        # Add sex-specific analysis if available
        if sex is not None:
            report['sex_specific'] = self._sex_specific_analysis(y_true, y_pred, sex)
        
        # Compare to literature
        overall = report['overall_performance']
        report['literature_comparison'] = self.benchmark_library.compare_to_literature(
            overall['r2'], overall['mae'], overall['rmse']
        )
        
        self.logger.info(f"Evaluation completed: R² = {overall['r2']:.3f}, MAE = {overall['mae']:.2f} years")
        
        return report
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate overall performance metrics."""
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1],
            'median_absolute_error': np.median(np.abs(y_true - y_pred)),
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def _age_stratified_analysis(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 age: np.ndarray) -> Dict:
        """
        Age-stratified performance analysis.
        
        Important for aging research as model performance often varies by age group.
        """
        # Define age groups matching aging research literature
        age_groups = {
            'young_adult': (25, 40),
            'middle_aged': (40, 60),
            'elderly': (60, 80)
        }
        
        stratified_results = {}
        
        for group_name, (age_min, age_max) in age_groups.items():
            mask = (age >= age_min) & (age < age_max)
            
            if np.sum(mask) > 10:  # Need sufficient samples
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                
                stratified_results[group_name] = {
                    'age_range': (age_min, age_max),
                    'n_samples': np.sum(mask),
                    'r2': r2_score(group_y_true, group_y_pred),
                    'mae': mean_absolute_error(group_y_true, group_y_pred),
                    'rmse': np.sqrt(mean_squared_error(group_y_true, group_y_pred)),
                    'correlation': np.corrcoef(group_y_true, group_y_pred)[0, 1]
                }
        
        return stratified_results
    
    def _sex_specific_analysis(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              sex: np.ndarray) -> Dict:
        """
        Sex-specific performance analysis.
        
        Aging rates can differ between sexes, requiring separate evaluation.
        """
        sex_results = {}
        
        for sex_label in np.unique(sex):
            mask = sex == sex_label
            
            if np.sum(mask) > 10:
                sex_y_true = y_true[mask]
                sex_y_pred = y_pred[mask]
                
                sex_results[str(sex_label)] = {
                    'n_samples': np.sum(mask),
                    'r2': r2_score(sex_y_true, sex_y_pred),
                    'mae': mean_absolute_error(sex_y_true, sex_y_pred),
                    'rmse': np.sqrt(mean_squared_error(sex_y_true, sex_y_pred)),
                    'correlation': np.corrcoef(sex_y_true, sex_y_pred)[0, 1]
                }
        
        return sex_results
    
    def _bootstrap_confidence_intervals(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       n_bootstrap: int = 1000,
                                       confidence_level: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Essential for understanding uncertainty in performance estimates.
        """
        self.logger.info(f"Calculating bootstrap confidence intervals ({n_bootstrap} iterations)")
        
        n_samples = len(y_true)
        bootstrap_metrics = {
            'r2': [],
            'mae': [],
            'rmse': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            bootstrap_metrics['r2'].append(r2_score(y_true_boot, y_pred_boot))
            bootstrap_metrics['mae'].append(mean_absolute_error(y_true_boot, y_pred_boot))
            bootstrap_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}
        
        for metric, values in bootstrap_metrics.items():
            values = np.array(values)
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            mean = np.mean(values)
            
            confidence_intervals[metric] = {
                'mean': mean,
                'lower': lower,
                'upper': upper,
                'std': np.std(values)
            }
        
        return confidence_intervals
    
    def _analyze_residuals(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          age: np.ndarray) -> Dict:
        """
        Analyze prediction residuals for patterns.
        
        Important for detecting systematic biases in aging prediction.
        """
        residuals = y_pred - y_true
        
        analysis = {
            'mean_residual': np.mean(residuals),
            'median_residual': np.median(residuals),
            'std_residual': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'age_correlation': np.corrcoef(age, residuals)[0, 1]
        }
        
        # Test for age-dependent bias
        if abs(analysis['age_correlation']) > 0.2:
            analysis['age_bias_warning'] = (
                f"Significant age-dependent bias detected (r={analysis['age_correlation']:.3f}). "
                "Model may systematically over/under-predict at certain ages."
            )
        
        return analysis
    
    def generate_evaluation_report(self, evaluation_results: Dict, output_path: str = None) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_comprehensive
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AGING MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall performance
        overall = evaluation_results['overall_performance']
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(f"  Samples: {overall['n_samples']}")
        report_lines.append(f"  R² Score: {overall['r2']:.4f}")
        report_lines.append(f"  MAE: {overall['mae']:.2f} years")
        report_lines.append(f"  RMSE: {overall['rmse']:.2f} years")
        report_lines.append(f"  Correlation: {overall['correlation']:.4f}")
        report_lines.append(f"  Median Absolute Error: {overall['median_absolute_error']:.2f} years")
        report_lines.append("")
        
        # Literature comparison
        lit_comp = evaluation_results['literature_comparison']
        report_lines.append("LITERATURE COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"  Performance Category: {lit_comp['performance_category'].upper()}")
        report_lines.append(f"  Description: {lit_comp['category_description']}")
        report_lines.append(f"  Context: {lit_comp['literature_context']}")
        report_lines.append("")
        report_lines.append(f"  Closest Published Clock: {lit_comp['closest_published_clock']['name']}")
        report_lines.append(f"  Reference: {lit_comp['closest_published_clock']['reference']}")
        report_lines.append(f"  R² Difference: {lit_comp['closest_published_clock']['r2_diff']:+.4f}")
        report_lines.append(f"  MAE Difference: {lit_comp['closest_published_clock']['mae_diff']:+.2f} years")
        report_lines.append("")
        
        # Confidence intervals
        ci = evaluation_results['confidence_intervals']
        report_lines.append("CONFIDENCE INTERVALS (95%)")
        report_lines.append("-" * 80)
        for metric, values in ci.items():
            report_lines.append(f"  {metric.upper()}:")
            report_lines.append(f"    Mean: {values['mean']:.4f}")
            report_lines.append(f"    95% CI: [{values['lower']:.4f}, {values['upper']:.4f}]")
            report_lines.append(f"    Std: {values['std']:.4f}")
        report_lines.append("")
        
        # Age-stratified results
        if 'age_stratified' in evaluation_results:
            report_lines.append("AGE-STRATIFIED PERFORMANCE")
            report_lines.append("-" * 80)
            for group_name, metrics in evaluation_results['age_stratified'].items():
                report_lines.append(f"  {group_name.replace('_', ' ').title()} ({metrics['age_range'][0]}-{metrics['age_range'][1]} years):")
                report_lines.append(f"    Samples: {metrics['n_samples']}")
                report_lines.append(f"    R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f} years")
            report_lines.append("")
        
        # Sex-specific results
        if 'sex_specific' in evaluation_results:
            report_lines.append("SEX-SPECIFIC PERFORMANCE")
            report_lines.append("-" * 80)
            for sex_label, metrics in evaluation_results['sex_specific'].items():
                report_lines.append(f"  Sex {sex_label}:")
                report_lines.append(f"    Samples: {metrics['n_samples']}")
                report_lines.append(f"    R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f} years")
            report_lines.append("")
        
        # Residual analysis
        residuals = evaluation_results['residual_analysis']
        report_lines.append("RESIDUAL ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"  Mean Residual: {residuals['mean_residual']:.2f} years")
        report_lines.append(f"  Median Residual: {residuals['median_residual']:.2f} years")
        report_lines.append(f"  Std Residual: {residuals['std_residual']:.2f} years")
        report_lines.append(f"  Age Correlation: {residuals['age_correlation']:.4f}")
        
        if 'age_bias_warning' in residuals:
            report_lines.append(f"  ⚠️  WARNING: {residuals['age_bias_warning']}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text


def demonstrate_aging_benchmarks():
    """Demonstrate the aging benchmarks and evaluation system."""
    print("=== Aging Research Benchmarks Library ===\n")
    
    # Show published benchmarks
    print("Published Aging Clock Benchmarks:")
    print("-" * 80)
    
    for clock_name, benchmark in AgingBenchmarkLibrary.get_all_benchmarks().items():
        print(f"\n{benchmark.name}")
        print(f"  R²: {benchmark.r2_score:.3f}, MAE: {benchmark.mae_years:.1f} years")
        print(f"  Sample Size: {benchmark.sample_size}, Tissue: {benchmark.tissue_type}")
        print(f"  Reference: {benchmark.reference}")
    
    print("\n" + "=" * 80)
    print("\nPerformance Categories:")
    print("-" * 80)
    
    for category, criteria in AgingBenchmarkLibrary.PERFORMANCE_CATEGORIES.items():
        print(f"\n{category.upper()}:")
        print(f"  R² Range: {criteria['r2_range']}")
        print(f"  MAE Range: {criteria['mae_range']} years")
        print(f"  {criteria['description']}")
    
    # Example comparison
    print("\n" + "=" * 80)
    print("\nExample Model Comparison:")
    print("-" * 80)
    
    example_r2 = 0.539
    example_mae = 8.2
    example_rmse = 10.5
    
    comparison = AgingBenchmarkLibrary.compare_to_literature(example_r2, example_mae, example_rmse)
    
    print(f"\nModel Performance: R² = {example_r2:.3f}, MAE = {example_mae:.1f} years")
    print(f"Category: {comparison['performance_category'].upper()}")
    print(f"Description: {comparison['category_description']}")
    print(f"Context: {comparison['literature_context']}")
    print(f"\nClosest Clock: {comparison['closest_published_clock']['name']}")
    print(f"  R² Difference: {comparison['closest_published_clock']['r2_diff']:+.3f}")
    print(f"  MAE Difference: {comparison['closest_published_clock']['mae_diff']:+.1f} years")


if __name__ == "__main__":
    demonstrate_aging_benchmarks()
