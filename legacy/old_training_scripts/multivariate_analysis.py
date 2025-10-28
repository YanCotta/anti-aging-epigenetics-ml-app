"""
Multivariate Statistical Analysis for Feature Groupings and Colinear Relationships (Issue #42)

This module implements clustering/grouping analysis and canonical correlation analysis
to discover colinear relationships between dataset variables. The hypothesis is that
variables can be grouped (genetic, lifestyle, demographic, health markers, environmental)
with different weights and impacts on model performance based on their relationships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class FeatureGroupAnalyzer:
    """
    Analyzer for feature groupings and colinear relationships in anti-aging dataset.
    """
    
    def __init__(self):
        self.feature_groups = {
            'genetic': [],
            'lifestyle': [],
            'demographic': [],
            'health_markers': [],
            'environmental': []
        }
        self.correlation_matrix = None
        self.cluster_labels = None
        self.canonical_correlations = None
        
    def define_feature_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features into predefined groups based on their names.
        """
        genetic_patterns = ['SNP', 'methylation', 'genetic_aging_score', 'longevity_alleles', 'risk_alleles']
        lifestyle_patterns = ['exercise_frequency', 'sleep_hours', 'stress_level', 'diet_quality', 'smoking', 'alcohol']
        demographic_patterns = ['age', 'gender', 'height', 'weight', 'bmi']
        health_patterns = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose', 'telomere_length']
        environmental_patterns = ['pollution_exposure', 'sun_exposure', 'occupation_stress']
        
        for feature in feature_names:
            if any(pattern in feature for pattern in genetic_patterns):
                self.feature_groups['genetic'].append(feature)
            elif any(pattern in feature for pattern in lifestyle_patterns):
                self.feature_groups['lifestyle'].append(feature)
            elif any(pattern in feature for pattern in demographic_patterns):
                self.feature_groups['demographic'].append(feature)
            elif any(pattern in feature for pattern in health_patterns):
                self.feature_groups['health_markers'].append(feature)
            elif any(pattern in feature for pattern in environmental_patterns):
                self.feature_groups['environmental'].append(feature)
        
        return self.feature_groups
    
    def compute_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix for all numeric features.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'pearson':
            self.correlation_matrix = df[numeric_cols].corr()
        elif method == 'spearman':
            self.correlation_matrix = df[numeric_cols].corr(method='spearman')
        
        return self.correlation_matrix
    
    def perform_hierarchical_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
        """
        Perform hierarchical clustering on features based on correlation.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Use correlation distance for clustering
        distance_matrix = 1 - corr_matrix
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.cluster_labels = clustering.fit_predict(distance_matrix)
        
        return self.cluster_labels
    
    def perform_kmeans_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
        """
        Perform K-means clustering on standardized features.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols].T)  # Transpose to cluster features
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(scaled_data)
        
        return self.cluster_labels
    
    def canonical_correlation_analysis(self, df: pd.DataFrame, group1_features: List[str], 
                                     group2_features: List[str]) -> Dict[str, Any]:
        """
        Perform canonical correlation analysis between two feature groups.
        """
        try:
            from sklearn.cross_decomposition import CCA
            
            # Get data for both groups
            X1 = df[group1_features].dropna()
            X2 = df[group2_features].dropna()
            
            # Ensure same indices
            common_idx = X1.index.intersection(X2.index)
            X1 = X1.loc[common_idx]
            X2 = X2.loc[common_idx]
            
            # Standardize the data
            scaler1 = StandardScaler()
            scaler2 = StandardScaler()
            X1_scaled = scaler1.fit_transform(X1)
            X2_scaled = scaler2.fit_transform(X2)
            
            # Perform CCA
            n_components = min(X1.shape[1], X2.shape[1], 3)  # Max 3 components
            cca = CCA(n_components=n_components)
            cca.fit(X1_scaled, X2_scaled)
            
            # Transform data
            X1_c, X2_c = cca.transform(X1_scaled, X2_scaled)
            
            # Calculate canonical correlations
            canonical_corrs = []
            for i in range(n_components):
                corr, _ = pearsonr(X1_c[:, i], X2_c[:, i])
                canonical_corrs.append(abs(corr))
            
            return {
                'canonical_correlations': canonical_corrs,
                'n_components': n_components,
                'group1_features': group1_features,
                'group2_features': group2_features,
                'x_weights': cca.x_weights_,
                'y_weights': cca.y_weights_
            }
            
        except ImportError:
            # Fallback to simple correlation if CCA not available
            corr_matrix = df[group1_features + group2_features].corr()
            cross_corr = corr_matrix.loc[group1_features, group2_features]
            return {
                'cross_correlation_matrix': cross_corr,
                'mean_cross_correlation': cross_corr.abs().mean().mean(),
                'max_cross_correlation': cross_corr.abs().max().max()
            }
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Analyze feature importance using mutual information.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(df[target_col].mean())
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        return dict(zip(feature_cols, mi_scores))
    
    def generate_correlation_heatmap(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Generate correlation heatmap with feature groupings highlighted.
        """
        corr_matrix = self.compute_correlation_matrix(df)
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    def generate_cluster_dendrogram(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Generate dendrogram for hierarchical clustering of features.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        distance_matrix = 1 - corr_matrix
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(linkage_matrix, labels=numeric_cols, ax=ax, orientation='top')
        ax.set_title('Feature Clustering Dendrogram')
        ax.tick_params(axis='x', rotation=90)
        plt.tight_layout()
        
        return fig
    
    def comprehensive_analysis(self, df: pd.DataFrame, target_col: str = 'biological_age') -> Dict[str, Any]:
        """
        Perform comprehensive multivariate analysis.
        """
        results = {}
        
        # Define feature groups
        feature_names = [col for col in df.columns if col not in ['user_id', target_col]]
        results['feature_groups'] = self.define_feature_groups(feature_names)
        
        # Correlation analysis
        results['correlation_matrix'] = self.compute_correlation_matrix(df)
        
        # Clustering analysis
        results['hierarchical_clusters'] = self.perform_hierarchical_clustering(df)
        results['kmeans_clusters'] = self.perform_kmeans_clustering(df)
        
        # Feature importance
        results['feature_importance'] = self.analyze_feature_importance(df, target_col)
        
        # Canonical correlation between major groups
        genetic_features = [f for f in results['feature_groups']['genetic'] if f in df.columns]
        lifestyle_features = [f for f in results['feature_groups']['lifestyle'] if f in df.columns]
        
        if len(genetic_features) > 1 and len(lifestyle_features) > 1:
            results['genetic_lifestyle_cca'] = self.canonical_correlation_analysis(
                df, genetic_features, lifestyle_features)
        
        demographic_features = [f for f in results['feature_groups']['demographic'] if f in df.columns]
        health_features = [f for f in results['feature_groups']['health_markers'] if f in df.columns]
        
        if len(demographic_features) > 1 and len(health_features) > 1:
            results['demographic_health_cca'] = self.canonical_correlation_analysis(
                df, demographic_features, health_features)
        
        return results


def generate_multivariate_report(df: pd.DataFrame, output_path: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive multivariate analysis report.
    """
    analyzer = FeatureGroupAnalyzer()
    results = analyzer.comprehensive_analysis(df)
    
    # Generate summary statistics
    summary = {
        'total_features': len([col for col in df.columns if col not in ['user_id', 'biological_age']]),
        'feature_group_sizes': {k: len(v) for k, v in results['feature_groups'].items()},
        'high_correlation_pairs': [],
        'top_features_by_importance': []
    }
    
    # Find highly correlated feature pairs
    corr_matrix = results['correlation_matrix']
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.7:  # High correlation threshold
                summary['high_correlation_pairs'].append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Top features by importance
    feature_importance = results['feature_importance']
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    summary['top_features_by_importance'] = sorted_features[:10]
    
    results['summary'] = summary
    
    if output_path:
        # Save results to file
        import json
        with open(output_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    json_results[key] = value.to_dict()
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Multivariate Analysis Module for Anti-Aging ML App")
    print("Usage: from multivariate_analysis import FeatureGroupAnalyzer, generate_multivariate_report")