#!/usr/bin/env python3
"""
Advanced Feature Engineering for Aging Biology

This module implements sophisticated feature engineering based on aging biology
research, including pathway-based features, gene-environment interactions,
polygenic risk scores, and aging-specific transformations.

Key Features:
1. Biological pathway-based feature grouping
2. Gene-environment interaction terms
3. Polygenic risk scores for aging
4. Epigenetic clock-inspired features
5. Non-linear age transformations
6. Sex-specific feature engineering
7. Composite aging biomarker indices

Scientific Foundation:
- Aging pathway databases (KEGG, Reactome, GO)
- Published gene-environment interactions
- Literature-based feature combinations
- Aging biomarker research standards

Author: Anti-Aging ML Project
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class AgingPathway:
    """Container for aging pathway information."""
    
    name: str
    description: str
    genes: List[str]
    weight: float  # Relative importance in aging (0-1)
    hallmark: str  # Which hallmark of aging


class AgingPathwayDatabase:
    """
    Database of aging-related biological pathways.
    
    Based on the Hallmarks of Aging (López-Otín et al., 2013, Cell)
    and related aging biology research.
    """
    
    # Hallmarks of Aging (López-Otín et al., 2013)
    AGING_HALLMARKS = [
        'genomic_instability',
        'telomere_attrition',
        'epigenetic_alterations',
        'loss_of_proteostasis',
        'deregulated_nutrient_sensing',
        'mitochondrial_dysfunction',
        'cellular_senescence',
        'stem_cell_exhaustion',
        'altered_intercellular_communication'
    ]
    
    # Aging pathways mapped to genes in our dataset
    PATHWAYS = {
        'dna_repair': AgingPathway(
            name='DNA Repair Pathway',
            description='DNA damage response and repair mechanisms',
            genes=['TP53', 'TERT', 'TERC'],
            weight=0.20,
            hallmark='genomic_instability'
        ),
        'telomere_maintenance': AgingPathway(
            name='Telomere Maintenance',
            description='Telomere length regulation and protection',
            genes=['TERT', 'TERC'],
            weight=0.20,
            hallmark='telomere_attrition'
        ),
        'cellular_senescence': AgingPathway(
            name='Cellular Senescence',
            description='Cell cycle arrest and senescence pathways',
            genes=['CDKN2A', 'TP53'],
            weight=0.25,
            hallmark='cellular_senescence'
        ),
        'longevity_signaling': AgingPathway(
            name='Longevity Signaling',
            description='Nutrient sensing and longevity pathways',
            genes=['FOXO3', 'SIRT1', 'IGF1', 'KLOTHO'],
            weight=0.35,
            hallmark='deregulated_nutrient_sensing'
        ),
        'oxidative_stress_response': AgingPathway(
            name='Oxidative Stress Response',
            description='Antioxidant and stress resistance pathways',
            genes=['FOXO3', 'SIRT1'],
            weight=0.15,
            hallmark='mitochondrial_dysfunction'
        ),
        'apoptosis_regulation': AgingPathway(
            name='Apoptosis Regulation',
            description='Programmed cell death pathways',
            genes=['TP53', 'APOE'],
            weight=0.10,
            hallmark='loss_of_proteostasis'
        )
    }
    
    @classmethod
    def get_pathway(cls, pathway_name: str) -> Optional[AgingPathway]:
        """Get specific aging pathway."""
        return cls.PATHWAYS.get(pathway_name)
    
    @classmethod
    def get_all_pathways(cls) -> Dict[str, AgingPathway]:
        """Get all aging pathways."""
        return cls.PATHWAYS.copy()
    
    @classmethod
    def get_gene_pathways(cls, gene: str) -> List[AgingPathway]:
        """Get all pathways containing a specific gene."""
        return [pathway for pathway in cls.PATHWAYS.values() if gene in pathway.genes]


class AdvancedAgingFeatureEngineer:
    """
    Advanced feature engineering for aging prediction.
    
    Implements biologically-informed feature engineering based on
    aging research and pathway biology.
    """
    
    def __init__(self, create_interactions: bool = True,
                 max_interaction_order: int = 2,
                 include_polynomial: bool = True,
                 polynomial_degree: int = 2):
        """
        Initialize advanced feature engineer.
        
        Args:
            create_interactions: Whether to create gene-environment interactions
            max_interaction_order: Maximum order for interaction terms
            include_polynomial: Whether to include polynomial features
            polynomial_degree: Degree for polynomial features
        """
        self.create_interactions = create_interactions
        self.max_interaction_order = max_interaction_order
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        
        self.pathway_db = AgingPathwayDatabase()
        self.logger = self._setup_logger()
        
        self.created_features: Dict[str, List[str]] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('aging_feature_engineer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive aging-specific feature engineering.
        
        Args:
            df: Input dataframe with genetic and phenotypic data
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting advanced aging feature engineering")
        
        df_engineered = df.copy()
        
        # 1. Pathway-based features
        df_engineered = self._create_pathway_features(df_engineered)
        
        # 2. Polygenic risk scores
        df_engineered = self._create_polygenic_risk_scores(df_engineered)
        
        # 3. Gene-environment interactions
        if self.create_interactions:
            df_engineered = self._create_gene_environment_interactions(df_engineered)
        
        # 4. Epigenetic aging features
        df_engineered = self._create_epigenetic_aging_features(df_engineered)
        
        # 5. Aging biomarker composites
        df_engineered = self._create_biomarker_composites(df_engineered)
        
        # 6. Non-linear age transformations
        df_engineered = self._create_age_transformations(df_engineered)
        
        # 7. Sex-specific features
        if 'gender' in df_engineered.columns:
            df_engineered = self._create_sex_specific_features(df_engineered)
        
        # 8. Lifestyle pattern features
        df_engineered = self._create_lifestyle_patterns(df_engineered)
        
        self.logger.info(f"Feature engineering completed: {df.shape[1]} → {df_engineered.shape[1]} features")
        
        return df_engineered
    
    def _create_pathway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pathway-level aggregated features.
        
        Aggregates genetic variants within biological pathways to create
        pathway activity scores.
        """
        self.logger.info("Creating pathway-based features")
        
        pathway_features = []
        
        for pathway_name, pathway in self.pathway_db.get_all_pathways().items():
            # Find available gene dosage columns
            gene_cols = []
            for gene in pathway.genes:
                # Look for dosage columns
                dosage_col = f'{gene}_dosage'
                if dosage_col in df.columns:
                    gene_cols.append(dosage_col)
                else:
                    # Try direct gene column
                    for col in df.columns:
                        if gene in col and 'dosage' in col:
                            gene_cols.append(col)
                            break
            
            if gene_cols:
                # Calculate pathway activity score (weighted mean)
                pathway_score = df[gene_cols].mean(axis=1) * pathway.weight
                feature_name = f'{pathway_name}_pathway_score'
                df[feature_name] = pathway_score
                pathway_features.append(feature_name)
                
                # Also create pathway risk indicator (high vs low activity)
                risk_threshold = pathway_score.median()
                df[f'{pathway_name}_high_risk'] = (pathway_score > risk_threshold).astype(int)
                pathway_features.append(f'{pathway_name}_high_risk')
        
        self.created_features['pathway'] = pathway_features
        self.logger.info(f"Created {len(pathway_features)} pathway features")
        
        return df
    
    def _create_polygenic_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create polygenic risk scores for aging-related outcomes.
        
        Weighted combination of genetic variants based on aging research.
        """
        self.logger.info("Creating polygenic risk scores")
        
        prs_features = []
        
        # Define aging-related polygenic risk scores
        prs_definitions = {
            'longevity_prs': {
                'variants': {
                    'APOE_e4_dosage': -0.30,  # Risk allele
                    'FOXO3_A_dosage': 0.25,   # Protective
                    'SIRT1_C_dosage': 0.20,   # Protective
                    'KLOTHO_G_dosage': 0.22   # Protective
                },
                'description': 'Polygenic risk score for longevity'
            },
            'cellular_aging_prs': {
                'variants': {
                    'TP53_G_dosage': -0.20,
                    'CDKN2A_G_dosage': -0.25,
                    'TERT_T_dosage': 0.15,
                    'TERC_G_dosage': 0.12
                },
                'description': 'Polygenic risk score for cellular aging'
            },
            'metabolic_aging_prs': {
                'variants': {
                    'IGF1_CA_dosage': -0.18,
                    'KLOTHO_G_dosage': 0.28,
                    'SIRT1_C_dosage': 0.15
                },
                'description': 'Polygenic risk score for metabolic aging'
            },
            'neurodegeneration_prs': {
                'variants': {
                    'APOE_e4_dosage': -0.40,  # Strong Alzheimer's risk
                    'FOXO3_A_dosage': 0.15,
                    'KLOTHO_G_dosage': 0.20
                },
                'description': 'Polygenic risk score for neurodegeneration'
            }
        }
        
        for prs_name, prs_info in prs_definitions.items():
            prs_score = 0
            n_variants = 0
            
            for variant, weight in prs_info['variants'].items():
                if variant in df.columns:
                    prs_score += df[variant] * weight
                    n_variants += 1
            
            if n_variants > 0:
                # Normalize by number of variants
                df[prs_name] = prs_score / n_variants
                prs_features.append(prs_name)
                
                # Create risk categories
                prs_tertiles = pd.qcut(df[prs_name], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                df[f'{prs_name}_category'] = prs_tertiles
        
        self.created_features['polygenic_risk'] = prs_features
        self.logger.info(f"Created {len(prs_features)} polygenic risk scores")
        
        return df
    
    def _create_gene_environment_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create gene × environment interaction terms.
        
        Models how genetic variants modulate environmental/lifestyle effects.
        """
        self.logger.info("Creating gene-environment interactions")
        
        interaction_features = []
        
        # Define biologically meaningful interactions
        interactions = [
            # Exercise interactions
            ('FOXO3_A_dosage', 'exercise_frequency', 'FOXO3 enhances exercise benefits'),
            ('SIRT1_C_dosage', 'exercise_frequency', 'SIRT1 modulates exercise response'),
            ('IGF1_CA_dosage', 'exercise_frequency', 'IGF1 affects exercise adaptation'),
            
            # Diet interactions
            ('SIRT1_C_dosage', 'diet_quality', 'SIRT1 mediates dietary restriction effects'),
            ('FOXO3_A_dosage', 'diet_quality', 'FOXO3 responds to nutritional status'),
            ('KLOTHO_G_dosage', 'diet_quality', 'Klotho affects nutrient metabolism'),
            
            # Stress interactions
            ('TP53_G_dosage', 'stress_level', 'TP53 responds to stress'),
            ('FOXO3_A_dosage', 'stress_level', 'FOXO3 mediates stress resistance'),
            
            # Sleep interactions
            ('SIRT1_C_dosage', 'sleep_quality', 'SIRT1 regulates circadian rhythms'),
            ('CDKN2A_G_dosage', 'sleep_quality', 'Sleep affects cellular senescence'),
            
            # Environmental exposure interactions
            ('TP53_G_dosage', 'pollution_exposure', 'TP53 responds to environmental damage'),
            ('TERT_T_dosage', 'pollution_exposure', 'Telomeres affected by pollution'),
            
            # Health marker interactions
            ('APOE_e4_dosage', 'health_score', 'APOE modulates health outcomes'),
            ('KLOTHO_G_dosage', 'health_score', 'Klotho protects against disease')
        ]
        
        for genetic_var, environment_var, description in interactions:
            if genetic_var in df.columns and environment_var in df.columns:
                interaction_name = f'{genetic_var}_x_{environment_var}'
                df[interaction_name] = df[genetic_var] * df[environment_var]
                interaction_features.append(interaction_name)
        
        # Create higher-order interactions for key variants
        key_lifestyle = ['exercise_frequency', 'diet_quality', 'sleep_quality']
        key_genetics = ['FOXO3_A_dosage', 'SIRT1_C_dosage']
        
        for genetic in key_genetics:
            if genetic in df.columns:
                # Three-way interactions: gene × lifestyle1 × lifestyle2
                for lifestyle1, lifestyle2 in combinations(key_lifestyle, 2):
                    if lifestyle1 in df.columns and lifestyle2 in df.columns:
                        interaction_name = f'{genetic}_x_{lifestyle1}_x_{lifestyle2}'
                        df[interaction_name] = (df[genetic] * 
                                               df[lifestyle1] * 
                                               df[lifestyle2])
                        interaction_features.append(interaction_name)
        
        self.created_features['gene_environment'] = interaction_features
        self.logger.info(f"Created {len(interaction_features)} gene-environment interactions")
        
        return df
    
    def _create_epigenetic_aging_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create epigenetic clock-inspired features.
        
        Mimics feature engineering from Horvath and Hannum aging clocks.
        """
        self.logger.info("Creating epigenetic aging features")
        
        epigenetic_features = []
        
        # Get methylation columns
        methylation_cols = [col for col in df.columns if col.startswith('CpG')]
        
        if methylation_cols:
            # Methylation variability (epigenetic entropy)
            df['methylation_variance'] = df[methylation_cols].var(axis=1)
            df['methylation_range'] = (df[methylation_cols].max(axis=1) - 
                                      df[methylation_cols].min(axis=1))
            df['methylation_cv'] = (df[methylation_cols].std(axis=1) / 
                                   (df[methylation_cols].mean(axis=1) + 1e-10))
            
            epigenetic_features.extend(['methylation_variance', 'methylation_range', 'methylation_cv'])
            
            # Hypermethylation index (proportion of highly methylated sites)
            hypermethylated = (df[methylation_cols] > 0.7).sum(axis=1) / len(methylation_cols)
            df['hypermethylation_index'] = hypermethylated
            epigenetic_features.append('hypermethylation_index')
            
            # Hypomethylation index
            hypomethylated = (df[methylation_cols] < 0.3).sum(axis=1) / len(methylation_cols)
            df['hypomethylation_index'] = hypomethylated
            epigenetic_features.append('hypomethylation_index')
            
            # Methylation drift (distance from median methylation)
            median_methylation = df[methylation_cols].median(axis=1)
            df['methylation_drift'] = np.abs(df[methylation_cols].subtract(median_methylation, axis=0)).mean(axis=1)
            epigenetic_features.append('methylation_drift')
        
        self.created_features['epigenetic'] = epigenetic_features
        self.logger.info(f"Created {len(epigenetic_features)} epigenetic features")
        
        return df
    
    def _create_biomarker_composites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite aging biomarker indices.
        
        Combines multiple biomarkers into composite scores used in aging research.
        Avoids using age-derived features to prevent target leakage when predicting age.
        """
        self.logger.info("Creating biomarker composite indices")
        
        composite_features = []
        
        # Skip age acceleration - causes leakage when predicting age
        # Only use when predicting other outcomes
        
        # Frailty index components (age-independent)
        frailty_components = []
        
        if 'health_score' in df.columns:
            frailty_components.append('health_score')
        
        if 'exercise_frequency' in df.columns:
            df['low_physical_activity'] = (df['exercise_frequency'] < df['exercise_frequency'].median()).astype(int)
            frailty_components.append('low_physical_activity')
        
        if frailty_components:
            df['frailty_index'] = df[frailty_components].mean(axis=1)
            composite_features.append('frailty_index')
        
        # Skip healthspan indicator - it uses age directly (leakage)
        
        # Allostatic load proxy (cumulative stress burden) - age-independent
        stress_indicators = []
        if 'stress_level' in df.columns:
            stress_indicators.append('stress_level')
        if 'pollution_exposure' in df.columns:
            stress_indicators.append('pollution_exposure')
        
        if stress_indicators:
            df['allostatic_load'] = df[stress_indicators].mean(axis=1)
            composite_features.append('allostatic_load')
        
        self.created_features['composite_biomarkers'] = composite_features
        self.logger.info(f"Created {len(composite_features)} composite biomarkers")
        
        return df
    
    def _create_age_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create non-linear age transformations.
        
        Captures non-linear aging patterns observed in biology.
        NOTE: These transformations should NOT be used when predicting chronological age
        as they cause direct data leakage. Only use when predicting biological age or
        age-related outcomes.
        """
        self.logger.info("Creating age transformations")
        
        age_features = []
        
        # Skip age transformations - they cause data leakage when predicting age
        # These should only be used when predicting OTHER outcomes (biological age, disease risk, etc.)
        self.logger.info("Skipping age transformations to prevent target leakage")
        
        self.created_features['age_transformations'] = age_features
        self.logger.info(f"Created {len(age_features)} age transformation features")
        
        return df
    
    def _create_sex_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sex-specific aging features.
        
        Men and women show different aging patterns requiring sex-specific features.
        Avoids age interactions to prevent target leakage when predicting age.
        """
        self.logger.info("Creating sex-specific features")
        
        sex_features = []
        
        # Encode gender if needed
        if 'gender' in df.columns and df['gender'].dtype == 'object':
            gender_encoded = pd.get_dummies(df['gender'], prefix='gender', drop_first=True)
            for col in gender_encoded.columns:
                df[col] = gender_encoded[col]
                sex_features.append(col)
        
        # Skip sex × age interactions - causes leakage when predicting age
        
        # Sex-specific genetic effects (age-independent)
        sex_specific_genes = ['APOE_e4_dosage', 'FOXO3_A_dosage', 'KLOTHO_G_dosage']
        
        for gene in sex_specific_genes:
            if gene in df.columns:
                for gender_col in [col for col in df.columns if col.startswith('gender_')]:
                    interaction = f'{gene}_x_{gender_col}'
                    df[interaction] = df[gene] * df[gender_col]
                    sex_features.append(interaction)
        
        self.created_features['sex_specific'] = sex_features
        self.logger.info(f"Created {len(sex_features)} sex-specific features")
        
        return df
    
    def _create_lifestyle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lifestyle pattern composite features.
        
        Captures synergistic effects of multiple lifestyle factors.
        """
        self.logger.info("Creating lifestyle pattern features")
        
        lifestyle_features = []
        
        # Healthy lifestyle score (composite of positive behaviors)
        healthy_components = []
        
        if 'exercise_frequency' in df.columns:
            df['high_exercise'] = (df['exercise_frequency'] > df['exercise_frequency'].median()).astype(int)
            healthy_components.append('high_exercise')
        
        if 'diet_quality' in df.columns:
            df['healthy_diet'] = (df['diet_quality'] > df['diet_quality'].median()).astype(int)
            healthy_components.append('healthy_diet')
        
        if 'sleep_quality' in df.columns:
            df['good_sleep'] = (df['sleep_quality'] > df['sleep_quality'].median()).astype(int)
            healthy_components.append('good_sleep')
        
        if 'stress_level' in df.columns:
            df['low_stress'] = (df['stress_level'] < df['stress_level'].median()).astype(int)
            healthy_components.append('low_stress')
        
        if healthy_components:
            df['healthy_lifestyle_score'] = df[healthy_components].sum(axis=1)
            lifestyle_features.append('healthy_lifestyle_score')
            
            # Lifestyle pattern categories
            df['optimal_lifestyle'] = (df['healthy_lifestyle_score'] >= len(healthy_components) - 1).astype(int)
            df['poor_lifestyle'] = (df['healthy_lifestyle_score'] <= 1).astype(int)
            lifestyle_features.extend(['optimal_lifestyle', 'poor_lifestyle'])
        
        # Mediterranean diet pattern proxy
        if 'diet_quality' in df.columns and 'exercise_frequency' in df.columns:
            df['mediterranean_pattern'] = (df['diet_quality'] * 0.6 + 
                                          df['exercise_frequency'] * 0.4)
            lifestyle_features.append('mediterranean_pattern')
        
        self.created_features['lifestyle_patterns'] = lifestyle_features
        self.logger.info(f"Created {len(lifestyle_features)} lifestyle pattern features")
        
        return df
    
    def get_feature_report(self) -> Dict:
        """
        Generate report of created features.
        
        Returns:
            Dictionary with feature creation summary
        """
        report = {
            'total_features_created': sum(len(features) for features in self.created_features.values()),
            'feature_groups': {group: len(features) for group, features in self.created_features.items()},
            'feature_details': self.created_features.copy()
        }
        
        return report


def demonstrate_feature_engineering():
    """Demonstrate advanced feature engineering."""
    print("=== Advanced Aging Feature Engineering Demo ===\n")
    
    # Load data
    data_path = "/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/data/datasets/train.csv"
    df = pd.read_csv(data_path)
    
    print(f"Input data shape: {df.shape}")
    print(f"Input features: {df.columns.tolist()[:10]}...\n")
    
    # Initialize feature engineer
    engineer = AdvancedAgingFeatureEngineer(
        create_interactions=True,
        include_polynomial=False  # Skip for speed in demo
    )
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    print(f"\nOutput data shape: {df_engineered.shape}")
    print(f"Features added: {df_engineered.shape[1] - df.shape[1]}")
    
    # Get report
    report = engineer.get_feature_report()
    
    print(f"\n=== Feature Engineering Report ===")
    print(f"Total new features: {report['total_features_created']}")
    print(f"\nFeatures by group:")
    for group, count in report['feature_groups'].items():
        print(f"  {group}: {count} features")
    
    print(f"\nExample new features:")
    for group, features in list(report['feature_details'].items())[:3]:
        print(f"\n{group}:")
        for feature in features[:5]:
            print(f"  - {feature}")


if __name__ == "__main__":
    demonstrate_feature_engineering()
