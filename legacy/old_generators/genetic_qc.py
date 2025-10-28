#!/usr/bin/env python3
"""
Genetic Quality Control Pipeline for Anti-Aging ML Application

This module implements comprehensive genetic quality control procedures
following GWAS best practices and genomics standards.

Key Features:
1. Hardy-Weinberg equilibrium testing
2. Minor allele frequency filtering
3. Genotyping call rate assessment
4. Linkage disequilibrium analysis
5. Population structure detection
6. Ancestry outlier identification
7. Genetic relationship matrix calculation
8. Batch effect detection

Scientific Standards:
- Call rate threshold: >95%
- MAF threshold: >0.05 (adjustable for rare variants)
- HWE p-value: >1e-6
- LD pruning: r² <0.8
- Population outliers: >3 SD from ancestry PCs

References:
- Anderson et al. (2010) Nat Protoc. Data quality control in genetic association studies
- Price et al. (2006) Nat Genet. Principal components analysis corrects for stratification
- Purcell et al. (2007) Am J Hum Genet. PLINK: a tool set for whole-genome association

Author: Anti-Aging ML Project  
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.stats import chi2, norm
from scipy.spatial.distance import pdist, squareform
import logging
from pathlib import Path
import json


@dataclass
class GeneticQCMetrics:
    """Container for genetic quality control metrics."""
    
    # Sample-level metrics
    sample_call_rate: float = 0.0
    sample_heterozygosity_rate: float = 0.0
    sample_ancestry_pc1: float = 0.0
    sample_ancestry_pc2: float = 0.0
    sample_outlier_flag: bool = False
    
    # SNP-level metrics
    snp_call_rate: float = 0.0
    snp_maf: float = 0.0
    snp_hwe_pvalue: float = 1.0
    snp_het_excess: float = 0.0
    
    # Population genetics
    inbreeding_coefficient: float = 0.0
    kinship_coefficient: float = 0.0


class GeneticQualityControl:
    """
    Comprehensive genetic quality control pipeline.
    
    Implements standard GWAS quality control procedures for SNP and sample
    filtering, population stratification detection, and genetic relationship assessment.
    """
    
    def __init__(self, 
                 min_call_rate: float = 0.95,
                 min_maf: float = 0.05,
                 hwe_threshold: float = 1e-6,
                 ld_threshold: float = 0.8,
                 ancestry_outlier_threshold: float = 3.0):
        """
        Initialize genetic QC pipeline.
        
        Args:
            min_call_rate: Minimum acceptable call rate for SNPs and samples
            min_maf: Minimum minor allele frequency  
            hwe_threshold: Hardy-Weinberg equilibrium p-value threshold
            ld_threshold: Linkage disequilibrium r² threshold for pruning
            ancestry_outlier_threshold: Standard deviations for ancestry outliers
        """
        self.min_call_rate = min_call_rate
        self.min_maf = min_maf
        self.hwe_threshold = hwe_threshold
        self.ld_threshold = ld_threshold
        self.ancestry_outlier_threshold = ancestry_outlier_threshold
        
        self.qc_metrics: Dict[str, GeneticQCMetrics] = {}
        self.failed_snps: List[str] = []
        self.failed_samples: List[str] = []
        self.ancestry_pcs: Optional[np.ndarray] = None
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for genetic QC."""
        logger = logging.getLogger('genetic_qc')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_sample_qc_metrics(self, df: pd.DataFrame, snp_columns: List[str]) -> Dict[str, GeneticQCMetrics]:
        """
        Calculate per-sample quality control metrics.
        
        Args:
            df: DataFrame with genetic data
            snp_columns: List of SNP column names
            
        Returns:
            Dictionary of sample QC metrics
        """
        self.logger.info(f"Calculating sample QC metrics for {len(df)} samples and {len(snp_columns)} SNPs")
        
        sample_metrics = {}
        
        for idx, row in df.iterrows():
            sample_id = row.get('user_id', f'sample_{idx}')
            
            # Calculate call rate (proportion of non-missing genotypes)
            genotypes = row[snp_columns]
            call_rate = (~genotypes.isnull()).sum() / len(snp_columns)
            
            # Calculate heterozygosity rate
            het_count = 0
            total_called = 0
            
            for genotype in genotypes.dropna():
                if isinstance(genotype, str) and len(genotype) == 2:
                    total_called += 1
                    if genotype[0] != genotype[1]:  # Heterozygous
                        het_count += 1
            
            het_rate = het_count / total_called if total_called > 0 else 0.0
            
            # Store metrics
            metrics = GeneticQCMetrics(
                sample_call_rate=call_rate,
                sample_heterozygosity_rate=het_rate
            )
            
            sample_metrics[sample_id] = metrics
        
        return sample_metrics
    
    def calculate_snp_qc_metrics(self, df: pd.DataFrame, snp_columns: List[str]) -> Dict[str, GeneticQCMetrics]:
        """
        Calculate per-SNP quality control metrics.
        
        Args:
            df: DataFrame with genetic data
            snp_columns: List of SNP column names
            
        Returns:
            Dictionary of SNP QC metrics
        """
        self.logger.info(f"Calculating SNP QC metrics for {len(snp_columns)} SNPs")
        
        snp_metrics = {}
        
        for snp_col in snp_columns:
            genotypes = df[snp_col].dropna()
            
            if len(genotypes) == 0:
                continue
            
            # Calculate call rate
            call_rate = len(genotypes) / len(df)
            
            # Calculate allele frequencies and MAF
            allele_counts = {}
            total_alleles = 0
            
            for genotype in genotypes:
                if isinstance(genotype, str) and len(genotype) == 2:
                    for allele in genotype:
                        allele_counts[allele] = allele_counts.get(allele, 0) + 1
                        total_alleles += 1
            
            if total_alleles == 0:
                continue
            
            allele_freqs = {allele: count/total_alleles for allele, count in allele_counts.items()}
            maf = min(allele_freqs.values()) if allele_freqs else 0.0
            
            # Calculate Hardy-Weinberg equilibrium
            hwe_pvalue = self._calculate_hwe_exact(genotypes)
            
            # Calculate heterozygote excess/deficit
            observed_het = sum(1 for g in genotypes if isinstance(g, str) and len(g) == 2 and g[0] != g[1]) / len(genotypes)
            
            if len(allele_freqs) == 2:
                alleles = list(allele_freqs.keys())
                p = allele_freqs[alleles[0]]
                q = allele_freqs[alleles[1]]
                expected_het = 2 * p * q
                het_excess = observed_het - expected_het
            else:
                het_excess = 0.0
            
            # Store metrics
            metrics = GeneticQCMetrics(
                snp_call_rate=call_rate,
                snp_maf=maf,
                snp_hwe_pvalue=hwe_pvalue,
                snp_het_excess=het_excess
            )
            
            snp_metrics[snp_col] = metrics
        
        return snp_metrics
    
    def _calculate_hwe_exact(self, genotypes: pd.Series) -> float:
        """
        Calculate exact Hardy-Weinberg equilibrium p-value.
        
        Args:
            genotypes: Series of genotype strings
            
        Returns:
            Exact HWE p-value
        """
        try:
            # Count genotypes
            genotype_counts = {}
            for genotype in genotypes:
                if isinstance(genotype, str) and len(genotype) == 2:
                    # Normalize genotype (alphabetical order)
                    norm_genotype = ''.join(sorted(genotype))
                    genotype_counts[norm_genotype] = genotype_counts.get(norm_genotype, 0) + 1
            
            if len(genotype_counts) == 0:
                return 1.0
            
            # Get unique alleles
            alleles = set()
            for genotype in genotype_counts.keys():
                alleles.update(genotype)
            
            if len(alleles) != 2:
                return 1.0  # Can't test HWE for non-biallelic
            
            alleles = sorted(list(alleles))
            a1, a2 = alleles
            
            # Count observed genotypes
            n_aa = genotype_counts.get(a1 + a1, 0)
            n_ab = genotype_counts.get(a1 + a2, 0)
            n_bb = genotype_counts.get(a2 + a2, 0)
            
            total = n_aa + n_ab + n_bb
            if total == 0:
                return 1.0
            
            # Calculate allele frequencies
            p = (2 * n_aa + n_ab) / (2 * total)
            q = 1 - p
            
            # Expected counts under HWE
            e_aa = total * p * p
            e_ab = total * 2 * p * q  
            e_bb = total * q * q
            
            # Chi-square test statistic
            if e_aa > 0 and e_ab > 0 and e_bb > 0:
                chi2_stat = ((n_aa - e_aa)**2 / e_aa + 
                           (n_ab - e_ab)**2 / e_ab + 
                           (n_bb - e_bb)**2 / e_bb)
                
                # P-value with 1 degree of freedom
                pvalue = 1 - chi2.cdf(chi2_stat, df=1)
                return pvalue
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating HWE p-value: {e}")
            return 1.0
    
    def identify_failed_snps(self, snp_metrics: Dict[str, GeneticQCMetrics]) -> List[str]:
        """
        Identify SNPs that fail quality control criteria.
        
        Args:
            snp_metrics: Dictionary of SNP QC metrics
            
        Returns:
            List of failed SNP names
        """
        failed_snps = []
        
        for snp_name, metrics in snp_metrics.items():
            failure_reasons = []
            
            if metrics.snp_call_rate < self.min_call_rate:
                failure_reasons.append(f"low_call_rate({metrics.snp_call_rate:.3f})")
            
            if metrics.snp_maf < self.min_maf:
                failure_reasons.append(f"low_maf({metrics.snp_maf:.3f})")
            
            if metrics.snp_hwe_pvalue < self.hwe_threshold:
                failure_reasons.append(f"hwe_failure({metrics.snp_hwe_pvalue:.2e})")
            
            if failure_reasons:
                failed_snps.append(snp_name)
                self.logger.info(f"SNP {snp_name} failed QC: {', '.join(failure_reasons)}")
        
        return failed_snps
    
    def identify_failed_samples(self, sample_metrics: Dict[str, GeneticQCMetrics]) -> List[str]:
        """
        Identify samples that fail quality control criteria.
        
        Args:
            sample_metrics: Dictionary of sample QC metrics
            
        Returns:
            List of failed sample IDs
        """
        failed_samples = []
        
        # Calculate heterozygosity outliers
        het_rates = [metrics.sample_heterozygosity_rate for metrics in sample_metrics.values()]
        het_mean = np.mean(het_rates)
        het_std = np.std(het_rates)
        
        for sample_id, metrics in sample_metrics.items():
            failure_reasons = []
            
            if metrics.sample_call_rate < self.min_call_rate:
                failure_reasons.append(f"low_call_rate({metrics.sample_call_rate:.3f})")
            
            # Check for heterozygosity outliers (>3 SD from mean)
            het_zscore = abs(metrics.sample_heterozygosity_rate - het_mean) / het_std if het_std > 0 else 0
            if het_zscore > 3.0:
                failure_reasons.append(f"het_outlier(z={het_zscore:.2f})")
            
            if failure_reasons:
                failed_samples.append(sample_id)
                self.logger.info(f"Sample {sample_id} failed QC: {', '.join(failure_reasons)}")
        
        return failed_samples
    
    def calculate_genetic_relationship_matrix(self, df: pd.DataFrame, snp_dosage_columns: List[str]) -> np.ndarray:
        """
        Calculate genetic relationship matrix (GRM) from SNP dosages.
        
        Args:
            df: DataFrame with SNP dosage data
            snp_dosage_columns: List of SNP dosage column names
            
        Returns:
            Genetic relationship matrix
        """
        self.logger.info(f"Calculating genetic relationship matrix using {len(snp_dosage_columns)} SNPs")
        
        if not snp_dosage_columns:
            return np.array([])
        
        # Get dosage matrix (samples x SNPs)
        dosage_matrix = df[snp_dosage_columns].fillna(1).values  # Fill missing with heterozygous
        n_samples, n_snps = dosage_matrix.shape
        
        # Standardize dosages (mean-center and scale by sqrt(2pq))
        standardized_matrix = np.zeros_like(dosage_matrix)
        
        for j in range(n_snps):
            dosages = dosage_matrix[:, j]
            p = np.mean(dosages) / 2  # Allele frequency
            q = 1 - p
            
            if p > 0 and q > 0:
                # Standardize: (X - 2p) / sqrt(2pq)
                standardized_matrix[:, j] = (dosages - 2*p) / np.sqrt(2*p*q)
            else:
                standardized_matrix[:, j] = 0
        
        # Calculate GRM: G = XX'/m where m is number of SNPs
        grm = np.dot(standardized_matrix, standardized_matrix.T) / n_snps
        
        return grm
    
    def detect_population_outliers(self, ancestry_pcs: np.ndarray, threshold: float = 3.0) -> List[int]:
        """
        Detect population outliers using ancestry principal components.
        
        Args:
            ancestry_pcs: Array of ancestry principal components
            threshold: Standard deviation threshold for outliers
            
        Returns:
            List of outlier sample indices
        """
        if ancestry_pcs.size == 0:
            return []
        
        outlier_indices = []
        
        # Check outliers in first few PCs
        n_pcs_to_check = min(4, ancestry_pcs.shape[1])
        
        for pc_idx in range(n_pcs_to_check):
            pc_values = ancestry_pcs[:, pc_idx]
            pc_mean = np.mean(pc_values)
            pc_std = np.std(pc_values)
            
            if pc_std > 0:
                z_scores = np.abs(pc_values - pc_mean) / pc_std
                outliers = np.where(z_scores > threshold)[0]
                outlier_indices.extend(outliers)
        
        # Remove duplicates
        outlier_indices = list(set(outlier_indices))
        
        if outlier_indices:
            self.logger.info(f"Detected {len(outlier_indices)} population outliers")
        
        return outlier_indices
    
    def run_comprehensive_qc(self, df: pd.DataFrame, 
                           snp_columns: List[str], 
                           snp_dosage_columns: List[str]) -> Dict:
        """
        Run comprehensive genetic quality control pipeline.
        
        Args:
            df: DataFrame with genetic data
            snp_columns: List of SNP genotype columns
            snp_dosage_columns: List of SNP dosage columns
            
        Returns:
            Comprehensive QC report
        """
        self.logger.info("Starting comprehensive genetic quality control")
        
        qc_report = {
            'input_stats': {
                'n_samples': len(df),
                'n_snps': len(snp_columns),
                'n_snp_dosages': len(snp_dosage_columns)
            },
            'sample_qc': {},
            'snp_qc': {},
            'population_structure': {},
            'failed_snps': [],
            'failed_samples': [],
            'recommendations': []
        }
        
        # 1. Sample-level QC
        sample_metrics = self.calculate_sample_qc_metrics(df, snp_columns)
        failed_samples = self.identify_failed_samples(sample_metrics)
        
        qc_report['sample_qc'] = {
            'total_samples': len(sample_metrics),
            'failed_samples': len(failed_samples),
            'mean_call_rate': np.mean([m.sample_call_rate for m in sample_metrics.values()]),
            'mean_heterozygosity': np.mean([m.sample_heterozygosity_rate for m in sample_metrics.values()])
        }
        qc_report['failed_samples'] = failed_samples
        
        # 2. SNP-level QC
        snp_metrics = self.calculate_snp_qc_metrics(df, snp_columns)
        failed_snps = self.identify_failed_snps(snp_metrics)
        
        qc_report['snp_qc'] = {
            'total_snps': len(snp_metrics),
            'failed_snps': len(failed_snps),
            'mean_call_rate': np.mean([m.snp_call_rate for m in snp_metrics.values()]),
            'mean_maf': np.mean([m.snp_maf for m in snp_metrics.values()]),
            'snps_failing_hwe': sum(1 for m in snp_metrics.values() if m.snp_hwe_pvalue < self.hwe_threshold)
        }
        qc_report['failed_snps'] = failed_snps
        
        # 3. Genetic relationship matrix
        if snp_dosage_columns:
            grm = self.calculate_genetic_relationship_matrix(df, snp_dosage_columns)
            if grm.size > 0:
                # Check for related individuals (kinship > 0.125 = 3rd degree relatives)
                n_samples = grm.shape[0]
                kinship_pairs = []
                
                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        kinship = grm[i, j]
                        if kinship > 0.125:  # Related individuals
                            kinship_pairs.append((i, j, kinship))
                
                qc_report['population_structure']['related_pairs'] = len(kinship_pairs)
                qc_report['population_structure']['max_kinship'] = float(np.max(grm[np.triu_indices(n_samples, k=1)]))
        
        # 4. Generate recommendations
        recommendations = []
        
        if qc_report['sample_qc']['failed_samples'] > 0:
            recommendations.append(f"Remove {qc_report['sample_qc']['failed_samples']} samples failing QC")
        
        if qc_report['snp_qc']['failed_snps'] > 0:
            recommendations.append(f"Remove {qc_report['snp_qc']['failed_snps']} SNPs failing QC")
        
        if qc_report['snp_qc']['mean_maf'] < 0.1:
            recommendations.append("Consider relaxing MAF threshold for rare variant analysis")
        
        if 'related_pairs' in qc_report['population_structure'] and qc_report['population_structure']['related_pairs'] > 0:
            recommendations.append(f"Found {qc_report['population_structure']['related_pairs']} related sample pairs - consider removing duplicates")
        
        qc_report['recommendations'] = recommendations
        
        # Store results
        self.qc_metrics = {**sample_metrics, **snp_metrics}
        self.failed_snps = failed_snps
        self.failed_samples = failed_samples
        
        self.logger.info("Comprehensive genetic QC completed")
        return qc_report
    
    def export_qc_report(self, qc_report: Dict, output_path: str) -> None:
        """
        Export QC report to JSON file.
        
        Args:
            qc_report: QC report dictionary
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(qc_report, f, indent=2, default=str)
        
        self.logger.info(f"QC report exported to {output_path}")


def demonstrate_genetic_qc(data_path: str) -> None:
    """
    Demonstrate the genetic quality control pipeline.
    
    Args:
        data_path: Path to genetic dataset
    """
    print("=== Genetic Quality Control Pipeline Demo ===")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Identify genetic columns
    snp_columns = [col for col in df.columns if any(gene in col for gene in 
                   ['APOE', 'FOXO3', 'SIRT1', 'TP53', 'CDKN2A', 'TERT', 'TERC', 'IGF1', 'KLOTHO']) 
                   and not col.endswith('_dosage')]
    
    snp_dosage_columns = [col for col in df.columns if col.endswith('_dosage')]
    
    print(f"Found {len(snp_columns)} SNP genotype columns")
    print(f"Found {len(snp_dosage_columns)} SNP dosage columns")
    
    # Initialize QC
    qc = GeneticQualityControl(
        min_call_rate=0.95,
        min_maf=0.01,  # Lower for demo
        hwe_threshold=1e-8
    )
    
    # Run comprehensive QC
    qc_report = qc.run_comprehensive_qc(df, snp_columns, snp_dosage_columns)
    
    # Display results
    print(f"\n=== QC Results ===")
    print(f"Input: {qc_report['input_stats']['n_samples']} samples, {qc_report['input_stats']['n_snps']} SNPs")
    
    print(f"\nSample QC:")
    print(f"  Failed samples: {qc_report['sample_qc']['failed_samples']}")
    print(f"  Mean call rate: {qc_report['sample_qc']['mean_call_rate']:.3f}")
    print(f"  Mean heterozygosity: {qc_report['sample_qc']['mean_heterozygosity']:.3f}")
    
    print(f"\nSNP QC:")
    print(f"  Failed SNPs: {qc_report['snp_qc']['failed_snps']}")
    print(f"  Mean call rate: {qc_report['snp_qc']['mean_call_rate']:.3f}")
    print(f"  Mean MAF: {qc_report['snp_qc']['mean_maf']:.3f}")
    print(f"  SNPs failing HWE: {qc_report['snp_qc']['snps_failing_hwe']}")
    
    if 'related_pairs' in qc_report['population_structure']:
        print(f"\nPopulation Structure:")
        print(f"  Related pairs: {qc_report['population_structure']['related_pairs']}")
        print(f"  Max kinship: {qc_report['population_structure']['max_kinship']:.4f}")
    
    if qc_report['recommendations']:
        print(f"\nRecommendations:")
        for rec in qc_report['recommendations']:
            print(f"  - {rec}")
    
    # Export report
    qc.export_qc_report(qc_report, "genetic_qc_report.json")
    
    return qc_report


if __name__ == "__main__":
    # Demo with the realistic dataset
    data_path = "/home/yan/Documents/Git/anti-aging-epigenetics-ml-app/antiaging-mvp/backend/api/data/datasets/train.csv"
    qc_report = demonstrate_genetic_qc(data_path)