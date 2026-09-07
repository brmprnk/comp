#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline aggregate functions for computing single aggregated features from raw data files.

This module supports aggregating selected genes/regions back from the original data files
to create a single scalar value per sample, which can be used directly for ROC analysis
without a machine learning classifier.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from util import calculate_rcov, desarkar_features


def aggregate_fslr_from_files(
    sample_file_paths: List[Path],
    selected_gene_indices: np.ndarray,
    features_dir: str = "./extracted_features/biomart_10kb",
    feature_type: str = 'fslr'
) -> np.ndarray:
    """
    Aggregate FSLR (Fragment Size Ratio) for selected genes from original _BIN.csv files.
    
    For each sample, computes:
        FSLR = sum(long_fragments) / sum(short_fragments) over selected genes
    
    Args:
        sample_file_paths: List of file paths or sample identifiers to load
        selected_gene_indices: Indices of genes to aggregate
        features_dir: Directory containing the _BIN.csv files
    
    Returns:
        Array of aggregated FSLR values, one per sample
    """
    aggregated_values = []
    
    for file_path in sample_file_paths:
        try:
            # Find the corresponding _BIN.csv file
            if isinstance(file_path, (str, Path)):
                file_name = Path(file_path).stem if Path(file_path).suffix else str(file_path)
            else:
                file_name = str(file_path)
            
            # Look for matching _BIN.csv file
            bin_file = Path(features_dir) / f"{file_name}_BIN.csv"
            
            if not bin_file.exists():
                # Try alternative patterns
                possible_files = list(Path(features_dir).glob(f"*{file_name}*_BIN.csv"))
                if possible_files:
                    bin_file = possible_files[0]
                else:
                    print(f"Warning: Could not find _BIN.csv for {file_name}, using NaN")
                    aggregated_values.append(np.nan)
                    continue
            
            # Load the file
            df = pd.read_csv(bin_file)

            if feature_type == 'fslr':
            
                # Check if required columns exist
                if 'absolute_short_fragments' not in df.columns or 'absolute_long_fragments' not in df.columns:
                    print(f"Warning: {bin_file.name} missing fragment columns, using NaN")
                    aggregated_values.append(np.nan)
                    continue
                
                # Extract selected genes
                long_frag = df['absolute_long_fragments'].values[selected_gene_indices]
                short_frag = df['absolute_short_fragments'].values[selected_gene_indices]
                
                # Aggregate: sum over selected genes
                total_long = np.sum(long_frag)
                total_short = np.sum(short_frag)

            else:

                # Check if required columns exist
                if 'mean_coverage_short' not in df.columns or 'mean_coverage_long' not in df.columns:
                    print(f"Warning: {bin_file.name} missing coverage columns, using NaN")
                    aggregated_values.append(np.nan)
                    continue
            
                # Extract selected genes
                long_frag = df['mean_coverage_long'].values[selected_gene_indices]
                short_frag = df['mean_coverage_short'].values[selected_gene_indices]

                # Aggregate: sum over selected genes
                total_long = np.sum(long_frag)
                total_short = np.sum(short_frag)

            # Calculate FSLR
            fslr = np.log2(max(total_short, 1) / max(total_long, 1))
                
            aggregated_values.append(fslr)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            aggregated_values.append(np.nan)
    
    return np.array(aggregated_values)


def aggregate_coverage_from_files(
    sample_file_paths: List[Path],
    selected_gene_indices: np.ndarray,
    features_dir: str = "./extracted_features/biomart_10kb",
    feature_type: str = 'griffin_diff'
) -> np.ndarray:
    """
    Aggregate coverage-based features from original coverage.npy files.
    
    For each sample:
    1. Load coverage.npy (shape: n_genes x region_size, typically n_genes x 10000)
    2. Extract rows for selected genes (top_genes x 10000)
    3. Aggregate across genes to get single coverage profile (1 x 10000)
    4. Calculate the feature (griffin_diff, mwh, or rcov) from this profile
    
    Args:
        sample_file_paths: List of file paths or sample identifiers
        selected_gene_indices: Indices of genes to aggregate
        features_dir: Directory containing coverage.npy files
        feature_type: Which feature to calculate ('griffin_diff', 'mwh', 'rcov')
    
    Returns:
        Array of feature values, one per sample
    """
    aggregated_values = []
    
    for file_path in sample_file_paths:
        try:
            # Find the corresponding coverage file
            if isinstance(file_path, (str, Path)):
                file_name = Path(file_path).stem if Path(file_path).suffix else str(file_path)
            else:
                file_name = str(file_path)
            
            # Look for matching coverage.npy file
            cov_file = Path(features_dir) / f"{file_name}_coverage.npy"
            
            if not cov_file.exists():
                # Try alternative patterns
                possible_files = list(Path(features_dir).glob(f"*{file_name}*_coverage.npy"))
                if possible_files:
                    cov_file = possible_files[0]
                else:
                    print(f"Warning: Could not find coverage.npy for {file_name}, using NaN")
                    aggregated_values.append(np.nan)
                    continue
            
            # Load coverage array (n_genes x region_size, e.g., 50000 x 10000)
            coverage_matrix = np.load(cov_file)
            
            # Handle 1D coverage arrays (aggregate mode from bin.py)
            if coverage_matrix.ndim == 1:
                print(f"Warning: {file_name} has 1D coverage (aggregate mode). Cannot select genes. Using as-is.")
                aggregated_coverage = coverage_matrix
            else:
                # Extract selected genes (top_genes x region_size)
                selected_coverage = coverage_matrix[selected_gene_indices, :]
                
                # Aggregate across genes: sum coverage from all selected genes
                # This gives us a single coverage profile (1 x region_size)
                aggregated_coverage = np.sum(selected_coverage, axis=0)
            
            # Normalize coverage by mean to get relative coverage
            mean_cov = np.mean(aggregated_coverage)
            if mean_cov > 0:
                normalized_coverage = aggregated_coverage / mean_cov
            else:
                # If mean coverage is 0, return NaN
                aggregated_values.append(np.nan)
                continue
            
            # Calculate the requested feature from the normalized coverage profile
            if feature_type == 'rcov':
                # Calculate rcov (relative coverage around TSS)
                feature_value = calculate_rcov(normalized_coverage, region_type='promoter')
                
            elif feature_type == 'mwh':
                # Calculate mwh (mean window height)
                _, _, mwh_value = desarkar_features(normalized_coverage)
                feature_value = mwh_value
                
            elif feature_type == 'griffin_diff':
                # Calculate griffin_diff (difference between outside and inside window)
                _, window_coverage, _ = desarkar_features(normalized_coverage)
                
                # Calculate mean coverage outside the 1000bp window
                midpoint = int(len(normalized_coverage) / 2)
                mean_outside_coverage = np.mean(
                    np.r_[
                        normalized_coverage[: midpoint - 990],
                        normalized_coverage[midpoint + 990 :],
                    ]
                )
                feature_value = mean_outside_coverage - window_coverage
                
            else:
                raise ValueError(f"Unknown feature type: {feature_type}. Must be 'rcov', 'mwh', or 'griffin_diff'")
            
            aggregated_values.append(feature_value)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            aggregated_values.append(np.nan)
    
    return np.array(aggregated_values)


def aggregate_feature_from_raw_data(
    sample_ids: List[str],
    selected_gene_indices: np.ndarray,
    feature_type: str,
    features_dir: str
) -> np.ndarray:
    """
    Main dispatcher function to aggregate features from raw data files.
    
    Aggregation method is implicit in the feature type:
    - fslr/fslr_coverage: sum(long)/sum(short) over selected genes
    - griffin_diff/mwh/rcov: sum coverage across genes, then calculate feature
    
    Args:
        sample_ids: List of sample identifiers
        selected_gene_indices: Indices of selected genes to aggregate
        feature_type: Type of feature ('fslr', 'fslr_coverage', 'griffin_diff', 'mwh', 'rcov')
        features_dir: Directory containing raw data files
    
    Returns:
        Array of aggregated feature values (one per sample)
    """
    if feature_type == 'fslr' or feature_type == 'fslr_coverage':
        return aggregate_fslr_from_files(sample_ids, selected_gene_indices, features_dir, feature_type)
    
    elif feature_type in ['griffin_diff', 'mwh', 'rcov']:
        return aggregate_coverage_from_files(
            sample_ids, 
            selected_gene_indices, 
            features_dir,
            feature_type
        )
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
