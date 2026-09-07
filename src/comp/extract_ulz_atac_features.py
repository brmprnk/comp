#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract ULZ-ATAC features from TFBS coverage profiles.

For each sample and each transcription factor (TF), loads the coverage_aggregate.npy,
normalizes it to mean=1, and applies the ulz_atac_feature function to extract a single
feature value. Creates a samples × TFs feature matrix.

Usage:
    python src/comp/extract_ulz_atac_features.py -c 64 -o ./extracted_features/ulz_atac_tfbs.npy
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

# Import the ulz_atac_features function
sys.path.append(str(Path(__file__).parent))
from util import ulz_atac_features


def find_tfbs_folders(base_dir="./extracted_features"):
    """Find all Top1000sites folders."""
    tfbs_folders = sorted(glob(f"{base_dir}/*Top1000sites"))
    print(f"Found {len(tfbs_folders)} TFBS folders")
    return tfbs_folders


def filter_complete_tfbs_folders(tfbs_folders, sample_ids):
    """
    Filter TF folders to only those where ALL samples have a coverage file.
    
    Args:
        tfbs_folders: List of TFBS folder paths
        sample_ids: List of sample IDs that must all be present
        
    Returns:
        list: Filtered list of TFBS folder paths with complete coverage
    """
    print(f"\nFiltering TF folders for completeness ({len(sample_ids)} samples must be present)...")
    complete_folders = []
    incomplete_folders = []
    
    for tf_folder in tfbs_folders:
        tf_name = get_tf_name(tf_folder)
        all_present = True
        for sample_id in sample_ids:
            coverage_file = Path(tf_folder) / f"{sample_id}.hg38.frag.tsv_coverage_aggregate.npy"
            if not coverage_file.exists():
                all_present = False
                break
        if all_present:
            complete_folders.append(tf_folder)
        else:
            incomplete_folders.append(tf_name)
    
    print(f"  Complete TF folders: {len(complete_folders)} / {len(tfbs_folders)}")
    print(f"  Incomplete TF folders (skipped): {len(incomplete_folders)}")
    if incomplete_folders:
        print(f"  First 10 skipped: {incomplete_folders[:10]}")
    
    if len(complete_folders) == 0:
        raise ValueError("No TF folders have coverage files for all samples!")
    
    return complete_folders


def get_tf_name(folder_path):
    """Extract TF name from folder path."""
    # e.g., "./extracted_features/CTCF_Top1000sites" -> "CTCF"
    return Path(folder_path).name.replace("_Top1000sites", "")


def normalize_coverage(coverage):
    """Normalize coverage to have mean = 1."""
    mean_cov = np.mean(coverage)
    if mean_cov == 0:
        return coverage  # Avoid division by zero
    return coverage / mean_cov


def process_single_sample_tf(sample_id, tf_folder):
    """
    Process a single sample for a single TF.
    
    Returns:
        tuple: (sample_id, tf_name, feature_value, normalized_coverage) or None if file not found
    """
    tf_name = get_tf_name(tf_folder)
    # Files are named: {sample_id}.hg38.frag.tsv_coverage_aggregate.npy
    coverage_file = Path(tf_folder) / f"{sample_id}.hg38.frag.tsv_coverage_aggregate.npy"
    
    if not coverage_file.exists():
        return None
    
    try:
        # Load coverage
        coverage = np.load(coverage_file)
        
        # Normalize to mean = 1
        normalized_coverage = normalize_coverage(coverage)
        
        # Extract ULZ-ATAC feature
        feature_value = ulz_atac_features(normalized_coverage)
        
        return (sample_id, tf_name, feature_value, normalized_coverage)
    
    except Exception as e:
        print(f"Error processing {sample_id} for {tf_name}: {e}")
        return None


def extract_features_parallel(sample_ids, tfbs_folders, n_jobs=64):
    """
    Extract features for all samples and TFs in parallel.
    
    Args:
        sample_ids: List of sample IDs
        tfbs_folders: List of TFBS folder paths
        n_jobs: Number of parallel jobs
        
    Returns:
        tuple: (features_df, coverage_dict)
            - features_df: DataFrame with columns [sample_id, tf_name, feature_value]
            - coverage_dict: Dict mapping sample_id -> {tf_name -> normalized_coverage_array}
    """
    print(f"\nExtracting features for {len(sample_ids)} samples × {len(tfbs_folders)} TFs...")
    print(f"Using {n_jobs} parallel jobs")
    
    # Create all (sample, tf_folder) combinations
    combinations = [(sample_id, tf_folder) 
                   for sample_id in sample_ids 
                   for tf_folder in tfbs_folders]
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_sample_tf)(sample_id, tf_folder)
        for sample_id, tf_folder in combinations
    )
    
    # Filter out None results and create DataFrame + coverage dict
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        raise ValueError("No valid features extracted!")
    
    # Build features DataFrame
    df = pd.DataFrame(
        [(r[0], r[1], r[2]) for r in valid_results], 
        columns=['sample_id', 'tf_name', 'feature_value']
    )
    
    # Build coverage dictionary: sample_id -> {tf_name -> coverage_array}
    coverage_dict = {}
    for sample_id, tf_name, feature_value, normalized_coverage in valid_results:
        if sample_id not in coverage_dict:
            coverage_dict[sample_id] = {}
        coverage_dict[sample_id][tf_name] = normalized_coverage
    
    print(f"\nSuccessfully extracted {len(df)} features")
    print(f"Expected: {len(sample_ids) * len(tfbs_folders)}")
    print(f"Coverage: {len(df) / (len(sample_ids) * len(tfbs_folders)) * 100:.2f}%")
    print(f"Coverage dict contains {len(coverage_dict)} samples")
    
    return df, coverage_dict


def create_feature_matrix(df, sample_ids, tf_names):
    """
    Convert long-format DataFrame to samples × TFs matrix.
    
    Args:
        df: DataFrame with columns [sample_id, tf_name, feature_value]
        sample_ids: Ordered list of sample IDs
        tf_names: Ordered list of TF names
        
    Returns:
        np.ndarray: Matrix of shape (n_samples, n_tfs)
    """
    # Pivot to wide format
    matrix = df.pivot(index='sample_id', columns='tf_name', values='feature_value')
    
    # Reindex to ensure consistent ordering and fill missing values with NaN
    matrix = matrix.reindex(index=sample_ids, columns=tf_names)
    
    return matrix.values


def plot_coverage_examples(sample_ids, tfbs_folders, output_dir, n_samples=5, n_tfs=3):
    """
    Create debug plots showing raw and normalized coverage for sample examples.
    
    Args:
        sample_ids: List of all sample IDs
        tfbs_folders: List of TFBS folder paths
        output_dir: Directory to save plots
        n_samples: Number of samples to plot
        n_tfs: Number of TFs to plot per sample
    """
    print(f"\nCreating debug plots for {n_samples} samples × {n_tfs} TFs...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples and TFs
    selected_samples = np.random.choice(sample_ids, size=min(n_samples, len(sample_ids)), replace=False)
    selected_tfs = np.random.choice(tfbs_folders, size=min(n_tfs, len(tfbs_folders)), replace=False)
    
    for sample_id in selected_samples:
        fig, axes = plt.subplots(n_tfs, 2, figsize=(14, 4*n_tfs))
        if n_tfs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f"Coverage Profiles for Sample: {sample_id}", fontsize=16, fontweight='bold')
        
        for idx, tf_folder in enumerate(selected_tfs):
            tf_name = get_tf_name(tf_folder)
            coverage_file = Path(tf_folder) / f"{sample_id}.hg38.frag.tsv_coverage_aggregate.npy"
            
            if coverage_file.exists():
                # Load and process
                coverage = np.load(coverage_file)
                normalized = normalize_coverage(coverage)
                feature_value = ulz_atac_features(normalized)
                
                # Plot raw coverage
                axes[idx, 0].plot(coverage, linewidth=1.5, color='steelblue')
                axes[idx, 0].set_title(f"{tf_name} - Raw Coverage")
                axes[idx, 0].set_xlabel("Position (bp)")
                axes[idx, 0].set_ylabel("Coverage")
                axes[idx, 0].grid(alpha=0.3)
                
                # Plot normalized coverage
                axes[idx, 1].plot(normalized, linewidth=1.5, color='coral')
                axes[idx, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Mean=1')
                axes[idx, 1].set_title(f"{tf_name} - Normalized (Feature={feature_value:.3f})")
                axes[idx, 1].set_xlabel("Position (bp)")
                axes[idx, 1].set_ylabel("Normalized Coverage")
                axes[idx, 1].legend()
                axes[idx, 1].grid(alpha=0.3)
            else:
                axes[idx, 0].text(0.5, 0.5, 'File not found', ha='center', va='center')
                axes[idx, 1].text(0.5, 0.5, 'File not found', ha='center', va='center')
                axes[idx, 0].set_title(f"{tf_name} - Raw Coverage")
                axes[idx, 1].set_title(f"{tf_name} - Normalized")
        
        plt.tight_layout()
        output_file = output_dir / f"coverage_debug_{sample_id}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")


def plot_feature_heatmap(feature_matrix, sample_ids, tf_names, output_dir):
    """
    Create heatmap visualizations of the feature matrix (raw and row-standardized).
    
    Args:
        feature_matrix: np.ndarray of shape (n_samples, n_tfs)
        sample_ids: List of sample IDs
        tf_names: List of TF names
        output_dir: Directory to save plot
    """
    print(f"\nCreating feature matrix heatmaps...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================
    # 1. RAW FEATURE HEATMAP (COMPLETE)
    # ========================
    print("  Creating raw feature heatmap (all samples × all TFs)...")
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # Use robust colormap limits (exclude extreme outliers)
    vmin, vmax = np.nanpercentile(feature_matrix, [5, 95])
    
    # Don't show tick labels for large matrices (too crowded)
    if len(sample_ids) > 50 or len(tf_names) > 50:
        show_labels = False
    else:
        show_labels = True
    
    sns.heatmap(feature_matrix, 
                xticklabels=tf_names if show_labels else False,
                yticklabels=sample_ids if show_labels else False,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': 'ULZ-ATAC Feature Value'},
                ax=ax)
    
    ax.set_xlabel('Transcription Factors', fontsize=14)
    ax.set_ylabel('Samples', fontsize=14)
    ax.set_title(f'ULZ-ATAC Features (Raw): {len(sample_ids)} Samples × {len(tf_names)} TFs',
                 fontsize=16, fontweight='bold')
    
    if show_labels:
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    
    plt.tight_layout()
    output_file = output_dir / "feature_matrix_heatmap_raw.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")
    
    # ========================
    # 2. ROW-STANDARDIZED HEATMAP (COMPLETE)
    # ========================
    print("  Creating row-standardized heatmap (z-score normalization)...")
    
    # Row-standardize: z = (x - mean) / std for each sample
    feature_matrix_std = np.copy(feature_matrix)
    for i in range(len(sample_ids)):
        row = feature_matrix[i, :]
        valid_values = row[~np.isnan(row)]
        if len(valid_values) > 1:
            row_mean = np.mean(valid_values)
            row_std = np.std(valid_values)
            if row_std > 0:
                feature_matrix_std[i, :] = (row - row_mean) / row_std
            else:
                feature_matrix_std[i, :] = row - row_mean
    
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # Use symmetric colormap around 0 for z-scores
    vmin_std = np.nanpercentile(feature_matrix_std, 5)
    vmax_std = np.nanpercentile(feature_matrix_std, 95)
    # Make symmetric
    vlim_std = max(abs(vmin_std), abs(vmax_std))
    
    sns.heatmap(feature_matrix_std, 
                xticklabels=tf_names if show_labels else False,
                yticklabels=sample_ids if show_labels else False,
                cmap='RdBu_r',  # Diverging colormap centered at 0
                vmin=-vlim_std,
                vmax=vlim_std,
                center=0,
                cbar_kws={'label': 'Z-score'},
                ax=ax)
    
    ax.set_xlabel('Transcription Factors', fontsize=14)
    ax.set_ylabel('Samples', fontsize=14)
    ax.set_title(f'ULZ-ATAC Features (Row-Standardized): {len(sample_ids)} Samples × {len(tf_names)} TFs',
                 fontsize=16, fontweight='bold')
    
    if show_labels:
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    
    plt.tight_layout()
    output_file = output_dir / "feature_matrix_heatmap_standardized.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")
    
    # ========================
    # 3. RAW FEATURE STATISTICS
    # ========================
    print("  Creating raw feature statistics plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distribution of feature values
    axes[0].hist(feature_matrix[~np.isnan(feature_matrix)].flatten(), bins=100, 
                 edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Feature Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Raw ULZ-ATAC Features', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Missing data per sample
    missing_per_sample = np.sum(np.isnan(feature_matrix), axis=1)
    axes[1].hist(missing_per_sample, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of Missing TFs', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('Missing Data per Sample', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Missing data per TF
    missing_per_tf = np.sum(np.isnan(feature_matrix), axis=0)
    axes[2].hist(missing_per_tf, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[2].set_xlabel('Number of Missing Samples', fontsize=12)
    axes[2].set_ylabel('Number of TFs', fontsize=12)
    axes[2].set_title('Missing Data per TF', fontsize=13, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "feature_statistics_raw.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")
    
    # ========================
    # 4. STANDARDIZED FEATURE STATISTICS
    # ========================
    print("  Creating standardized feature statistics plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distribution of z-scores
    valid_zscores = feature_matrix_std[~np.isnan(feature_matrix_std)].flatten()
    if len(valid_zscores) > 0:
        axes[0].hist(valid_zscores, bins=100, 
                     edgecolor='black', alpha=0.7, color='purple')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Mean=0')
        axes[0].set_xlabel('Z-score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Row-Standardized Features', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No valid z-scores', ha='center', va='center', transform=axes[0].transAxes)
    
    # Per-sample mean after standardization (should be ~0)
    sample_means = np.nanmean(feature_matrix_std, axis=1)
    valid_means = sample_means[~np.isnan(sample_means)]
    
    if len(valid_means) > 0:
        # Check if data has sufficient range for binning
        data_range = np.ptp(valid_means)  # peak-to-peak (max - min)
        unique_values = len(np.unique(valid_means))
        
        if data_range < 1e-10 or unique_values == 1:
            # All values are essentially the same
            axes[1].text(0.5, 0.5, f'All samples have mean ≈ {valid_means[0]:.4f}\n(no variation)', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        else:
            # Use automatic binning with fallback
            try:
                n_bins = min(unique_values, max(10, len(valid_means) // 5))  # Adaptive bin count
                axes[1].hist(valid_means, bins=n_bins, edgecolor='black', alpha=0.7, color='orange')
            except (ValueError, RuntimeError):
                # Last resort: just plot unique values as bars
                unique_vals, counts = np.unique(valid_means, return_counts=True)
                axes[1].bar(unique_vals, counts, edgecolor='black', alpha=0.7, color='orange', width=data_range/20 if data_range > 0 else 0.1)
            
            axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target=0')
            axes[1].legend()
        
        axes[1].set_xlabel('Mean Z-score per Sample', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title('Per-Sample Mean (Should be ~0)', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No valid means', ha='center', va='center', transform=axes[1].transAxes)
    
    # Per-sample std after standardization (should be ~1)
    sample_stds = np.nanstd(feature_matrix_std, axis=1)
    valid_stds = sample_stds[~np.isnan(sample_stds)]
    
    if len(valid_stds) > 1:  # Need at least 2 values for histogram
        # Check if data has sufficient range for binning
        data_range = np.ptp(valid_stds)  # peak-to-peak (max - min)
        unique_values = len(np.unique(valid_stds))
        
        if data_range < 1e-10 or unique_values == 1:
            # All values are essentially the same
            axes[2].text(0.5, 0.5, f'All samples have std ≈ {valid_stds[0]:.4f}\n(no variation)', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        else:
            # Use automatic binning with fallback
            try:
                n_bins = min(unique_values, max(10, len(valid_stds) // 5))  # Adaptive bin count
                axes[2].hist(valid_stds, bins=n_bins, edgecolor='black', alpha=0.7, color='teal')
            except (ValueError, RuntimeError):
                # Last resort: just plot unique values as bars
                unique_vals, counts = np.unique(valid_stds, return_counts=True)
                axes[2].bar(unique_vals, counts, edgecolor='black', alpha=0.7, color='teal', width=data_range/20 if data_range > 0 else 0.1)
            
            axes[2].axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target=1')
            axes[2].legend()
        
        axes[2].set_xlabel('Std Dev per Sample', fontsize=12)
        axes[2].set_ylabel('Number of Samples', fontsize=12)
        axes[2].set_title('Per-Sample Std Dev (Should be ~1)', fontsize=13, fontweight='bold')
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Insufficient valid data', ha='center', va='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    output_file = output_dir / "feature_statistics_standardized.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ULZ-ATAC features from TFBS coverage profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--features_dir",
        type=str,
        default="./extracted_features",
        help="Directory containing Top1000sites folders"
    )
    
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="accessory_files/cris_metadata.csv",
        help="Path to CRIS metadata CSV file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./extracted_features/ulz_atac_tfbs_features.npy",
        help="Output path for feature matrix (.npy file)"
    )
    
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./plots/ulz_atac_features",
        help="Directory to save debug plots and heatmaps"
    )
    
    parser.add_argument(
        "--aggregate_coverage_dir",
        type=str,
        default="./extracted_features/ulz_atac_aggregate_coverage",
        help="Directory to save per-sample aggregate coverage profiles (mean-normalized sum across all TFs)"
    )
    
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=64,
        help="Number of CPU cores for parallel processing"
    )
    
    parser.add_argument(
        "--n_plot_samples",
        type=int,
        default=5,
        help="Number of samples to plot for debugging"
    )
    
    args = parser.parse_args()
    
    print("="*90)
    print("ULZ-ATAC FEATURE EXTRACTION FROM TFBS COVERAGE PROFILES")
    print("="*90)
    
    # Find TFBS folders
    tfbs_folders = find_tfbs_folders(args.features_dir)
    if len(tfbs_folders) == 0:
        raise ValueError(f"No Top1000sites folders found in {args.features_dir}")
    
    print(f"TF names (first 10): {[get_tf_name(f) for f in tfbs_folders[:10]]}")
    
    # Load metadata
    print(f"\nLoading metadata from: {args.metadata_file}")
    metadata = pd.read_csv(args.metadata_file)

    # Deduplicate metadata rows (some metadata files contain duplicate entries)
    n_before_dedup = len(metadata)
    metadata = metadata.drop_duplicates(subset=['ID']).copy()
    n_after_dedup = len(metadata)
    if n_before_dedup != n_after_dedup:
        print(f"  Removed {n_before_dedup - n_after_dedup} duplicate rows (kept first occurrence)")

    # Keep all disease classes present in the metadata
    # (filtering is dataset-specific and handled via the metadata file itself)
    sample_ids = metadata['ID'].tolist()
    print(f"Found {len(sample_ids)} unique samples in metadata")
    print(f"Disease distribution:")
    for disease, count in metadata['disease'].value_counts().items():
        print(f"  - {disease}: {count}")
    
    # Filter TF folders to only those with complete coverage for all samples
    tfbs_folders = filter_complete_tfbs_folders(tfbs_folders, sample_ids)
    tf_names = [get_tf_name(folder) for folder in tfbs_folders]
    print(f"Using {len(tf_names)} complete TF folders")
    
    # Create debug plots first (before heavy computation)
    plot_coverage_examples(sample_ids, tfbs_folders, args.plot_dir, 
                          n_samples=args.n_plot_samples, n_tfs=3)
    
    # Extract features in parallel (now also returns coverage_dict)
    features_df, coverage_dict = extract_features_parallel(sample_ids, tfbs_folders, n_jobs=args.cores)
    
    # Create feature matrix
    print(f"\nCreating feature matrix...")
    feature_matrix = create_feature_matrix(features_df, sample_ids, tf_names)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Expected shape: ({len(sample_ids)}, {len(tf_names)})")
    
    # Statistics
    n_missing = np.sum(np.isnan(feature_matrix))
    total_values = feature_matrix.size
    print(f"\nMissing values: {n_missing} / {total_values} ({n_missing/total_values*100:.2f}%)")
    print(f"Feature value range: [{np.nanmin(feature_matrix):.3f}, {np.nanmax(feature_matrix):.3f}]")
    print(f"Feature value mean: {np.nanmean(feature_matrix):.3f}")
    print(f"Feature value std: {np.nanstd(feature_matrix):.3f}")
    
    # ============================================================
    # DETAILED DIAGNOSTIC ANALYSIS
    # ============================================================
    print("\n" + "="*90)
    print("DIAGNOSTIC ANALYSIS: IDENTIFYING PROBLEMATIC SAMPLES AND TFs")
    print("="*90)
    
    # 1. Identify samples with too much missing data
    print("\n1. SAMPLES WITH HIGH MISSING DATA:")
    missing_per_sample = np.sum(np.isnan(feature_matrix), axis=1)
    missing_pct_per_sample = (missing_per_sample / len(tf_names)) * 100
    problematic_samples = np.where(missing_pct_per_sample > 10)[0]  # >10% missing
    if len(problematic_samples) > 0:
        print(f"   Found {len(problematic_samples)} samples with >10% missing TFs:")
        for idx in problematic_samples[:20]:  # Show first 20
            print(f"   - Sample {idx}: {sample_ids[idx]}: {missing_per_sample[idx]}/{len(tf_names)} missing ({missing_pct_per_sample[idx]:.1f}%)")
        if len(problematic_samples) > 20:
            print(f"   ... and {len(problematic_samples)-20} more")
    else:
        print("   ✅ No samples with >10% missing data")
    
    # 2. Identify completely empty samples
    print("\n2. COMPLETELY EMPTY SAMPLES:")
    empty_samples = np.where(missing_per_sample == len(tf_names))[0]
    if len(empty_samples) > 0:
        print(f"   ⚠️  Found {len(empty_samples)} samples with ALL TFs missing:")
        for idx in empty_samples:
            print(f"   - Sample {idx}: {sample_ids[idx]}")
    else:
        print("   ✅ No completely empty samples")
    
    # 3. Identify TFs with too much missing data
    print("\n3. TFs WITH HIGH MISSING DATA:")
    missing_per_tf = np.sum(np.isnan(feature_matrix), axis=0)
    missing_pct_per_tf = (missing_per_tf / len(sample_ids)) * 100
    problematic_tfs = np.where(missing_pct_per_tf > 10)[0]  # >10% missing
    if len(problematic_tfs) > 0:
        print(f"   Found {len(problematic_tfs)} TFs with >10% missing samples:")
        for idx in problematic_tfs[:20]:  # Show first 20
            print(f"   - TF {idx}: {tf_names[idx]}: {missing_per_tf[idx]}/{len(sample_ids)} missing ({missing_pct_per_tf[idx]:.1f}%)")
        if len(problematic_tfs) > 20:
            print(f"   ... and {len(problematic_tfs)-20} more")
    else:
        print("   ✅ No TFs with >10% missing data")
    
    # 4. Identify samples with very low variance (uniform values)
    print("\n4. SAMPLES WITH LOW VARIANCE (Nearly Uniform):")
    sample_vars = np.nanvar(feature_matrix, axis=1)
    low_var_threshold = np.nanpercentile(sample_vars, 5)  # Bottom 5%
    low_var_samples = np.where(sample_vars < low_var_threshold)[0]
    if len(low_var_samples) > 0:
        print(f"   Found {len(low_var_samples)} samples with variance < {low_var_threshold:.6f} (bottom 5%):")
        for idx in low_var_samples[:20]:
            valid_vals = feature_matrix[idx, ~np.isnan(feature_matrix[idx, :])]
            if len(valid_vals) > 0:
                print(f"   - Sample {idx}: {sample_ids[idx]}: var={sample_vars[idx]:.6f}, mean={np.mean(valid_vals):.3f}, range=[{np.min(valid_vals):.3f}, {np.max(valid_vals):.3f}]")
            else:
                print(f"   - Sample {idx}: {sample_ids[idx]}: ALL NaN")
        if len(low_var_samples) > 20:
            print(f"   ... and {len(low_var_samples)-20} more")
    
    # 5. Identify TFs with very low variance (uniform across samples)
    print("\n5. TFs WITH LOW VARIANCE (Nearly Uniform):")
    tf_vars = np.nanvar(feature_matrix, axis=0)
    low_var_tf_threshold = np.nanpercentile(tf_vars, 5)  # Bottom 5%
    low_var_tfs = np.where(tf_vars < low_var_tf_threshold)[0]
    if len(low_var_tfs) > 0:
        print(f"   Found {len(low_var_tfs)} TFs with variance < {low_var_tf_threshold:.6f} (bottom 5%):")
        for idx in low_var_tfs[:20]:
            valid_vals = feature_matrix[:, idx][~np.isnan(feature_matrix[:, idx])]
            if len(valid_vals) > 0:
                print(f"   - TF {idx}: {tf_names[idx]}: var={tf_vars[idx]:.6f}, mean={np.mean(valid_vals):.3f}, range=[{np.min(valid_vals):.3f}, {np.max(valid_vals):.3f}]")
        if len(low_var_tfs) > 20:
            print(f"   ... and {len(low_var_tfs)-20} more")
    
    # 6. Identify samples with shifted distributions (very high mean)
    print("\n6. SAMPLES WITH SHIFTED DISTRIBUTIONS (High Mean):")
    sample_means = np.nanmean(feature_matrix, axis=1)
    high_mean_threshold = np.nanpercentile(sample_means, 95)  # Top 5%
    high_mean_samples = np.where(sample_means > high_mean_threshold)[0]
    if len(high_mean_samples) > 0:
        print(f"   Found {len(high_mean_samples)} samples with mean > {high_mean_threshold:.3f} (top 5%):")
        for idx in high_mean_samples[:20]:
            valid_vals = feature_matrix[idx, ~np.isnan(feature_matrix[idx, :])]
            if len(valid_vals) > 0:
                print(f"   - Sample {idx}: {sample_ids[idx]}: mean={sample_means[idx]:.3f}, median={np.median(valid_vals):.3f}, max={np.max(valid_vals):.3f}")
        if len(high_mean_samples) > 20:
            print(f"   ... and {len(high_mean_samples)-20} more")
    
    # 7. Identify extreme outlier values
    print("\n7. EXTREME OUTLIER VALUES:")
    q99 = np.nanpercentile(feature_matrix, 99)
    extreme_mask = feature_matrix > q99
    extreme_coords = np.where(extreme_mask)
    if len(extreme_coords[0]) > 0:
        print(f"   Found {len(extreme_coords[0])} values > 99th percentile ({q99:.3f}):")
        for i in range(min(20, len(extreme_coords[0]))):
            sample_idx = extreme_coords[0][i]
            tf_idx = extreme_coords[1][i]
            value = feature_matrix[sample_idx, tf_idx]
            print(f"   - Sample {sample_idx} ({sample_ids[sample_idx]}), TF {tf_idx} ({tf_names[tf_idx]}): value={value:.3f}")
        if len(extreme_coords[0]) > 20:
            print(f"   ... and {len(extreme_coords[0])-20} more")
    
    # 8. Check metadata for disease distribution
    print("\n8. SAMPLE DISEASE DISTRIBUTION:")
    disease_counts = metadata['disease'].value_counts()
    print(f"   Total samples: {len(sample_ids)}")
    for disease, count in disease_counts.items():
        print(f"   - {disease}: {count} samples ({count/len(sample_ids)*100:.1f}%)")
    
    print("\n" + "="*90)
    
    # ============================================================
    # SAVE ALL OUTPUT FILES FIRST (before plotting)
    # ============================================================
    print("\n" + "="*90)
    print("SAVING OUTPUT FILES")
    print("="*90)
    
    # Save feature matrix
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature_matrix)
    print(f"✅ Saved feature matrix to: {output_path}")
    
    # Also save metadata about the matrix in human-readable formats
    metadata_csv = output_path.with_suffix('.metadata.csv')
    mapping_df = metadata.copy()
    # Ensure we write the row index mapping so downstream tools know the exact row order
    mapping_df = mapping_df.reset_index(drop=True)
    mapping_df['row_index'] = mapping_df.index
    mapping_df.to_csv(metadata_csv, index=False)
    print(f"✅ Saved metadata CSV to: {metadata_csv}")

    # Save TF names as a simple CSV for reproducible column ordering
    tf_csv = output_path.with_suffix('.tf_names.csv')
    pd.DataFrame({'tf_name': tf_names}).to_csv(tf_csv, index=False)
    print(f"✅ Saved TF names to: {tf_csv}")

    # Also keep the original binary NPZ for quick programmatic reload if desired
    metadata_output = output_path.with_suffix('.metadata.npz')
    np.savez(metadata_output,
             sample_ids=sample_ids,
             tf_names=tf_names,
             feature_matrix=feature_matrix)
    print(f"✅ Saved metadata (npz) to: {metadata_output}")
    
    # ============================================================
    # SAVE AGGREGATE COVERAGE PROFILES (SUM ACROSS ALL TFs)
    # ============================================================
    print("\n" + "="*90)
    print("SAVING AGGREGATE COVERAGE PROFILES")
    print("="*90)
    print(f"Creating aggregate coverage profiles (sum across all TFs, normalized to mean=1)...")
    
    aggregate_cov_dir = Path(args.aggregate_coverage_dir)
    aggregate_cov_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {aggregate_cov_dir}")
    
    saved_count = 0
    failed_count = 0
    
    for sample_id in sample_ids:
        if sample_id not in coverage_dict:
            print(f"  ⚠ Sample {sample_id} not in coverage_dict (no TFs found)")
            failed_count += 1
            continue
        
        tf_coverages = coverage_dict[sample_id]
        
        if len(tf_coverages) == 0:
            print(f"  ⚠ Sample {sample_id} has no TF coverage data")
            failed_count += 1
            continue
        
        try:
            # Get coverage arrays (already normalized to mean=1 per TF)
            coverage_arrays = list(tf_coverages.values())
            
            # Sum across all TFs (element-wise sum)
            aggregate_coverage = np.sum(coverage_arrays, axis=0)
            
            # Normalize the aggregate to mean=1
            mean_cov = np.mean(aggregate_coverage)
            if mean_cov > 0:
                aggregate_coverage_normalized = aggregate_coverage / mean_cov
            else:
                print(f"  ⚠ Sample {sample_id}: aggregate coverage has mean={mean_cov}, skipping normalization")
                aggregate_coverage_normalized = aggregate_coverage
            
            # Save to file
            output_file = aggregate_cov_dir / f"{sample_id}_ulz_atac_aggregate_coverage.npy"
            np.save(output_file, aggregate_coverage_normalized)
            saved_count += 1
            
            # Debug info for first 5 samples
            if saved_count <= 5:
                print(f"  ✓ {sample_id}: {len(tf_coverages)} TFs, "
                      f"aggregate shape={aggregate_coverage_normalized.shape}, "
                      f"mean={np.mean(aggregate_coverage_normalized):.4f}, "
                      f"range=[{np.min(aggregate_coverage_normalized):.4f}, {np.max(aggregate_coverage_normalized):.4f}]")
        
        except Exception as e:
            print(f"  ✗ Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    print(f"\n✅ Saved aggregate coverage for {saved_count}/{len(sample_ids)} samples")
    if failed_count > 0:
        print(f"⚠  Failed to save {failed_count} samples")
    print(f"Files saved to: {aggregate_cov_dir}")
    
    # ============================================================
    # NOW CREATE VISUALIZATIONS
    # ============================================================
    print("\n" + "="*90)
    print("CREATING VISUALIZATIONS")
    print("="*90)
    plot_feature_heatmap(feature_matrix, sample_ids, tf_names, args.plot_dir)
    
    print("\n" + "="*90)
    print("EXTRACTION COMPLETE!")
    print("="*90)
    print(f"\nTo load the feature matrix:")
    print(f"  feature_matrix = np.load('{output_path}')")
    print(f"  # Or with metadata:")
    print(f"  data = np.load('{metadata_output}')")
    print(f"  feature_matrix = data['feature_matrix']")
    print(f"  sample_ids = data['sample_ids']")
    print(f"  tf_names = data['tf_names']")


if __name__ == "__main__":
    main()
