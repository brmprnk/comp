#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract aggregate features from coverage profiles and add to metadata CSV.

This script:
1. Loads cris_stage_vaf_coverage_ichor.csv
2. For each sample with a coverage_aggregate.npy file in extracted_features/biomart_10kb:
   - Extracts griffin_diff, rcov, mwh from aggregated coverage profile
   - Extracts FSLR from fragment counts in _BIN.csv
   - Extracts FSLR_coverage from coverage in _BIN.csv
3. Adds these 5 features as new columns to the CSV
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from baseline_aggregate import aggregate_coverage_from_files, aggregate_fslr_from_files
from util import calculate_rcov, desarkar_features


def extract_single_sample_griffin_diff(coverage_profile: np.ndarray) -> float:
    """Extract Griffin Difference from a single coverage profile."""
    try:
        # Normalize by mean
        mean_cov = np.mean(coverage_profile)
        if mean_cov <= 0:
            print(f"  Warning: Mean coverage is {mean_cov}, returning NaN")
            return np.nan
        
        normalized = coverage_profile / mean_cov
        
        # Calculate Griffin difference (outside - inside window coverage)
        _, window_coverage, _ = desarkar_features(normalized)
        
        midpoint = int(len(normalized) / 2)
        mean_outside_coverage = np.mean(
            np.r_[
                normalized[: midpoint - 990],
                normalized[midpoint + 990 :],
            ]
        )
        
        griffin_diff = mean_outside_coverage - window_coverage
        print(f"  Griffin diff: window={window_coverage:.4f}, outside={mean_outside_coverage:.4f}, diff={griffin_diff:.4f}")
        return griffin_diff
        
    except Exception as e:
        print(f"  Error calculating griffin_diff: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def extract_single_sample_rcov(coverage_profile: np.ndarray) -> float:
    """Extract RCOV from a single coverage profile."""
    try:
        # Normalize by mean
        mean_cov = np.mean(coverage_profile)
        if mean_cov <= 0:
            print(f"  Warning: Mean coverage is {mean_cov}, returning NaN")
            return np.nan
        
        normalized = coverage_profile / mean_cov
        
        # Calculate RCOV
        rcov = calculate_rcov(normalized, region_type='promoter')
        print(f"  RCOV: {rcov:.4f}")
        return rcov
        
    except Exception as e:
        print(f"  Error calculating rcov: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def extract_single_sample_mwh(coverage_profile: np.ndarray) -> float:
    """Extract MWH (Max Wave Height) from a single coverage profile."""
    try:
        # Normalize by mean
        mean_cov = np.mean(coverage_profile)
        if mean_cov <= 0:
            print(f"  Warning: Mean coverage is {mean_cov}, returning NaN")
            return np.nan
        
        normalized = coverage_profile / mean_cov
        
        # Calculate MWH
        _, _, mwh = desarkar_features(normalized)
        print(f"  MWH: {mwh:.4f}")
        return mwh
        
    except Exception as e:
        print(f"  Error calculating mwh: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def extract_single_sample_fslr(bin_csv_path: Path) -> float:
    """Extract FSLR from fragment counts in _BIN.csv."""
    try:
        df = pd.read_csv(bin_csv_path)
        
        if 'absolute_short_fragments' not in df.columns or 'absolute_long_fragments' not in df.columns:
            print(f"  Warning: Missing fragment columns in {bin_csv_path.name}")
            return np.nan
        
        # Sum all fragments across all genes
        total_short = df['absolute_short_fragments'].sum()
        total_long = df['absolute_long_fragments'].sum()
        
        # Calculate FSLR = log2(short / long)
        fslr = np.log2(max(total_short, 1) / max(total_long, 1))
        print(f"  FSLR: short={total_short}, long={total_long}, fslr={fslr:.4f}")
        return fslr
        
    except Exception as e:
        print(f"  Error calculating FSLR: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def extract_single_sample_fslr_coverage(bin_csv_path: Path) -> float:
    """Extract FSLR_coverage from coverage in _BIN.csv."""
    try:
        df = pd.read_csv(bin_csv_path)
        
        if 'mean_coverage_short' not in df.columns or 'mean_coverage_long' not in df.columns:
            print(f"  Warning: Missing coverage columns in {bin_csv_path.name}")
            return np.nan
        
        # Sum all coverage across all genes
        total_short_cov = df['mean_coverage_short'].sum()
        total_long_cov = df['mean_coverage_long'].sum()
        
        # Calculate FSLR_coverage = log2(short_cov / long_cov)
        fslr_cov = np.log2(max(total_short_cov, 1) / max(total_long_cov, 1))
        print(f"  FSLR_coverage: short_cov={total_short_cov:.2f}, long_cov={total_long_cov:.2f}, fslr_cov={fslr_cov:.4f}")
        return fslr_cov
        
    except Exception as e:
        print(f"  Error calculating FSLR_coverage: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def main():
    """Main function to extract features and update CSV."""
    
    # Paths
    metadata_csv = Path("accessory_files/cris_stage_vaf_coverage_ichor.csv")
    features_dir = Path("extracted_features/biomart_10kb")
    output_csv = Path("accessory_files/cris_stage_vaf_coverage_ichor.csv")
    
    print("="*80)
    print("EXTRACTING AGGREGATE FEATURES FROM COVERAGE PROFILES")
    print("="*80)
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Features directory: {features_dir}")
    print(f"Output CSV: {output_csv}")
    print()
    
    # Load metadata
    print(f"Loading metadata from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print()
    
    # Initialize new columns
    df['aggregate_griffin_diff'] = np.nan
    df['aggregate_rcov'] = np.nan
    df['aggregate_mwh'] = np.nan
    df['aggregate_fslr'] = np.nan
    df['aggregate_fslr_coverage'] = np.nan
    
    # Track statistics
    total_samples = len(df)
    processed_samples = 0
    found_coverage_aggregate = 0
    found_bin_csv = 0
    successful_griffin = 0
    successful_rcov = 0
    successful_mwh = 0
    successful_fslr = 0
    successful_fslr_cov = 0
    
    # Process each sample
    print(f"Processing {total_samples} samples...")
    print("-"*80)
    
    for idx, row in df.iterrows():
        sample_id = row['ID']
        print(f"\n[{idx+1}/{total_samples}] Processing sample: {sample_id}")
        
        # Look for coverage_aggregate.npy file
        # Try different naming patterns
        coverage_patterns = [
            f"{sample_id}_coverage_aggregate.npy",
            f"{sample_id}.hg38.frag.tsv_coverage_aggregate.npy",
            f"*{sample_id}*coverage_aggregate.npy"
        ]
        
        coverage_file = None
        for pattern in coverage_patterns:
            if '*' in pattern:
                matches = list(features_dir.glob(pattern))
                if matches:
                    coverage_file = matches[0]
                    break
            else:
                candidate = features_dir / pattern
                if candidate.exists():
                    coverage_file = candidate
                    break
        
        if coverage_file is None:
            print(f"  ⚠ No coverage_aggregate.npy found (tried {len(coverage_patterns)} patterns)")
            continue
        
        print(f"  ✓ Found coverage file: {coverage_file.name}")
        found_coverage_aggregate += 1
        
        # Load coverage profile
        try:
            coverage_profile = np.load(coverage_file)
            print(f"  ✓ Loaded coverage profile: shape={coverage_profile.shape}, "
                  f"mean={np.mean(coverage_profile):.2f}, "
                  f"min={np.min(coverage_profile):.2f}, "
                  f"max={np.max(coverage_profile):.2f}")
        except Exception as e:
            print(f"  ✗ Error loading coverage file: {e}")
            continue
        
        # Extract coverage-based features
        print(f"  Extracting coverage-based features...")
        
        # Griffin diff
        griffin_diff = extract_single_sample_griffin_diff(coverage_profile)
        if not np.isnan(griffin_diff):
            df.at[idx, 'aggregate_griffin_diff'] = griffin_diff
            successful_griffin += 1
        
        # RCOV
        rcov = extract_single_sample_rcov(coverage_profile)
        if not np.isnan(rcov):
            df.at[idx, 'aggregate_rcov'] = rcov
            successful_rcov += 1
        
        # MWH
        mwh = extract_single_sample_mwh(coverage_profile)
        if not np.isnan(mwh):
            df.at[idx, 'aggregate_mwh'] = mwh
            successful_mwh += 1
        
        # Look for _BIN.csv file for fragment-based features
        bin_patterns = [
            f"{sample_id}_BIN.csv",
            f"{sample_id}.hg38.frag.tsv_BIN.csv",
            f"*{sample_id}*_BIN.csv"
        ]
        
        bin_file = None
        for pattern in bin_patterns:
            if '*' in pattern:
                matches = list(features_dir.glob(pattern))
                if matches:
                    bin_file = matches[0]
                    break
            else:
                candidate = features_dir / pattern
                if candidate.exists():
                    bin_file = candidate
                    break
        
        if bin_file is None:
            print(f"  ⚠ No _BIN.csv found (tried {len(bin_patterns)} patterns)")
            processed_samples += 1
            continue
        
        print(f"  ✓ Found BIN file: {bin_file.name}")
        found_bin_csv += 1
        
        # Extract fragment-based features
        print(f"  Extracting fragment-based features...")
        
        # FSLR
        fslr = extract_single_sample_fslr(bin_file)
        if not np.isnan(fslr):
            df.at[idx, 'aggregate_fslr'] = fslr
            successful_fslr += 1
        
        # FSLR_coverage
        fslr_cov = extract_single_sample_fslr_coverage(bin_file)
        if not np.isnan(fslr_cov):
            df.at[idx, 'aggregate_fslr_coverage'] = fslr_cov
            successful_fslr_cov += 1
        
        processed_samples += 1
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples in CSV: {total_samples}")
    print(f"Samples with coverage_aggregate.npy: {found_coverage_aggregate}")
    print(f"Samples with _BIN.csv: {found_bin_csv}")
    print(f"Samples processed: {processed_samples}")
    print()
    print(f"Successful extractions:")
    print(f"  Griffin Diff: {successful_griffin}/{found_coverage_aggregate} ({100*successful_griffin/max(found_coverage_aggregate,1):.1f}%)")
    print(f"  RCOV: {successful_rcov}/{found_coverage_aggregate} ({100*successful_rcov/max(found_coverage_aggregate,1):.1f}%)")
    print(f"  MWH: {successful_mwh}/{found_coverage_aggregate} ({100*successful_mwh/max(found_coverage_aggregate,1):.1f}%)")
    print(f"  FSLR: {successful_fslr}/{found_bin_csv} ({100*successful_fslr/max(found_bin_csv,1):.1f}%)")
    print(f"  FSLR_coverage: {successful_fslr_cov}/{found_bin_csv} ({100*successful_fslr_cov/max(found_bin_csv,1):.1f}%)")
    print()
    
    # Show sample of results
    print("Sample of extracted features (first 10 rows with data):")
    feature_cols = ['ID', 'disease', 'aggregate_griffin_diff', 'aggregate_rcov', 
                   'aggregate_mwh', 'aggregate_fslr', 'aggregate_fslr_coverage']
    sample_df = df[feature_cols].dropna(subset=feature_cols[2:], how='all').head(10)
    print(sample_df.to_string(index=False))
    print()
    
    # Save updated CSV
    print(f"Saving updated CSV to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df)} rows with {len(df.columns)} columns")
    print(f"  New columns added: {feature_cols[2:]}")
    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
