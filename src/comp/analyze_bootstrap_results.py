#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bootstrap Study Analysis Script

This script analyzes the results from a bootstrap robustness study.
It identifies challenging samples, robust samples, and provides
comprehensive statistics.

Usage:
    python src/comp/analyze_bootstrap_results.py <path_to_bootstrap_summary.pkl>
    
Example:
    python src/comp/analyze_bootstrap_results.py ./bootstrap_results/bootstrap_best_model_robustness/bootstrap_summary.pkl
"""

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_bootstrap_results(pkl_path):
    """Load bootstrap summary pickle file."""
    with open(pkl_path, 'rb') as f:
        summary = pickle.load(f)
    return summary

def print_configuration(config):
    """Print study configuration."""
    print("="*80)
    print("BOOTSTRAP STUDY CONFIGURATION")
    print("="*80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

def analyze_overall_performance(results_df):
    """Analyze overall classification performance."""
    print("="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    total_appearances = results_df['test_appearances'].sum()
    total_misclass = results_df['misclassifications'].sum()
    overall_rate = total_misclass / total_appearances
    
    print(f"Total test appearances: {total_appearances}")
    print(f"Total misclassifications: {total_misclass}")
    print(f"Overall misclassification rate: {overall_rate:.4f} ({overall_rate*100:.2f}%)")
    print(f"Overall accuracy: {1-overall_rate:.4f} ({(1-overall_rate)*100:.2f}%)")
    print()
    
    # By true label
    print("By True Label:")
    for label in results_df['true_label'].unique():
        subset = results_df[results_df['true_label'] == label]
        rate = subset['misclassifications'].sum() / subset['test_appearances'].sum()
        label_name = "Cancer" if label == 1 else "Healthy"
        print(f"  {label_name}: {rate:.4f} ({rate*100:.2f}%)")
    print()
    
    # By cancer type
    print("By Cancer Type:")
    for cancer_type in sorted(results_df['cancer_type'].unique()):
        subset = results_df[results_df['cancer_type'] == cancer_type]
        rate = subset['misclassifications'].sum() / subset['test_appearances'].sum()
        print(f"  {cancer_type}: {rate:.4f} ({rate*100:.2f}%) - {len(subset)} samples")
    print()

def identify_challenging_samples(results_df, n=20):
    """Identify most challenging samples."""
    print("="*80)
    print(f"TOP {n} MOST CHALLENGING SAMPLES")
    print("="*80)
    
    challenging = results_df.nlargest(n, 'misclassification_rate')
    print(challenging[['sample_id', 'cancer_type', 'test_appearances', 
                       'misclassifications', 'misclassification_rate',
                       'mean_predicted_proba']].to_string(index=False))
    print()
    
    # Count by threshold
    print("Samples by Misclassification Rate:")
    print(f"  >50%: {len(results_df[results_df['misclassification_rate'] > 0.5])}")
    print(f"  30-50%: {len(results_df[(results_df['misclassification_rate'] > 0.3) & (results_df['misclassification_rate'] <= 0.5)])}")
    print(f"  15-30%: {len(results_df[(results_df['misclassification_rate'] > 0.15) & (results_df['misclassification_rate'] <= 0.3)])}")
    print(f"  5-15%: {len(results_df[(results_df['misclassification_rate'] > 0.05) & (results_df['misclassification_rate'] <= 0.15)])}")
    print(f"  0-5%: {len(results_df[results_df['misclassification_rate'] <= 0.05])}")
    print()

def identify_robust_samples(results_df, n=20):
    """Identify most robust samples."""
    print("="*80)
    print(f"TOP {n} MOST ROBUST SAMPLES")
    print("="*80)
    
    robust = results_df.nsmallest(n, 'misclassification_rate')
    print(robust[['sample_id', 'cancer_type', 'test_appearances', 
                  'misclassifications', 'misclassification_rate',
                  'mean_predicted_proba']].to_string(index=False))
    print()
    
    perfect = results_df[results_df['misclassification_rate'] == 0.0]
    print(f"Samples with 0% misclassification rate: {len(perfect)}")
    if len(perfect) > 0:
        print("  By cancer type:")
        for cancer_type in perfect['cancer_type'].value_counts().items():
            print(f"    {cancer_type[0]}: {cancer_type[1]}")
    print()

def analyze_prediction_consistency(results_df, n=20):
    """Analyze prediction consistency (variance)."""
    print("="*80)
    print("PREDICTION CONSISTENCY ANALYSIS")
    print("="*80)
    
    print(f"\nOverall prediction variance:")
    print(f"  Mean std: {results_df['std_predicted_proba'].mean():.4f}")
    print(f"  Median std: {results_df['std_predicted_proba'].median():.4f}")
    print()
    
    print(f"Most Inconsistent Samples (high variance):")
    inconsistent = results_df.nlargest(n, 'std_predicted_proba')
    print(inconsistent[['sample_id', 'cancer_type', 'mean_predicted_proba', 
                        'std_predicted_proba', 'misclassification_rate']].to_string(index=False))
    print()
    
    print(f"Most Consistent Samples (low variance):")
    consistent = results_df.nsmallest(n, 'std_predicted_proba')
    print(consistent[['sample_id', 'cancer_type', 'mean_predicted_proba', 
                      'std_predicted_proba', 'misclassification_rate']].to_string(index=False))
    print()

def create_visualizations(results_df, output_dir):
    """Create visualization plots."""
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution of misclassification rates
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['misclassification_rate'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Misclassification Rate', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Sample Misclassification Rates', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'misclassification_rate_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 2. Misclassification rate by cancer type (boxplot)
    plt.figure(figsize=(14, 6))
    cancer_types_sorted = results_df.groupby('cancer_type')['misclassification_rate'].median().sort_values().index
    sns.boxplot(data=results_df, x='cancer_type', y='misclassification_rate', order=cancer_types_sorted)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Misclassification Rate', fontsize=12)
    plt.xlabel('Cancer Type', fontsize=12)
    plt.title('Misclassification Rate by Cancer Type', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'misclassification_by_cancer_type.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 3. Mean probability vs std (prediction consistency)
    plt.figure(figsize=(10, 8))
    cancer_mask = results_df['true_label'] == 1
    plt.scatter(results_df[cancer_mask]['mean_predicted_proba'], 
               results_df[cancer_mask]['std_predicted_proba'],
               alpha=0.5, label='Cancer', c='red', s=30)
    plt.scatter(results_df[~cancer_mask]['mean_predicted_proba'], 
               results_df[~cancer_mask]['std_predicted_proba'],
               alpha=0.5, label='Healthy', c='blue', s=30)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Std Predicted Probability (Uncertainty)', fontsize=12)
    plt.title('Prediction Consistency: Mean vs Std', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'prediction_consistency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 4. Test appearances distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['test_appearances'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Test Set Appearances', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Test Set Appearances per Sample', fontsize=14, fontweight='bold')
    plt.axvline(results_df['test_appearances'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["test_appearances"].mean():.1f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'test_appearances_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    print()

def export_detailed_results(results_df, output_dir):
    """Export detailed results to CSV files."""
    print("="*80)
    print("EXPORTING DETAILED RESULTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Challenging samples
    challenging = results_df.nlargest(50, 'misclassification_rate')
    output_path = output_dir / 'challenging_samples_top50.csv'
    challenging.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Robust samples
    robust = results_df.nsmallest(50, 'misclassification_rate')
    output_path = output_dir / 'robust_samples_top50.csv'
    robust.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # High variance (inconsistent) samples
    inconsistent = results_df.nlargest(50, 'std_predicted_proba')
    output_path = output_dir / 'inconsistent_samples_top50.csv'
    inconsistent.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Summary statistics by cancer type
    cancer_type_stats = results_df.groupby('cancer_type').agg({
        'sample_id': 'count',
        'misclassification_rate': ['mean', 'median', 'std'],
        'mean_predicted_proba': ['mean', 'std'],
        'std_predicted_proba': ['mean', 'median']
    }).round(4)
    cancer_type_stats.columns = ['_'.join(col).strip() for col in cancer_type_stats.columns.values]
    output_path = output_dir / 'cancer_type_summary.csv'
    cancer_type_stats.to_csv(output_path)
    print(f"  Saved: {output_path}")
    
    print()

def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python src/comp/analyze_bootstrap_results.py <path_to_bootstrap_summary.pkl>")
        print("\nExample:")
        print("  python src/comp/analyze_bootstrap_results.py ./bootstrap_results/bootstrap_best_model_robustness/bootstrap_summary.pkl")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    
    if not Path(pkl_path).exists():
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)
    
    print(f"\nLoading bootstrap results from: {pkl_path}\n")
    summary = load_bootstrap_results(pkl_path)
    
    # Extract components
    config = summary['configuration']
    results_df = summary['results_dataframe']
    
    # Print configuration
    print_configuration(config)
    
    # Analyze overall performance
    analyze_overall_performance(results_df)
    
    # Identify challenging samples
    identify_challenging_samples(results_df, n=20)
    
    # Identify robust samples
    identify_robust_samples(results_df, n=20)
    
    # Analyze prediction consistency
    analyze_prediction_consistency(results_df, n=15)
    
    # Create visualizations
    output_dir = Path(pkl_path).parent / 'analysis_plots'
    create_visualizations(results_df, output_dir)
    
    # Export detailed results
    export_dir = Path(pkl_path).parent / 'detailed_exports'
    export_detailed_results(results_df, export_dir)
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Plots: {output_dir}")
    print(f"  CSV files: {export_dir}")
    print()

if __name__ == '__main__':
    main()
