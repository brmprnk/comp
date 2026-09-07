#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Best Model Results

This script loads all best_model_GridSearchCV results from a directory,
creates an overview of all models, and identifies the single best performing model.

Usage:
    python src/comp/analyze_best_models.py <results_directory>
    python src/comp/analyze_best_models.py ./nested_cv_results/best_model_myexp/

Example:
    python src/comp/analyze_best_models.py ./nested_cv_results/best_model_brca_crc/ --save_summary
"""

import argparse
import sys
from pathlib import Path

# Import the analysis function from model_hpc
from model_hpc import analyze_best_models_and_get_best  # sibling module; run as: python src/comp/analyze_best_models.py


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best model GridSearchCV results and identify the overall best model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to directory containing best_model_GridSearchCV_*.pkl files"
    )
    
    parser.add_argument(
        "--save_summary",
        action='store_true',
        help="Save the overview DataFrame to CSV in the results directory"
    )
    
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top models to display in detail"
    )
    
    parser.add_argument(
        "--export_best_model",
        type=str,
        default=None,
        help="Path to save the best model's GridSearchCV object (as .pkl)"
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    # Run analysis
    try:
        results = analyze_best_models_and_get_best(
            results_dir=str(results_dir),
            verbose=True
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)
    
    # Save summary if requested
    if args.save_summary:
        overview_df = results['overview_df']
        # Drop the grid_search_object column for CSV export
        export_df = overview_df.drop(columns=['grid_search_object', 'filepath'])
        
        # Convert best_params dict to string for CSV
        export_df['best_params'] = export_df['best_params'].apply(str)
        
        output_path = results_dir / "best_models_summary.csv"
        export_df.to_csv(output_path, index=False)
        print(f"\n✅ Summary saved to: {output_path}")
    
    # Export best model if requested
    if args.export_best_model:
        import pickle
        export_path = Path(args.export_best_model)
        with open(export_path, 'wb') as f:
            pickle.dump(results['best_model'], f)
        print(f"\n✅ Best model exported to: {export_path}")
    
    # Print additional statistics
    overview_df = results['overview_df']
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90)
    print(f"Total models analyzed: {len(overview_df)}")
    print(f"\nCV AUC Statistics:")
    print(f"  Mean:   {overview_df['cv_auc'].mean():.4f}")
    print(f"  Median: {overview_df['cv_auc'].median():.4f}")
    print(f"  Std:    {overview_df['cv_auc'].std():.4f}")
    print(f"  Min:    {overview_df['cv_auc'].min():.4f}")
    print(f"  Max:    {overview_df['cv_auc'].max():.4f}")
    
    print(f"\nBreakdown by Feature:")
    feature_stats = overview_df.groupby('feature')['cv_auc'].agg(['count', 'mean', 'max'])
    print(feature_stats.to_string())
    
    print(f"\nBreakdown by Classifier:")
    classifier_stats = overview_df.groupby('classifier')['cv_auc'].agg(['count', 'mean', 'max'])
    print(classifier_stats.to_string())
    
    print(f"\nBreakdown by Number of Genes:")
    genes_stats = overview_df.groupby('n_genes')['cv_auc'].agg(['count', 'mean', 'max']).sort_index()
    print(genes_stats.to_string())
    
    print("\n" + "="*90)
    print("✅ Analysis complete!")
    print("="*90 + "\n")
    
    # Provide guidance on next steps
    print("Next steps:")
    print(f"  1. Use the best model object: results['best_model']")
    print(f"  2. Access best estimator: results['best_model'].best_estimator_")
    print(f"  3. Best model info: results['best_model_info']")
    print(f"  4. Train on full dataset using the best hyperparameters")


if __name__ == "__main__":
    main()
