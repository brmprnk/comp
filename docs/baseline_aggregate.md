# Baseline Aggregate Analysis

## Overview

The baseline aggregate mode provides a simple alternative to complex machine learning classifiers by testing whether a basic aggregated biomarker can achieve comparable performance. This helps answer the question: **"Do we need ML, or is a simple metric sufficient?"**

## How It Works

Instead of training a classifier, this mode:

1. **Iterates through all gene counts** from `DATA_PARAM_GRIDS['top_genes']` (e.g., 10, 25, 50, 100, 250, 500, 1000, etc.)
2. **For each gene count**: Selects top N genes on training data using feature selection (fold change, p-value, etc.)
3. **Aggregates those genes** from raw data files to create a single scalar value per sample
4. **Creates ROC curves** directly from the aggregated values (no ML classifier)
5. **Evaluates at multiple operating points**: median threshold, 95% specificity, 90% specificity, and Youden's optimal
6. **Saves one results file per gene count** for easy comparison across configurations

## Key Features

- **Nested Cross-Validation**: Uses the same 10-fold stratified nested CV as other experiments to avoid data leakage
- **Gene Selection on Training Data**: Top genes are selected only from training folds
- **Raw Data Aggregation**: Goes back to original files (_BIN.csv for FSLR, coverage.npy for others)
- **Per-Cancer-Type Metrics**: Evaluates each cancer type separately
- **Sample-Level Tracking**: Records which samples are correctly/incorrectly classified at each threshold

## Supported Features

### FSLR (Fragment Size Ratio)
Loads `_BIN.csv` files and computes:
```
FSLR = sum(long_fragments) / sum(short_fragments)
```
over the selected gene regions.

### Coverage-Based Features
Loads `coverage.npy` files and computes aggregation (mean, sum, or median) over selected genes:
- `fslr_coverage`
- `griffin_diff`
- `mwh` (Mean Window Height)
- `rcov` (Relative Coverage)

## Usage

### Basic Usage

```bash
# This will automatically test ALL top_genes values from DATA_PARAM_GRIDS
# (10, 25, 50, 100, 250, 500, 1000, 1500, 2500, 5000, 10000, 15000)
apptainer exec --bind $(pwd) comp.sif python -m comp.model_hpc \
    --run_baseline_aggregate \
    --baseline_feature fslr \
    --baseline_selection fold_change \
    --baseline_aggregation sum \
    --features_dir ./extracted_features/biomart_10kb \
    --metadata_file accessory_files/cris_metadata.csv \
    --experiment_name cris_fslr_baseline \
    --output_dir nested_cv_results_cris \
    --cores 64
```

### Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|------|
| `--run_baseline_aggregate` | Enable baseline aggregate mode | False | flag |
| `--baseline_feature` | Feature type to use | `fslr` | Any feature name |
| `--baseline_selection` | Gene selection method | `fold_change` | `p_value`, `fold_change`, `weighted`, `mean_diff`, `random` |
| `--baseline_aggregation` | Aggregation method | `sum` | `sum`, `mean`, `median` |
| `--features_dir` | Directory with raw feature files | Required | Path |
| `--metadata_file` | Sample metadata CSV | Required | Path |
| `--experiment_name` | Name for this experiment | Required | String |
| `--output_dir` | Output directory | Required | Path |
| `--cores` | Number of CPU cores | `64` | Integer |

**Note**: The number of genes is automatically determined by `DATA_PARAM_GRIDS['top_genes']` (typically: 10, 25, 50, 100, 250, 500, 1000, 1500, 2500, 5000, 10000, 15000). The study will iterate through all values and produce one results file per gene count.

### Selection Methods

- **`fold_change`**: Selects genes with largest fold change between cancer and healthy
- **`p_value`**: Selects genes with most significant p-values from t-tests
- **`weighted`**: Combines fold change and p-value
- **`mean_diff`**: Selects genes with largest mean difference
- **`random`**: Random selection (control)

### Aggregation Methods

- **`sum`**: Total signal across selected genes (recommended for FSLR)
- **`mean`**: Average signal across selected genes (recommended for coverage)
- **`median`**: Median signal across selected genes (robust to outliers)

## Output Format

Results are saved as a pickle file containing a DataFrame with one row per fold:

```python
{
    'feature': str,                      # Feature name
    'top_genes': int,                    # Number of genes used
    'selection_type': str,               # Selection method
    'aggregation_method': str,           # Aggregation method
    'fold': int,                         # Fold number (1-10)
    'num_genes_used': int,              # Actual number of genes selected
    
    # Sample tracking
    'test_sample_ids': list,            # IDs of test samples
    'aggregated_scores': list,          # Aggregated values per sample
    'misclassified_sample_ids': dict,   # Misclassified at each threshold
    
    # Overall metrics
    'auc_full_model': float,            # AUC
    'fpr_full_model': array,            # False positive rates
    'tpr_full_model': array,            # True positive rates
    'precision_full_model': array,      # Precision values
    'recall_full_model': array,         # Recall values
    
    # Multi-threshold evaluation
    'classification_report_full_model': dict,  # Reports at 4 operating points
    'confusion_matrix_full_model': dict,       # Confusion matrices at 4 points
    'full_model_thresholds': dict,             # Threshold values used
    
    # Per-cancer-type metrics
    'auc_per_cancer_type': dict,              # AUC per cancer type
    'roc_per_cancer_type': dict,              # ROC curves per type
    'confusion_matrix_per_cancer_type': dict, # Confusion matrices per type
    'classification_report_per_cancer_type': dict,  # Reports per type
    'sample_classification_per_type': dict,   # Sample tracking per type
}
```

### Operating Points

1. **Median Threshold**: Uses median of aggregated scores (default 0.5-like behavior)
2. **95% Specificity**: Threshold targeting 95% specificity (5% false positive rate)
3. **90% Specificity**: Threshold targeting 90% specificity (10% false positive rate)
4. **Youden's Optimal**: Threshold maximizing sensitivity + specificity - 1

## Example Workflows

### Compare Feature Types

```bash
# Each run will test all gene counts automatically
for feature in fslr fslr_coverage griffin_diff; do
    apptainer exec --bind $(pwd) comp.sif python -m comp.model_hpc \
        --run_baseline_aggregate \
        --baseline_feature $feature \
        --baseline_selection fold_change \
        --baseline_aggregation sum \
        --features_dir ./extracted_features/biomart_10kb \
        --metadata_file accessory_files/cris_metadata.csv \
        --experiment_name cris_${feature}_baseline \
        --output_dir nested_cv_results_cris \
        --cores 64
done
```

### Test Aggregation Methods

```bash
for agg in sum mean median; do
    apptainer exec --bind $(pwd) comp.sif python -m comp.model_hpc \
        --run_baseline_aggregate \
        --baseline_feature griffin_diff \
        --baseline_selection fold_change \
        --baseline_aggregation $agg \
        --features_dir ./extracted_features/biomart_10kb \
        --metadata_file accessory_files/cris_metadata.csv \
        --experiment_name cris_griffin_${agg} \
        --output_dir nested_cv_results_cris \
        --cores 64
done
```

## Interpreting Results

### High Performance (AUC > 0.85)
A simple aggregated biomarker performs well, suggesting:
- The signal is strong and doesn't require complex modeling
- A clinical test could be simpler (just measure total signal over selected genes)
- ML may provide marginal improvement over simpler approach

### Low Performance (AUC < 0.70)
ML models likely add value through:
- Capturing complex interactions between genes
- Modeling non-linear relationships
- Better handling of heterogeneous samples

### Comparison with ML Models
Compare baseline aggregate AUC with nested CV results for ML classifiers:
- Similar performance → Simple biomarker is sufficient
- Large gap → ML complexity is justified

## Technical Details

### Avoiding Data Leakage
Gene selection happens **only on training data** within each CV fold. The selected indices are then used to aggregate from raw files for both train and test samples.

### Missing Data Handling
If a sample's raw data file is missing or corrupt, the aggregation function returns `np.nan` for that sample, which is reported in the output.

### File Matching
Sample files are matched using glob patterns:
- FSLR: `{sample_id}*_BIN.csv`
- Coverage: `{sample_id}*.coverage.npy`

### Memory Efficiency
Raw files are loaded per-sample on-the-fly, avoiding loading the entire dataset into memory.

## Troubleshooting

### Issue: "Feature not found in feature arrays"
**Solution**: The baseline feature must be loaded in the initial feature loading step. Check that the feature name matches one in your features directory.

### Issue: "No genes selected"
**Solution**: 
- Check if your data has sufficient variance
- Try a different selection method
- Reduce `--baseline_n_genes` if dataset is small

### Issue: Many NaN values in aggregated scores
**Solution**:
- Verify raw data files exist in `--features_dir`
- Check file naming conventions match expected patterns
- Ensure sample IDs in metadata match file names

### Issue: Low AUC across all folds
**Solution**:
- Try different `--baseline_selection` methods
- Increase `--baseline_n_genes`
- Check data quality in diagnostic plots
- Consider that the feature may not be informative for your cancer types

## Related Documentation

- [Nested CV Documentation](nested_cv.md)
- [Feature Extraction Guide](feature_extraction.md)
- [Bootstrap Study](bootstrap_study.md)
