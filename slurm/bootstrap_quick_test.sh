#!/bin/bash

# Quick Bootstrap Test - Faster execution for testing

# This runs a quick test with fewer iterations to verify the pipeline works
# Use this before committing to a full 100-iteration study

python src/comp/model_hpc.py \
    --run_bootstrap_study \
    --bootstrap_feature griffin_diff \
    --bootstrap_classifier "Support Vector Machine" \
    --bootstrap_selection fold_change \
    --bootstrap_n_genes 15000 \
    --bootstrap_iterations 20 \
    --bootstrap_n_splits 5 \
    --metadata_file accessory_files/cris_metadata.csv \
    -o ./bootstrap_test \
    -e quick_test \
    -t PANCAN

# This will:
# - Run 20 iterations × 5 folds = 100 total trainings
# - Each sample appears in test set ~20 times
# - Estimated time: ~30-45 minutes

echo ""
echo "Bootstrap test complete! Results in: ./bootstrap_test/bootstrap_quick_test/"
echo ""
echo "Quick analysis:"
python -c "
import pandas as pd
results = pd.read_csv('./bootstrap_test/bootstrap_quick_test/sample_robustness_results.csv')
print(f'Total samples: {len(results)}')
print(f'Overall misclassification rate: {results[\"misclassification_rate\"].mean():.4f}')
print(f'Samples with 0% misclassification: {len(results[results[\"misclassification_rate\"] == 0.0])}')
print(f'Samples with >50% misclassification: {len(results[results[\"misclassification_rate\"] > 0.5])}')
"
