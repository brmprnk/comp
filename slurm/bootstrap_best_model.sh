#!/bin/bash

# Bootstrap Robustness Study Example
# Runs a single model configuration multiple times to assess sample classification robustness

python src/comp/model_hpc.py \
    --run_bootstrap_study \
    --bootstrap_feature griffin_diff \
    --bootstrap_classifier "Support Vector Machine" \
    --bootstrap_selection fold_change \
    --bootstrap_n_genes 15000 \
    --bootstrap_iterations 100 \
    --bootstrap_n_splits 5 \
    --metadata_file accessory_files/cris_metadata.csv \
    -o ./bootstrap_results \
    -e best_model_robustness \
    -t PANCAN

# Output: bootstrap_results/bootstrap_best_model_robustness/bootstrap_*.pkl
# Contains per-sample statistics: test appearances, misclassifications, predicted probabilities

