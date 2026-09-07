#!/bin/bash

# Bootstrap study with custom hyperparameters
# Example: SVM with specific C and kernel parameters

python src/comp/model_hpc.py \
    --run_bootstrap_study \
    --bootstrap_feature griffin_diff \
    --bootstrap_classifier "Support Vector Machine" \
    --bootstrap_selection fold_change \
    --bootstrap_n_genes 15000 \
    --bootstrap_iterations 100 \
    --bootstrap_n_splits 5 \
    --bootstrap_hyperparams '{"C": 1.0, "kernel": "rbf", "gamma": "scale"}' \
    --metadata_file accessory_files/cris_metadata.csv \
    -o ./bootstrap_custom \
    -e svm_rbf_c1 \
    -t PANCAN

# Other examples:

# Random Forest with 500 trees
# --bootstrap_classifier "Random Forest" \
# --bootstrap_hyperparams '{"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 5, "max_features": "sqrt"}'

# Logistic Regression with ElasticNet
# --bootstrap_classifier "Logistic Regression" \
# --bootstrap_hyperparams '{"C": 0.1, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5}'

# KNN with 15 neighbors
# --bootstrap_classifier "K-Nearest Neighbors" \
# --bootstrap_hyperparams '{"n_neighbors": 15, "weights": "distance", "metric": "euclidean"}'

# MLP with specific architecture
# --bootstrap_classifier "Multi-Layer Perceptron" \
# --bootstrap_hyperparams '{"hidden_layer_sizes": [256, 128], "alpha": 0.001, "learning_rate_init": 0.001}'
