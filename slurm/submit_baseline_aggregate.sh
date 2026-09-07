#!/bin/bash

# Submit baseline aggregate jobs for all feature and selection method combinations
# This tests simple aggregated biomarkers (no ML) across different configurations
# 
# Each job will automatically iterate through ALL top_genes values from DATA_PARAM_GRIDS
# (10, 25, 50, 100, 250, 500, 1000, 1500, 2500, 5000, 10000, 15000)
# and save separate results files for each gene count.

DATASET="cris"  # Change to jiang, lucas, or mathios as needed

# Features to test (each has its own inherent aggregation method)
FEATURES=("fslr" "fslr_coverage" "griffin_diff" "mwh" "rcov")

# Selection methods to test
SELECTIONS=("p_value" "fold_change" "weighted" "mean_diff" "random")

echo "============================================"
echo "Submitting baseline aggregate jobs"
echo "Dataset: ${DATASET}"
echo "Features: ${FEATURES[@]}"
echo "Selection methods: ${SELECTIONS[@]}"
echo "Total jobs: $((${#FEATURES[@]} * ${#SELECTIONS[@]}))"
echo "Note: Each job tests all top_genes values (12 configurations per job)"
echo "============================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for feature in "${FEATURES[@]}"; do
    for selection in "${SELECTIONS[@]}"; do
        echo "Submitting: ${feature} + ${selection}"
        
        # Submit job with feature and selection as positional arguments
        # slurm/baseline_aggregate.sh expects: $1=DATASET, $2=FEATURE, $3=STRATEGY
        sbatch slurm/baseline_aggregate.sh "${DATASET}" "${feature}" "${selection}"
        
        # Small delay to avoid overwhelming the scheduler
        sleep 0.2
    done
done

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: logs/"
echo "============================================"
