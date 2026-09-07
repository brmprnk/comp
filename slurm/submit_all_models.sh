#!/bin/bash

DATASETS="jiang"
STRATEGIES="mean_diff random p_value fold_change weighted"
# Loop over each dataset
for dataset in $DATASETS; do

  # Loop over each strategy for that dataset
  for strategy in $STRATEGIES; do

    echo "Submitting job: DATASET=${dataset}, STRATEGY=${strategy}"
    sbatch slurm/nested_cv_dd.sh "$dataset" "$strategy"
    sleep 1;
  done
done
