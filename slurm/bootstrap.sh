#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=bootstrap
#SBATCH --output=logs/%j_bootstrap.out
#SBATCH --error=logs/%j_bootstrap.out

# ============================================================================
# Bootstrap robustness study (per-sample misclassification stability)
# (replaces run_model.sh MODE 3)
# ============================================================================
# Writes <output_dir>/bootstrap_<EXPERIMENT_NAME>/bootstrap_*.pkl.
#
# Usage (submit from the repo root):
#   sbatch slurm/bootstrap.sh [DATASET] [STRATEGY]
#
# NOTE (verbatim from run_model.sh MODE 3): --bootstrap_classifier is "SVM"
# here, whereas the local slurm/bootstrap_*.sh scripts pass
# "Support Vector Machine". Kept as-is; unify only after checking which
# spelling model_hpc.py's bootstrap mode actually accepts.
# ============================================================================

MODE="bootstrap"
JOB_NAME="${MODE}"
DATASET=${1:-cris}
STRATEGY=${2:-fold_change}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-bootstrap_study}
MODE_ARGS="--run_bootstrap_study --bootstrap_feature griffin_diff --bootstrap_classifier SVM --bootstrap_selection fold_change --bootstrap_n_genes 15000 --bootstrap_iterations 100 --bootstrap_n_splits 5"

. slurm/_model_hpc_exec.sh
print_job_header

set -- \
	-c 64 -t ${CANCER_TYPES:-PANCAN} \
	${MODE_ARGS} \
	--metadata_file accessory_files/${DATASET}_metadata.csv \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR:-nested_cv_results_${DATASET}} \
	-s ${STRATEGY:-mean_diff}

run_model_hpc "$@"
