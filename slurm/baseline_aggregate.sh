#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=baseline_aggregate
#SBATCH --output=logs/%j_baseline_aggregate.out
#SBATCH --error=logs/%j_baseline_aggregate.out

# ============================================================================
# Baseline aggregate — classifier-free aggregated biomarker (Fig 4c inputs)
# (replaces run_model.sh MODE 6; see docs/baseline_aggregate.md)
# ============================================================================
# Writes <output_dir>/baseline_aggregate_<EXPERIMENT_NAME>/ with one results
# file per gene count. Each job iterates all top_genes values internally.
#
# Usage (submit from the repo root; same positional contract as the old
# `sbatch run_model.sh <dataset> <feature> <selection>` used by
# slurm/submit_baseline_aggregate.sh):
#   sbatch slurm/baseline_aggregate.sh [DATASET] [FEATURE] [SELECTION]
#   FEATURE:   fslr | fslr_coverage | griffin_diff | mwh | rcov
#   SELECTION: p_value | fold_change | weighted | mean_diff | random
# ============================================================================

MODE="baseline_aggregate"
DATASET=${1:-cris}
FEATURE=${2:-griffin_diff}
STRATEGY=${3:-fold_change}
JOB_NAME="${MODE}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-baseline_${FEATURE}_${STRATEGY}}
MODE_ARGS="--run_baseline_aggregate --baseline_feature ${FEATURE} --baseline_selection ${STRATEGY} --features_dir extracted_features/biomart_10kb"

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
