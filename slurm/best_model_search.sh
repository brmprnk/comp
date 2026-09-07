#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=best_model_search
#SBATCH --output=logs/%j_best_model_search.out
#SBATCH --error=logs/%j_best_model_search.out

# ============================================================================
# Best-model search — trains/saves the single best model per configuration
# (replaces run_model.sh MODE 4)
# ============================================================================
# Produces <output_dir>/best_model_<EXPERIMENT_NAME>/ with
# best_model_GridSearchCV_*.pkl etc., consumed by the external-validation and
# label-shuffle experiments.
#
# Usage (submit from the repo root):
#   sbatch slurm/best_model_search.sh [DATASET] [FEATURE_TYPE]
#   FEATURE_TYPE: dd (biomart TSS) | rna (zhu_all_after_lift) | atac (ulz TFBS)
#
# Overridable via environment: CANCER_TYPES, EXPERIMENT_NAME.
# NOTE (verbatim from run_model.sh): the atac branch uses the *_pancan* matrix
# (ulz_atac_tfbs_features_pancan.npy), unlike MODE 5 / label_shuffle which use
# ulz_atac_tfbs_features.npy.
# ============================================================================

MODE="best_model_search"
DATASET=${1:-cris}
FEATURE_TYPE=${2:-dd}
JOB_NAME="${MODE}_${FEATURE_TYPE}"
STRATEGY="mean_diff"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-best_model_search_${FEATURE_TYPE}}
MODE_ARGS="--run_best_model_search"
# Add feature-specific arguments
if [ "${FEATURE_TYPE}" = "rna" ]; then
    MODE_ARGS="${MODE_ARGS} --no_gene_selection --features_dir extracted_features/zhu_all_after_lift --feature_subset rcov"
elif [ "${FEATURE_TYPE}" = "atac" ]; then
    PREDEFINED_MATRIX="extracted_features/ulz_atac_tfbs_features_pancan.npy"
    PREDEFINED_META="extracted_features/ulz_atac_tfbs_features_pancan.metadata.csv"
    MODE_ARGS="${MODE_ARGS} --predefined_feature_matrix ${PREDEFINED_MATRIX} --predefined_metadata_csv ${PREDEFINED_META}"
else
    MODE_ARGS="${MODE_ARGS} --features_dir extracted_features/biomart_10kb"
fi

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
