#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=ext_val_single
#SBATCH --output=logs/%j_ext_val_single.out
#SBATCH --error=logs/%j_ext_val_single.out

# ============================================================================
# External validation — single best model only (replaces run_model.sh MODE 7)
# ============================================================================
# Applies the stored best CRIS model to an external cohort, without the
# comprehensive all-features/all-gene-counts sweep. For the comprehensive
# sweep used in the manuscript (Fig 5b,c / S7), use
# slurm/external_validation.sh (formerly run_external_validation.sh) or
# slurm/submit_external_val_all.sh.
#
# Usage (submit from the repo root):
#   sbatch slurm/external_validation_single.sh [TEST_DATASET] [FEATURE_TYPE]
#   TEST_DATASET: jiang | lucas | mathios      FEATURE_TYPE: dd | rna | atac
#
# Requires the matching slurm/best_model_search.sh run to exist under
# nested_cv_results_cris/best_model_best_model_search_<ft>/.
# ============================================================================

MODE="external_validation"
TRAIN_DATASET="cris"                # Dataset the model was trained on
TEST_DATASET=${1:-jiang}            # Dataset to validate on (jiang, lucas, mathios)
DATASET=${TRAIN_DATASET}            # Set DATASET for original training metadata
FEATURE_TYPE=${2:-dd}
JOB_NAME="${MODE}_${FEATURE_TYPE}_${TEST_DATASET}"
EXTERNAL_METADATA_FILE="accessory_files/${TEST_DATASET}_metadata.csv"
STRATEGY="mean_diff"
# Set model directory and feature-loading args based on feature type
if [ "${FEATURE_TYPE}" = "rna" ]; then
    TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_rna"
    EXPERIMENT_NAME="external_val_${TRAIN_DATASET}_on_${TEST_DATASET}_rna"
    FEATURE_ARGS="--features_dir extracted_features/zhu_all_after_lift --no_gene_selection --feature_subset rcov --filter_feature rcov"
elif [ "${FEATURE_TYPE}" = "atac" ]; then
    TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_atac"
    EXPERIMENT_NAME="external_val_${TRAIN_DATASET}_on_${TEST_DATASET}_atac"
    FEATURE_ARGS="--predefined_feature_matrix extracted_features/ulz_atac_tfbs_features_${TEST_DATASET}.npy --predefined_metadata_csv extracted_features/ulz_atac_tfbs_features_${TEST_DATASET}.metadata.csv"
else
    TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_dd"
    EXPERIMENT_NAME="external_val_${TRAIN_DATASET}_on_${TEST_DATASET}_dd"
    FEATURE_ARGS="--features_dir extracted_features/biomart_10kb"
fi
OUTPUT_DIR="nested_cv_results_${TRAIN_DATASET}"     # Save to training dataset directory
MODE_ARGS="--validate_external --external_model_dir ${TRAIN_MODEL_DIR} --external_metadata_file ${EXTERNAL_METADATA_FILE} ${FEATURE_ARGS}"

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
