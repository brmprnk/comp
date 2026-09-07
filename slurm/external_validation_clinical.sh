#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=ext_val_clinical
#SBATCH --output=logs/%j_ext_val_clinical.out
#SBATCH --error=logs/%j_ext_val_clinical.out

# ============================================================================
# External validation — CLINICAL mode (no test-set StandardScaler refit)
# (replaces run_model.sh MODEs 7c and 7d)
# ============================================================================
# Uses the pancan best model (best_model_best_model_search_pancan) and
# --external_validation_mode clinical. DD features only, matching the retired
# modes verbatim.
#
# Usage (submit from the repo root):
#   sbatch slurm/external_validation_clinical.sh [TEST_DATASET] [VARIANT]
#   TEST_DATASET: jiang | lucas | mathios
#   VARIANT: comprehensive (default; = old MODE 7d, all features & gene counts)
#            single        (= old MODE 7c, best model only)
# ============================================================================

TEST_DATASET=${1:-jiang}
VARIANT=${2:-comprehensive}
TRAIN_DATASET="cris"
DATASET=${TRAIN_DATASET}
FEATURE_TYPE="dd"
EXTERNAL_METADATA_FILE="accessory_files/${TEST_DATASET}_metadata.csv"
STRATEGY="mean_diff"
TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_pancan"
OUTPUT_DIR="nested_cv_results_${TRAIN_DATASET}"

if [ "${VARIANT}" = "single" ]; then
    MODE="clinical_validation"
    EXPERIMENT_NAME="clinical_val_${TRAIN_DATASET}_on_${TEST_DATASET}"
    MODE_ARGS="--validate_external --external_validation_mode clinical --external_model_dir ${TRAIN_MODEL_DIR} --external_metadata_file ${EXTERNAL_METADATA_FILE} --features_dir extracted_features/biomart_10kb"
else
    MODE="comprehensive_clinical_validation"
    EXPERIMENT_NAME="comprehensive_clinical_${TRAIN_DATASET}_on_${TEST_DATASET}"
    MODE_ARGS="--validate_external --comprehensive_external_validation --external_validation_mode clinical --external_model_dir ${TRAIN_MODEL_DIR} --external_metadata_file ${EXTERNAL_METADATA_FILE} --features_dir extracted_features/biomart_10kb"
fi
JOB_NAME="${MODE}_${TEST_DATASET}"

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
