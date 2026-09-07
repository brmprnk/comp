#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_cv_atac
#SBATCH --output=logs/%j_nested_cv_atac.out
#SBATCH --error=logs/%j_nested_cv_atac.out

# ============================================================================
# Nested CV — ATAC baseline (predefined Ulz TFBS feature matrix)
# (replaces run_model.sh MODE 5)
# ============================================================================
# Verified against archived logs (logs/slurm_archive):
#   Mode Arguments: --run_nested_cv --predefined_feature_matrix extracted_features/ulz_atac_tfbs_features.npy --predefined_metadata_csv extracted_features/ulz_atac_tfbs_features.metadata.csv
#
# Usage (submit from the repo root):
#   sbatch slurm/nested_cv_atac.sh [DATASET] [STRATEGY]
#
# Overridable via environment: CANCER_TYPES, EXPERIMENT_NAME (see nested_cv_dd.sh).
# Historical manuscript runs of this mode:
#   EXPERIMENT_NAME=brca_crc_atac  CANCER_TYPES="breast colorectal"  (Fig 3)
#   EXPERIMENT_NAME=pancan_atac    CANCER_TYPES unset (PANCAN)       (Fig 5a)
# The matrix comes from extract_ulz_atac_features.py (slurm/ulz_atac_extraction.sh).
# ============================================================================

MODE="predefined_ulz_atac"
JOB_NAME="nested_cv_atac"
DATASET=${1:-cris}
STRATEGY=${2:-mean_diff}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-brca_crc_pancan_postprecisionfix}
PREDEFINED_MATRIX="extracted_features/ulz_atac_tfbs_features.npy"
PREDEFINED_META="extracted_features/ulz_atac_tfbs_features.metadata.csv"
MODE_ARGS="--run_nested_cv --predefined_feature_matrix ${PREDEFINED_MATRIX} --predefined_metadata_csv ${PREDEFINED_META}"

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
