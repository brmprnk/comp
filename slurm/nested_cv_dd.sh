#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_cv_dd
#SBATCH --output=logs/%j_nested_cv_dd.out
#SBATCH --error=logs/%j_nested_cv_dd.out

# ============================================================================
# Nested CV — data-driven TSS features (replaces run_model.sh MODE 1)
# ============================================================================
# Verified against archived logs (logs/slurm_archive):
#   Mode Arguments: --run_nested_cv --features_dir extracted_features/biomart_10kb
#
# Usage (submit from the repo root):
#   sbatch slurm/nested_cv_dd.sh [DATASET] [STRATEGY]
#
# Overridable via environment:
#   CANCER_TYPES     e.g. "breast colorectal" for the healthy-vs-BRCA/CRC task;
#                    empty (default) = PANCAN (all cancers vs healthy)
#   EXPERIMENT_NAME  results land in <output_dir>/<EXPERIMENT_NAME>/
#
# Historical manuscript runs of this mode (per archived logs / result dirs):
#   EXPERIMENT_NAME=brca_crc_dd   CANCER_TYPES="breast colorectal"   (Fig 3)
#   EXPERIMENT_NAME=pancan_dd     CANCER_TYPES unset (PANCAN)        (Fig 5a)
# ============================================================================

MODE="nested_cv"
JOB_NAME="${MODE}_dd"
DATASET=${1:-cris}
STRATEGY=${2:-mean_diff}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-brca_crc_pancan_postprecisionfix}
MODE_ARGS="--run_nested_cv --features_dir extracted_features/biomart_10kb"

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
