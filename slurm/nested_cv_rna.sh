#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_cv_rna
#SBATCH --output=logs/%j_nested_cv_rna.out
#SBATCH --error=logs/%j_nested_cv_rna.out

# ============================================================================
# Nested CV — RNA baseline (Zhu et al. regions, RCOV only, no gene selection)
# (replaces run_model.sh MODE 2)
# ============================================================================
# Verified against archived logs (logs/slurm_archive):
#   Mode Arguments: --run_nested_cv --no_gene_selection --features_dir extracted_features/zhu_all_after_lift
#
# Usage (submit from the repo root):
#   sbatch slurm/nested_cv_rna.sh [DATASET] [STRATEGY]
#
# Overridable via environment: CANCER_TYPES, EXPERIMENT_NAME (see nested_cv_dd.sh).
# Historical manuscript runs of this mode:
#   EXPERIMENT_NAME=brca_crc_rna       CANCER_TYPES="breast colorectal"  (Fig 3)
#   EXPERIMENT_NAME=nested_rna_pancan  CANCER_TYPES unset (PANCAN)       (Fig 5a)
# ============================================================================

MODE="no_gene_selection"
JOB_NAME="nested_cv_rna"
DATASET=${1:-cris}
STRATEGY=${2:-mean_diff}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-brca_crc_pancan_postprecisionfix}
MODE_ARGS="--run_nested_cv --no_gene_selection --features_dir extracted_features/zhu_all_after_lift"

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
