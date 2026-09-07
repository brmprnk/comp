#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=within_gen_compare
#SBATCH --output=logs/%j_within_gen_compare.out
#SBATCH --error=logs/%j_within_gen_compare.out

# ============================================================================
# Within-dataset generalization — comparison only (DD vs RNA vs ATAC)
# (replaces run_model.sh MODE 9)
# ============================================================================
# Compares three completed within-gen training runs (see slurm/within_gen.sh /
# slurm/submit_within_gen_all.sh); writes
# <output_dir>/within_gen_comparison/within_gen_dd_rna_atac_comparison.png.
#
# Usage (submit from the repo root):
#   sbatch slurm/within_gen_compare.sh [DATASET] [DD_DIR] [RNA_DIR] [ATAC_DIR]
# ============================================================================

MODE="within_gen_compare"
JOB_NAME="${MODE}"
DATASET=${1:-cris}
STRATEGY="mean_diff"
# Point to the three training directories (or their comprehensive_evaluation subdirs)
COMPARE_DD_DIR=${2:-"results/within_gen_Breast_cancer_Colorectal_cancer_dd"}
COMPARE_RNA_DIR=${3:-"results/within_gen_Breast_cancer_Colorectal_cancer_rna"}
COMPARE_ATAC_DIR=${4:-"results/within_gen_Breast_cancer_Colorectal_cancer_atac"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-within_gen_compare}
MODE_ARGS="--run_within_dataset_generalization_compare --generalization_compare_dirs ${COMPARE_DD_DIR} ${COMPARE_RNA_DIR} ${COMPARE_ATAC_DIR}"

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
