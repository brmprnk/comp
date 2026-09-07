#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --job-name=label_shuffle
#SBATCH --output=logs/%j_label_shuffle.out
#SBATCH --error=logs/%j_label_shuffle.out

# ============================================================================
# Label-shuffle stability analysis (Supplementary Fig S5)
# (replaces run_model.sh MODE 4b — the mode that was active when the script
#  was retired)
# ============================================================================
# Verified against archived logs (logs/slurm_archive):
#   Mode Arguments: --run_label_shuffle_experiment --label_shuffle_n_shuffles 10 --label_shuffle_cv_folds 10 --label_shuffle_top_k 1000 --features_dir extracted_features/biomart_10kb
#
# Writes <output_dir>/label_shuffle_<EXPERIMENT_NAME>/ incl.
# label_shuffle_stability.png.
#
# Usage (submit from the repo root):
#   sbatch slurm/label_shuffle.sh [DATASET] [FEATURE_TYPE]
#   FEATURE_TYPE: dd | rna | atac
#
# Overridable via environment: CANCER_TYPES (default "breast colorectal",
# matching the manuscript's healthy-vs-BRCA/CRC setting), EXPERIMENT_NAME.
# ============================================================================

MODE="label_shuffle"
DATASET=${1:-cris}
FEATURE_TYPE=${2:-dd}
JOB_NAME="${MODE}_${FEATURE_TYPE}"
STRATEGY="mean_diff"
CANCER_TYPES=${CANCER_TYPES:-"breast colorectal"}   # Healthy vs BRCA+CRC only (set to PANCAN for all cancers)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-best_model_search_${FEATURE_TYPE}}
MODE_ARGS="--run_label_shuffle_experiment --label_shuffle_n_shuffles 10 --label_shuffle_cv_folds 10 --label_shuffle_top_k 1000"
# Add feature-specific arguments
if [ "${FEATURE_TYPE}" = "rna" ]; then
    MODE_ARGS="${MODE_ARGS} --no_gene_selection --features_dir extracted_features/zhu_all_after_lift --feature_subset rcov"
elif [ "${FEATURE_TYPE}" = "atac" ]; then
    PREDEFINED_MATRIX="extracted_features/ulz_atac_tfbs_features.npy"
    PREDEFINED_META="extracted_features/ulz_atac_tfbs_features.metadata.csv"
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
