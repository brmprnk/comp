#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END
#SBATCH --job-name=ext_val
#SBATCH --output=logs/%j_ext_val.out
#SBATCH --error=logs/%j_ext_val.out

# =================================================================================
# External Dataset Validation Script
# =================================================================================
#
# This script validates a model trained on one dataset (e.g., CRIS BRCA+CRC)
# on a completely different external dataset (e.g., Jiang Liver Cancer).
#
# Usage:
#   sbatch slurm/external_validation.sh [FEATURE_TYPE] [TEST_DATASET]
#   sbatch slurm/external_validation.sh dd              # DD on jiang (default)
#   sbatch slurm/external_validation.sh rna              # RNA on jiang
#   sbatch slurm/external_validation.sh atac             # ATAC on jiang
#   sbatch slurm/external_validation.sh dd lucas         # DD on lucas
#
# What it does:
#   1. Loads the best model from CRIS training results
#   2. Applies it to Jiang dataset (never seen during training)
#   3. Generates comprehensive evaluation metrics (all features & gene counts)
#
# =================================================================================

# Configuration
TRAIN_DATASET="cris"
FEATURE_TYPE=${1:-dd}   # dd, rna, or atac
TEST_DATASET=${2:-jiang} # jiang, lucas, mathios
STRATEGY="mean_diff"
EXTERNAL_METADATA_FILE="accessory_files/${TEST_DATASET}_metadata.csv"

# Set model directory and feature-loading args based on feature type
if [ "${FEATURE_TYPE}" = "rna" ]; then
    TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_rna"
    EXPERIMENT_NAME="comprehensive_external_${TRAIN_DATASET}_on_${TEST_DATASET}_rna"
    FEATURE_ARGS="--features_dir extracted_features/zhu_all_after_lift --no_gene_selection --feature_subset rcov"
    EXTRA_ARGS="--filter_feature rcov"
    COMPREHENSIVE="--comprehensive_external_validation"
elif [ "${FEATURE_TYPE}" = "atac" ]; then
    TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_atac"
    EXPERIMENT_NAME="comprehensive_external_${TRAIN_DATASET}_on_${TEST_DATASET}_atac"
    FEATURE_ARGS="--predefined_feature_matrix extracted_features/ulz_atac_tfbs_features_${TEST_DATASET}.npy --predefined_metadata_csv extracted_features/ulz_atac_tfbs_features_${TEST_DATASET}.metadata.csv"
    EXTRA_ARGS=""
    COMPREHENSIVE="--comprehensive_external_validation"
else
    TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_dd"
    EXPERIMENT_NAME="comprehensive_external_${TRAIN_DATASET}_on_${TEST_DATASET}_dd"
    FEATURE_ARGS="--features_dir extracted_features/biomart_10kb"
    EXTRA_ARGS=""
    COMPREHENSIVE="--comprehensive_external_validation"
fi

OUTPUT_DIR="nested_cv_results_${TRAIN_DATASET}"

# Job name
JOB_NAME="ext_val_${FEATURE_TYPE}_${TEST_DATASET}"

echo "============================================"
echo "External Dataset Validation"
echo "Training Dataset: ${TRAIN_DATASET}"
echo "Test Dataset: ${TEST_DATASET}"
echo "Feature Type: ${FEATURE_TYPE}"
echo "Model Directory: ${TRAIN_MODEL_DIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Feature Args: ${FEATURE_ARGS}"
echo "============================================"
echo ""

# Execute
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
	--bind nested_cv_results_jiang/:/opt/app/nested_cv_results_jiang/ \
	--bind nested_cv_results_lucas/:/opt/app/nested_cv_results_lucas/ \
	--bind nested_cv_results_mathios/:/opt/app/nested_cv_results_mathios/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/:/opt/app/cris/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/jiang/:/opt/app/jiang/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/lucas/:/opt/app/lucas/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/mathios/:/opt/app/mathios/ \
	--bind extracted_features/:/opt/app/extracted_features/ \
		./comp.sif pixi run python -u src/comp/model_hpc.py \
		-c 4 \
		--validate_external \
		${COMPREHENSIVE} \
		--external_model_dir ${TRAIN_MODEL_DIR} \
		--external_metadata_file ${EXTERNAL_METADATA_FILE} \
		${FEATURE_ARGS} \
		${EXTRA_ARGS} \
		-e "${EXPERIMENT_NAME}" \
		-o ${OUTPUT_DIR} \
		-s ${STRATEGY} \
		-t PANCAN
