#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=24:38:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --mail-type=END
#SBATCH --job-name=within_gen
#SBATCH --output=logs/%j_within_gen.out
#SBATCH --error=logs/%j_within_gen.out

# ============================================================================
# WITHIN-DATASET GENERALIZATION: Single Feature Type (for Parallel Execution)
# ============================================================================
# Submit 3 separate jobs to run DD, RNA, and ATAC in parallel (3×64 CPUs)
#
# Usage:
#   sbatch slurm/within_gen.sh dd "Breast cancer" "Colorectal cancer"
#   sbatch slurm/within_gen.sh rna "Breast cancer" "Colorectal cancer"
#   sbatch slurm/within_gen.sh atac "Breast cancer" "Colorectal cancer"
#
# To run ONLY evaluation (skip training):
#   sbatch slurm/within_gen.sh dd "Breast cancer" "Colorectal cancer" true
#   sbatch slurm/within_gen.sh rna "Breast cancer" "Colorectal cancer" true
#   sbatch slurm/within_gen.sh atac "Breast cancer" "Colorectal cancer" true
#
# Or use the helper to submit all three:
#   ./slurm/submit_within_gen_all.sh "Breast cancer" "Colorectal cancer"
#
# Each job automatically:
#   1. Trains models for the specified feature type
#   2. Runs comprehensive evaluation
#   3. Generates plots and summaries
# ============================================================================

FEATURE_TYPE=${1:-dd}  # dd, rna, or atac
CANCER_TYPE_1=${2:-"Breast cancer"}
CANCER_TYPE_2=${3:-"Colorectal cancer"}
SKIP_TRAINING=${4:-false}  # Set to 'true' to skip training and only run evaluation

DATASET="cris"
EXPERIMENT_BASE="within_gen_${CANCER_TYPE_1// /_}_${CANCER_TYPE_2// /_}"
TRAIN_DIR="results/${EXPERIMENT_BASE}_${FEATURE_TYPE}"

echo "============================================"
echo "Within-Dataset Generalization"
echo "============================================"
echo "Feature Type: ${FEATURE_TYPE}"
echo "Dataset: ${DATASET}"
echo "Training Types: ${CANCER_TYPE_1}, ${CANCER_TYPE_2}"
echo "Output Directory: ${TRAIN_DIR}"
echo "============================================"
echo ""

# Common apptainer setup
APPTAINER_CMD="apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind results/:/opt/app/results/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/:/opt/app/cris/ \
	--bind extracted_features/:/opt/app/extracted_features/ \
	./comp.sif pixi run python -u src/comp/model_hpc.py"

# ============================================================================
# Configure feature-specific parameters
# ============================================================================
case "$FEATURE_TYPE" in
    dd)
        echo "Configuring DD (biomarker) features..."
        FEATURES_DIR="extracted_features/biomart_10kb"
        EXTRA_FLAGS=""
        ;;
    rna)
        echo "Configuring RNA (expression) features..."
        FEATURES_DIR="extracted_features/zhu_all_after_lift"
        EXTRA_FLAGS="--no_gene_selection --feature_subset rcov"
        ;;
    atac)
        echo "Configuring ATAC (chromatin) features..."
        FEATURES_DIR=""
        EXTRA_FLAGS="--predefined_feature_matrix extracted_features/ulz_atac_tfbs_features.npy --predefined_metadata_csv extracted_features/ulz_atac_tfbs_features.metadata.csv"
        ;;
    *)
        echo "❌ Error: Unknown feature type '${FEATURE_TYPE}'"
        echo "   Valid options: dd, rna, atac"
        exit 1
        ;;
esac

# ============================================================================
# STEP 1: TRAINING
# ============================================================================
if [ "$SKIP_TRAINING" = "true" ]; then
    echo ""
    echo "========================================"
    echo "⏭️  SKIPPING TRAINING (--skip_training)"
    echo "========================================"
    echo "Using existing models in: ${TRAIN_DIR}"
    echo ""
else
    echo ""
    echo "========================================"
    echo "STEP 1: TRAINING ${FEATURE_TYPE^^} MODELS"
    echo "========================================"
    echo ""

    $APPTAINER_CMD \
        -c 64 -t "PANCAN" \
        --run_within_dataset_generalization_train \
        --within_gen_feature_types ${FEATURE_TYPE} \
        ${EXTRA_FLAGS} \
        $([ ! -z "$FEATURES_DIR" ] && echo "--features_dir $FEATURES_DIR") \
        --metadata_file accessory_files/${DATASET}_metadata.csv \
        -e "${EXPERIMENT_BASE}" \
        -o "results" \
        -s mean_diff \
        --generalization_training_types "${CANCER_TYPE_1}" "${CANCER_TYPE_2}"

    if [ $? -ne 0 ]; then
        echo "❌ Training failed for ${FEATURE_TYPE}"
        exit 1
    fi

    echo ""
    echo "✓ Training complete for ${FEATURE_TYPE}"
fi

# ============================================================================
# STEP 2: COMPREHENSIVE EVALUATION (Automatic)
# ============================================================================
echo ""
echo "========================================"
echo "STEP 2: EVALUATING ${FEATURE_TYPE^^} MODELS"
echo "========================================"
echo "Running comprehensive analysis..."
echo ""

$APPTAINER_CMD \
    -c 64 -t "PANCAN" \
    --run_within_dataset_generalization_eval \
    --within_gen_feature_types ${FEATURE_TYPE} \
    --comprehensive_within_dataset_generalization \
    --generalization_training_dir "${TRAIN_DIR}" \
    ${EXTRA_FLAGS} \
    $([ ! -z "$FEATURES_DIR" ] && echo "--features_dir $FEATURES_DIR") \
    --metadata_file accessory_files/${DATASET}_metadata.csv \
    -e "eval" \
    -o "results" \
    -s mean_diff

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed for ${FEATURE_TYPE}"
    exit 1
fi

echo ""
echo "✓ Comprehensive evaluation complete for ${FEATURE_TYPE}"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================"
echo "✅ ${FEATURE_TYPE^^} PIPELINE COMPLETE!"
echo "============================================"
echo ""
echo "📊 Results available in:"
echo "   ${TRAIN_DIR}/"
echo "   ${TRAIN_DIR}/comprehensive_evaluation/"
echo ""
echo "📈 Files generated:"
echo "   - best_model_GridSearchCV_*.pkl (trained models)"
echo "   - summary_per_feature.csv"
echo "   - summary_per_gene_count.csv"
echo "   - roc_curves_per_feature.png"
echo "   - roc_curves_per_gene_count.png"
echo ""
echo "🔄 Next steps:"
echo "   If all 3 feature types are complete, run:"
echo "   jupyter notebook plot_within_generalization.ipynb"
echo "============================================"
