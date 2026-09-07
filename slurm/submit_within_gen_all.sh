#!/bin/bash
# ============================================================================
# Submit all 3 feature types as separate parallel jobs
# ============================================================================
# Usage:
#   ./slurm/submit_within_gen_all.sh "Breast cancer" "Colorectal cancer"
#   ./slurm/submit_within_gen_all.sh  # Uses defaults
#
# To run ONLY evaluation (skip training):
#   ./slurm/submit_within_gen_all.sh "Breast cancer" "Colorectal cancer" true
#
# To run ONLY a single feature type (dd/rna/atac):
#   ./slurm/submit_within_gen_all.sh "Breast cancer" "Colorectal cancer" atac
#   ./slurm/submit_within_gen_all.sh "Breast cancer" "Colorectal cancer" atac true
# ============================================================================

CANCER_TYPE_1=${1:-"Breast cancer"}
CANCER_TYPE_2=${2:-"Colorectal cancer"}
FEATURE_ONLY=${3:-""}
SKIP_TRAINING=${4:-false}

if [ "$FEATURE_ONLY" = "true" ] || [ "$FEATURE_ONLY" = "false" ]; then
    SKIP_TRAINING=${FEATURE_ONLY}
    FEATURE_ONLY=""
fi

echo "============================================"
echo "Submitting 3 Parallel Jobs"
echo "============================================"
echo "Cancer Types: ${CANCER_TYPE_1}, ${CANCER_TYPE_2}"
if [ ! -z "$FEATURE_ONLY" ]; then
    echo "Feature Only: ${FEATURE_ONLY}"
fi
echo "Skip Training: ${SKIP_TRAINING}"
echo ""

JOB_DD=""
JOB_RNA=""
JOB_ATAC=""

if [ -z "$FEATURE_ONLY" ] || [ "$FEATURE_ONLY" = "dd" ]; then
    echo "Submitting DD job..."
    JOB_DD=$(sbatch --parsable slurm/within_gen.sh dd "${CANCER_TYPE_1}" "${CANCER_TYPE_2}" "${SKIP_TRAINING}")
    echo "  Job ID: ${JOB_DD}"
fi

if [ -z "$FEATURE_ONLY" ] || [ "$FEATURE_ONLY" = "rna" ]; then
    echo "Submitting RNA job..."
    JOB_RNA=$(sbatch --parsable slurm/within_gen.sh rna "${CANCER_TYPE_1}" "${CANCER_TYPE_2}" "${SKIP_TRAINING}")
    echo "  Job ID: ${JOB_RNA}"
fi

if [ -z "$FEATURE_ONLY" ] || [ "$FEATURE_ONLY" = "atac" ]; then
    echo "Submitting ATAC job..."
    JOB_ATAC=$(sbatch --parsable slurm/within_gen.sh atac "${CANCER_TYPE_1}" "${CANCER_TYPE_2}" "${SKIP_TRAINING}")
    echo "  Job ID: ${JOB_ATAC}"
fi

echo ""
echo "============================================"
if [ -z "$FEATURE_ONLY" ]; then
    echo "✅ All 3 jobs submitted!"
else
    echo "✅ ${FEATURE_ONLY} job submitted!"
fi
echo "============================================"
echo "Job IDs: ${JOB_DD} (DD), ${JOB_RNA} (RNA), ${JOB_ATAC} (ATAC)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
if [ "$SKIP_TRAINING" = "true" ]; then
    echo "Each job will:"
    echo "  1. ⏭️ Skip training (using existing models)"
    echo "  2. Run comprehensive evaluation"
    echo "  3. Generate plots and summaries"
else
    echo "Each job will:"
    echo "  1. Train models (using 64 CPUs)"
    echo "  2. Run comprehensive evaluation"
    echo "  3. Generate plots and summaries"
fi
echo ""
echo "When all complete, run:"
echo "  jupyter notebook plot_within_generalization.ipynb"
echo "============================================"
