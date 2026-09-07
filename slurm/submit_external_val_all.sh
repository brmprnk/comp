#!/bin/bash
# ============================================================================
# Submit external validation jobs for all 3 feature types (or one specific type)
# ============================================================================
# Usage:
#   ./slurm/submit_external_val_all.sh                        # All 3 types on jiang
#   ./slurm/submit_external_val_all.sh jiang                  # All 3 types on jiang
#   ./slurm/submit_external_val_all.sh jiang atac             # Only ATAC on jiang
#   ./slurm/submit_external_val_all.sh lucas dd               # Only DD on lucas
# ============================================================================

TEST_DATASET=${1:-jiang}
FEATURE_ONLY=${2:-""}

echo "============================================"
echo "Submitting External Validation Jobs"
echo "============================================"
echo "Test Dataset: ${TEST_DATASET}"
if [ ! -z "$FEATURE_ONLY" ]; then
    echo "Feature Only: ${FEATURE_ONLY}"
else
    echo "Feature Types: dd, rna, atac"
fi
echo ""

JOB_DD=""
JOB_RNA=""
JOB_ATAC=""

if [ -z "$FEATURE_ONLY" ] || [ "$FEATURE_ONLY" = "dd" ]; then
    echo "Submitting DD external validation..."
    JOB_DD=$(sbatch --parsable slurm/external_validation.sh dd ${TEST_DATASET})
    echo "  Job ID: ${JOB_DD}"
fi

if [ -z "$FEATURE_ONLY" ] || [ "$FEATURE_ONLY" = "rna" ]; then
    echo "Submitting RNA external validation..."
    JOB_RNA=$(sbatch --parsable slurm/external_validation.sh rna ${TEST_DATASET})
    echo "  Job ID: ${JOB_RNA}"
fi

if [ -z "$FEATURE_ONLY" ] || [ "$FEATURE_ONLY" = "atac" ]; then
    echo "Submitting ATAC external validation..."
    JOB_ATAC=$(sbatch --parsable slurm/external_validation.sh atac ${TEST_DATASET})
    echo "  Job ID: ${JOB_ATAC}"
fi

echo ""
echo "============================================"
if [ -z "$FEATURE_ONLY" ]; then
    echo "All 3 jobs submitted!"
else
    echo "${FEATURE_ONLY} job submitted!"
fi
echo "============================================"
echo "Job IDs: ${JOB_DD:--} (DD), ${JOB_RNA:--} (RNA), ${JOB_ATAC:--} (ATAC)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "============================================"
