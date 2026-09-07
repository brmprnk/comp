#!/bin/bash
# Submit the Jiang coverage array job, then submit the summary job with a dependency.
# Usage: bash slurm/submit_jiang_coverage.sh

set -e

echo "Submitting Jiang coverage array job (249 tasks)..."
ARRAY_JOB_ID=$(sbatch --parsable slurm/jiang_coverage.sh)
echo "  Array job ID: $ARRAY_JOB_ID"

echo "Submitting summary job (depends on array job completing)..."
SUMMARY_JOB_ID=$(sbatch --parsable \
    --account=ewi-insy-prb \
    --partition=insy,general \
    --job-name=jiang_coverage_summary \
    --qos=short \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem=2GB \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --output=logs/jiang_coverage_summary_%j.out \
    --error=logs/jiang_coverage_summary_%j.out \
    --wrap="apptainer exec --writable-tmpfs --pwd /opt/app --containall \
        --bind src/:/opt/app/src/ \
        --bind results/:/opt/app/results/ \
        ./comp.sif pixi run python -u src/comp/summarize_coverage.py \
            /opt/app/results/jiang_coverage \
            /opt/app/results/jiang_coverage/jiang_mean_coverage.csv")

echo "  Summary job ID: $SUMMARY_JOB_ID"
echo ""
echo "Done. Final CSV will be at: results/jiang_coverage/jiang_mean_coverage.csv"
echo "Summary stats will appear in: logs/jiang_coverage_summary_${SUMMARY_JOB_ID}.out"
