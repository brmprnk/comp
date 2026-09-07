#!/bin/bash

# Script to submit multiple parallel jobs for BED file processing
# Usage: ./submit_parallel_jobs.sh <total_jobs>
# Example: ./submit_parallel_jobs.sh 5

TOTAL_JOBS=${1:-5}  # Default to 5 jobs if not specified

echo "Submitting $TOTAL_JOBS parallel jobs..."

for ((i=0; i<$TOTAL_JOBS; i++)); do
    echo "Submitting job $((i+1))/$TOTAL_JOBS (job_index=$i)"
    sbatch --job-name="bin_job_$i" slurm/extract_features.sh $i $TOTAL_JOBS
    sleep 1
done

echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
