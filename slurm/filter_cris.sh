#!/bin/bash

# Script to filter CRIS samples for Healthy, Breast cancer, and Colorectal cancer only
# Creates cris_bam_filtered.txt and cris_gc_filtered.txt

echo "Filtering CRIS samples..."
python src/comp/filter_cris_samples.py

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS! Filtered sample lists created."
    echo ""
    echo "Next steps:"
    echo "1. Check the filtered files in input/ directory"
    echo "2. Update slurm/extract_features.sh to use the filtered files"
    echo "3. Run: ./slurm/submit_parallel_jobs.sh 5"
else
    echo "ERROR: Filtering failed!"
    exit 1
fi
