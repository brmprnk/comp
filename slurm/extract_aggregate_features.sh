#!/bin/bash
#SBATCH --job-name=extract_agg_features
#SBATCH --output=logs/extract_aggregate_features_%j.out
#SBATCH --error=logs/extract_aggregate_features_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --qos=short

# Extract aggregate features from coverage profiles and add to metadata CSV
# This script processes all CRIS samples with coverage_aggregate.npy files

echo "=================================="
echo "Extract Aggregate Features"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Run the extraction script
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
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/:/opt/app/GC \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/TSSClassification/data/:/opt/app/data/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		./comp.sif pixi run python -u src/comp/extract_aggregate_features.py

echo ""
echo "End time: $(date)"
echo "Job completed successfully!"
