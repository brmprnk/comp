#!/bin/bash
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general # Request partition. Default is 'general'
#SBATCH --job-name=tfbs
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=3:38:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=32   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=64GB          # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=logs/%j_ulz_atac.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/%j_ulz_atac.out # Set name of error log. %j is the Slurm jobId

# ULZ-ATAC Feature Extraction from TFBS Coverage Profiles
# Usage: sbatch run_ulz_atac_extraction.sh

echo "=========================================="
echo "ULZ-ATAC Feature Extraction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="
echo ""

# Run the extraction script
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
    --bind plots/:/opt/app/plots/ \
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
        ./comp.sif pixi run python -u src/comp/extract_ulz_atac_features.py \
        --features_dir ./extracted_features \
        --metadata_file accessory_files/jiang_metadata.csv \
        --output ./extracted_features/ulz_atac_tfbs_features_jiang.npy \
        --plot_dir ./plots/ulz_atac_features_jiang \
        --cores ${SLURM_CPUS_PER_TASK} \
        --n_plot_samples 5

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

