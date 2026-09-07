#!/bin/bash
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --job-name=coverage_profile
#SBATCH --qos=short
#SBATCH --time=03:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --output=logs/%j_coverage_profile.out
#SBATCH --error=logs/%j_coverage_profile.out

# Plot aggregated coverage profiles over gene sets (PAU and housekeeping)
# using first N healthy samples as panel-of-normals
# Usage: sbatch slurm/plot_coverage.sh [PON_N]

echo "=========================================="
echo "Coverage Profile Plotting Job — Panel of Normals"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="
echo ""

PON_N=${1:-50}
echo "PoN size: $PON_N"
echo ""

# Run the plotting script
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind beds/:/opt/app/beds/ \
	--bind plots/:/opt/app/plots/ \
	--bind extracted_features/:/opt/app/extracted_features/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
        ./comp.sif pixi run python -u src/comp/plot_aggregated_coverage.py \
        --metadata accessory_files/cris_metadata.csv \
        --pon_n ${PON_N} \
        --biomart_bed beds/biomart_10kb.bed \
        --pau_bed beds/pau.bed \
        --hk_bed beds/hk_genes.bed \
        --features_dir extracted_features/biomart_10kb \
        --output plots/aggregated_coverage_pon.png \
        --window_size 10000

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Plot saved to: plots/aggregated_coverage_pon.png"
else
    echo "❌ Job failed with exit code $EXIT_CODE"
fi
