#!/bin/bash
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --job-name=hmmcopy
#SBATCH --qos=medium
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --mail-type=END
#SBATCH --output=logs/%j_hmmcopy.out
#SBATCH --error=logs/%j_hmmcopy.out

# Specify the directory containing the .bam files
bam_directory="/tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris"
wig_directory="$bam_directory/wigs_5mb"

# Number of parallel jobs
NUM_JOBS=64

echo "============================================"
echo "Running HMMcopy with $NUM_JOBS parallel jobs"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "============================================"

# Check if the directory exists
if [ ! -d "$bam_directory" ]; then
  echo "Error: Directory not found!"
  exit 1
fi

# Create wig directory if it doesn't exist
mkdir -p "$wig_directory"

# Collect all .bam files into an array
bam_files=("$bam_directory"/*.bam)

# Check if any .bam files exist
if [ ! -f "${bam_files[0]}" ]; then
  echo "No .bam files found in $bam_directory"
  exit 1
fi

total_files=${#bam_files[@]}

echo "Found $total_files .bam files"
echo "Processing with $NUM_JOBS parallel jobs"
echo "============================================"

# Function to process a single BAM file
process_bam() {
  local bam_file="$1"
  local wig_file="$wig_directory/$(basename "$bam_file" .bam).wig"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing $(basename "$bam_file")..."
  
  /tudelft.net/staff-umbrella/KWFcfDNA/emc/tools/hmmcopy_utils/bin/readCounter \
    --window 1000000 \
    --quality 10 \
    --chromosome chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX \
    "$bam_file" > "$wig_file" 2>&1
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done with $(basename "$bam_file")"
}

# Process files in parallel using background jobs
job_count=0
pids=()

for bam_file in "${bam_files[@]}"; do
  # Run in background
  process_bam "$bam_file" &
  pids+=($!)
  
  # Increment job counter
  ((job_count++))
  
  # Wait if we've reached the max number of parallel jobs
  if [ $job_count -ge $NUM_JOBS ]; then
    # Wait for the oldest job to finish (compatible with older bash)
    wait ${pids[0]}
    pids=("${pids[@]:1}")  # Remove first element
    ((job_count--))
  fi
done

# Wait for all remaining background jobs to complete
wait

echo "============================================"
echo "All $total_files files processed!"
echo "============================================"
