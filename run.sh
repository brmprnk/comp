#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general # Request partition. Default is 'general'
#SBATCH --job-name=wps-test
#SBATCH --qos=short        # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=3:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=1   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=8GB         # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=%j_all.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=%j_all.out # Set name of error log. %j is the Slurm jobId
# /usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working


# Build the command with optional job splitting parameters
# CMD_ARGS="-I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b hg38_extended_chrom1.bed"
CMD_ARGS="-I cris/EE87922.hg38.frag.tsv.bam --gc_file GC/Output_Bam/EE87922.hg38.frag.tsv__correction_factors.csv -c 1 -F WPS -b beds/hg38_extended_chrom1.bed"
# CMD_ARGS="-I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/biomart_10kb.bed --aggregate"  # Add this if you want to aggregate coverage across loci (.npy)
# CMD_ARGS="-I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/biomart_10kb.bed --coverage"  # Add this if you want to store coverage for each locus (.npy), but be cautious of large file sizes, especially with many loci!!!

# Get job index and total jobs from command line arguments (default to single job)
# only used when running multiple parallel jobs via submit_parallel_jobs.sh (to split TFBS workload)
JOB_INDEX=${1:-""}
TOTAL_JOBS=${2:-1}
if [ -n "$JOB_INDEX" ]; then
    CMD_ARGS="$CMD_ARGS --job_index $JOB_INDEX --total_jobs $TOTAL_JOBS"
fi

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/:/opt/app/cris/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/jiang/:/opt/app/jiang/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/lucas/:/opt/app/lucas/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/mathios/:/opt/app/mathios/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/:/opt/app/GC \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/TSSClassification/data/:/opt/app/data/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		./comp.sif pixi run python features.py $CMD_ARGS
