#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general # Request partition. Default is 'general'
#SBATCH --job-name=cpg
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=35:59:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=8   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=60GB          # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=%j_all.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=%j_all.out # Set name of error log. %j is the Slurm jobId
# /usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/Output_Bam/:/opt/app/GC \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/TSSClassification/data/:/opt/app/data/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		./comp.sif pixi run python features.py -I input/run.txt --gc_file input/run_gc.txt -c 8
