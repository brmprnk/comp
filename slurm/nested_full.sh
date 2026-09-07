#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_full
#SBATCH --output=logs/%A_%a_nested_full.out
#SBATCH --error=logs/%A_%a_nested_full.out

# ============================================================================
# FULLY NESTED CV - one outer fold per array task (data-driven features only)
# ============================================================================
# Feature type, ranking function, top_genes and the classifier hyperparameters are
# ALL chosen inside each outer fold. The ten outer folds are independent, so this
# runs as a 10-task array and a dependent aggregation step.
#
# Usage (normally via ./slurm/submit_nested_full.sh):
#   sbatch --array=1-10 slurm/nested_full.sh          # the ten outer folds
#   sbatch slurm/nested_full.sh aggregate             # combine the partials
#   sbatch slurm/nested_full.sh estimate              # cost estimate, no fitting
#   sbatch --array=1-10 slurm/nested_full.sh test     # reduced grid, correctness check
#
# $1 : "" (fold, uses SLURM_ARRAY_TASK_ID) | aggregate | estimate | test
# ============================================================================

STEP=${1:-fold}
DATASET=${DATASET:-cris}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-brca_crc_nested_full}
CANCER_TYPES=${CANCER_TYPES:-"breast colorectal"}
OUTPUT_DIR=${OUTPUT_DIR:-nested_cv_results_${DATASET}}

# Data-driven TSS features only. RNA and ATAC are not part of this experiment.
FEATURES_DIR="extracted_features/biomart_10kb"

# Only SVM is searched.
CLASSIFIER="Support Vector Machine"

# Ranking functions searched. Table S2 of the manuscript lists four; p_value and weighted
# are omitted here for a quick turnaround because both run mannwhitneyu per gene (14,224
# calls per ranking fit) and, with the pipeline cache disabled, that is recomputed for
# every candidate sharing it. mean_diff and fold_change are fully vectorised.
# Restore the full set with:
#   RANKING_FUNCTIONS="mean_diff fold_change p_value weighted"
RANKING_FUNCTIONS="mean_diff fold_change"

MODE_ARGS="--run_nested_cv_full --features_dir ${FEATURES_DIR}"
MODE_ARGS="${MODE_ARGS} --nested_full_selection_types ${RANKING_FUNCTIONS}"

case "${STEP}" in
	estimate)
		MODE_ARGS="${MODE_ARGS} --nested_full_estimate_only"
		;;
	aggregate)
		MODE_ARGS="${MODE_ARGS} --nested_full_aggregate"
		;;
	test)
		# Cheap correctness run: one feature, reduced k grid, still all 4 ranking functions.
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_test"
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-1}"
		MODE_ARGS="${MODE_ARGS} --nested_full_features griffin_diff"
		MODE_ARGS="${MODE_ARGS} --nested_full_top_genes 25 100 250"
		;;
	test_aggregate)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_test"
		MODE_ARGS="${MODE_ARGS} --nested_full_aggregate"
		MODE_ARGS="${MODE_ARGS} --nested_full_features griffin_diff"
		MODE_ARGS="${MODE_ARGS} --nested_full_top_genes 25 100 250"
		;;
	ksweep)
		# One complete nested selection per k, so every point of the k trend is
		# leakage-free. The k loop runs inside each task, so the feature matrices are
		# loaded 10 times (once per outer fold) rather than 120 times.
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-1}"
		;;
	ksweep_aggregate)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_ksweep --nested_full_aggregate"
		;;
	*)
		# One outer fold per array task.
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-1}"
		# NOTE: no --nested_full_cache_dir here. Pipeline(memory=...) makes joblib hash
		# TopGeneSelector on every fit, and because model_hpc.py runs as a script the
		# class lives in '__main__', which the parallel workers cannot resolve -> every
		# fit fails with PicklingError. The ranking is recomputed per candidate instead.
		;;
esac

echo "============================================"
echo "Fully nested CV | step=${STEP} | array task=${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Classifier: ${CLASSIFIER}"
echo "Args: ${MODE_ARGS}"
echo "============================================"

set -- \
	-c 32 -t ${CANCER_TYPES} \
	${MODE_ARGS} \
	--metadata_file accessory_files/${DATASET}_metadata.csv \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR} \
	-s mean_diff \
	--nested_full_classifiers "${CLASSIFIER}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
	--bind results/:/opt/app/results/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/:/opt/app/cris/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/:/opt/app/GC \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/TSSClassification/data/:/opt/app/data/ \
	--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		./comp.sif pixi run python -u src/comp/model_hpc.py "$@"
