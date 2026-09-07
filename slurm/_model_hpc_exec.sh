# Shared execution helper for the slurm/ model-experiment wrappers.
# Not an sbatch script — source it from a wrapper, then call run_model_hpc "$@".
#
# The apptainer invocation (bind list included) is copied verbatim from the
# retired run_model.sh, so every wrapper runs model_hpc.py in the exact same
# container environment as the historical runs.
#
# REQUIREMENT: submit jobs from the repository root (all binds are relative),
# i.e. `sbatch slurm/<wrapper>.sh ...`. This is the same requirement the old
# run_model.sh already had.

run_model_hpc() {
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
		--bind results/:/opt/app/results/ \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/:/opt/app/cris/ \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/jiang/:/opt/app/jiang/ \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/lucas/:/opt/app/lucas/ \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/mathios/:/opt/app/mathios/ \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/:/opt/app/GC \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/TSSClassification/data/:/opt/app/data/ \
		--bind /tudelft.net/staff-umbrella/KWFcfDNA/emc/GCfix_Software/hg38/:/opt/app/hg38 \
		--bind extracted_features/:/opt/app/extracted_features/ \
			./comp.sif pixi run python -u src/comp/model_hpc.py "$@"
}

# Standard log header shared by the wrappers (mirrors the old run_model.sh echo block,
# so new logs stay grep-compatible with logs/slurm_archive/).
print_job_header() {
	echo "============================================"
	echo "Running job: ${JOB_NAME}"
	echo "Mode: ${MODE}"
	echo "Dataset: ${DATASET:-N/A}"
	echo "Feature Type: ${FEATURE_TYPE:-N/A}"
	echo "Strategy: ${STRATEGY:-N/A}"
	echo "Experiment Name: ${EXPERIMENT_NAME}"
	echo "Mode Arguments: ${MODE_ARGS}"
	if [ ! -z "${CANCER_TYPES}" ]; then
		echo "Cancer Types: ${CANCER_TYPES}"
	fi
	echo "============================================"
	echo ""
}
