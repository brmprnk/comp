#!/bin/bash
# ============================================================================
# Submit the fully nested CV as a 10-task array + a dependent aggregation job.
# Data-driven TSS features only; SVM only.
# ============================================================================
# Usage:
#   ./slurm/submit_nested_full.sh estimate   # cost estimate only, no fitting (seconds)
#   ./slurm/submit_nested_full.sh test       # reduced grid correctness run (1 feature, 3 k values)
#   ./slurm/submit_nested_full.sh full       # the real run
#
# The aggregation job is submitted with --dependency=afterok on the array, so it
# starts by itself once all ten folds have finished.
# ============================================================================

STEP=${1:-test}

case "${STEP}" in
	estimate)
		echo "Submitting cost-estimate job (no fitting)..."
		JOB=$(sbatch --parsable slurm/nested_full.sh estimate)
		echo "  Job ID: ${JOB}"
		echo "  Output: ${JOB}_*_nested_full.out"
		;;

	test)
		echo "Submitting REDUCED correctness run: griffin_diff only, k in {25,100,250}, SVM."
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh test)
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold)"
		AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/nested_full.sh test_aggregate)
		echo "  Aggregation job ID: ${AGG_JOB} (waits for all 10 folds)"
		echo ""
		echo "Results will land in nested_cv_results_cris/nested_full_brca_crc_nested_full_test/"
		;;

	full)
		echo "Submitting FULL run: 5 data-driven features x 4 ranking functions x 12 k values x SVM."
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh)
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold)"
		AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/nested_full.sh aggregate)
		echo "  Aggregation job ID: ${AGG_JOB} (waits for all 10 folds)"
		echo ""
		echo "Results will land in nested_cv_results_cris/nested_full_brca_crc_nested_full/"
		;;

	ksweep)
		echo "Submitting k-SWEEP: one complete nested selection per k value."
		echo "  Feature, ranking function and SVM hyperparameters are still chosen inside"
		echo "  each fold; only k is held fixed, so every point of the trend is leakage-free."
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh ksweep)
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold, each looping all k)"
		AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/nested_full.sh ksweep_aggregate)
		echo "  Aggregation job ID: ${AGG_JOB} (waits for all 10 folds)"
		echo ""
		echo "Results will land in nested_cv_results_cris/nested_full_brca_crc_nested_full_ksweep/"
		echo "  per k       : ksweep_k<k>/nested_full_per_fold.csv"
		echo "  the trend   : nested_full_ksweep.csv"
		;;

	*)
		echo "Unknown step '${STEP}'. Use: estimate | test | full | ksweep"
		exit 1
		;;
esac

echo ""
echo "Monitor with:  squeue -u \$USER"
