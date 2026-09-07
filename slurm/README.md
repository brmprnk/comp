# slurm/ — cluster entry points for the model experiments

One sbatch wrapper per experiment. These replace the retired `run_model.sh`,
whose behavior depended on which MODE block was last uncommented; every mode now
has its own file with the argument list preserved **verbatim** (verified against
the echoed `Mode Arguments:` lines in `logs/slurm_archive/*.out` and dry-runs of
the assembled `model_hpc.py` argv).

**Always submit from the repository root** (`sbatch slurm/<wrapper>.sh ...`):
the apptainer bind paths, the `logs/` output paths, and the sourcing of
`_model_hpc_exec.sh` are all relative to the submission directory.

## Mapping from the retired run_model.sh

| Old mode | New entry point | Notes |
|---|---|---|
| MODE 1 (nested_cv, DD) | `nested_cv_dd.sh [DATASET] [STRATEGY]` | `CANCER_TYPES`/`EXPERIMENT_NAME` via env; historical MS runs: `brca_crc_dd`, `pancan_dd` |
| MODE 2 (RNA, no gene selection) | `nested_cv_rna.sh [DATASET] [STRATEGY]` | historical: `brca_crc_rna`, `nested_rna_pancan` |
| MODE 5 (ATAC, predefined matrix) | `nested_cv_atac.sh [DATASET] [STRATEGY]` | historical: `brca_crc_atac`, `pancan_atac` |
| MODE 2b (fully nested CV) | `./slurm/submit_nested_full.sh estimate\|test\|full\|ksweep` | 10-task array + dependent aggregation, runs `nested_full.sh` |
| MODE 3 (bootstrap) | `bootstrap.sh [DATASET] [STRATEGY]` | container variant; `bootstrap_best_model.sh` / `bootstrap_custom_hyperparams.sh` / `bootstrap_quick_test.sh` are local bare-python variants (no `#SBATCH`) |
| MODE 4 (best-model search) | `best_model_search.sh [DATASET] [dd\|rna\|atac]` | atac uses the `_pancan` matrix (verbatim quirk) |
| MODE 4b (label shuffle, Fig S5) | `label_shuffle.sh [DATASET] [dd\|rna\|atac]` | defaults `CANCER_TYPES="breast colorectal"` |
| MODE 6 (baseline aggregate, Fig 4c) | `baseline_aggregate.sh [DATASET] [FEATURE] [SELECTION]` | swept by `submit_baseline_aggregate.sh` |
| MODE 7 (external val, single model) | `external_validation_single.sh [TEST] [dd\|rna\|atac]` | |
| MODE 7b (comprehensive external val, Fig 5b/c, S7) | `external_validation.sh [dd\|rna\|atac] [TEST]` | formerly `run_external_validation.sh`; swept by `submit_external_val_all.sh` |
| MODE 7c/7d (clinical external val) | `external_validation_clinical.sh [TEST] [single\|comprehensive]` | pancan best model, DD only |
| MODE 10 (external val, all 3 types) | `./slurm/submit_external_val_all.sh [TEST] [type]` | |
| MODE 8 (within-gen train+eval, Fig 5d–f) | `./slurm/submit_within_gen_all.sh "<type1>" "<type2>" [type] [skip_training]` | runs `within_gen.sh` per feature type |
| MODE 9 (within-gen compare) | `within_gen_compare.sh [DATASET] [DD_DIR] [RNA_DIR] [ATAC_DIR]` | old mode passed an empty `-e`; this one defaults it to `within_gen_compare` (unused by the compare mode's fixed output dir) |

`_model_hpc_exec.sh` is not a job: it holds the shared apptainer invocation
(bind list copied verbatim from `run_model.sh`) and the common log header, and
is sourced by the wrappers above. `nested_full.sh`, `within_gen.sh`, and
`external_validation.sh` keep their own historical apptainer blocks unchanged.

## Feature extraction / QC stage

The upstream wrappers also live here (formerly `run_*.sh` at the repo root):

| Entry point | What it runs |
|---|---|
| `filter_cris.sh` | filters the CRIS BAM/GC input lists (Healthy+BRCA+CRC) via `src/comp/filter_cris_samples.py` |
| `extract_features.sh [JOB_INDEX] [TOTAL_JOBS]` | `features.py -F BIN` feature/coverage extraction (formerly `run.sh`; edit `CMD_ARGS` inside — the manuscript invocations are the commented lines) |
| `submit_parallel_jobs.sh [N]` | submits N parallel `extract_features.sh` jobs to split a TFBS workload |
| `ulz_atac_extraction.sh` | builds the Ulz TFBS feature matrix (`src/comp/extract_ulz_atac_features.py`) |
| `extract_aggregate_features.sh` | appends per-sample aggregate features to the ichorCNA table (`src/comp/extract_aggregate_features.py`) |
| `jiang_coverage.sh` + `submit_jiang_coverage.sh` | per-sample genome-wide mean coverage for the Jiang cohort (array job + summary) |
| `plot_coverage.sh [PON_N]` | Fig 1a/b coverage profiles (`src/comp/plot_aggregated_coverage.py`) |

The ichorCNA arm's `run_hmmcopy.sh` / `ichorcha.sh` remain in
`src/comp/scripts/`.
