# src/figures — manuscript figure notebooks (NARGAB-2026-181)

One notebook per manuscript figure. Every notebook:

- reads **only** the committed source-data CSVs in `data/` — no result pickles,
  no cluster paths, no network access;
- regenerates the figure end to end in a fresh kernel (run top to bottom from
  this directory: `jupyter nbconvert --to notebook --execute --inplace fig3.ipynb`,
  or open and Run All);
- writes the rendered figure to `output/` (git-ignored; the committed notebooks
  carry the rendered outputs inline);
- ends with a check cell comparing computed headline numbers against the
  submitted manuscript values.

Requirements: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`matplotlib-venn` (figS2 only).

| Notebook | Figure | Source data (in `data/`) | Exporter |
|---|---|---|---|
| `fig1.ipynb` | Fig 1a–c (TSS coverage HK vs PAU; per-locus coverage; feature–expression correlations) | `fig1a_coverage_profiles.csv`, `fig1b_locus_coverage.csv`, `fig1c_expression_correlations.csv` | `export/export_fig1.py` |
| — | Fig 2 | *(method overview drawn in BioRender — no code, no notebook)* | — |
| `fig3.ipynb` | Fig 3a–c (nested-CV ROC; metrics at 95% specificity; posterior differences by stage) | `fig3_predictions.csv`, `fig3_config.csv` | `export/export_fig3.py` |
| `fig4.ipynb` | Fig 4a–c (AUROC vs k by feature / classifier; locus-level vs aggregates) | `fig4_auc_sweep.csv`, `fig4c_aggregate_scores.csv` | `export/export_fig4.py` |
| `fig5.ipynb` | Fig 5a–f (PANCAN; external validation on Jiang; within-dataset generalization) | `fig5a_predictions.csv` (+config), `fig5bc_external_predictions.csv` (+config), `fig5def_roc_curves.csv`, `fig5def_metrics.csv` | `export/export_fig5.py` |
| — | Fig S1 | *(correlation vs WGS coverage — original generating code not present in the repository; per project decision the figure is not reconstructed)* | — |
| `figS2.ipynb` | Fig S2a–c (blacklist coverage; venn; QC filter scatter) | `figS2a_blacklist_per_chrom.csv`, `figS2b_venn_counts.csv`, `figS2c_tss_qc.csv` | `export/export_figS2.py` |
| `figS3.ipynb` | Fig S3 (metrics with ichorCNA at TF ≥ 0.03) | `fig3_predictions.csv`, `fig3_config.csv` (shared with Fig 3) | `export/export_fig3.py` |
| `figS4.ipynb` | Fig S4 (AUROC vs k by ranking function) | `fig4_auc_sweep.csv` (shared with Fig 4) | `export/export_fig4.py` |
| `figS5.ipynb` | Fig S5 (label-shuffle stability) | `figS5_label_shuffle.csv`, `figS5_config.csv` | `export/export_figS5.py` |
| `figS6.ipynb` | Fig S6 (PANCAN per-cancer-type ROC grid) | `fig5a_predictions.csv` (shared with Fig 5a) | `export/export_fig5.py` |
| `figS7.ipynb` | Fig S7 (external AUROC vs number of TSSs) | `figS7_auc_vs_k.csv` | `export/export_fig5.py` |

## The export layer (`export/`)

The exporters are the only code that touches raw experiment outputs
(`nested_cv_results_cris/`, `results/`, `accessory_files/`,
`extracted_features/`). They run on the machine holding those outputs, from
the repository root:

```
python src/figures/export/export_fig3.py
```

Each exporter **validates what it writes** against the stored pipeline
results (recomputed AUCs and confusion matrices must equal the stored values;
filter-category counts must equal the submitted panel's counts) and records
the selected model configuration per method in a `*_config.csv`. The
result-directory manifest lives in `export/_common.py`.

After the planned post-fix cluster rerun, point the manifest at the new result
directories, re-run the exporters, and re-execute the notebooks — the
notebooks themselves need no changes (their final check cells compare against
the *submitted* values and will flag any number that moved).

## Extending for the revision

New reviewer-driven experiments follow the same three-layer pattern — nothing
goes back into monolithic notebooks:

1. **Run** the experiment via a `slurm/` wrapper (new mode → new wrapper).
2. **Export**: extend the relevant exporter (or add `export_figR<n>.py` for
   response-letter figures) so the new numbers land as validated CSVs in
   `data/`. All pickle-reading, model selection and cohort bookkeeping belongs
   here, with assertions against the stored pipeline values.
3. **Plot**: extend the figure notebook (or add `figR<n>.ipynb`). The check
   cells then double as the record of what changed relative to the submission.

The pre-revision originals of these figures live in `notebooks/archived/`
(git-ignored, local only) and serve as a function library to lift from — most
of their plotting code runs unmodified on the `data/` CSVs, because the CSVs
carry per-sample predictions rather than pre-aggregated curves.

### Where un-ported legacy logic lives (the mining map)

All in `notebooks/archived/` unless marked otherwise:

| Legacy logic (not yet in this layer) | Original notebook |
|---|---|
| Full 4×3 comparison grids (ROC + PR + confusion matrices, per scenario: full / BRCA-only / CRC-only) | `hypo_nestedcv_plots.ipynb` (`compare_all_approaches_comprehensive`) |
| Honest per-fold-selected variant of Fig 3 (`select_nested_cv_folds` + `plot_full_model_compact_nested`) | `hypo_nestedcv_plots.ipynb` |
| Per-classifier comparison plots (`plot_profile_classifiers_comparison`) | `hypo_nestedcv_plots.ipynb` |
| Posteriors-by-stage 5-panel supplement, detection-rate-by-stage lines, per-fold-threshold detection collectors | `stage_plot.ipynb` |
| Per-cancer-type PR curves and metric panels (S6 shows ROC only) | `hypo_nestedcv_plots_pancan.ipynb` |
| Overfitting diagnostics (`section33_overfit_diag`), optimal-gene-threshold analysis | `nested_cv_plots copy.ipynb` |
| Fully-nested-CV / k-sweep analysis (no counterpart here yet) | `nested_full_plots.ipynb` — still active in `src/comp/` |
| ichorCNA raw-params QC audit | `ichorcna_precision_audit.ipynb` — still active in `src/comp/` |
| GSEA on best-model coefficients (null result, Discussion-adjacent) | `gsea_best_model.ipynb` |

## Provenance notes and known deviations

- **Fig 3 / S3**: the data-driven arm is pinned to the post-fix `mean_diff`
  run: at k = 14,224 (all TSSs) the ranking function selects nothing, so all
  five ranking runs are the same model (asserted by the exporter); only the
  post-fix run carries leak-free train-derived operating thresholds, and it
  reproduces the submitted Fig 3b bars exactly.
- **ichorCNA duplicates**: the ichorCNA tables contain 16 byte-identical
  duplicate healthy rows; the original analysis used the files as-is, and the
  `n_rows_in_source` column preserves that weighting (it is what reproduces
  the submitted AUC of 0.600 ± 0.086; deduplicated it would read 0.602).
- **Fig 4**: the submitted figure was assembled from two notebooks plus a
  manual paste-in of panel c; `fig4.ipynb` draws all three panels directly.
  Panel a/b AUCs for the four non-`mean_diff` ranking functions come from the
  pre-fix run (their only source; AUC-identical except `fslr`,
  max |Δ mean AUROC| = 0.0065).
- **Fig 5a**: a pure argmax over the PANCAN grid selects Logistic Regression
  (0.9468); the submitted legend (0.946 ± 0.032) corresponds to the SVM at
  k = 14,224, which the exporter pins (consistent with the manuscript's model
  description).
- **Fig S5**: sourced from `label_shuffle_best_model_search_dd/` — its
  observed CV AUC (0.9595 → "0.960") and shuffle distribution are the ones in
  the submitted panel; the `*_currentlyinmsjune2026` snapshot carries an older
  run (observed 0.9468) despite its name.
- **Fig S2c**: the notebook that drew the submitted panel is not in the
  repository; `figS2.ipynb` reconstructs it from the filter artifacts. All
  category counts reproduce the submitted numbers exactly (asserted); the
  "(<5kb)" in the legend is literal — overlapping TSSs are windows whose TSS
  is within 5,000 bp of another. Panel a's ENCODE total differs from the
  submitted annotation by ~0.02% (blacklist-file revision level).
- **Fig S1**: generating code absent repo-wide; intentionally not
  reconstructed (project decision).
