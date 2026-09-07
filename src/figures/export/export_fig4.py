"""Export source data for manuscript Figure 4 (and Supplementary Figure S4).

Figure 4: analysis of the data-driven model on healthy vs BRCA+CRC:
  a) best/mean AUROC vs number of selected TSSs (k), stratified by fragmentomic
     feature; b) the same stratified by classifier; c) ROC of the locus-level
     data-driven model vs the five aggregate (pooled-coverage) baselines.
Figure S4: best/mean AUROC vs k stratified by ranking (weighting) function.

Writes:
  src/figures/data/fig4_auc_sweep.csv
      one row per (feature, classifier, weighting_function, top_genes, fold)
      with the outer-fold test AUROC — the full nested-CV sweep behind
      Fig 4a/b and Fig S4.
  src/figures/data/fig4c_aggregate_scores.csv
      one row per sample: outer fold, label, the five aggregate feature values
      (single pooled value per sample; from the ichorCNA/aggregate table) and
      the data-driven best-model out-of-fold score.

Sweep manifest: mean_diff from the post-fix directory, the other four ranking
functions from the pre-fix directory (their only source; AUCs were audited to
be identical between the runs except for `fslr`, max |delta mean AUC| 0.0065 —
see src/comp/hypo_nestedcv_plots.ipynb). Only AUCs are exported from the
pre-fix rows, never operating-point thresholds.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    DATA_DIR,
    DD_RESULT_DIRS,
    ICHOR_AGG_CSV,
    load_results,
    rp,
)

AGG_FEATURES = ["aggregate_griffin_diff", "aggregate_rcov", "aggregate_mwh",
                "aggregate_fslr", "aggregate_fslr_coverage"]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    keep = ["feature", "classifier", "top_genes", "fold", "auc_full_model",
            "test_sample_ids", "y_pred_proba", "y_test", "full_model_thresholds"]
    frames = []
    for wf, dd_dir in DD_RESULT_DIRS.items():
        frames.append(load_results(f"{dd_dir}/results*{wf}.pkl", keep,
                                   extra={"weighting_function": wf,
                                          "source_dir": os.path.basename(dd_dir)}))
    dd = pd.concat(frames, ignore_index=True)
    dd["top_genes"] = dd["top_genes"].replace(15000, 14224)

    sweep = dd[["feature", "classifier", "weighting_function", "top_genes",
                "fold", "auc_full_model", "source_dir"]].rename(
        columns={"auc_full_model": "auc"})
    out = os.path.join(DATA_DIR, "fig4_auc_sweep.csv")
    sweep.to_csv(out, index=False)
    n_cfg = sweep.groupby(["feature", "classifier", "weighting_function", "top_genes"]).ngroups
    print(f"wrote {out}: {len(sweep)} fold-level rows, {n_cfg} configs, "
          f"best mean AUROC = "
          f"{sweep.groupby(['feature','classifier','weighting_function','top_genes'])['auc'].mean().max():.4f}")

    # ---------------- Fig 4c: DD best model OOF scores + aggregate values ----
    # DD best model = SVM + griffin_diff + all 14,224 TSSs; the mean_diff run
    # is used (post-fix; at k=all the ranking function is inert — see
    # export_fig3.py). Same selection as Figure 3.
    dd_sel = dd[(dd["feature"] == "griffin_diff")
                & (dd["classifier"] == "Support Vector Machine")
                & (dd["weighting_function"] == "mean_diff")
                & (dd["top_genes"] == 14224)]
    assert len(dd_sel) == 10, f"expected 10 folds, got {len(dd_sel)}"

    ichor = pd.read_csv(rp(ICHOR_AGG_CSV), index_col=0)
    multiplicity = ichor.groupby("ID").size()
    dup_ids = set(multiplicity[multiplicity > 1].index)
    for sid in dup_ids:
        grp = ichor[ichor["ID"] == sid]
        assert (grp.nunique() <= 1).all(), f"duplicate rows for {sid} are not identical"
    ichor = ichor.drop_duplicates(subset="ID").set_index("ID")

    records = []
    for _, row in dd_sel.iterrows():
        for sid, y, s in zip(row["test_sample_ids"], row["y_test"], row["y_pred_proba"]):
            rec = {"sample_id": sid, "fold": int(row["fold"]), "y_true": int(y),
                   "dd_score": float(s),
                   "n_rows_in_source": 2 if sid in dup_ids else 1}
            for f in AGG_FEATURES:
                rec[f] = ichor.loc[sid, f] if sid in ichor.index else np.nan
            records.append(rec)
    fig4c = pd.DataFrame(records)
    missing = fig4c[AGG_FEATURES].isna().all(axis=1).sum()
    print(f"fig4c: {len(fig4c)} task samples, {missing} without aggregate values")

    out_c = os.path.join(DATA_DIR, "fig4c_aggregate_scores.csv")
    fig4c.to_csv(out_c, index=False)
    print(f"wrote {out_c}")

    # quick reference check against the submitted panel legend
    from sklearn.metrics import roc_auc_score
    print("\nfig4c per-feature mean AUC over folds (manuscript legend for comparison):")
    ref = {"aggregate_griffin_diff": "0.373 ± 0.126", "aggregate_rcov": "0.577 ± 0.140",
           "aggregate_mwh": "0.496 ± 0.089", "aggregate_fslr": "0.670 ± 0.121",
           "aggregate_fslr_coverage": "0.673 ± 0.124"}
    for f in AGG_FEATURES:
        aucs = []
        for _, g in fig4c.groupby("fold"):
            g = g.loc[g.index.repeat(g["n_rows_in_source"])].dropna(subset=[f])
            aucs.append(roc_auc_score(g["y_true"], g[f]))
        print(f"  {f:<28} {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   (MS: {ref[f]})")


if __name__ == "__main__":
    main()
