"""Export source data for manuscript Figure 3 (and Supplementary Figure S3).

Figure 3: nested-CV comparison of the data-driven model against the ATAC and
RNA baselines and ichorCNA on healthy vs BRCA+CRC (Cristiano cohort):
  a) mean ROC over the 10 outer folds,
  b) classification metrics at the fold-specific 95%-specificity threshold,
  c) paired posterior differences relative to the data-driven model by stage.
Figure S3 re-uses the same source data with ichorCNA at the fixed TF >= 0.03
operating point.

Writes:
  src/figures/data/fig3_predictions.csv
      one row per (method, sample): out-of-fold prediction score, fold id,
      label, fold-specific train-derived 95%-specificity threshold
      (data_driven / rna / atac), plus every ichorCNA sample (score =
      ichorCNA tumor fraction; samples outside the BRCA/CRC task carry
      fold = NA and enter only the per-fold ichorCNA threshold derivation,
      exactly as in the original notebook).
  src/figures/data/fig3_config.csv
      the selected configuration per method (provenance).

Source experiments (see src/figures/export/_common.py for the manifest):
  data_driven : nested_cv_results_cris/brca_crc_dd (+ pre-fix dir for the
                other ranking functions, used only for model selection)
  rna         : nested_cv_results_cris/brca_crc_rna
  atac        : nested_cv_results_cris/brca_crc_atac
  ichorcna    : accessory_files/cris_stage_vaf_coverage_ichor_aggregate.csv
Selection replicates src/comp/hypo_nestedcv_plots.ipynb: best mean AUC over
the grid, with the data-driven gene count then pinned to all 14,224 TSSs.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    ATAC_RESULT_DIR,
    DATA_DIR,
    DD_RESULT_DIRS,
    ICHOR_AGG_CSV,
    ICHOR_CSV,
    RNA_RESULT_DIR,
    assert_predictions_match_stored,
    best_mean_auc_config,
    load_results,
    rp,
)

KEEP = [
    "feature", "classifier", "top_genes", "fold", "auc_full_model",
    "test_sample_ids", "y_pred_proba", "y_test",
    "full_model_thresholds", "confusion_matrix_full_model",
]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------ load
    dd_frames = []
    for wf, dd_dir in DD_RESULT_DIRS.items():
        dd_frames.append(load_results(f"{dd_dir}/results*{wf}.pkl", KEEP,
                                      extra={"weighting_function": wf,
                                             "source_dir": os.path.basename(dd_dir)}))
    dd = pd.concat(dd_frames, ignore_index=True)
    dd["top_genes"] = dd["top_genes"].replace(15000, 14224)
    rna = load_results(f"{RNA_RESULT_DIR}/results*.pkl", KEEP)
    atac = load_results(f"{ATAC_RESULT_DIR}/results*.pkl", KEEP)
    print(f"loaded: dd {len(dd)} rows, rna {len(rna)} rows, atac {len(atac)} rows")

    # ------------------------------------------------------- model selection
    dd_pool = dd[dd["feature"] == "griffin_diff"]
    best_dd = best_mean_auc_config(dd_pool, ["classifier", "weighting_function", "top_genes"])
    # The manuscript figure pins the gene count to all retained TSSs (14,224).
    # At that k the ranking function selects nothing (every gene is kept), so
    # all five weighting functions are literally the same model — asserted
    # below. We pin weighting_function = "mean_diff" because only that run
    # lives in the post-fix directory (train-derived operating thresholds;
    # the pre-fix rows carry test-derived thresholds, i.e. the leak the fix
    # removed) and because its stored thresholds reproduce the submitted
    # Fig 3b bars exactly (precision .87 / recall .79 / F1 .82 / acc .90 /
    # spec .95).
    tie_check = dd[(dd["feature"] == "griffin_diff")
                   & (dd["classifier"] == best_dd["classifier"])
                   & (dd["top_genes"] == 14224)]
    tie_aucs = tie_check.groupby("weighting_function")["auc_full_model"].mean()
    assert np.allclose(tie_aucs, tie_aucs.iloc[0]), (
        "expected identical AUCs across weighting functions at top_genes=14224 "
        f"(all genes kept), got: {tie_aucs.to_dict()}")
    best_dd["weighting_function"] = "mean_diff"
    dd_sel = dd[(dd["feature"] == "griffin_diff")
                & (dd["classifier"] == best_dd["classifier"])
                & (dd["weighting_function"] == "mean_diff")
                & (dd["top_genes"] == 14224)]
    best_rna = best_mean_auc_config(rna[rna["feature"] == "rcov"], ["classifier"])
    rna_sel = rna[(rna["feature"] == "rcov") & (rna["classifier"] == best_rna["classifier"])]
    best_atac = best_mean_auc_config(atac[atac["feature"] == "ulz_atac_tfbs_features"], ["classifier"])
    atac_sel = atac[(atac["feature"] == "ulz_atac_tfbs_features")
                    & (atac["classifier"] == best_atac["classifier"])]

    for name, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        assert len(sel) == 10, f"{name}: expected 10 folds, got {len(sel)}"

    print("selected configs:")
    print(f"  data_driven: {best_dd} -> pinned top_genes=14224, "
          f"mean AUC {dd_sel['auc_full_model'].mean():.4f} ± {dd_sel['auc_full_model'].std(ddof=0):.4f}")
    print(f"  rna:         {best_rna}")
    print(f"  atac:        {best_atac}")

    # --------------------------------------------------- fold-partition check
    def fold_map(sel: pd.DataFrame) -> dict[int, frozenset]:
        return {int(r["fold"]): frozenset(r["test_sample_ids"]) for _, r in sel.iterrows()}

    dd_folds, rna_folds, atac_folds = fold_map(dd_sel), fold_map(rna_sel), fold_map(atac_sel)
    assert dd_folds == rna_folds == atac_folds, (
        "outer-fold partitions differ between the three arms; the ichorCNA fold "
        "assignment below would be ambiguous")
    print("fold partitions identical across data_driven / rna / atac")

    # ------------------------------------------------ score/threshold checks
    for name, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        assert_predictions_match_stored(sel, name)
        has_train_fields = all("train_fpr_at_95spec" in r["full_model_thresholds"]
                               for _, r in sel.iterrows())
        print(f"  {name}: stored AUC + 95%-spec confusion matrices reproduced; "
              f"post-fix (train-derived) threshold fields present: {has_train_fields}")

    # ----------------------------------------------------------- assemble
    records = []
    for method, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        for _, row in sel.iterrows():
            thr = float(row["full_model_thresholds"]["95_percent_specificity"])
            for sid, y, s in zip(row["test_sample_ids"], row["y_test"], row["y_pred_proba"]):
                records.append({"sample_id": sid, "method": method,
                                "fold": int(row["fold"]), "y_true": int(y),
                                "score": float(s), "threshold_95spec": thr})
    pred = pd.DataFrame(records)

    # ichorCNA: every sample in the aggregate CSV, with the fold in which it
    # appears as a test sample (NA when outside the BRCA/CRC task). The
    # original notebook derives fold thresholds from ALL samples not in the
    # fold's test set — including cancer types outside this task — so those
    # rows must be exported too.
    #
    # The aggregate CSV contains 16 healthy samples twice (byte-identical
    # rows). The original analysis used the file as-is, and the duplicate
    # weighting is what reproduces the submitted ichorCNA AUC of
    # 0.600 ± 0.086 (deduplicated it becomes 0.602). We export one row per
    # sample with n_rows_in_source recording the multiplicity; the figure
    # notebook repeats rows by that column to match the original computation.
    ichor = pd.read_csv(rp(ICHOR_AGG_CSV), index_col=0)
    multiplicity = ichor.groupby("ID").size()
    dup_ids = multiplicity[multiplicity > 1]
    for sid in dup_ids.index:
        grp = ichor[ichor["ID"] == sid]
        assert (grp.nunique() <= 1).all(), f"duplicate rows for {sid} are not identical"
    ichor = ichor.drop_duplicates(subset="ID")
    print(f"ichorCNA CSV: {len(ichor)} unique samples "
          f"({len(dup_ids)} duplicated, all byte-identical)")

    sample_to_fold = {sid: f for f, ids in dd_folds.items() for sid in ids}
    ichor_rows = pd.DataFrame({
        "sample_id": ichor["ID"],
        "method": "ichorcna",
        "fold": ichor["ID"].map(sample_to_fold),
        "y_true": (ichor["disease"] != "Healthy").astype(int),
        "score": ichor["ichorcna_tf"],
        "threshold_95spec": np.nan,
    })
    pred = pd.concat([pred, ichor_rows], ignore_index=True)
    pred["n_rows_in_source"] = np.where(
        (pred["method"] == "ichorcna") & pred["sample_id"].isin(dup_ids.index), 2, 1)

    # stage + disease annotations (Cristiano et al. publication metadata)
    ann = ichor.set_index("ID")
    pred["stage"] = pred["sample_id"].map(ann["Stage"])
    pred["disease"] = pred["sample_id"].map(ann["disease"])

    n_task = pred[pred["method"] == "data_driven"]["sample_id"].nunique()
    print(f"task samples: {n_task}; ichorCNA rows: {len(ichor_rows)} "
          f"({ichor_rows['fold'].notna().sum()} inside the task folds)")

    out = os.path.join(DATA_DIR, "fig3_predictions.csv")
    pred.to_csv(out, index=False)
    print(f"wrote {out} ({len(pred)} rows)")

    cfg = pd.DataFrame([
        {"method": "data_driven", "feature": "griffin_diff",
         "classifier": best_dd["classifier"], "weighting_function": best_dd["weighting_function"],
         "top_genes": 14224, "source_dir": DD_RESULT_DIRS["mean_diff"],
         "selection_mean_auc": best_dd["mean_auc"]},
        {"method": "rna", "feature": "rcov", "classifier": best_rna["classifier"],
         "weighting_function": "", "top_genes": "", "source_dir": RNA_RESULT_DIR,
         "selection_mean_auc": best_rna["mean_auc"]},
        {"method": "atac", "feature": "ulz_atac_tfbs_features", "classifier": best_atac["classifier"],
         "weighting_function": "", "top_genes": "", "source_dir": ATAC_RESULT_DIR,
         "selection_mean_auc": best_atac["mean_auc"]},
        {"method": "ichorcna", "feature": "ichorcna_tf", "classifier": "",
         "weighting_function": "", "top_genes": "", "source_dir": ICHOR_AGG_CSV,
         "selection_mean_auc": ""},
    ])
    out_cfg = os.path.join(DATA_DIR, "fig3_config.csv")
    cfg.to_csv(out_cfg, index=False)
    print(f"wrote {out_cfg}")


if __name__ == "__main__":
    main()
