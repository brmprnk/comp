"""Export source data for manuscript Figure 5 (and Supplementary Figs S6, S7).

Figure 5:
  a) PANCAN nested CV (all 7 cancer types vs healthy) — mean ROC per method.
  b) External validation on the Jiang cohort — ROC per method.
  c) Sensitivity at 95% specificity on the Jiang cohort.
  d) Within-dataset generalization — ROC on withheld CRC/BRCA samples.
  e) Within-dataset generalization — ROC on novel cancer types.
  f) Sensitivity at 95% specificity for the within-dataset generalization.
Figure S6: PANCAN per-cancer-type ROC grid (same source data as 5a).
Figure S7: external-validation AUC vs number of TSSs (DD models).

Writes to src/figures/data/:
  fig5a_predictions.csv / fig5a_config.csv     (drives 5a and S6)
  fig5bc_external_predictions.csv / fig5bc_config.csv   (drives 5b, 5c)
  figS7_auc_vs_k.csv
  fig5def_roc_curves.csv / fig5def_metrics.csv  (drives 5d, 5e, 5f)

Source experiments:
  5a/S6 : nested_cv_results_cris/{pancan_dd, nested_rna_pancan, pancan_atac - Copy}
          + accessory_files/cris_stage_vaf_coverage_ichor.csv (ichorCNA)
  5b/c/S7: nested_cv_results_cris/external_validation_comprehensive_external_
          cris_on_jiang_{dd,rna,atac}/ + the deployed models' training
          summaries (for the train-derived 95%-spec threshold)
  5d-f  : results/within_gen_Breast_cancer_Colorectal_cancer_{dd,rna,atac}/
          comprehensive_evaluation/comprehensive_generalization_summary.pkl
"""

import os
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, best_mean_auc_config, load_results, rp  # noqa: E402

PANCAN_DD_DIR = "nested_cv_results_cris/pancan_dd"
PANCAN_RNA_DIR = "nested_cv_results_cris/nested_rna_pancan"
PANCAN_ATAC_DIR = "nested_cv_results_cris/pancan_atac - Copy"
ICHOR_CSV = "accessory_files/cris_stage_vaf_coverage_ichor.csv"
EXT_DIRS = {ft: f"nested_cv_results_cris/external_validation_comprehensive_external_cris_on_jiang_{ft}"
            for ft in ["dd", "rna", "atac"]}
WITHIN_BASE = "results/within_gen_Breast_cancer_Colorectal_cancer"

KEEP = ["feature", "classifier", "top_genes", "fold", "auc_full_model",
        "auc_per_cancer_type", "test_sample_ids", "y_pred_proba", "y_test"]


def export_fig5a() -> None:
    frames = []
    for wf in ["mean_diff", "random", "p_value", "fold_change", "weighted"]:
        frames.append(load_results(f"{PANCAN_DD_DIR}/results*{wf}.pkl", KEEP,
                                   extra={"weighting_function": wf}))
    dd = pd.concat(frames, ignore_index=True)
    dd["top_genes"] = dd["top_genes"].replace(15000, 14224)
    rna = load_results(f"{PANCAN_RNA_DIR}/results*.pkl", KEEP)
    atac = load_results(f"{PANCAN_ATAC_DIR}/results*.pkl", KEEP)

    # A pure argmax over the grid lands on Logistic Regression at k=14,224
    # (mean AUC 0.9468 ± 0.034); the submitted Figure 5a, however, shows the
    # SVM at k=14,224 (0.9459 -> "0.946 ± 0.032" in the legend), consistent
    # with the manuscript describing the best model as an SVM throughout. We
    # pin the SVM accordingly, and mean_diff as the (inert at k=all) ranking.
    best_argmax = best_mean_auc_config(dd[dd["feature"] == "griffin_diff"],
                                       ["classifier", "weighting_function", "top_genes"])
    print(f"  (pure argmax config would be: {best_argmax})")
    best_dd = {"classifier": "Support Vector Machine",
               "weighting_function": "mean_diff", "top_genes": 14224}
    dd_sel = dd[(dd["feature"] == "griffin_diff")
                & (dd["classifier"] == best_dd["classifier"])
                & (dd["weighting_function"] == best_dd["weighting_function"])
                & (dd["top_genes"] == best_dd["top_genes"])]
    best_dd["mean_auc"] = float(dd_sel["auc_full_model"].mean())
    best_rna = best_mean_auc_config(rna[rna["feature"] == "rcov"], ["classifier"])
    rna_sel = rna[(rna["feature"] == "rcov") & (rna["classifier"] == best_rna["classifier"])]
    atac_feat = "ulz_atac_tfbs_features_pancan"
    best_atac = best_mean_auc_config(atac[atac["feature"] == atac_feat], ["classifier"])
    atac_sel = atac[(atac["feature"] == atac_feat) & (atac["classifier"] == best_atac["classifier"])]

    print("PANCAN selected configs:")
    print(f"  data_driven: {best_dd}")
    print(f"  rna:         {best_rna}")
    print(f"  atac:        {best_atac}")
    for name, sel in [("dd", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        assert len(sel) == 10, f"{name}: expected 10 folds, got {len(sel)}"

    def fold_map(sel):
        return {int(r["fold"]): frozenset(r["test_sample_ids"]) for _, r in sel.iterrows()}

    assert fold_map(dd_sel) == fold_map(rna_sel) == fold_map(atac_sel), \
        "PANCAN fold partitions differ between arms"

    ichor = pd.read_csv(rp(ICHOR_CSV), index_col=0)
    mult = ichor.groupby("ID").size()
    dup_ids = set(mult[mult > 1].index)
    for sid in dup_ids:
        grp = ichor[ichor["ID"] == sid]
        assert (grp.nunique() <= 1).all(), f"duplicate rows for {sid} differ"
    ichor = ichor.drop_duplicates(subset="ID")
    print(f"  ichorCNA table: {len(ichor)} unique samples ({len(dup_ids)} duplicated)")
    ann = ichor.set_index("ID")

    # validate that per-sample scores reproduce stored full-model and
    # per-cancer-type AUCs (the latter is what Figure S6 recomputes)
    for name, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        for _, row in sel.iterrows():
            y = np.asarray(row["y_test"], dtype=int)
            s = np.asarray(row["y_pred_proba"], dtype=float)
            assert np.isclose(roc_auc_score(y, s), row["auc_full_model"], atol=1e-10)
            disease = pd.Series(row["test_sample_ids"]).map(ann["disease"]).to_numpy()
            for ct, auc_stored in row["auc_per_cancer_type"].items():
                m = (disease == "Healthy") | (disease == ct)
                if len(np.unique(y[m])) < 2:
                    continue
                auc_re = roc_auc_score(y[m], s[m])
                assert np.isclose(auc_re, auc_stored, atol=1e-10), (
                    f"{name} fold {row['fold']} {ct}: {auc_re} != stored {auc_stored}")
        print(f"  {name}: stored full-model and per-cancer-type AUCs reproduced")

    records = []
    for method, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        for _, row in sel.iterrows():
            for sid, y, s in zip(row["test_sample_ids"], row["y_test"], row["y_pred_proba"]):
                records.append({"sample_id": sid, "method": method,
                                "fold": int(row["fold"]), "y_true": int(y),
                                "score": float(s)})
    pred = pd.DataFrame(records)

    dd_folds = fold_map(dd_sel)
    sample_to_fold = {sid: f for f, ids in dd_folds.items() for sid in ids}
    ichor_rows = pd.DataFrame({
        "sample_id": ichor["ID"], "method": "ichorcna",
        "fold": ichor["ID"].map(sample_to_fold),
        "y_true": (ichor["disease"] != "Healthy").astype(int),
        "score": ichor["ichorcna_tf"],
    })
    pred = pd.concat([pred, ichor_rows], ignore_index=True)
    pred["n_rows_in_source"] = np.where(
        (pred["method"] == "ichorcna") & pred["sample_id"].isin(dup_ids), 2, 1)
    pred["disease"] = pred["sample_id"].map(ann["disease"])

    out = os.path.join(DATA_DIR, "fig5a_predictions.csv")
    pred.to_csv(out, index=False)
    print(f"wrote {out} ({len(pred)} rows; "
          f"{pred[pred.method == 'data_driven']['sample_id'].nunique()} task samples)")

    cfg = pd.DataFrame([
        {"method": "data_driven", "feature": "griffin_diff", **{k: best_dd[k] for k in
         ["classifier", "weighting_function", "top_genes"]},
         "source_dir": PANCAN_DD_DIR, "selection_mean_auc": best_dd["mean_auc"]},
        {"method": "rna", "feature": "rcov", "classifier": best_rna["classifier"],
         "weighting_function": "", "top_genes": "", "source_dir": PANCAN_RNA_DIR,
         "selection_mean_auc": best_rna["mean_auc"]},
        {"method": "atac", "feature": atac_feat, "classifier": best_atac["classifier"],
         "weighting_function": "", "top_genes": "", "source_dir": PANCAN_ATAC_DIR,
         "selection_mean_auc": best_atac["mean_auc"]},
        {"method": "ichorcna", "feature": "ichorcna_tf", "classifier": "",
         "weighting_function": "", "top_genes": "", "source_dir": ICHOR_CSV,
         "selection_mean_auc": ""},
    ])
    cfg.to_csv(os.path.join(DATA_DIR, "fig5a_config.csv"), index=False)
    print("wrote fig5a_config.csv")


def _train_threshold_95spec(best_model_path: str) -> float | None:
    """Mean per-cancer-type 95%-spec threshold from the deployed model's
    training summary (same logic as analyze_external_generalization.ipynb)."""
    summary_path = Path(rp(str(best_model_path).replace("best_model_GridSearchCV_", "summary_")))
    if not summary_path.exists():
        # the stored path may be absolute from the cluster; retry inside the repo
        parts = str(best_model_path).replace("\\", "/").split("/")
        if "nested_cv_results_cris" in parts:
            rel = "/".join(parts[parts.index("nested_cv_results_cris"):])
            summary_path = Path(rp(rel.replace("best_model_GridSearchCV_", "summary_")))
    if not summary_path.exists():
        print(f"  ! training summary not found for {best_model_path}")
        return None
    with open(summary_path, "rb") as fh:
        summ = pickle.load(fh)
    roc_ct = summ["full_dataset_evaluation"]["roc_per_cancer_type"]
    thrs = [v["thresholds"]["95_percent_specificity"] for v in roc_ct.values()
            if "thresholds" in v and "95_percent_specificity" in v["thresholds"]]
    return float(np.mean(thrs)) if thrs else None


def export_fig5bc() -> None:
    pred_rows, cfg_rows = [], []
    for ft, ext_dir in EXT_DIRS.items():
        with open(rp(f"{ext_dir}/comprehensive_validation_summary.pkl"), "rb") as fh:
            summary = pickle.load(fh)

        # best feature per type by TRAINING CV AUC (selected before seeing
        # external data — same as the analysis notebook)
        fr = {k: v for k, v in summary["feature_results"].items() if v is not None}
        best_feat = max(fr, key=lambda k: fr[k].get("training_cv_auc", 0))

        detail = {}
        for p in sorted(glob(rp(f"{ext_dir}/external_validation_*_results.pkl"))):
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            detail[d.get("best_model_info", {}).get("feature", os.path.basename(p))] = d
        bp = detail[best_feat]

        y = np.asarray(bp["y_true"], dtype=int)
        s = np.asarray(bp["y_pred_proba"], dtype=float)
        assert np.isclose(roc_auc_score(y, s), bp["overall_auc"], atol=1e-9), \
            f"{ft}: recomputed external AUC != stored"
        sample_ids = bp.get("sample_ids", [None] * len(y))

        for sid, yi, si in zip(sample_ids, y, s):
            pred_rows.append({"method": ft, "sample_id": sid, "y_true": int(yi),
                              "score": float(si)})

        thr_train = _train_threshold_95spec(bp.get("best_model_path", ""))
        cfg_rows.append({
            "method": ft, "feature": best_feat,
            "classifier": bp.get("best_model_info", {}).get("classifier", ""),
            "n_genes": bp.get("best_model_info", {}).get("n_genes", ""),
            "train_cv_auc": bp.get("best_model_info", {}).get("cv_auc", np.nan),
            "external_auc": bp.get("overall_auc", np.nan),
            "threshold_95spec_external": bp.get("threshold_95spec", np.nan),
            "threshold_95spec_train": thr_train if thr_train is not None else np.nan,
            "proba_degenerate": bool(bp.get("proba_degenerate", False)),
            "source_dir": ext_dir,
        })
        print(f"  {ft}: best_feat={best_feat}, external AUC={bp['overall_auc']:.3f}, "
              f"n={len(y)} samples, thr_train={thr_train}")

    pd.DataFrame(pred_rows).to_csv(
        os.path.join(DATA_DIR, "fig5bc_external_predictions.csv"), index=False)
    pd.DataFrame(cfg_rows).to_csv(
        os.path.join(DATA_DIR, "fig5bc_config.csv"), index=False)
    print("wrote fig5bc_external_predictions.csv + fig5bc_config.csv")

    # S7: external AUC vs number of genes for the DD models
    with open(rp(f"{EXT_DIRS['dd']}/comprehensive_validation_summary.pkl"), "rb") as fh:
        summary = pickle.load(fh)
    rows = []
    for k, r in sorted(summary.get("gene_count_results", {}).items()):
        if not r:
            continue
        rows.append({"n_genes": int(k), "external_auc": r.get("auc", np.nan),
                     "training_cv_auc": r.get("training_cv_auc", np.nan),
                     "feature": r.get("feature", ""), "classifier": r.get("classifier", "")})
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "figS7_auc_vs_k.csv"), index=False)
    print(f"wrote figS7_auc_vs_k.csv ({len(rows)} gene counts)")


def export_fig5def() -> None:
    curve_rows, metric_rows = [], []
    for ft in ["dd", "rna", "atac"]:
        pkl = rp(f"{WITHIN_BASE}_{ft}/comprehensive_evaluation/comprehensive_generalization_summary.pkl")
        with open(pkl, "rb") as fh:
            s = pickle.load(fh)
        roc = s.get("best_feature_roc", {})
        bf_auc = s.get("best_feature_auc", {})
        best_feat = s.get("best_feature", "?")
        r = (s.get("feature_results", {}) or {}).get(best_feat, {}) or {}
        for subset in ["withheld", "novel"]:
            rd, ci = roc.get(subset, {}), roc.get(f"{subset}_ci", {})
            for f_, t_ in zip(rd.get("fpr", []), rd.get("tpr", [])):
                curve_rows.append({"method": ft, "subset": subset, "series": "mean",
                                   "fpr": f_, "tpr": t_, "tpr_lower": np.nan, "tpr_upper": np.nan})
            if ci:
                for f_, lo, hi in zip(ci.get("fpr", []), ci.get("tpr_lower", []),
                                      ci.get("tpr_upper", [])):
                    curve_rows.append({"method": ft, "subset": subset, "series": "ci",
                                       "fpr": f_, "tpr": np.nan, "tpr_lower": lo, "tpr_upper": hi})
            metric_rows.append({
                "method": ft, "subset": subset, "best_feature": best_feat,
                "auc_mean": bf_auc.get(f"{subset}_mean", bf_auc.get(subset, np.nan)),
                "auc_std": bf_auc.get(f"{subset}_std", np.nan),
                "sens_95": r.get(f"{subset}_sens_at_95spec", np.nan),
                "sens_95_std": r.get(f"{subset}_sens_at_95spec_std", np.nan),
                "n_splits": s.get("n_splits", np.nan),
                "n_samples": s.get("eval_n_withheld" if subset == "withheld"
                                   else "eval_n_novel_cancer", np.nan),
            })
        print(f"  {ft}: best_feature={best_feat}, "
              f"withheld AUC={bf_auc.get('withheld_mean', np.nan):.3f}, "
              f"novel AUC={bf_auc.get('novel_mean', np.nan):.3f}")

    pd.DataFrame(curve_rows).to_csv(os.path.join(DATA_DIR, "fig5def_roc_curves.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(DATA_DIR, "fig5def_metrics.csv"), index=False)
    print("wrote fig5def_roc_curves.csv + fig5def_metrics.csv")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Fig 5a / S6 (PANCAN) ===")
    export_fig5a()
    print("\n=== Fig 5b/c / S7 (external validation, Jiang) ===")
    export_fig5bc()
    print("\n=== Fig 5d-f (within-dataset generalization) ===")
    export_fig5def()
