"""Shared helpers for the figure source-data exporters.

These exporters read the raw experiment outputs (result pickles under
nested_cv_results_cris/, results/, and accessory_files/) and write the
per-figure source-data CSVs in src/figures/data/. They are the ONLY layer
allowed to touch result pickles or machine-specific paths; the figure
notebooks in src/figures/ read the CSVs alone.

Run from the repository root, e.g.:
    python src/figures/export/export_fig3.py

Every exporter validates what it writes against the numbers stored in the
pickles (AUCs, confusion matrices) so that a regenerated figure is guaranteed
to match the original pipeline output.
"""

from __future__ import annotations

import os
import pickle
from glob import glob

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "src", "figures", "data")

# ---------------------------------------------------------------------------
# Result-directory manifest (mirrors the figure notebooks as of Sep 2026).
#
# brca_crc_dd/ holds ONLY the post-operating-point-fix mean_diff re-run
# (June 2026); the other ranking functions exist only in the pre-fix
# directory. AUCs/predictions are identical between the two runs except for
# `fslr` (see the audit note in src/comp/hypo_nestedcv_plots.ipynb), but the
# pre-fix rows carry test-fold-derived operating points, so anything
# threshold-keyed must come from the post-fix rows.
# ---------------------------------------------------------------------------
DD_RESULT_DIRS = {
    "mean_diff": "nested_cv_results_cris/brca_crc_dd",
    "random": "nested_cv_results_cris/brca_crc_dd_before_prec_change_jun122026",
    "p_value": "nested_cv_results_cris/brca_crc_dd_before_prec_change_jun122026",
    "fold_change": "nested_cv_results_cris/brca_crc_dd_before_prec_change_jun122026",
    "weighted": "nested_cv_results_cris/brca_crc_dd_before_prec_change_jun122026",
}
RNA_RESULT_DIR = "nested_cv_results_cris/brca_crc_rna"
ATAC_RESULT_DIR = "nested_cv_results_cris/brca_crc_atac"
ICHOR_AGG_CSV = "accessory_files/cris_stage_vaf_coverage_ichor_aggregate.csv"
ICHOR_CSV = "accessory_files/cris_stage_vaf_coverage_ichor.csv"


def rp(*parts: str) -> str:
    """Path relative to the repo root."""
    return os.path.join(REPO_ROOT, *parts)


def load_results(pattern: str, keep_cols: list[str], extra: dict | None = None) -> pd.DataFrame:
    """Load all result pickles matching `pattern` (repo-root relative), keep
    only `keep_cols`, and concatenate."""
    paths = sorted(glob(rp(pattern)))
    if not paths:
        raise FileNotFoundError(f"No result pickles match {pattern!r}")
    frames = []
    for p in paths:
        with open(p, "rb") as fh:
            df = pickle.load(fh)
        df = df[[c for c in keep_cols if c in df.columns]]
        if extra:
            for k, v in extra.items():
                df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def best_mean_auc_config(df: pd.DataFrame, group_cols: list[str]) -> dict:
    """Configuration (group_cols) with the highest mean auc_full_model.

    Same selection as find_best_datadriven_config / find_best_profile_classifier
    in src/comp/hypo_nestedcv_plots.ipynb.
    """
    grouped = df.groupby(group_cols)["auc_full_model"].mean().reset_index()
    best = grouped.loc[grouped["auc_full_model"].idxmax()]
    out = {c: best[c] for c in group_cols}
    out["mean_auc"] = float(best["auc_full_model"])
    return out


def assert_predictions_match_stored(rows: pd.DataFrame, label: str) -> None:
    """For each fold row of a results DataFrame, recompute AUC and the
    95%-specificity confusion matrix from (y_test, y_pred_proba) and assert
    they equal the stored values. Guarantees the exported per-sample scores
    reproduce the pipeline numbers."""
    from sklearn.metrics import roc_auc_score

    for _, row in rows.iterrows():
        y = np.asarray(row["y_test"], dtype=int)
        s = np.asarray(row["y_pred_proba"], dtype=float)
        auc_re = roc_auc_score(y, s)
        assert np.isclose(auc_re, row["auc_full_model"], atol=1e-10), (
            f"{label} fold {row['fold']}: recomputed AUC {auc_re} != stored {row['auc_full_model']}"
        )
        thr = row["full_model_thresholds"]["95_percent_specificity"]
        cm_stored = row["confusion_matrix_full_model"]["95_percent_specificity"]
        cm_stored = np.asarray(cm_stored, dtype=int)
        y_pred = (s >= thr).astype(int)
        tn = int(np.sum((y == 0) & (y_pred == 0)))
        fp = int(np.sum((y == 0) & (y_pred == 1)))
        fn = int(np.sum((y == 1) & (y_pred == 0)))
        tp = int(np.sum((y == 1) & (y_pred == 1)))
        cm_re = np.array([[tn, fp], [fn, tp]])
        assert (cm_re == cm_stored).all(), (
            f"{label} fold {row['fold']}: confusion matrix at 95% specificity does not "
            f"reproduce from scores + stored threshold ({cm_re.tolist()} vs {cm_stored.tolist()}); "
            "check >= convention / threshold provenance"
        )
