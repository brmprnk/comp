"""Export source data for Supplementary Figure S5 (label-shuffle stability).

S5: class-label-shuffle analysis of the best data-driven model (griffin_diff +
SVM + all TSSs, healthy vs BRCA+CRC): a) mean 10-fold CV AUROC under 10 random
label permutations vs the observed value; b) Pearson correlation of the
permuted models' weights with the true model's weights; c) Jaccard overlap of
the top-1000 TSSs by weight with the true model's top-1000.

Writes:
  src/figures/data/figS5_label_shuffle.csv  (one row per shuffle)
  src/figures/data/figS5_config.csv         (true-model reference values)

Source: nested_cv_results_cris/label_shuffle_best_model_search_dd/ — chosen
over the sibling "..._currentlyinmsjune2026" snapshot because its observed CV
AUC (0.9595 -> "0.960") and shuffle-AUC distribution are the ones shown in the
submitted S5; the snapshot carries an older run (observed 0.9468).
"""

import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402

SRC_DIR = "nested_cv_results_cris/label_shuffle_best_model_search_dd"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    results = pd.read_csv(rp(f"{SRC_DIR}/label_shuffle_results.csv"))
    out = os.path.join(DATA_DIR, "figS5_label_shuffle.csv")
    results.to_csv(out, index=False)
    print(f"wrote {out} ({len(results)} shuffles)")

    with open(rp(f"{SRC_DIR}/label_shuffle_summary.pkl"), "rb") as fh:
        s = pickle.load(fh)
    info = s.get("best_model_info", {})
    cfg = pd.DataFrame([{
        "observed_cv_auc": s.get("observed_cv_auc", s.get("true_cv_auc")),
        "true_full_auc": s.get("true_full_auc"),
        "n_shuffles": s.get("n_shuffles"),
        "top_k": s.get("top_k"),
        "weights_type": s.get("weights_type"),
        "feature": info.get("feature"),
        "classifier": info.get("classifier"),
        "n_genes": info.get("n_genes"),
        "selection_type": info.get("selection_type"),
        "source_dir": SRC_DIR,
    }])
    out_cfg = os.path.join(DATA_DIR, "figS5_config.csv")
    cfg.to_csv(out_cfg, index=False)
    print(f"wrote {out_cfg}")
    print(f"observed CV AUC = {cfg['observed_cv_auc'].iloc[0]:.4f} (manuscript S5a: 0.960); "
          f"shuffle AUC mean = {results['cv_auc_mean'].mean():.4f}")


if __name__ == "__main__":
    main()
