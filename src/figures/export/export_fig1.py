"""Export source data for manuscript Figure 1.

Figure 1: a) normalized cfDNA coverage profiles at TSSs for housekeeping (HK)
vs unexpressed (PAU) genes across the panel-of-normals; b) distribution of
per-locus mean coverage in the two gene sets; c) mean per-sample Pearson
correlation of each fragmentomic feature with Esfahani et al. gene-group
expression, for all TSSs vs QC-filtered TSSs.

Writes to src/figures/data/:
  fig1a_coverage_profiles.csv   (position, gene_set, mean, sd over PoN samples)
  fig1b_locus_coverage.csv      (sample_id, gene_set, per-locus mean coverage,
                                 after the per-sample low-coverage mask used by
                                 the original script)
  fig1c_expression_correlations.csv (sample_index, feature, tss_set, pearson_r)

Panel a/b replicate src/comp/plot_aggregated_coverage.py (PoN = first 50
healthy samples with coverage arrays; per-sample aggregate over the gene-set
loci, normalized to mean 1; QC filter applied to the loci). Panel c replicates
notebooks/esfahani.ipynb (per-sample Pearson r of gene-group feature means vs
TPM[log2]; "QC TSSs" = gene groups with at most one QC-removed gene, using the
non-removed loci).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402

FEATURES_DIR = "extracted_features/biomart_10kb"
METADATA = "accessory_files/cris_metadata.csv"
BIOMART_BED = "beds/biomart_10kb.bed"
PAU_BED = "beds/pau.bed"
HK_BED = "beds/hk_genes.bed"
FINAL_NPY = "accessory_files/final_tss_indices_to_remove_hg38.npy"
ESFAHANI_XLSX = "accessory_files/esfahani_supplement.xlsx"
PON_N = 50
WINDOW = 10000
TSS_FEATURES = ["fslr", "fslr_coverage", "griffin_diff", "rcov", "mwh"]


def load_bed_genes(path):
    bed = pd.read_csv(rp(path), sep="\t", header=None)
    return bed[3].unique()


def coverage_path(sid):
    p = Path(rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_coverage.npy"))
    if p.exists():
        return p
    alt = Path(rp(f"{FEATURES_DIR}/{sid}_coverage.npy"))
    return alt if alt.exists() else None


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    metadata = pd.read_csv(rp(METADATA))
    healthy_ids = metadata[metadata["disease"] == "Healthy"]["ID"].tolist()
    pon_ids = []
    for sid in healthy_ids:
        if len(pon_ids) >= PON_N:
            break
        if coverage_path(sid) is not None:
            pon_ids.append(sid)
    print(f"PoN: {len(pon_ids)} healthy samples with coverage arrays")

    biomart = pd.read_csv(rp(BIOMART_BED), sep="\t", header=None,
                          names=["chrom", "start", "end", "gene_name", "strand", "GC"])
    pau_genes = load_bed_genes(PAU_BED)
    hk_genes = load_bed_genes(HK_BED)
    removed = np.load(rp(FINAL_NPY))
    removed_set = set(removed.tolist())
    blacklisted_genes = set(biomart.iloc[removed]["gene_name"].unique())

    def gene_indices(genes):
        idx = biomart.index[biomart["gene_name"].isin(set(genes))].to_numpy()
        return np.array([i for i in idx if i not in removed_set])

    pau_idx = gene_indices(pau_genes)
    hk_idx = gene_indices(hk_genes)
    print(f"loci after QC filter: PAU {len(pau_idx)}, HK {len(hk_idx)} "
          f"(panel legend: 529 / 1423)")

    # ---------------------------------------------------- panel a: profiles
    pau_profiles, hk_profiles = [], []
    locus_rows = []
    hk_set, pau_set = set(hk_genes), set(pau_genes)
    for n, sid in enumerate(pon_ids, 1):
        cov = np.load(coverage_path(sid), mmap_mode="r")
        for indices, sink in [(pau_idx, pau_profiles), (hk_idx, hk_profiles)]:
            agg = np.asarray(cov[np.sort(indices), :]).sum(axis=0)
            if agg.shape[0] > WINDOW:  # center-trim (11001 -> 10000)
                start = (agg.shape[0] - WINDOW) // 2
                agg = agg[start:start + WINDOW]
            mean_cov = agg.mean()
            sink.append(agg / mean_cov if mean_cov > 0 else agg)
        del cov

        # panel b: per-locus mean coverage from the BIN.csv, with the original
        # per-sample low-coverage mask (drop loci below the 1st percentile of
        # positive coverage among HK+PAU loci)
        bin_df = pd.read_csv(rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_BIN.csv"),
                             usecols=["gene_name", "mean_coverage"])
        bl_mask = bin_df["gene_name"].isin(blacklisted_genes)
        hk_mask = bin_df["gene_name"].isin(hk_set) & ~bl_mask
        pau_mask = bin_df["gene_name"].isin(pau_set) & ~bl_mask
        all_cov = bin_df.loc[hk_mask | pau_mask, "mean_coverage"].dropna()
        min_thr = all_cov[all_cov > 0].quantile(0.01)
        cov_mask = bin_df["mean_coverage"] > min_thr
        for gs, mask in [("HK", hk_mask), ("PAU", pau_mask)]:
            for v in bin_df.loc[mask & cov_mask, "mean_coverage"].dropna():
                locus_rows.append({"sample_id": sid, "gene_set": gs, "mean_coverage": v})
        print(f"  [{n}/{len(pon_ids)}] {sid} done")

    x = np.arange(-WINDOW // 2, WINDOW // 2)
    rows = []
    for gs, profiles in [("HK", hk_profiles), ("PAU", pau_profiles)]:
        stack = np.array(profiles)
        rows.append(pd.DataFrame({"position": x, "gene_set": gs,
                                  "mean_normalized_coverage": stack.mean(axis=0),
                                  "sd_normalized_coverage": stack.std(axis=0)}))
    prof = pd.concat(rows, ignore_index=True)
    prof.to_csv(os.path.join(DATA_DIR, "fig1a_coverage_profiles.csv"), index=False)
    print(f"wrote fig1a_coverage_profiles.csv ({len(prof)} rows, n_samples={len(pon_ids)}, "
          f"n_hk_tss={len(hk_idx)}, n_pau_tss={len(pau_idx)})")

    locus = pd.DataFrame(locus_rows)
    locus.to_csv(os.path.join(DATA_DIR, "fig1b_locus_coverage.csv"), index=False)
    print(f"wrote fig1b_locus_coverage.csv ({len(locus)} rows)")

    # ------------------------------------------------- panel c: correlations
    print("panel c: loading feature matrices for the PoN...")
    feature_arrays = {}
    for feat in TSS_FEATURES:
        cols = []
        for sid in pon_ids:
            df = pd.read_csv(rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_BIN.csv"),
                             usecols=[feat])
            cols.append(df[feat].to_numpy())
        feature_arrays[feat] = np.vstack(cols)
        print(f"  {feat}: {feature_arrays[feat].shape}")

    esf = pd.read_excel(rp(ESFAHANI_XLSX), sheet_name="Supplementary Table 2",
                        header=6, usecols=["Gene group", "TPM [log2]"])
    groups = []
    for _, row in esf.iterrows():
        genes = [g.strip() for g in row["Gene group"].split("|")]
        all_idx, unfiltered_idx, n_filtered = [], [], 0
        for gene in genes:
            idx = biomart.index[biomart["gene_name"] == gene].tolist()
            if idx:
                all_idx.extend(idx)
                if gene in blacklisted_genes:
                    n_filtered += 1
                else:
                    unfiltered_idx.extend(idx)
        if all_idx:
            groups.append({"tpm": row["TPM [log2]"], "n_filtered": n_filtered,
                           "all_idx": all_idx, "unfiltered_idx": unfiltered_idx})
    print(f"panel c: {len(groups)} Esfahani gene groups mapped")

    corr_rows = []
    for feat in TSS_FEATURES:
        fa = feature_arrays[feat]
        # "all": every mapped group, all loci
        tpm_all = np.array([g["tpm"] for g in groups])
        means_all = np.vstack([np.nanmean(fa[:, g["all_idx"]], axis=1) for g in groups])
        # "qc": groups with <=1 filtered gene, non-removed loci only
        sel = [g for g in groups if g["n_filtered"] <= 1 and g["unfiltered_idx"]]
        tpm_qc = np.array([g["tpm"] for g in sel])
        means_qc = np.vstack([np.nanmean(fa[:, g["unfiltered_idx"]], axis=1) for g in sel])
        for label, tpm_v, means in [("all", tpm_all, means_all), ("qc", tpm_qc, means_qc)]:
            for si in range(means.shape[1]):
                fv = means[:, si]
                mask = ~(np.isnan(tpm_v) | np.isnan(fv))
                r = pearsonr(tpm_v[mask], fv[mask])[0] if mask.sum() >= 3 else np.nan
                corr_rows.append({"sample_index": si, "sample_id": pon_ids[si],
                                  "feature": feat, "tss_set": label, "pearson_r": r})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(DATA_DIR, "fig1c_expression_correlations.csv"), index=False)
    print("wrote fig1c_expression_correlations.csv")
    print("\npanel c summary (manuscript Fig 1c values for comparison):")
    for feat in TSS_FEATURES:
        for label in ["all", "qc"]:
            v = corr[(corr.feature == feat) & (corr.tss_set == label)]["pearson_r"].dropna()
            print(f"  {feat:<14} {label:<4} r = {v.mean():+.3f} ± {v.std(ddof=0):.3f}")


if __name__ == "__main__":
    main()
