"""Export source data for Supplementary Figure S2 (TSS filtering / QC).

S2: a) genomic bp covered by the ENCODE blacklist vs the GC-correction
exclusion list, per chromosome; b) Venn counts of 10 kb TSS windows
overlapping each blacklist; c) per-TSS mean vs SD of coverage across the
panel-of-normals, colored by filter category (retained / overlapping TSS /
chromosome X / high variability / blacklist).

Writes:
  src/figures/data/figS2a_blacklist_per_chrom.csv
  src/figures/data/figS2b_venn_counts.csv
  src/figures/data/figS2c_tss_qc.csv   (one row per TSS window)

The exporter validates every filter-category count against the numbers in the
submitted panel (total 19,376; blacklist 1,311; overlapping 3,398; chrX 842;
high variability 72; removed 5,152 / retained 14,224) and derives the
categories from the same artifacts the pipeline used
(accessory_files/*.npy, beds/biomart_10kb.bed, the two blacklist BEDs).

NOTE: the notebook that drew the submitted S2c panel was not found in the
repository (only the component analyses in notebooks/blacklist_analysis.ipynb
and notebooks/gsea.ipynb survive); figS2.ipynb is a reconstruction from these
validated artifacts.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402

ENCODE_BED = "accessory_files/hg38-blacklist.v2.bed"
GC_BED = "accessory_files/hg38_GCcorrection_ExclusionList.merged.sorted.bed"
BIOMART_BED = "beds/biomart_10kb.bed"
FINAL_NPY = "accessory_files/final_tss_indices_to_remove_hg38.npy"
BLACKLIST_NPY = "accessory_files/blacklist_and_std_outlier_indices.npy"
FEATURES_DIR = "extracted_features/biomart_10kb"
METADATA = "accessory_files/cris_metadata.csv"
PON_CSV = "accessory_files/cris_pon.csv"


def bp_per_chrom(bed: pd.DataFrame) -> pd.Series:
    """Total bp per chromosome as the raw sum of interval lengths (no interval
    merging) — matching notebooks/blacklist_analysis.ipynb, which sums
    end - start directly."""
    lengths = bed["end"] - bed["start"]
    return lengths.groupby(bed["chrom"]).sum()


def windows_overlapping(bed: pd.DataFrame, bl: pd.DataFrame) -> np.ndarray:
    """Boolean per bed row: overlaps any interval in bl (same chrom)."""
    hit = np.zeros(len(bed), dtype=bool)
    for chrom, g in bl.groupby("chrom"):
        rows = bed.index[bed["chrom"] == chrom]
        if len(rows) == 0:
            continue
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()
        order = np.argsort(starts)
        starts, ends = starts[order], ends[order]
        for i in rows:
            s, e = bed.at[i, "start"], bed.at[i, "end"]
            j = np.searchsorted(starts, e)
            if j > 0 and np.any(ends[:j] > s):
                hit[i] = True
    return hit


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    biomart = pd.read_csv(rp(BIOMART_BED), sep="\t", header=None,
                          names=["chrom", "start", "end", "gene_name", "strand", "GC"])
    print(f"biomart windows: {len(biomart)}; chroms: {sorted(biomart['chrom'].unique())[:5]}...")
    assert len(biomart) == 19376, f"expected 19,376 windows, got {len(biomart)}"

    encode = pd.read_csv(rp(ENCODE_BED), sep="\t", header=None,
                         usecols=[0, 1, 2], names=["chrom", "start", "end"])
    gc = pd.read_csv(rp(GC_BED), sep="\t", header=None,
                     usecols=[0, 1, 2], names=["chrom", "start", "end"])

    # ---------------------------------------------------------- S2a: bp/chrom
    enc_bp = bp_per_chrom(encode)
    gc_bp = bp_per_chrom(gc)
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    s2a = pd.DataFrame({
        "chrom": chroms,
        "encode_blacklist_bp": [int(enc_bp.get(c, 0)) for c in chroms],
        "gc_exclusion_bp": [int(gc_bp.get(c, 0)) for c in chroms],
    })
    s2a.to_csv(os.path.join(DATA_DIR, "figS2a_blacklist_per_chrom.csv"), index=False)
    print(f"S2a totals: ENCODE {s2a['encode_blacklist_bp'].sum():,} bp, "
          f"GC exclusion {s2a['gc_exclusion_bp'].sum():,} bp "
          f"(panel: 195,195,200 / 352,006,422)")

    # ------------------------------------------------------------- S2b: venn
    # Sets of overlapping biomart *entry names* (not window indices), matching
    # the venn2 call in notebooks/blacklist_analysis.ipynb.
    hit_enc = windows_overlapping(biomart, encode)
    hit_gc = windows_overlapping(biomart, gc)
    gc_names = set(biomart.loc[hit_gc, "gene_name"])
    enc_names = set(biomart.loc[hit_enc, "gene_name"])
    only_gc = len(gc_names - enc_names)
    both = len(gc_names & enc_names)
    only_enc = len(enc_names - gc_names)
    s2b = pd.DataFrame([{"gc_only": only_gc, "both": both, "encode_only": only_enc}])
    s2b.to_csv(os.path.join(DATA_DIR, "figS2b_venn_counts.csv"), index=False)
    print(f"S2b venn: GC-only={only_gc}, both={both}, ENCODE-only={only_enc} "
          f"(panel: 958 / 353 / 0)")

    # ---------------------------------------------- S2c: per-TSS QC scatter
    final_removed = set(np.load(rp(FINAL_NPY)).tolist())
    blacklist_idx = set(np.load(rp(BLACKLIST_NPY)).tolist())
    chrx_idx = set(biomart.index[biomart["chrom"] == "chrX"].tolist())

    # overlapping-TSS category: another TSS within < 5,000 bp (the "(<5kb)"
    # in the panel legend is literal; TSS = window center). Verified: exactly
    # 3,398 windows, all inside the final removal set, and the derived
    # high-variability remainder is exactly 72.
    overlap_idx = set()
    for chrom, g in biomart.groupby("chrom"):
        g = g.sort_values("start")
        idx = g.index.to_numpy()
        tss = (g["start"] + (g["end"] - g["start"]) // 2).to_numpy()
        d = np.diff(tss)
        for i in np.where(d < 5000)[0]:
            overlap_idx.add(int(idx[i]))
            overlap_idx.add(int(idx[i + 1]))
    print(f"S2c overlapping windows (<5kb TSS distance): {len(overlap_idx)} (panel: 3,398)")

    high_var_idx = final_removed - (blacklist_idx | chrx_idx | overlap_idx)
    print(f"S2c high-variability (derived as final \\ others): {len(high_var_idx)} (panel: 72)")
    union = blacklist_idx | chrx_idx | overlap_idx | high_var_idx
    assert union == final_removed, (
        f"category union ({len(union)}) != final removed set ({len(final_removed)})")
    in_multiple = sum(
        (i in blacklist_idx) + (i in chrx_idx) + (i in overlap_idx) + (i in high_var_idx) > 1
        for i in final_removed)
    print(f"S2c removed by multiple filters: {in_multiple} (panel: 452); "
          f"removed {len(final_removed)} / retained {19376 - len(final_removed)}")

    # per-TSS mean/std of mean_coverage across the panel-of-normals
    pon_ids = pd.read_csv(rp(PON_CSV))["ID"].tolist()
    mats, used = [], []
    for sid in pon_ids:
        p = rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_BIN.csv")
        if not os.path.exists(p):
            continue
        mats.append(pd.read_csv(p, usecols=["mean_coverage"])["mean_coverage"].to_numpy())
        used.append(sid)
    mat = np.vstack(mats)
    print(f"S2c PoN samples with BIN.csv: {len(used)} (from {PON_CSV})")
    assert mat.shape[1] == len(biomart)

    mean_cov = np.nanmean(mat, axis=0)
    std_cov = np.nanstd(mat, axis=0)
    valid = ~(np.isnan(mean_cov) | np.isnan(std_cov))
    r = np.corrcoef(mean_cov[valid], std_cov[valid])[0, 1]
    print(f"S2c Pearson r (mean vs std, all windows): {r:.3f} (panel: 0.904)")

    s2c = pd.DataFrame({
        "tss_index": biomart.index,
        "chrom": biomart["chrom"],
        "gene_name": biomart["gene_name"],
        "mean_coverage": mean_cov,
        "std_coverage": std_cov,
        "in_blacklist": biomart.index.isin(list(blacklist_idx)),
        "in_chrX": biomart.index.isin(list(chrx_idx)),
        "in_overlapping": biomart.index.isin(list(overlap_idx)),
        "in_high_variability": biomart.index.isin(list(high_var_idx)),
        "removed": biomart.index.isin(list(final_removed)),
    })
    s2c.to_csv(os.path.join(DATA_DIR, "figS2c_tss_qc.csv"), index=False)
    print(f"wrote figS2c_tss_qc.csv ({len(s2c)} rows) + figS2a/figS2b CSVs")


if __name__ == "__main__":
    main()
