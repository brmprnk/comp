"""
Aggregate ichorCNA 1MB logR bins into 5MB windows defined in mathios_metadata.xlsx (sheet s12).

Output: CSV at extracted_features/ichorCNA_5mb_logR.csv
        rows = samples (from cris_metadata.csv), columns = 5MB bin labels (chr:start-end)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
ICHORCNA_DIR = BASE / "results" / "ichorCNA"
METADATA_FILE = BASE / "accessory_files" / "cris_metadata.csv"
MATHIOS_FILE = BASE / "accessory_files" / "mathios_metadata.xlsx"
OUTPUT_FILE = BASE / "extracted_features" / "ichorCNA_5mb_logR.csv"

# --- Load 5MB bins (sheet s12) ---
bins_5mb = pd.read_excel(MATHIOS_FILE, sheet_name="s12", header=1)
bins_5mb["start"] = bins_5mb["start"].astype(int)
bins_5mb["end"] = bins_5mb["end"].astype(int)
# normalize chr: "chr1" → "1" to match ichorCNA convention
bins_5mb["chr_norm"] = bins_5mb["chr"].str.replace("chr", "", regex=False)
bins_5mb["bin_label"] = (
    bins_5mb["chr"] + ":" + bins_5mb["start"].astype(str) + "-" + bins_5mb["end"].astype(str)
)
print(f"Loaded {len(bins_5mb)} 5MB bins from s12")

# --- Load sample IDs from cris_metadata ---
metadata = pd.read_csv(METADATA_FILE)
sample_ids = metadata["ID"].tolist()
print(f"Loaded {len(sample_ids)} samples from cris_metadata.csv")

# --- Precompute 1MB → 5MB bin mapping (genome coordinates are shared across samples) ---
# Use first available sample as reference to enumerate all 1MB bin coordinates
ref_id = next(
    s for s in sample_ids
    if (ICHORCNA_DIR / f"{s}.hg38.frag.tsv" / f"{s}.hg38.frag.tsv.correctedDepth.txt").exists()
)
ref_file = ICHORCNA_DIR / f"{ref_id}.hg38.frag.tsv" / f"{ref_id}.hg38.frag.tsv.correctedDepth.txt"
ref_coords = pd.read_csv(ref_file, sep="\t", usecols=["chr", "start", "end"])
ref_coords["chr"] = ref_coords["chr"].astype(str)
ref_coords["start"] = ref_coords["start"].astype(int)
ref_coords["end"] = ref_coords["end"].astype(int)

bins_for_join = bins_5mb[["chr_norm", "start", "end", "bin_label"]].rename(
    columns={"chr_norm": "chr", "start": "bin_start", "end": "bin_end"}
)

# Cross-join within chromosome, then keep only contained 1MB bins
mapping = (
    ref_coords.rename(columns={"start": "depth_start", "end": "depth_end"})
    .merge(bins_for_join, on="chr")
    .query("depth_start >= bin_start and depth_end <= bin_end")
    [["chr", "depth_start", "depth_end", "bin_label"]]
    .rename(columns={"depth_start": "start", "depth_end": "end"})
    .reset_index(drop=True)
)
print(
    f"Mapping covers {len(mapping)} 1MB bins -> {mapping['bin_label'].nunique()} 5MB bins"
)

# --- Aggregate per sample ---
ordered_bins = bins_5mb["bin_label"].tolist()
all_results: dict[str, pd.Series] = {}
missing: list[str] = []

for i, sample_id in enumerate(sample_ids, 1):
    depth_file = (
        ICHORCNA_DIR / f"{sample_id}.hg38.frag.tsv"
        / f"{sample_id}.hg38.frag.tsv.correctedDepth.txt"
    )
    if not depth_file.exists():
        missing.append(sample_id)
        continue

    depth = pd.read_csv(depth_file, sep="\t")
    depth["chr"] = depth["chr"].astype(str)
    depth["start"] = depth["start"].astype(int)
    depth["end"] = depth["end"].astype(int)

    labeled = depth.merge(mapping, on=["chr", "start", "end"])
    bin_means = labeled.groupby("bin_label")["log2_TNratio_corrected"].mean()
    all_results[sample_id] = bin_means

    if i % 50 == 0:
        print(f"  Processed {i}/{len(sample_ids)} samples")

if missing:
    print(f"Warning: {len(missing)} samples had no ichorCNA results: {missing}")

# --- Build output DataFrame: samples × 5MB bins ---
result_df = pd.DataFrame(all_results).T
result_df = result_df.reindex(columns=ordered_bins)
result_df.index.name = "sample_id"

print(f"\nFinal shape: {result_df.shape}  (samples × 5MB bins)")
print(result_df.iloc[:3, :5])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_FILE)
print(f"\nSaved to {OUTPUT_FILE}")
