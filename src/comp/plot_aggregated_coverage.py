#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate coverage over gene sets (PAU and housekeeping genes) across a
panel-of-normals (PoN) and plot mean +/- SD profiles and per-locus distributions.

Usage:
    python src/comp/plot_aggregated_coverage.py \
        --metadata accessory_files/cris_metadata.csv \
        --pon_n 50 \
        --biomart_bed beds/biomart_10kb.bed \
        --pau_bed beds/pau.bed \
        --hk_bed beds/hk_genes.bed \
        --features_dir extracted_features/biomart_10kb \
        --output plots/aggregated_coverage_pon.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu
from pathlib import Path
import sys


def load_bed_file(bed_path):
    """Load BED file and return gene names."""
    print(f"\n📂 Loading BED file: {bed_path}")
    bed = pd.read_csv(bed_path, sep="\t", header=None)
    if bed.shape[1] >= 4:
        bed.columns = ["chrom", "start", "end", "gene_name"] + [f"col_{i}" for i in range(4, bed.shape[1])]
        gene_names = bed["gene_name"].unique()
    else:
        print(f"⚠️  Warning: BED file has only {bed.shape[1]} columns, expected at least 4")
        return []
    print(f"  Found {len(bed)} regions, {len(gene_names)} unique genes")
    return gene_names


def load_biomart_bed(bed_path):
    """Load biomart BED file to get gene name to index mapping."""
    print(f"\n📂 Loading biomart BED file: {bed_path}")
    bed = pd.read_csv(bed_path, sep="\t", header=None)
    if bed.shape[1] >= 4:
        bed.columns = ["chrom", "start", "end", "gene_name"] + [f"col_{i}" for i in range(4, bed.shape[1])]
    else:
        bed.columns = ["chrom", "start", "end"] + [f"col_{i}" for i in range(3, bed.shape[1])]
        bed["gene_name"] = None
    print(f"  Total regions: {len(bed)}, unique genes: {len(bed['gene_name'].unique())}")
    return bed


def find_gene_indices(biomart_bed, target_genes):
    """Find indices in biomart bed that match target genes."""
    gene_to_indices = {}
    for idx, row in biomart_bed.iterrows():
        gene = row["gene_name"]
        if gene not in gene_to_indices:
            gene_to_indices[gene] = []
        gene_to_indices[gene].append(idx)

    matched_indices = []
    matched_genes = []
    for gene in target_genes:
        if gene in gene_to_indices:
            matched_indices.extend(gene_to_indices[gene])
            matched_genes.append(gene)

    print(f"  Matched {len(matched_genes)}/{len(target_genes)} genes → {len(matched_indices)} regions")
    return np.array(matched_indices)


def load_coverage_array(features_dir, sample_id):
    """Load coverage array for a sample. Returns None if not found."""
    coverage_path = Path(features_dir) / f"{sample_id}.hg38.frag.tsv_coverage.npy"
    if not coverage_path.exists():
        alt_path = Path(features_dir) / f"{sample_id}_coverage.npy"
        if alt_path.exists():
            coverage_path = alt_path
        else:
            return None
    return np.load(coverage_path)


def normalize_coverage(coverage):
    """Normalize coverage to mean = 1."""
    mean_cov = np.mean(coverage)
    if mean_cov == 0:
        return coverage
    return coverage / mean_cov


def trim_to_window(arr, target_size):
    """Trim array to target_size around center."""
    if arr.shape[0] == target_size:
        return arr
    elif arr.shape[0] < target_size:
        return arr
    start = (arr.shape[0] - target_size) // 2
    return arr[start:start + target_size]


def plot_pon_coverage(pau_profiles, hk_profiles, pau_locus_all, hk_locus_all,
                      output_path, n_samples, n_pau_tss, n_hk_tss, window_size=10000):
    """Plot mean±SD coverage profiles and aggregated density distributions."""
    print(f"\n📈 Creating PoN coverage plots (n={n_samples} samples)...")

    from matplotlib.patches import Patch

    pau_color = (194 / 255, 106 / 255, 119 / 255)
    hk_color = (148 / 255, 203 / 255, 236 / 255)

    x = np.arange(-window_size // 2, window_size // 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left plot: Mean ± SD coverage profiles ----
    for profiles, label, color in [
        (hk_profiles, 'Housekeeping genes', hk_color),
        (pau_profiles, 'PAU genes', pau_color),
    ]:
        if len(profiles) == 0:
            continue
        stack = np.array(profiles)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        ax1.plot(x, mean, color=color, linewidth=2, label=f'{label}\n(mean ± SD, n={n_samples})', alpha=0.9)
        ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    feature_colors = {
        'GD': (93 / 255, 168 / 255, 153 / 255),
        'RCOV': (220 / 255, 205 / 255, 125 / 255),
        'MWH': (159 / 255, 74 / 255, 150 / 255),
    }
    feature_patches = [
        Patch(facecolor=feature_colors['GD'], label='Griffin Difference (GD)'),
        Patch(facecolor=feature_colors['RCOV'], label='Relative Coverage (RCOV)'),
        Patch(facecolor=feature_colors['MWH'], label='Max Wave Height (MWH)'),
    ]

    ax1.set_xlabel('Distance from TSS (bp)', fontsize=14)
    ax1.set_ylabel('Normalized coverage', fontsize=14)
    ax1.tick_params(labelsize=12)
    handles, _ = ax1.get_legend_handles_labels()
    legend1 = ax1.legend(handles=handles, loc='lower right', fontsize=11, framealpha=0.95)
    ax1.legend(handles=feature_patches, loc='lower left', fontsize=10, framealpha=0.95)
    ax1.add_artist(legend1)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax1.set_xlim(-window_size // 2, window_size // 2)
    ax1.set_xticks([-5000, -2500, 0, 2500, 5000])
    ax1.set_xticklabels(['-5kb', '-2.5kb', 'TSS', '+2.5kb', '+5kb'])

    # ---- Right plot: Density distributions with quartiles + significance ----
    density_sets = []
    if len(hk_locus_all) > 0:
        density_sets.append(('HK genes', hk_locus_all, hk_color, n_hk_tss))
    if len(pau_locus_all) > 0:
        density_sets.append(('PAU genes', pau_locus_all, pau_color, n_pau_tss))

    y_max = 0
    means = {}
    if density_sets:
        x_max = 5  # Cap visible range to focus on main distributions
        for label, data, color, n_tss in density_sets:
            kde = gaussian_kde(data, bw_method='scott')
            x_range = np.linspace(max(data.min(), 0), x_max, 500)
            density = kde(x_range)
            y_max = max(y_max, density.max())

            ax2.plot(x_range, density, color=color, linewidth=2, label=f'{label} (#TSS={n_tss})')
            ax2.fill_between(x_range, density, alpha=0.15, color=color)

            q25, q50, q75 = np.percentile(data, [25, 50, 75])
            for q_val, q_label in [(q25, 'Q1'), (q50, 'Median'), (q75, 'Q3')]:
                q_density = kde(q_val)[0]
                lw = 1.5 if q_label == 'Median' else 1.0
                ax2.vlines(q_val, 0, q_density, colors=color, linestyles='dashed', linewidth=lw, alpha=0.8)

            means[label] = np.mean(data)

        # Significance bracket between the two distributions
        if len(density_sets) == 2:
            _, data_a, color_a, _ = density_sets[0]
            _, data_b, color_b, _ = density_sets[1]
            stat, pval = mannwhitneyu(data_a, data_b, alternative='two-sided')

            if pval < 0.001:
                p_text = f'p < 0.001'
            else:
                p_text = f'p = {pval:.3f}'
            print(f"  Mann-Whitney U test: U={stat:.0f}, {p_text}")

            # Draw bracket between the two distribution means
            mean_a = means[density_sets[0][0]]
            mean_b = means[density_sets[1][0]]
            left, right = min(mean_a, mean_b), max(mean_a, mean_b)
            tick_h = y_max * 0.03
            bracket_y = y_max * 1.08

            ax2.plot([left, left], [bracket_y - tick_h, bracket_y],
                     color='black', linewidth=1.2, clip_on=False)
            ax2.plot([left, right], [bracket_y, bracket_y],
                     color='black', linewidth=1.2, clip_on=False)
            ax2.plot([right, right], [bracket_y - tick_h, bracket_y],
                     color='black', linewidth=1.2, clip_on=False)
            ax2.text((left + right) / 2, bracket_y * 1.01, p_text,
                     ha='center', va='bottom', fontsize=12, fontstyle='italic')

            y_max = bracket_y * 1.15

        ax2.legend(fontsize=11, framealpha=0.95)
    else:
        ax2.text(0.5, 0.5, 'No per-locus data available', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=14)

    ax2.set_xlabel('Mean coverage per TSS 10k bp window', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.set_xlim(left=0, right=5)
    ax2.set_ylim(bottom=0, top=y_max if y_max > 0 else None)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate coverage over gene sets across PoN healthy samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=str, default="accessory_files/cris_metadata.csv",
                        help="Path to metadata CSV with ID and disease columns")
    parser.add_argument("--pon_n", type=int, default=50,
                        help="Number of healthy samples to use (first N)")
    parser.add_argument("--biomart_bed", type=str, default="beds/biomart_10kb.bed")
    parser.add_argument("--pau_bed", type=str, default="beds/pau.bed")
    parser.add_argument("--hk_bed", type=str, default="beds/hk_genes.bed")
    parser.add_argument("--features_dir", type=str, default="extracted_features/biomart_10kb")
    parser.add_argument("--output", type=str, default="plots/aggregated_coverage_pon.png")
    parser.add_argument("--remove_indices", type=str,
                        default="accessory_files/final_tss_indices_to_remove_hg38.npy")
    parser.add_argument("--window_size", type=int, default=10000)
    args = parser.parse_args()

    print("=" * 90)
    print("AGGREGATED COVERAGE PROFILE ANALYSIS — PANEL OF NORMALS")
    print("=" * 90)

    # Load metadata and get PoN sample IDs (skip missing files, draw more from pool)
    metadata = pd.read_csv(args.metadata)
    healthy = metadata[metadata['disease'] == 'Healthy']
    all_healthy_ids = healthy['ID'].tolist()
    pon_ids = []
    for sid in all_healthy_ids:
        if len(pon_ids) >= args.pon_n:
            break
        cov_path = Path(args.features_dir) / f"{sid}.hg38.frag.tsv_coverage.npy"
        alt_path = Path(args.features_dir) / f"{sid}_coverage.npy"
        if cov_path.exists() or alt_path.exists():
            pon_ids.append(sid)
        else:
            print(f"  ⚠️  Skipping {sid}: no coverage file, selecting next healthy sample")
    print(f"PoN samples: {len(pon_ids)} (from {args.metadata})")
    print(f"Selected IDs: {pon_ids}")

    # Load BED files and find indices
    pau_genes = load_bed_file(args.pau_bed)
    hk_genes = load_bed_file(args.hk_bed)
    biomart_bed = load_biomart_bed(args.biomart_bed)

    print(f"\n🔍 Finding PAU gene indices...")
    pau_indices = find_gene_indices(biomart_bed, pau_genes)
    print(f"🔍 Finding HK gene indices...")
    hk_indices = find_gene_indices(biomart_bed, hk_genes)

    # Apply QC blacklist
    remove_path = Path(args.remove_indices)
    blacklisted_genes = set()
    if remove_path.exists():
        blacklist = set(np.load(str(remove_path)))
        pau_before, hk_before = len(pau_indices), len(hk_indices)
        pau_indices = np.array([i for i in pau_indices if i not in blacklist])
        hk_indices = np.array([i for i in hk_indices if i not in blacklist])
        print(f"\n🚫 QC blacklist ({len(blacklist)} indices): PAU {pau_before}→{len(pau_indices)}, HK {hk_before}→{len(hk_indices)}")
        blacklist_indices = np.load(str(remove_path))
        valid_bl = blacklist_indices[blacklist_indices < len(biomart_bed)]
        blacklisted_genes = set(biomart_bed.iloc[valid_bl]["gene_name"].unique())
    else:
        print(f"\n⚠️  Blacklist not found: {args.remove_indices}")

    # Process each sample
    pau_profiles = []
    hk_profiles = []
    pau_locus_all = []
    hk_locus_all = []
    n_loaded = 0

    for sample_id in pon_ids:
        coverage = load_coverage_array(args.features_dir, sample_id)
        if coverage is None:
            print(f"  ⚠️  Skipping {sample_id}: coverage file not found")
            continue

        n_loaded += 1

        # Aggregate and normalize coverage profiles
        if len(pau_indices) > 0:
            pau_agg = np.sum(coverage[pau_indices, :], axis=0)
            pau_agg = trim_to_window(pau_agg, args.window_size)
            pau_profiles.append(normalize_coverage(pau_agg))

        if len(hk_indices) > 0:
            hk_agg = np.sum(coverage[hk_indices, :], axis=0)
            hk_agg = trim_to_window(hk_agg, args.window_size)
            hk_profiles.append(normalize_coverage(hk_agg))

        # Load per-locus coverage from BIN.csv
        bin_path = Path(args.features_dir) / f"{sample_id}.hg38.frag.tsv_BIN.csv"
        if bin_path.exists():
            bin_df = pd.read_csv(bin_path)
            if 'mean_coverage' in bin_df.columns and 'gene_name' in bin_df.columns:
                bl_mask = bin_df['gene_name'].isin(blacklisted_genes)
                hk_mask = bin_df['gene_name'].isin(hk_genes) & ~bl_mask
                pau_mask = bin_df['gene_name'].isin(pau_genes) & ~bl_mask
                all_cov = bin_df.loc[hk_mask | pau_mask, 'mean_coverage'].dropna()
                min_thr = all_cov[all_cov > 0].quantile(0.01)
                cov_mask = bin_df['mean_coverage'] > min_thr
                hk_locus_all.extend(bin_df.loc[hk_mask & cov_mask, 'mean_coverage'].dropna().values)
                pau_locus_all.extend(bin_df.loc[pau_mask & cov_mask, 'mean_coverage'].dropna().values)

    print(f"\n📊 Loaded {n_loaded}/{len(pon_ids)} samples")
    print(f"   PAU profiles: {len(pau_profiles)}, HK profiles: {len(hk_profiles)}")
    print(f"   PAU loci pooled: {len(pau_locus_all)}, HK loci pooled: {len(hk_locus_all)}")

    pau_locus_all = np.array(pau_locus_all)
    hk_locus_all = np.array(hk_locus_all)

    # Plot
    plot_pon_coverage(
        pau_profiles, hk_profiles,
        pau_locus_all, hk_locus_all,
        args.output, n_loaded,
        n_pau_tss=len(pau_indices), n_hk_tss=len(hk_indices),
        window_size=args.window_size,
    )

    print("\n" + "=" * 90)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
