import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm


def get_filtered_alignments(bam, args, chrom=None, start=None, end=None):
    """
    Fetch alignments from a BAM file with optional filtering by chromosome and position.
    """

    # From GCparagon:
    # 4 (0x4): The read is unmapped.
    # 8 (0x8): The read's mate is unmapped.
    # 256 (0x100): The alignment is not primary.
    #              (This filters out secondary alignments, which can occur when a read maps to multiple locations).
    # 512 (0x200): The read failed quality control checks.
    # 1024 (0x400): The read is a PCR or optical duplicate.
    # 2048 (0x800): The alignment is supplementary.
    #              (This is another type of non-primary alignment, often used for chimeric reads).
    exclude_flags = np.uint32(3852)  # = 256 + 2048 + 512 + 1024 + 4 + 8
    exclude_flags_binary = bin(exclude_flags)

    # Fetch reads overlapping the region, or genome-wide if no region is specified
    try:
        filtered_alignments = filter(
            lambda a: a.is_paired and bin(~np.uint32(a.flag) & exclude_flags) == exclude_flags_binary and a.mapping_quality >= 5 and (args.min_frag_len <= abs(a.template_length) <= args.max_frag_len) and not a.is_unmapped,
            bam.fetch(
                contig=chrom if chrom is not None else None,
                start=start if start is not None else None,
                stop=end if end is not None else None,
            ),
        )
    except Exception as e:
        print(f"Something went wrong while fetching reads for locus {chrom}:{start}-{end}")
        print(e)
        return None

    return filtered_alignments


def aggregate_tfbs(results_dir="./extracted_features"):
    """Aggregate TFBS data from multiple directories into a single directory."""

    tfbs_dirs = glob(f"{results_dir}/*Top1000sites")

    print(len(tfbs_dirs), "TFBS directories found")

    # Open the first folder to get all the dfs
    aggregate_dfs = {}
    files = glob(f"{tfbs_dirs[0]}/*.csv")
    for file in files:
        df = pd.read_csv(file, index_col=0)
        sample_name = file.split("/")[-1]
        aggregate_dfs[sample_name] = df

    # Loop through all the directories and aggregate the dataframes
    for tfbs_dir in tqdm(tfbs_dirs[1:], desc="Aggregating TFBS data"):
        files = glob(f"{tfbs_dir}/*.csv")
        for file in files:
            df = pd.read_csv(file, index_col=0)
            sample_name = file.split("/")[-1]
            if sample_name in aggregate_dfs:
                aggregate_dfs[sample_name].add(df)
            else:
                print("File not found in aggregate_dfs, perhaps now doesn't include all TFBS:", file)
                aggregate_dfs[sample_name] = df

    # Write the aggregated dataframes to a new directory
    output_dir = f"{results_dir}/aggregated_tfbs"
    os.makedirs(output_dir, exist_ok=True)
    for sample_name, df in aggregate_dfs.items():
        output_file = f"{output_dir}/{sample_name}"
        df.to_csv(output_file)
        print(f"Wrote {sample_name} to {output_file}")


def calculate_rcov(coverage, region_type="promoter") -> float:
    """Calculate relative coverage for each region in the coverage matrix

    For the promoter region, the coverage is calculated as the mean of the coverage at -150 to +50 bp
    normalized by the mean of the coverage at -2000 to -1000 bp and +1000 to +2000 bp.

    For junctions, the coverage is calculated as the mean of the coverage at the junction -300 to -100 bp
    normalized by the mean of the coverage at -2000 to -1000 bp and +1000 to +2000 bp.

    Parameters
    ----------
    cov_matrix : np.array of shape (n_samples, n_regions, n_bp)

    Returns
    -------
    np.array of shape (n_samples, n_regions)
    """
    midpoint = len(coverage) // 2
    region_coverage = np.mean(coverage[midpoint - 150 : midpoint + 50]) if region_type == "promoter" else np.mean(coverage[midpoint - 300 : midpoint - 100])

    # Calculate normalization factor
    up_flank = coverage[midpoint - 2000 : midpoint - 1000]
    down_flank = coverage[midpoint + 1000 : midpoint + 2000]
    norm_factor = np.mean(np.concatenate((up_flank, down_flank)))

    if norm_factor == 0:
        return 0

    return region_coverage / norm_factor


def desarkar_features(coverage) -> tuple[float, float, float]:
    """Calculate features from de Sarkar et al. (2023) https://doi.org/10.1158/2159-8290.CD-22-0692

    a) “mean central coverage,” the mean coverage between minus 30 and 30 bp relative to the site center,

    b) “mean window coverage,” the mean coverage between minus 990 and 990 bp relative to the site center, and

    c) “max wave height,” the absolute difference between the minimum coverage within the window from
    minus 120 to 30 bp and maximum coverage in the window from 31 to 195 bp relative to the TSS.

    Parameters
    ----------
    coverage : np.array of shape (n_bp)

    Returns
    -------
    tuple[float, float, float]
    """
    midpoint = len(coverage) // 2
    mean_central_coverage = np.mean(coverage[midpoint - 30 : midpoint + 30])
    mean_window_coverage = np.mean(coverage[midpoint - 990 : midpoint + 990])
    max_wave_height = np.abs(np.min(coverage[midpoint - 120 : midpoint + 30]) - np.max(coverage[midpoint + 31 : midpoint + 195]))

    return mean_central_coverage, mean_window_coverage, max_wave_height


def ulz_atac_features(coverage) -> float:
    # Only retain the +- 1000 bp around the TSS
    mid = len(coverage) // 2
    coverage = coverage[mid - 1000 : mid + 1000]

    # Normalize the profile
    coverage = coverage / np.mean(coverage)

    # Get the low signal
    low_signal = savgol_filter(coverage, window_length=1001, polyorder=3)

    # Get the high signal
    high_signal = savgol_filter(coverage, window_length=51, polyorder=3)

    # Adjust high signal
    high_signal = high_signal / low_signal

    # Get the amplitude
    return max(high_signal) - min(high_signal)
