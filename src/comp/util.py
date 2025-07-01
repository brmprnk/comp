import os
from glob import glob

import numpy as np
import pandas as pd
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
                start=(start - args.max_frag_len) if start is not None else None,
                stop=(end + args.max_frag_len) if end is not None else None,
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
