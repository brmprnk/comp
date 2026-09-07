import math
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import src.comp.gc as gc_module
import src.comp.util as util
import matplotlib.pyplot as plt
from src.comp.em import initialize_kmer_dictionary, reverse_complement


def calculate_mds(kmer_counts):
    """
    Calculates the Motif Diversity Score (MDS) for a dictionary of k-mer counts.

    The MDS is the normalized Shannon entropy, as described in the paper
      https://aacrjournals.org/cancerdiscovery/article/10/5/664/2457/Plasma-DNA-End-Motif-Profiling-as-a-Fragmentomic.
    MDS = (Σ -P_i * log(P_i)) / log(N)
    where P_i is the frequency of motif i, and N is the total number of possible motifs.

    Args:
        kmer_counts (dict): A dictionary where keys are k-mers (str) and
                            values are their counts (int or float).

    Returns:
        float: The Motif Diversity Score, bounded between 0 and 1.
    """
    # Get the total number of possible motifs from the input dictionary size.
    # For 3-mers, this would be 4^3 = 64.
    num_possible_motifs = len(kmer_counts)
    if num_possible_motifs == 0:
        return 0.0

    # Calculate the total number of observed k-mers
    total_counts = sum(kmer_counts.values())
    if total_counts == 0:
        return 0.0

    # Calculate Shannon entropy part: Σ -P_i * log(P_i)
    shannon_entropy = 0.0
    for kmer in kmer_counts:
        count = kmer_counts[kmer]
        if count > 0:
            # Calculate the frequency (probability) of the k-mer
            p_i = count / total_counts
            # Add to the entropy sum.
            shannon_entropy += p_i * math.log(p_i)

    # Normalize the entropy by log(N) where N is the number of possible motifs.
    # The base of the logarithm doesn't matter as long as it's consistent.
    # We use math.log (natural log) here.
    # The normalization factor is log(N).
    normalization_factor = math.log(num_possible_motifs)

    # Handle the edge case where there is only one possible motif (log(1)=0)
    if normalization_factor == 0:
        return 1.0  # By definition, if there's only one outcome, diversity is maximal (or minimal, depending on definition, but 1 is common for normalized entropy)

    return shannon_entropy / normalization_factor


def calculate_bin(bam_path, output_path, bed_path, gc_file, args):
    """
    Calculate BIN features for a given BAM file and BED file.
    Bin features are defined as:
    - Absolute number of reads in each bin
    - Relative number of reads in each bin
    - FSLR (log2(short/long) ratio) in each bin
    - fragment length distribution in each bin
    - GC content in each bin
    - MDS (Motif Diversity Score) in each bin
    - Each 3-mer frequency in each bin
    """
    # Placeholder for actual BIN feature extraction logic
    print(f"Calculating BIN features for {bam_path} and saving to {output_path}")
    output_save_file = output_path / (Path(bam_path).stem + "_NOGC_BIN.csv") if not args.gc else output_path / (Path(bam_path).stem + "_BIN.csv")
    print(f"Output will be saved to {output_save_file}")
    
    # Check if aggregate coverage already exists and is valid
    if hasattr(args, 'aggregate') and args.aggregate:
        coverage_save_path = output_path / (Path(bam_path).stem + "_coverage_aggregate.npy")
        if coverage_save_path.exists():
            try:
                existing_coverage = np.load(coverage_save_path)
                # Check if coverage is not all zeros and has reasonable size
                if existing_coverage.size > 0 and np.sum(existing_coverage) > 0:
                    print(f"✓ Valid aggregate coverage already exists at {coverage_save_path}")
                    print(f"  Coverage sum: {np.sum(existing_coverage):.2f}, Mean: {np.mean(existing_coverage):.4f}")
                    print(f"  Skipping calculation for {bam_path}")
                    return
                else:
                    print(f"⚠ Existing coverage at {coverage_save_path} is empty or all zeros")
                    print(f"  Will recalculate...")
            except Exception as e:
                print(f"⚠ Error loading existing coverage from {coverage_save_path}: {e}")
                print(f"  Will recalculate...")
        else:
            print(f"No existing coverage found at {coverage_save_path}")

    if args.gc:
        gc_matrix = gc_module.load_gc_matrix(gc_file, args.min_frag_len, args.max_frag_len)
        print(f"Loaded GC matrix from {gc_file}")

    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
        ref_fasta = pysam.FastaFile(args.fasta)
    except Exception as e:
        print(f"Something went wrong while opening the BAM file: {bam_path}")
        print(e)
        return

    if bed_path is None or not Path(bed_path).exists():
        bed = pd.DataFrame({"chrom": [None], "start": [None], "end": [None], "gene_name": [None], "region_type": [None]})
    else:
        # Read the bed file and check how many columns it has
        bed = pd.read_csv(bed_path, sep="\t", header=None)
        num_cols = bed.shape[1]
        
        if num_cols >= 5:
            # If 5 or more columns, use first 5 as chrom, start, end, gene_name, region_type
            bed = bed.iloc[:, :5]
            bed.columns = ["chrom", "start", "end", "gene_name", "region_type"]
        elif num_cols == 4:
            # If 4 columns, use first 4 as chrom, start, end, gene_name
            bed = bed.iloc[:, :4]
            bed.columns = ["chrom", "start", "end", "gene_name"]
            bed["region_type"] = None
        else:
            # If 3 or fewer columns, use as chrom, start, end
            bed = bed.iloc[:, :3]
            bed.columns = ["chrom", "start", "end"]
            bed["gene_name"] = None
            bed["region_type"] = None

    # Extend single-base regions to a window around that location
    # Get window size from args, default to 10000 (4999 upstream + 5000 downstream)
    window_size = getattr(args, 'window_size', 10000)
    upstream = (window_size - 1) // 2  # 4999 for window_size=10000
    downstream = window_size - upstream  # 5001 for window_size=10000, but we don't add the base itself
    
    # Check if regions are single-base (end - start == 1)
    region_sizes = bed["end"] - bed["start"]
    if len(region_sizes) > 0 and region_sizes.iloc[0] == 1:
        print(f"Detected single-base regions. Extending to {window_size}bp windows ({upstream}bp upstream and {downstream-1}bp downstream)")
        # Extend around the start position
        # For a position at 'start', we want [start - upstream, start + downstream)
        # This gives us exactly 'window_size' bases
        # CRITICAL: Make a copy to avoid SettingWithCopyWarning and ensure proper DataFrame modification
        bed = bed.copy()
        bed["end"] = bed["start"] + downstream
        bed["start"] = bed["start"] - upstream
        # Ensure start doesn't go negative
        bed.loc[bed["start"] < 0, "start"] = 0
        # Recalculate region sizes after extension
        region_sizes = bed["end"] - bed["start"]
    
    # Determine the region size (assumes all regions are the same size after potential extension)
    region_size = region_sizes.iloc[0] if len(region_sizes) > 0 else 0
    
    # Sanity check: ensure all region sizes are positive
    if len(region_sizes) > 0 and (region_sizes <= 0).any():
        print("WARNING: Some regions have non-positive sizes after extension:")
        problematic = bed[region_sizes <= 0]
        print(problematic)
        # Filter out problematic regions
        bed = bed[region_sizes > 0].copy()
        # CRITICAL: Reset index after filtering to ensure indices are 0, 1, 2, ... len(bed)-1
        # This is necessary because itertuples() uses the index, and arrays are sized by len(bed)
        bed = bed.reset_index(drop=True)
        region_sizes = bed["end"] - bed["start"]
        region_size = region_sizes.iloc[0] if len(region_sizes) > 0 else 0
        print(f"Filtered to {len(bed)} valid regions with reset indices")
    
    # Initialize coverage array for all loci (if aggregate, we'll sum; otherwise, we'll store per-locus)
    if hasattr(args, 'aggregate') and args.aggregate:
        # For aggregate mode, create a 1D array to accumulate coverage
        coverage_array = np.zeros(np.abs(region_size), dtype=np.float32)
    elif hasattr(args, 'coverage') and args.coverage:
        # For coverage mode, create a 2D array: (n_loci, region_size)
        coverage_array = np.zeros((len(bed), np.abs(region_size)), dtype=np.float32)
    else:
        # For non-aggregate mode, don't store coverage
        coverage_array = None

    absolute_fragment_counts = np.zeros(len(bed), dtype=np.float32)
    # Relative read counts will be calculated at the end
    absolute_short_fragments = np.zeros(len(bed), dtype=np.float32)
    absolute_long_fragments = np.zeros(len(bed), dtype=np.float32)
    fslr_values = np.zeros(len(bed), dtype=float)
    mean_coverage_values = np.zeros(len(bed), dtype=float)
    mean_coverage_short = np.zeros(len(bed), dtype=float)
    mean_coverage_long = np.zeros(len(bed), dtype=float)
    fslr_coverage_values = np.zeros(len(bed), dtype=float)
    coverage_spread = np.zeros(len(bed), dtype=float)
    fragment_length_distribution = {}
    for length in range(100, 220 + 1):
        fragment_length_distribution[length] = np.zeros(len(bed), dtype=np.float32)
    kmer_distribution = {kmer: np.zeros(len(bed), dtype=np.float32) for kmer in initialize_kmer_dictionary(3)}
    bg_kmer_distribution = {f"bg_{kmer}": np.zeros(len(bed), dtype=np.float32) for kmer in initialize_kmer_dictionary(3)}  # background kmers
    gc_bin_length_distribution = np.zeros((101, 121), dtype=np.float32)  # 100 GC content bins for fragments of lengths 100-220
    gc_fragment_length_distribution = np.zeros((101, 121), dtype=np.float32)  # 100 GC content bins for fragments of lengths 100-220
    # kmer_length_distribution = np.zeros((len(list(initialize_kmer_dictionary(3))), len(bed), 121), dtype=np.float32)
    mds_values = np.zeros(len(bed), dtype=float)
    gc_content = np.zeros(len(bed), dtype=float)
    mean_fragment_gc_content = np.zeros(len(bed), dtype=float)
    mean_fragment_gc_content_short = np.zeros(len(bed), dtype=float)
    mean_fragment_gc_content_long = np.zeros(len(bed), dtype=float)
    rcov = np.zeros(len(bed), dtype=float)
    griffin_diff = np.zeros(len(bed), dtype=float)
    mwh = np.zeros(len(bed), dtype=float)

    background_kmer_counts = None
    for locus in bed.itertuples():
        bin_index = locus.Index
        chrom = locus.chrom
        start = locus.start
        end = locus.end
        coverage = np.zeros(end - start)
        coverage_short = np.zeros(end - start)
        coverage_long = np.zeros(end - start)
        gc_content_per_fragment = []
        gc_content_per_fragment_short = []
        gc_content_per_fragment_long = []

        if chrom is not None and chrom not in bam.references:
            continue

        # Calculate GC content for the whole region
        background_kmer_counts = None
        if chrom is not None:
            ref_seq_full = ref_fasta.fetch(chrom, start, end).upper()
            gc_content[bin_index] = gc_module.get_gc_content(ref_seq_full)
            background_kmer_counts = gc_module.count_kmers(ref_seq_full, k=3)
            # Add 'bg_' prefix to the kmer counts for background
            for kmer, count in background_kmer_counts.items():
                bg_kmer_name = f"bg_{kmer}"
                if bg_kmer_name in bg_kmer_distribution:
                    bg_kmer_distribution[bg_kmer_name][bin_index] = count

        # Fetch alignments with buffer to capture fragments that overlap the region
        # Use max fragment length as buffer to get fragments that start outside but overlap
        max_frag_len = getattr(args, 'max_frag_len', 220)
        alignment_fetch_start = max(0, start - max_frag_len)
        alignment_fetch_end = end + max_frag_len
        
        filtered_alignments = util.get_filtered_alignments(bam, args, chrom=chrom, start=alignment_fetch_start, end=alignment_fetch_end)
        if filtered_alignments is None:
            print(f"No alignments found for {chrom}:{start}-{end}")
            continue

        for read in filtered_alignments:
            # Process each fragment once using the first read in the pair
            if not read.is_read1 or read.mapping_quality <= args.mapq:
                continue

            tlen = abs(read.template_length)  # Template length is the length of the fragment

            # Determine fragment coordinates and the extended region to fetch
            if not read.is_reverse:  # Fragment is on the forward strand
                frag_start = read.reference_start
                frag_end = frag_start + tlen
            else:  # Fragment is on the reverse strand
                frag_end = read.reference_end
                frag_start = frag_end - tlen

            # Define the window to fetch from the reference genome for motif analysis
            motif_fetch_start = frag_start - 3
            motif_fetch_end = frag_end + 3

            ref_seq = ref_fasta.fetch(read.reference_name, motif_fetch_start, motif_fetch_end).upper()

            # Check if we got the expected length; if not, it's at a contig boundary
            if len(ref_seq) != (motif_fetch_end - motif_fetch_start):
                continue

            # If the fragment is on the reverse strand, we need to reverse complement the sequence
            if read.is_reverse:
                ref_seq = reverse_complement(ref_seq)

            read_value = 1
            frag_gc_content = gc_module.get_gc_content(ref_seq[3:-3])
            if args.gc:
                read_value = gc_matrix.get(str(int(frag_gc_content)), {}).get(tlen, 0)

            if tlen >= 100 and tlen <= 220:  ## ADD GCFIX READ VALUE
                # Calculate overlap between fragment and region, clipping to region boundaries
                cov_start = max(0, frag_start - start)
                cov_end = min(end - start, frag_end - start)
                if cov_start < cov_end:  # Only add if there's actual overlap
                    coverage[cov_start:cov_end] += read_value
            if tlen >= 100 and tlen <= 150:
                cov_start = max(0, frag_start - start)
                cov_end = min(end - start, frag_end - start)
                if cov_start < cov_end:
                    coverage_short[cov_start:cov_end] += read_value
            elif tlen >= 151 and tlen <= 220:
                cov_start = max(0, frag_start - start)
                cov_end = min(end - start, frag_end - start)
                if cov_start < cov_end:
                    coverage_long[cov_start:cov_end] += read_value

            if tlen >= 100 and tlen <= 220:
                gc_content_per_fragment.append(frag_gc_content * read_value)
                gc_fragment_length_distribution[min(int(frag_gc_content * read_value), 100)][tlen - 100] += read_value
                gc_bin_length_distribution[int(gc_content[bin_index])][tlen - 100] += read_value
            if tlen >= 100 and tlen <= 150:
                gc_content_per_fragment_short.append(frag_gc_content * read_value)
            elif tlen >= 151 and tlen <= 220:
                gc_content_per_fragment_long.append(frag_gc_content * read_value)


            if tlen >= 100 and tlen <= 220:
                absolute_fragment_counts[bin_index] += read_value
            if tlen >= 100 and tlen <= 150:
                absolute_short_fragments[bin_index] += read_value
            elif tlen >= 151 and tlen <= 220:
                absolute_long_fragments[bin_index] += read_value

            if tlen in fragment_length_distribution:
                fragment_length_distribution[tlen][bin_index] += read_value

            s3_motif = ref_seq[3:6]
            if s3_motif in kmer_distribution:
                kmer_distribution[s3_motif][bin_index] += read_value
                # s3_motif_pos = list(initialize_kmer_dictionary(3).keys()).index(s3_motif)
                # kmer_length_distribution[s3_motif_pos][bin_index][tlen - 100] += read_value


        fslr_values[bin_index] = np.log2(max(absolute_short_fragments[bin_index], 1) / max(absolute_long_fragments[bin_index], 1))

        # calculate mds using only the kmer distribution in this bin
        mds_values[bin_index] = calculate_mds({kmer: count[bin_index] for kmer, count in kmer_distribution.items()})

        # Calculate the mean coverage in the bin
        mean_coverage_values[bin_index] = np.sum(coverage) / len(coverage) if len(coverage) > 0 else 0
        mean_coverage_short[bin_index] = np.sum(coverage_short) / len(coverage_short) if len(coverage_short) > 0 else 0
        mean_coverage_long[bin_index] = np.sum(coverage_long) / len(coverage_long) if len(coverage_long) > 0 else 0
        if mean_coverage_short[bin_index] == 0 or mean_coverage_long[bin_index] == 0:
            fslr_coverage_values[bin_index] = 0
        else:
            fslr_coverage_values[bin_index] = np.log2(mean_coverage_short[bin_index] / mean_coverage_long[bin_index])


        # Calculate the coverage spread in the bin
        number_of_bases_covered = np.sum(coverage > 0)
        coverage_spread[bin_index] = number_of_bases_covered / len(coverage) if len(coverage) > 0 else 0

        # Calculate the mean gc content of all the mapped fragments
        mean_fragment_gc_content[bin_index] = np.mean(gc_content_per_fragment) if gc_content_per_fragment else 0
        mean_fragment_gc_content_short[bin_index] = np.mean(gc_content_per_fragment_short) if gc_content_per_fragment_short else 0
        mean_fragment_gc_content_long[bin_index] = np.mean(gc_content_per_fragment_long) if gc_content_per_fragment_long else 0

        # Calculate TSS Features
        normalized_coverage = coverage / np.mean(coverage) if np.mean(coverage) > 0 else coverage
        region_type = bed.loc[bin_index, "region_type"] if "region_type" in bed.columns and pd.notna(bed.loc[bin_index, "region_type"]) else "promoter"
        rcov[bin_index] = util.calculate_rcov(normalized_coverage, region_type=region_type)
        _, window_coverage, mwh[bin_index] = util.desarkar_features(normalized_coverage)
        # Extra feature based on difference between mean coverage inside and outside the 1000bp window
        midpoint = int(len(coverage) / 2)
        mean_outside_coverage = np.mean(
            np.r_[
                normalized_coverage[: midpoint - 990],
                normalized_coverage[midpoint + 990 :],
            ]
        )
        griffin_diff[bin_index] = mean_outside_coverage - window_coverage

        # Store coverage array
        if hasattr(args, 'aggregate') and args.aggregate:
            # Aggregate mode: sum the coverage across all loci
            coverage_array += coverage
        elif hasattr(args, 'coverage') and args.coverage:
            # Coverage mode: store each locus coverage in the 2D array
            coverage_array[bin_index, :] = coverage
        else:
            # Non-aggregate mode: don't store coverage
            pass

    relative_read_counts = absolute_fragment_counts / np.sum(absolute_fragment_counts) if np.sum(absolute_fragment_counts) > 0 else np.zeros_like(absolute_fragment_counts)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        {
            "chrom": bed["chrom"],
            "start": bed["start"],
            "end": bed["end"],
            "gene_name": bed["gene_name"],
            "absolute_fragment_counts": absolute_fragment_counts,
            "relative_fragment_counts": relative_read_counts,
            "absolute_short_fragments": absolute_short_fragments,
            "absolute_long_fragments": absolute_long_fragments,
            "mean_coverage": mean_coverage_values,
            "mean_coverage_short": mean_coverage_short,
            "mean_coverage_long": mean_coverage_long,
            "coverage_spread": coverage_spread,
            "fslr": fslr_values,
            "fslr_coverage": fslr_coverage_values,
            "rcov": rcov,
            "griffin_diff": griffin_diff,
            "mwh": mwh,
            "gc_content": gc_content,
            "mean_fragment_gc_content": mean_fragment_gc_content,
            "mean_fragment_gc_content_short": mean_fragment_gc_content_short,
            "mean_fragment_gc_content_long": mean_fragment_gc_content_long,
            "mds": mds_values,
        }
    )
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(fragment_length_distribution)], axis=1, ignore_index=False)
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(kmer_distribution)], axis=1, ignore_index=False)
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(bg_kmer_distribution)], axis=1, ignore_index=False) if background_kmer_counts else results_df

    # Save the results to a CSV file
    results_df.to_csv(output_save_file, index=False)
    print(f"BIN features saved to {output_save_file}")

    # Save coverage array
    if hasattr(args, 'aggregate') and args.aggregate:
        # Save aggregated (summed) coverage as a 1D array
        coverage_save_path = output_path / (Path(bam_path).stem + "_coverage_aggregate.npy")
        np.save(coverage_save_path, coverage_array)
        print(f"Aggregated coverage array saved to {coverage_save_path}")
    elif hasattr(args, 'coverage') and args.coverage:
        # Save full coverage matrix (one row per locus)
        coverage_save_path = output_path / (Path(bam_path).stem + "_coverage.npy")
        np.save(coverage_save_path, coverage_array)
        print(f"Coverage array (shape: {coverage_array.shape}) saved to {coverage_save_path}")
    else:
        pass

    # # Save kmer length distribution
    # np.save(output_path / (Path(bam_path).stem + "_kmer_length_distribution.npy"), kmer_length_distribution)
    # print(f"Kmer length distribution saved to {output_path / (Path(bam_path).stem + '_kmer_length_distribution.npy')}")

    # Save GC length distribution
    np.save(output_path / (Path(bam_path).stem + "_gc_bin_length_distribution.npy"), gc_bin_length_distribution)

    # Save GC fragment length distribution
    np.save(output_path / (Path(bam_path).stem + "_gc_fragment_length_distribution.npy"), gc_fragment_length_distribution)

    bam.close()
    ref_fasta.close()
