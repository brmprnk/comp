import math
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import src.comp.gc as gc_module
import src.comp.util as util
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

    bed = pd.DataFrame({"chrom": [None], "start": [None], "end": [None]}) if bed_path is None or not Path(bed_path).exists() else pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])

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

        filtered_alignments = util.get_filtered_alignments(bam, args, chrom=chrom, start=start, end=end)
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

            # Define the window to fetch from the reference genome
            fetch_start = frag_start - 3
            fetch_end = frag_end + 3

            ref_seq = ref_fasta.fetch(read.reference_name, fetch_start, fetch_end).upper()

            # Check if we got the expected length; if not, it's at a contig boundary
            if len(ref_seq) != (fetch_end - fetch_start):
                continue

            # If the fragment is on the reverse strand, we need to reverse complement the sequence
            if read.is_reverse:
                ref_seq = reverse_complement(ref_seq)

            read_value = 1
            frag_gc_content = gc_module.get_gc_content(ref_seq[3:-3])
            if args.gc:
                read_value = gc_matrix.get(str(int(frag_gc_content)), {}).get(tlen, 0)

            if tlen >= 100 and tlen <= 220:  ## ADD GCFIX READ VALUE
                coverage[frag_start - start : frag_end - start] += read_value
            if tlen >= 100 and tlen <= 150:
                coverage_short[frag_start - start : frag_end - start] += read_value
            elif tlen >= 151 and tlen <= 220:
                coverage_long[frag_start - start : frag_end - start] += read_value

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
            else:
                print(f"Warning: {s3_motif} not found in kmer distribution, skipping")

        fslr_values[bin_index] = np.log2(max(absolute_short_fragments[bin_index], 1) / max(absolute_long_fragments[bin_index], 1))

        # calculate mds using only the kmer distribution in this bin
        mds_values[bin_index] = calculate_mds({kmer: count[bin_index] for kmer, count in kmer_distribution.items()})

        # Calculate the mean coverage in the bin
        mean_coverage_values[bin_index] = np.sum(coverage) / len(coverage) if len(coverage) > 0 else 0
        mean_coverage_short[bin_index] = np.sum(coverage_short) / len(coverage_short) if len(coverage_short) > 0 else 0
        mean_coverage_long[bin_index] = np.sum(coverage_long) / len(coverage_long) if len(coverage_long) > 0 else 0
        fslr_coverage_values[bin_index] = np.log2(max(mean_coverage_short[bin_index], 1) / max(mean_coverage_long[bin_index], 1))

        # Calculate the coverage spread in the bin
        number_of_bases_covered = np.sum(coverage > 0)
        coverage_spread[bin_index] = number_of_bases_covered / len(coverage) if len(coverage) > 0 else 0

        # Calculate the mean gc content of all the mapped fragments
        mean_fragment_gc_content[bin_index] = np.mean(gc_content_per_fragment) if gc_content_per_fragment else 0
        mean_fragment_gc_content_short[bin_index] = np.mean(gc_content_per_fragment_short) if gc_content_per_fragment_short else 0
        mean_fragment_gc_content_long[bin_index] = np.mean(gc_content_per_fragment_long) if gc_content_per_fragment_long else 0

        # Calculate TSS Features
        normalized_coverage = coverage / np.mean(coverage) if np.mean(coverage) > 0 else coverage
        rcov[bin_index] = util.calculate_rcov(normalized_coverage, region_type="promoter")
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

    relative_read_counts = absolute_fragment_counts / np.sum(absolute_fragment_counts) if np.sum(absolute_fragment_counts) > 0 else np.zeros_like(absolute_fragment_counts)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        {
            "chrom": bed["chrom"],
            "start": bed["start"],
            "end": bed["end"],
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

    # # Save kmer length distribution
    # np.save(output_path / (Path(bam_path).stem + "_kmer_length_distribution.npy"), kmer_length_distribution)
    # print(f"Kmer length distribution saved to {output_path / (Path(bam_path).stem + '_kmer_length_distribution.npy')}")

    # Save GC length distribution
    np.save(output_path / (Path(bam_path).stem + "_gc_bin_length_distribution.npy"), gc_bin_length_distribution)

    # Save GC fragment length distribution
    np.save(output_path / (Path(bam_path).stem + "_gc_fragment_length_distribution.npy"), gc_fragment_length_distribution)

    bam.close()
    ref_fasta.close()
