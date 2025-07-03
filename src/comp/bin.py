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

    return (shannon_entropy / normalization_factor) * -1  # Return the negative of the entropy to match the original definition


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
    fragment_length_distribution = {}
    for length in range(100, 220 + 1):
        fragment_length_distribution[length] = np.zeros(len(bed), dtype=np.float32)
    kmer_distribution = {kmer: np.zeros(len(bed), dtype=np.float32) for kmer in initialize_kmer_dictionary(3)}
    mds_values = np.zeros(len(bed), dtype=float)
    gc_content = np.zeros(len(bed), dtype=float)

    for locus in bed.itertuples():
        bin_index = locus.Index
        chrom = locus.chrom
        start = locus.start
        end = locus.end

        if chrom is not None and chrom not in bam.references:
            continue

        filtered_alignments = util.get_filtered_alignments(bam, args, chrom=chrom, start=start, end=end)
        if filtered_alignments is None:
            print(f"No alignments found for {chrom}:{start}-{end}")
            continue

        num_short_fragments = 0  # Defined to be lengths [100, 150]
        num_long_fragments = 0  # Defined to be lenghts [151, 220]
        for read in filtered_alignments:
            # Process each fragment once using the first read in the pair
            if not read.is_read1:
                continue

            tlen = abs(read.template_length)

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
            if args.gc:
                frag_gc_content = gc_module.get_gc_content(ref_seq[3:-3])
                read_value = gc_matrix.get(frag_gc_content, {}).get(tlen, 0)

            absolute_fragment_counts[bin_index] += read_value

            if tlen >= 100 and tlen <= 150:
                num_short_fragments += read_value
            elif tlen >= 151 and tlen <= 220:
                num_long_fragments += read_value

            if tlen in fragment_length_distribution:
                fragment_length_distribution[tlen][bin_index] += read_value

            s3_motif = ref_seq[3:6]
            kmer_distribution[s3_motif][bin_index] += read_value

        absolute_short_fragments[bin_index] = num_short_fragments
        absolute_long_fragments[bin_index] = num_long_fragments
        fslr_values[bin_index] = np.log2(max(num_short_fragments, 1) / max(num_long_fragments, 1))

        # Calculate GC content for the whole region
        if chrom is not None:
            ref_seq_full = ref_fasta.fetch(chrom, start, end).upper()
            gc_content[bin_index] = gc_module.get_gc_content(ref_seq_full)

        # Calculate MDS (Motif Diversity Score) as
        # MDS = sum of kmer 1-64 -Pi*log(Pi)/log(64), where Pi is the frequency of kmer i in the bin
        # Defined to be in [0, 1]
        MDS = 0
        number_of_kmers = len(list(kmer_distribution.keys()))
        for _, counts in kmer_distribution.items():
            if counts[bin_index] > 0:
                # Should Pi be relative to the total number of reads or absolute?
                Pi = counts[bin_index] / absolute_fragment_counts[bin_index]
                MDS += Pi * np.log2(Pi) / np.log2(number_of_kmers)
        mds_values[bin_index] = MDS

    relative_read_counts = absolute_fragment_counts / np.sum(absolute_fragment_counts) if np.sum(absolute_fragment_counts) > 0 else np.zeros_like(absolute_fragment_counts)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        {
            "chrom": bed["chrom"],
            "start": bed["start"],
            "end": bed["end"],
            "absolute_fragment_counts": absolute_fragment_counts,
            "relative_read_counts": relative_read_counts,
            "absolute_short_fragments": absolute_short_fragments,
            "absolute_long_fragments": absolute_long_fragments,
            "fslr": fslr_values,
            "gc_content": gc_content,
            "mds": mds_values,
        }
    )
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(fragment_length_distribution)], axis=1, ignore_index=False)
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(kmer_distribution)], axis=1, ignore_index=False)

    # Save the results to a CSV file
    results_df.to_csv(output_save_file, index=False)
    print(f"BIN features saved to {output_save_file}")
    bam.close()
    ref_fasta.close()
