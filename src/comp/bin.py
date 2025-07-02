from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import src.comp.gc as gc_module
import src.comp.util as util
from src.comp.em import initialize_kmer_dictionary, reverse_complement


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
    output_save_file = output_path / (Path(bam_path).stem + "_BIN.csv")
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

    absolute_fragment_counts = np.zeros(len(bed), dtype=int)
    # Relative read counts will be calculated at the end
    absolute_short_fragments = np.zeros(len(bed), dtype=int)
    absolute_long_fragments = np.zeros(len(bed), dtype=int)
    fslr_values = np.zeros(len(bed), dtype=float)
    fragment_length_distribution = {}
    # For args.min_frag_len to args.max_frag_len, fill with zeros
    for length in range(args.min_frag_len, args.max_frag_len + 1):
        fragment_length_distribution[length] = np.zeros(len(bed), dtype=int)
    kmer_distribution = {kmer: np.zeros(len(bed), dtype=int) for kmer in initialize_kmer_dictionary(3)}
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

            fragment_length_distribution[tlen][bin_index] += read_value

            s3_motif = ref_seq[3:6]
            kmer_distribution[s3_motif][bin_index] += read_value

        absolute_short_fragments[bin_index] = num_short_fragments
        absolute_long_fragments[bin_index] = num_long_fragments
        fslr_values[bin_index] = np.log2(num_short_fragments / num_long_fragments) if num_long_fragments > 0 else 0

        # Calculate GC content for the whole region
        if chrom is not None:
            ref_seq_full = ref_fasta.fetch(chrom, start, end).upper()
            gc_content[bin_index] = gc_module.get_gc_content(ref_seq_full)

        # Calculate MDS (Motif Diversity Score) as https://aacrjournals.org/cancerdiscovery/article/10/5/664/2457/Plasma-DNA-End-Motif-Profiling-as-a-Fragmentomic
        # MDS = sum of kmer 1-64 -Pi*log(Pi)/log(64), where Pi is the frequency of kmer i in the bin
        # Defined to be in [0, 1]
        MDS = 0
        number_of_kmers = len(list(kmer_distribution.keys()))
        for _, counts in kmer_distribution.items():
            if counts[bin_index] > 0:
                # Should Pi be relative to the total number of reads or absolute?
                Pi = counts[bin_index]  # / absolute_fragment_counts[bin_index]
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
    for length, counts in fragment_length_distribution.items():
        results_df[f"{length}"] = counts
    for kmer, counts in kmer_distribution.items():
        results_df[f"{kmer}"] = counts

    # Save the results to a CSV file
    results_df.to_csv(output_save_file, index=False)
    print(f"BIN features saved to {output_save_file}")
    bam.close()
    ref_fasta.close()
