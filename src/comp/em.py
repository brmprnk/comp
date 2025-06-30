import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import src.comp.gc as gc_module
import src.comp.util as util


def initialize_kmer_dictionary(k: int) -> dict:
    """
    Creates a dictionary of all possible k-mers with counts initialized to 0.

    Args:
        k: The length of the k-mers to generate. Must be a positive integer.

    Returns:
        A dictionary where keys are all possible k-mers of length k
        (composed of 'A', 'C', 'T', 'G') and all values are 0.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or k <= 0:
        msg = "k must be a positive integer."
        raise ValueError(msg)

    bases = ["A", "C", "T", "G"]
    kmer_dict = {}

    # itertools.product generates the cartesian product of the input iterables.
    # repeat=k means it will compute the product of the 'bases' list with itself k times.
    # For k=3, this is equivalent to:
    # ((x,y,z) for x in bases for y in bases for z in bases)
    for p in itertools.product(bases, repeat=k):
        # The product is a tuple of characters, e.g., ('A', 'A', 'T')
        # We join them to form the k-mer string, e.g., "AAT"
        kmer = "".join(p)
        kmer_dict[kmer] = 0

    return kmer_dict


def reverse_complement(dna_seq: str) -> str:
    """Computes the reverse complement of a DNA sequence."""
    complement_map = str.maketrans("ATCGN", "TAGCN")
    return dna_seq.upper().translate(complement_map)[::-1]


def calculate_em(bam_path, output_path, bed_file, gc_file, args):
    """
    Calculate k-mer motifs in the given window.

    See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03607-5#Fig5 for definition of motifs.

    Args:
        bam_path (str): Path to the BAM file.
        output_path (str): Path to save the output features.
        bed_file (str): Path to the BED file containing regions of interest.
        gc_file (str): Path to the GC correction file.
        args: Additional arguments, including k-mer size and other parameters.
    """
    print(f"Calculating EM features for {bam_path} and saving to {output_path}")
    output_save_file = output_path / (Path(bam_path).stem + "_EM.csv")
    print(f"Output will be saved to {output_save_file}")

    if args.gc:
        gc_matrix = gc_module.load_gc_matrix(gc_file, args.min_frag_len, args.max_frag_len)
        print(f"Loaded GC matrix from {gc_file}")

    s3 = initialize_kmer_dictionary(3)
    u3 = initialize_kmer_dictionary(3)
    e3 = initialize_kmer_dictionary(3)
    d3 = initialize_kmer_dictionary(3)

    # Open the BAM and reference FASTA files
    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
        ref_fasta = pysam.FastaFile(args.fasta)
    except Exception as e:
        print(f"Something went wrong while opening the BAM file: {bam_path}")
        print(e)
        return None

    bed = pd.DataFrame({"chrom": [None], "start": [None], "end": [None]}) if bed_file is None or not Path(bed_file).exists() \
          else pd.read_csv(bed_file, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])

    for locus in bed.itertuples(index=False):
        chrom = locus.chrom
        start = locus.start
        end = locus.end

        if chrom is not None and chrom not in bam.references:
            continue  # Skip if the chromosome is not in the BAM file

        filtered_alignments = util.get_filtered_alignments(bam, args, chrom=chrom, start=start, end=end)
        if filtered_alignments is None:
            print(f"No alignments found for {chrom}:{start}-{end}")
            continue

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

            # Ensure we don't fetch from negative coordinates
            if fetch_start < 0:
                continue

            try:
                # OPTIMIZATION: Fetch the entire sequence (upstream+fragment+downstream) in one go
                ref_seq = ref_fasta.fetch(read.reference_name, fetch_start, fetch_end).upper()

                # Check if we got the expected length; if not, it's at a contig boundary
                if len(ref_seq) != (fetch_end - fetch_start):
                    continue

                # If the fragment is on the reverse strand, we need to reverse complement the sequence
                if read.is_reverse:
                    ref_seq = reverse_complement(ref_seq)

                u3_motif, s3_motif, e3_motif, d3_motif = ref_seq[0:3], ref_seq[3:6], ref_seq[-6:-3], ref_seq[-3:]

                # Check for gc correction of read
                read_value = 1
                if args.gc:
                    gc_content = gc_module.get_gc_content(ref_seq[3:-3])
                    read_value = gc_matrix.get(gc_content, {}).get(tlen, 0)

                # Increment counts if the motifs are valid keys
                if s3_motif in s3:
                    s3[s3_motif] += read_value
                if u3_motif in u3:
                    u3[u3_motif] += read_value
                if e3_motif in e3:
                    e3[e3_motif] += read_value
                if d3_motif in d3:
                    d3[d3_motif] += read_value

            except Exception as e:
                # Catch errors from fetching sequence, e.g., at chromosome ends
                print(
                    f"Could not process motif for read {read.query_name} at {chrom}:{read.reference_start}. Error: {e}"
                )
                continue

    bam.close()
    ref_fasta.close()

    # Create a DataFrame for each motif dictionary
    motif_df = pd.DataFrame(
        {
            "s3": pd.Series(s3),
            # 'u3': pd.Series(u3),
            # 'e3': pd.Series(e3),
            # 'd3': pd.Series(d3)
        }
    )
    motif_df.index.name = "motif"
    motif_df.to_csv(output_save_file, sep=",", index=True)
    print(motif_df)

    print(f"EM features saved to {output_save_file}")
    return s3
