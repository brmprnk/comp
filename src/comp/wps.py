import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import src.comp.gc as gc_module
import src.comp.util as util
from src.comp.em import reverse_complement


def calculate_wps(bam_path, output_path, bed_path, gc_file, args, wps_window):
    print(f"Calculating  WPS for {bam_path} and saving to {output_path}")
    output_base = Path(bam_path).stem

    # Check gc
    if args.gc:
        gc_matrix = gc_module.load_gc_matrix(gc_file, args.min_frag_len, args.max_frag_len)
        print(f"Loaded GC matrix from {gc_file}")

    # Read bam and reference files
    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
        ref_fasta = pysam.FastaFile(args.fasta)
    except Exception as e:
        print(f"Something went wrong while opening the BAM file: {bam_path}")
        print(e)
        return

    # Read the bed file
    bed = pd.DataFrame({"chrom": [None], "start": [None], "end": [None]}) if bed_path is None or not Path(bed_path).exists() else pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    print(bed.head())

    for locus in bed.itertuples():
        chrom = locus.chrom
        start = int(locus.start)
        end = int(locus.end)
        print(f"Processing region {chrom}:{start}-{end}")

        # Output file per region to avoid huge csv
        region_file = output_path / output_base / f"{output_base}_{'WPS' if args.gc else 'NOGC_WPS'}_{chrom}_{start}_{end}.csv"

        with open(region_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["chrom", "pos", "wps"])

            # Initialize the data structures for the WPS
            length = end - start + 1
            span_count = np.zeros(length, dtype=float)
            end_count = np.zeros(length, dtype=float)

            if chrom is not None and chrom not in bam.references:
                continue

            # Fetch alignments with buffer to capture fragments that overlap the region
            # Use max fragment length as buffer to get fragments that start outside but overlap
            max_frag_len = getattr(args, "max_frag_len", 220)
            alignment_fetch_start = max(0, start - max_frag_len)
            alignment_fetch_end = end + max_frag_len

            # Filter for the reads quality etc
            filtered_alignments = util.get_filtered_alignments(bam, args, chrom=chrom, start=alignment_fetch_start, end=alignment_fetch_end)
            if filtered_alignments is None:
                print(f"No alignments found for {chrom}:{start}-{end}")
                continue

            half_window = wps_window // 2

            # Go over the reads in this bed region
            for read in filtered_alignments:
                # Process each fragment once using the first read in the pair
                if not read.is_read1 or read.mapping_quality <= args.mapq:
                    continue

                # Template length is the length of the fragment
                tlen = abs(read.template_length)

                # Determine fragment coordinates and the extended region to fetch
                if not read.is_reverse:  # Fragment is on the forward strand
                    frag_start = read.reference_start
                    frag_end = frag_start + tlen
                else:  # Fragment is on the reverse strand
                    frag_end = read.reference_end
                    frag_start = frag_end - tlen

                # Retrieve reference sequence - no motif analysis  => no 3 bp upstream and downstream
                ref_seq = ref_fasta.fetch(read.reference_name, frag_start, frag_end).upper()

                # Check if we got the expected length; if not, it's at a contig boundary
                if len(ref_seq) != (frag_end - frag_start):
                    continue

                # If the fragment is on the reverse strand, we need to reverse complement the sequence
                if read.is_reverse:
                    ref_seq = reverse_complement(ref_seq)

                read_value = 1
                frag_gc_content = gc_module.get_gc_content(ref_seq)

                if args.gc:
                    read_value = gc_matrix.get(str(int(frag_gc_content)), {}).get(tlen, 0)

                # Fragment end counting logic

                # Window is at centered at each base, so subtract/ add half to find range where this fragment end is counted
                end_start = max(frag_end - half_window, start)
                end_end = min(frag_end + half_window, end + 1)
                # subtract - start to match 0 based indexing of end_count
                idx = np.arange(end_start, end_end) - start
                end_count[idx] += read_value

                # Spanning positions
                span_start = max(frag_start + half_window, start)
                span_end = min(frag_end - half_window, end)
                if span_start < span_end:  # Sanity check
                    idx = np.arange(span_start, span_end) - start
                    span_count[idx] += read_value

            # Compute WPS = span_count - end_count
            wps_region = span_count - end_count

            # Write to file without accumulating to decrease memory usage
            for i, wps_val in enumerate(wps_region):
                writer.writerow([chrom, start + i, wps_val])

        print(f"Saved region → {region_file}")

    bam.close()
    ref_fasta.close()
    print("All regions processed.")
