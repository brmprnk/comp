#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Placeholder imports for feature extraction modules.
# In a real implementation, you would create these files in `comp/src/`
# and implement the feature extraction logic within them.
# # from comp.src.cna import calculate_cna
from src.comp.em import calculate_em

# from comp.src.em import calculate_em
# from comp.src.em import calculate_em_gw
# from comp.src.fp import calculate_fp
# from comp.src.nof import calculate_nof
# from comp.src.np import calculate_np
# from comp.src.wps import calculate_wps
# from comp.src.ocf import calculate_ocf
# from comp.src.emr import calculate_emr
# from comp.src.fpr import calculate_fpr
# from comp.src.pfe import calculate_pfe
# from comp.src.tssc import calculate_tssc


def get_args():
    """
    Parses command-line arguments for the feature extractor.
    """
    parser = argparse.ArgumentParser(
        description="Toolkit for generating fragmentomic features from .bam files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- General options ---
    general_group = parser.add_argument_group("General options")
    general_group.add_argument(
        "-I",
        "--input_file",
        required=True,
        type=str,
        help="A text file containing all input BAM files with one BAM file per line, \
            a single BAM file, or a directory containing BAM files.\n",
    )
    general_group.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./extracted_features",
        help="Output directory for all the results. Default: [./extracted_features]",
    )
    general_group.add_argument(
        "-F",
        "--features",
        type=str,
        default="CNA,NOF,WPS,EM,EMR,FP,FPR,NP,OCF,PFE,TSSC",
        help="Features to extract, separated by commas (e.g., CNA,NOF).\nAvailable: CNA, NOF, WPS, EM, EMR, FP, FPR, NP, OCF, PFE, TSSC.\nDefault: All features will be extracted.",
    )
    general_group.add_argument("--mapq", type=int, default=30, help="Minimum mapping quality for reads to be considered. Default: [30]")
    general_group.add_argument("--min_frag_len", type=int, default=51, help="Minimum fragment length to consider. Default: [51]")
    general_group.add_argument("--max_frag_len", type=int, default=400, help="Maximum fragment length to consider. Default: [400]")
    general_group.add_argument(
        "-g",
        "--genome_version",
        type=str,
        default="hg38",
        choices=["hg19", "hg38"],
        help="Genome version of input BAM files (hg19/hg38). Default: [hg38]",
    )
    general_group.add_argument(
        "--bl",
        "--blacklist",
        type=str,
        default="hg38/hg38_GCcorrection_ExclusionList.merged.sorted.bed",
        help="Path to a blacklist BED file to filter out regions. \
            Default: [hg38/hg38_GCcorrection_ExclusionList.merged.sorted.bed]\n",
    )
    general_group.add_argument(
        "-b",
        "--bed_file",
        type=str,
        help="A BED3 file specifying the regions to extract features, \
            or a directory containing BED files or a .txt file with BED file paths.\n",
    )
    general_group.add_argument("-gw", "--genome_wide", action="store_true", default=False, help="If set, the script will extract features genome-wide without using a BED file. Default: [False]")
    general_group.add_argument("-c", "--cpu", type=int, default=1, help="Number of CPU to use. Default: [1]")
    general_group.add_argument(
        "--gc",
        action="store_true",
        default=True,
        help="Use the GCFix output file to incorporate GC Bias correction. Default: [True]",
    )
    general_group.add_argument("--gc_file", type=str, help="Path to the GCFix output file for GC Bias correction.")

    # --- CNA specific options ---
    cna_group = parser.add_argument_group("Options specific for Copy Number Alterations (CNA)")
    cna_group.add_argument(
        "-B",
        "--bin_size",
        type=int,
        default=1000,
        choices=[10, 50, 500, 1000],
        help="Bin size in kilobases (10, 50, 500, or 1000). Default: [1000]",
    )
    cna_group.add_argument("--CNA_params", type=str, help="Additional parameter string for CNA analysis.")

    # --- NOF specific options ---
    nof_group = parser.add_argument_group("Options specific for Nucleosome Occupancy and Fuzziness (NOF)")
    nof_group.add_argument("--NOF_params", type=str, help="Additional parameter string for NOF analysis.")

    # --- WPS specific options ---
    wps_group = parser.add_argument_group("Options specific for Windowed Protection Score (WPS)")
    wps_group.add_argument("-x", "--min_len_long", type=int, default=120, help="Min fragment length for long fragments WPS. Default: [120]")
    wps_group.add_argument("-X", "--max_len_long", type=int, default=180, help="Max fragment length for long fragments WPS. Default: [180]")
    wps_group.add_argument("-w", "--win_size_long", type=int, default=120, help="Window size for long fragments WPS. Default: [120]")
    wps_group.add_argument("-m", "--min_len_short", type=int, default=35, help="Min fragment length for short fragments WPS. Default: [35]")
    wps_group.add_argument("-M", "--max_len_short", type=int, default=80, help="Max fragment length for short fragments WPS. Default: [80]")
    wps_group.add_argument("-W", "--win_size_short", type=int, default=16, help="Window size for short fragments WPS. Default: [16]")

    # --- EM specific options ---
    em_group = parser.add_argument_group("Options specific for End Motif (EM)")
    em_group.add_argument("-f", "--fasta", type=str, default="hg38/ref.fa", help="Reference genome in FASTA format.")
    em_group.add_argument("-k", "--kmer_size", type=int, default=3, help="K-mer size for motif extraction. Default: [3]")

    # --- NP specific options ---
    np_group = parser.add_argument_group("Options specific for Nucleosome Profile (NP)")
    np_group.add_argument("-l", "--sites_path", type=str, help="Directory containing a list of files with each file for a set of sites.")

    # --- PFE specific options ---
    pfe_group = parser.add_argument_group("Options for Promoter Fragmentation Entropy (PFE)")
    pfe_group.add_argument("-T", "--tss_info", type=str, help="A TAB-delimited TSS information file without any header.")
    pfe_group.add_argument("--PFE_params", type=str, help="Additional parameter string for PFE analysis.")

    # --- TSSC specific options ---
    tssc_group = parser.add_argument_group("Options for TSS Coverage (TSSC)")
    tssc_group.add_argument("-u", "--upstream", type=int, default=1000, help="Number of base pairs upstream of TSS. Default: [1000]")
    tssc_group.add_argument("-d", "--downstream", type=int, default=1000, help="Number of base pairs downstream of TSS. Default: [1000]")
    tssc_group.add_argument("-S", "--tss_file", type=str, help="A BED6 file specifying the coordinates of TSSs.")
    tssc_group.add_argument(
        "-n",
        "--normalization",
        type=str,
        default="RPKM",
        choices=["RPKM", "CPM", "BPM", "RPGC"],
        help="Normalization method for bamCoverage. Default: [RPKM]",
    )
    tssc_group.add_argument("--bamCoverage_params", type=str, help="Additional parameter string for bamCoverage.")
    tssc_group.add_argument("--multiBigwigSummary_params", type=str, help="Additional parameter string for multiBigwigSummary.")

    return parser.parse_args()


def run_analysis(args):
    """
    Main function to run the feature extraction analysis.
    """
    print("--- Starting cfDNA Analysis ---")

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_path.resolve()}")

    # Validate input file
    input_file_path = Path(args.input_file)
    if not input_file_path.is_file():
        print(f"Error: Input file not found at {input_file_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Read BAM files from the input list
    if input_file_path.is_dir():
        # If input is a directory, list all BAM files in it
        bam_files = list(input_file_path.rglob("*.bam"))
        if not bam_files:
            print("Error: No BAM files found in the specified directory.", file=sys.stderr)
            sys.exit(1)
        bam_files = [str(bam_file) for bam_file in bam_files]
    if input_file_path.suffix == ".bam":
        # If input is a single BAM file, use it directly
        bam_files = [str(input_file_path)]
    if input_file_path.suffix == ".txt":
        # If input is a text file, read BAM files listed in it
        with open(input_file_path) as f:
            bam_files = [line.strip() for line in f if line.strip()]

    if not bam_files:
        print("Error: No BAM files listed in the input file.", file=sys.stderr)
        sys.exit(1)

    # Check for GC
    if args.gc:
        if not args.gc_file:
            print("Error: --gc is set to True but --gc_file is not provided.", file=sys.stderr)
            sys.exit(1)
        gc_file_path = Path(args.gc_file)
        if gc_file_path.is_dir():
            gc_files = list(gc_file_path.rglob("*.csv"))
            if not gc_files:
                print("Error: No GCFix output files found in the specified directory.", file=sys.stderr)
                sys.exit(1)
            gc_files = [str(gc_file) for gc_file in gc_files]
        if gc_file_path.suffix == ".csv":
            # If a single GCFix output file is provided
            gc_files = [str(gc_file_path)]
        if gc_file_path.suffix == ".txt":
            # If a text file is provided, read GCFix output files listed in it
            with open(gc_file_path) as f:
                gc_files = [line.strip() for line in f if line.strip()]
        if not gc_files:
            print("Error: No GCFix output files listed in the GC file.", file=sys.stderr)
            sys.exit(1)
    else:
        gc_files = [None] * len(bam_files)  # No GC correction if not specified

    # Check for BED file
    if args.bed_file:
        bed_file_path = Path(args.bed_file)
        if bed_file_path.is_dir():
            # If a directory is provided, list all BED files in it
            bed_files = list(bed_file_path.rglob("*.bed"))
            # Continue adding bed files recursively
            if not bed_files:
                print("Error: No BED files found in the specified directory.", file=sys.stderr)
                sys.exit(1)
            bed_files = [str(bed_file) for bed_file in bed_files]
        if bed_file_path.suffix == ".bed":
            # If a single BED file is provided
            bed_files = [str(bed_file_path)]
        if bed_file_path.suffix == ".txt":
            # If a text file is provided, read BED files listed in it
            with open(bed_file_path) as f:
                bed_files = [line.strip() for line in f if line.strip()]
        if not bed_files:
            print("Error: No BED files listed in the BED file.", file=sys.stderr)
            sys.exit(1)

    # Get the list of features to extract
    features_to_extract = {feature.strip().upper() for feature in args.features.split(",")}
    print(f"Requested features: {', '.join(sorted(features_to_extract))}")

    nr_of_processes = min(mp.cpu_count(), len(bam_files), args.cpu)
    nr_of_processes = int(nr_of_processes) if nr_of_processes > 0 else 1

    if args.bed_file:
        print(f"Using BED files: {', '.join(bed_files)}")

        for bed_file in bed_files:
            # Add dir with bed file name to the output path
            bed_file_path = Path(bed_file)
            output_path_bed = output_path / bed_file_path.stem
            os.makedirs(output_path_bed, exist_ok=True)

            print(f"Processing BED file: {bed_file}")
            starmap_args = []
            for i in range(len(bam_files)):
                starmap_args.append((bam_files[i], output_path_bed, bed_file, gc_files[i], args))

            print(f"Using {nr_of_processes} threads for processing.")

            with mp.Pool(processes=nr_of_processes) as pool:
                if "EM" in features_to_extract:
                    print("  -> Extracting EM...")
                    pool.starmap(calculate_em, starmap_args)

        # After processing all BED files, print a message
        print("Finished processing all BED files.")

    else:
        print("No BED files provided, calculating features for the entire genome.")

    # Process each BAM file
    # @TODO: Make genome-wide feature extraction
    if not args.bed_file or args.genome_wide:
        print("Processing genome-wide BAM files...")

        # Create a specific output directory for genome-wide features
        genome_wide_output_dir = output_path / "genome_wide"
        genome_wide_output_dir.mkdir(parents=True, exist_ok=True)

        starmap_args = []
        for i in range(len(bam_files)):
            starmap_args.append((bam_files[i], genome_wide_output_dir, None, gc_files[i], args))

        print(f"Using {nr_of_processes} threads for processing.")

        with mp.Pool(processes=nr_of_processes) as pool:
            if "EM" in features_to_extract:
                print("  -> Extracting EM...")
                pool.starmap(calculate_em, starmap_args)
    # for bam_file in bam_files:
    #     bam_path = Path(bam_file)
    #     if not bam_path.is_file():
    #         print(f"Warning: BAM file not found, skipping: {bam_file}", file=sys.stderr)
    #         continue

    #     print(f"\nProcessing GENOME-WIDE BAM file: {bam_file}")

    #     # Create a specific output directory for the current sample
    #     sample_name = bam_path.stem
    #     sample_output_dir = output_path / sample_name
    #     sample_output_dir.mkdir(parents=True, exist_ok=True)

    #     # --- Call feature extraction functions based on user selection ---
    #     # Each function would be imported from comp/src/ and would handle
    #     # the specific logic for that feature.

    #     # if 'CNA' in features_to_extract:
    #     #     print("  -> Extracting CNA...")
    #     #     # calculate_cna(bam_path, sample_output_dir, args)
    #     #     pass

    #     if 'EM' in features_to_extract:
    #         print("  -> Extracting EM...")
    #         calculate_em(bam_path, sample_output_dir, args)
    #         pass

    # if 'FP' in features_to_extract:
    #     print("  -> Extracting FP...")
    #     # calculate_fp(bam_path, sample_output_dir, args)
    #     pass

    # if 'NOF' in features_to_extract:
    #     print("  -> Extracting NOF...")
    #     # calculate_nof(bam_path, sample_output_dir, args)
    #     pass

    # if 'NP' in features_to_extract:
    #     print("  -> Extracting NP...")
    #     # calculate_np(bam_path, sample_output_dir, args)
    #     pass

    # if 'WPS' in features_to_extract:
    #     print("  -> Extracting WPS...")
    #     # calculate_wps(bam_path, sample_output_dir, args)
    #     pass

    # if 'OCF' in features_to_extract:
    #     print("  -> Extracting OCF...")
    #     # calculate_ocf(bam_path, sample_output_dir, args)
    #     pass

    # if 'EMR' in features_to_extract:
    #     print("  -> Extracting EMR...")
    #     # calculate_emr(bam_path, sample_output_dir, args)
    #     pass

    # if 'FPR' in features_to_extract:
    #     print("  -> Extracting FPR...")
    #     # calculate_fpr(bam_path, sample_output_dir, args)
    #     pass

    # if 'PFE' in features_to_extract:
    #     print("  -> Extracting PFE...")
    #     # calculate_pfe(bam_path, sample_output_dir, args)
    #     pass

    # if 'TSSC' in features_to_extract:
    #     print("  -> Extracting TSSC...")
    #     # calculate_tssc(bam_path, sample_output_dir, args)
    #     pass

    # print(f"Finished processing {sample_name}.")

    print("\n--- cfDNA Analysis Complete ---")
    print("Results are saved in the following directory:")
    print(f"  {output_path.resolve()}")


def main():
    """
    Main entry point for the script.
    """
    args = get_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
