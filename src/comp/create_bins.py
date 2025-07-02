import sys

import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
BLACKLIST_FILE = "accessory_files/hg38_GCcorrection_ExclusionList.merged.sorted.tsv"
OUTPUT_FILE = "hg38_1mb_filtered_bins.bed"
BIN_SIZE = 1_000_000
HISTOGRAM_FILE = "bin_size_distribution.png"

# hg38 chromosome sizes (chr1-22, chrX)
# Sourced from UCSC Genome Browser
CHROMOSOME_SIZES = {"chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167, "chr21": 46709983, "chr22": 50818468, "chrX": 156040895}


def parse_and_merge_blacklist(blacklist_path):
    """
    Parses the blacklist file and merges overlapping intervals for each chromosome.

    Args:
        blacklist_path (str): Path to the blacklist TSV file.

    Returns:
        dict: A dictionary where keys are chromosome names and values are lists
              of merged (start, end) blacklist intervals.
    """
    blacklist_regions = {}
    try:
        with open(blacklist_path) as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])

                if chrom not in blacklist_regions:
                    blacklist_regions[chrom] = []
                blacklist_regions[chrom].append((start, end))
    except FileNotFoundError:
        print(f"Error: Blacklist file not found at '{blacklist_path}'")
        sys.exit(1)

    # Merge overlapping intervals for efficiency
    for chrom, intervals in blacklist_regions.items():
        if not intervals:
            continue
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start < last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        blacklist_regions[chrom] = merged

    print(f"Successfully parsed and merged blacklist regions from '{blacklist_path}'.")
    return blacklist_regions


def subtract_intervals(intervals, blacklist):
    """
    Subtracts a list of blacklist intervals from a list of source intervals.

    Args:
        intervals (list): A list of (start, end) tuples to be filtered.
        blacklist (list): A sorted, merged list of (start, end) blacklist tuples.

    Returns:
        list: A new list of (start, end) tuples representing the valid regions.
    """
    if not blacklist:
        return intervals

    valid_regions = []
    for interval_start, interval_end in intervals:
        current_regions = [(interval_start, interval_end)]

        for black_start, black_end in blacklist:
            next_regions = []
            for r_start, r_end in current_regions:
                if r_end <= black_start or r_start >= black_end:
                    next_regions.append((r_start, r_end))
                    continue

                if r_start < black_start:
                    next_regions.append((r_start, black_start))
                if r_end > black_end:
                    next_regions.append((black_end, r_end))
            current_regions = next_regions

        valid_regions.extend(current_regions)

    return valid_regions


def plot_bin_size_distribution(bed_file, output_image_file):
    """
    Reads a BED file, calculates interval sizes, and plots a histogram.

    Args:
        bed_file (str): Path to the input BED file.
        output_image_file (str): Path to save the output histogram image.
    """
    print(f"\nReading '{bed_file}' to generate histogram...")
    try:
        # Read the BED file into a pandas DataFrame
        df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])

        # Calculate the size of each bin
        df["size"] = df["end"] - df["start"]

        # Create the histogram
        plt.figure(figsize=(12, 7))
        plt.hist(df["size"], bins=50, edgecolor="black")

        # Add titles and labels for clarity
        plt.title("Distribution of Filtered Bin Sizes", fontsize=16)
        plt.xlabel("Bin Size (Base Pairs)", fontsize=12)
        plt.ylabel("Frequency (Number of Bins)", fontsize=12)
        plt.grid(axis="y", alpha=0.75)

        # Add a vertical line at the target bin size for reference
        plt.axvline(BIN_SIZE, color="r", linestyle="dashed", linewidth=2, label=f"Unaffected Bin Size ({BIN_SIZE} bp)")
        plt.legend()

        # Save the plot to a file
        plt.savefig(output_image_file, dpi=300)
        print(f"Histogram saved to '{output_image_file}'.")

    except FileNotFoundError:
        print(f"Error: BED file not found at '{bed_file}'. Cannot generate plot.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")


def main():
    """
    Main function to generate binned genome BED file and plot the distribution.
    """
    # 1. Load and process the blacklist file
    blacklist = parse_and_merge_blacklist(BLACKLIST_FILE)

    # 2. Get the list of chromosomes to process
    target_chromosomes = list(CHROMOSOME_SIZES.keys())

    # 3. Generate bins and write to output file
    with open(OUTPUT_FILE, "w") as out_f:
        print(f"Generating bins of size {BIN_SIZE}bp and writing to '{OUTPUT_FILE}'...")

        for chrom in target_chromosomes:
            print(f"  Processing {chrom}...")
            chrom_size = CHROMOSOME_SIZES[chrom]
            chrom_blacklist = blacklist.get(chrom, [])

            for start in range(0, chrom_size, BIN_SIZE):
                end = min(start + BIN_SIZE, chrom_size)
                initial_bin = [(start, end)]
                final_bins = subtract_intervals(initial_bin, chrom_blacklist)

                for bin_start, bin_end in final_bins:
                    if bin_start < bin_end:
                        out_f.write(f"{chrom}\t{bin_start}\t{bin_end}\n")

    print("\nBED file generation complete.")

    # 4. Generate and save the histogram
    plot_bin_size_distribution(OUTPUT_FILE, HISTOGRAM_FILE)


if __name__ == "__main__":
    main()
