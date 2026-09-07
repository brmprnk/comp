#!/usr/bin/env python3
"""Compute genome-wide mean coverage for a single BAM file using pysam."""

import os
import sys

import pysam


def compute_mean_coverage(bam_path: str) -> float:
    # pysam.coverage() uses pysam's bundled samtools binary, no PATH dependency
    output = pysam.coverage(bam_path)
    total_bases = 0
    weighted_depth = 0.0
    for line in output.splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        chrom_len = int(fields[2])   # endpos = chromosome length
        mean_depth = float(fields[6])  # meandepth column
        total_bases += chrom_len
        weighted_depth += mean_depth * chrom_len
    return weighted_depth / total_bases if total_bases > 0 else 0.0


def extract_sample_name(bam_path: str) -> str:
    basename = os.path.basename(bam_path)
    for suffix in (".hg38.frag.tsv.bam", ".bam"):
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return basename


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <bam_path> <output_csv>", file=sys.stderr)
        sys.exit(1)

    bam_path = sys.argv[1]
    output_path = sys.argv[2]

    sample_name = extract_sample_name(bam_path)
    mean_cov = compute_mean_coverage(bam_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("sample_name,mean_coverage\n")
        f.write(f"{sample_name},{mean_cov:.6f}\n")

    print(f"{sample_name}: {mean_cov:.6f}x")


if __name__ == "__main__":
    main()
