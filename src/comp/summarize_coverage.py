#!/usr/bin/env python3
"""Aggregate per-sample coverage CSVs into one file and print summary statistics."""

import glob
import os
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <per_sample_dir> <output_csv>", file=sys.stderr)
        sys.exit(1)

    per_sample_dir = sys.argv[1]
    output_csv = sys.argv[2]

    csv_files = sorted(glob.glob(os.path.join(per_sample_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {per_sample_dir}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, float]] = []
    for path in csv_files:
        with open(path) as f:
            lines = f.read().splitlines()
        # Skip header line, read data line
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            sample_name, mean_cov = line.split(",", 1)
            rows.append((sample_name, float(mean_cov)))

    if not rows:
        print("No data found in CSV files.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r[0])
    coverages = [r[1] for r in rows]

    with open(output_csv, "w") as f:
        f.write("sample_name,mean_coverage\n")
        for sample_name, mean_cov in rows:
            f.write(f"{sample_name},{mean_cov:.6f}\n")

    zero_coverage = [sample for sample, cov in rows if cov == 0.0]

    print(f"Combined coverage written to: {output_csv}")
    print(f"Samples: {len(rows)}")
    print(f"\nSummary statistics:")
    print(f"  Min  : {min(coverages):.6f}x")
    print(f"  Max  : {max(coverages):.6f}x")
    print(f"  Mean : {sum(coverages) / len(coverages):.6f}x")

    if zero_coverage:
        print(f"\nZero-coverage samples ({len(zero_coverage)}) — need reprocessing:")
        for sample in zero_coverage:
            print(f"  {sample}")


if __name__ == "__main__":
    main()
