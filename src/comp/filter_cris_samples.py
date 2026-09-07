#!/usr/bin/env python3
"""
Filter CRIS samples to include only Healthy, Breast cancer, and Colorectal cancer samples.
Creates filtered cris_bam.txt and cris_gc.txt files.
"""

import pandas as pd
from pathlib import Path

# Define paths
METADATA_PATH = Path("accessory_files/cris_metadata.csv")
INPUT_BAM_PATH = Path("input/cris_bam.txt")
INPUT_GC_PATH = Path("input/cris_gc.txt")
OUTPUT_BAM_PATH = Path("input/cris_bam_filtered.txt")
OUTPUT_GC_PATH = Path("input/cris_gc_filtered.txt")

# Disease categories to include
DISEASES_TO_INCLUDE = ["Healthy", "Breast cancer", "Colorectal cancer"]

def main():
    print("=" * 60)
    print("CRIS Sample Filtering Script")
    print("=" * 60)
    
    # Load metadata
    print(f"\n1. Loading metadata from: {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"   Total samples in metadata: {len(metadata)}")
    
    # Filter by disease
    print(f"\n2. Filtering for diseases: {', '.join(DISEASES_TO_INCLUDE)}")
    filtered_metadata = metadata[metadata['disease'].isin(DISEASES_TO_INCLUDE)]
    print(f"   Samples after filtering: {len(filtered_metadata)}")
    
    # Show disease breakdown
    print("\n   Disease breakdown:")
    for disease in DISEASES_TO_INCLUDE:
        count = len(filtered_metadata[filtered_metadata['disease'] == disease])
        print(f"     - {disease}: {count} samples")
    
    # Get list of sample IDs to keep
    sample_ids_to_keep = set(filtered_metadata['ID'].values)
    print(f"\n3. Sample IDs to keep: {len(sample_ids_to_keep)}")
    
    # Load existing BAM file list
    print(f"\n4. Loading BAM file list from: {INPUT_BAM_PATH}")
    with open(INPUT_BAM_PATH, 'r') as f:
        bam_files = [line.strip() for line in f if line.strip()]
    print(f"   Total BAM files: {len(bam_files)}")
    
    # Filter BAM files
    print(f"\n5. Filtering BAM files...")
    filtered_bam_files = []
    for bam_file in bam_files:
        # Extract sample ID from path (e.g., "cris/EE87786.hg38.frag.tsv.bam" -> "EE87786")
        sample_id = Path(bam_file).stem.split('.')[0]
        if sample_id in sample_ids_to_keep:
            filtered_bam_files.append(bam_file)
    
    print(f"   Filtered BAM files: {len(filtered_bam_files)}")
    
    # Load existing GC file list
    print(f"\n6. Loading GC file list from: {INPUT_GC_PATH}")
    with open(INPUT_GC_PATH, 'r') as f:
        gc_files = [line.strip() for line in f if line.strip()]
    print(f"   Total GC files: {len(gc_files)}")
    
    # Filter GC files
    print(f"\n7. Filtering GC files...")
    filtered_gc_files = []
    for gc_file in gc_files:
        # Extract sample ID from path (e.g., "GC/Output_Bam/EE87786.hg38.frag.tsv__correction_factors.csv" -> "EE87786")
        sample_id = Path(gc_file).stem.split('.')[0]
        if sample_id in sample_ids_to_keep:
            filtered_gc_files.append(gc_file)
    
    print(f"   Filtered GC files: {len(filtered_gc_files)}")
    
    # Verify counts match
    if len(filtered_bam_files) != len(filtered_gc_files):
        print(f"\n   WARNING: BAM and GC file counts don't match!")
        print(f"   BAM files: {len(filtered_bam_files)}")
        print(f"   GC files: {len(filtered_gc_files)}")
    
    # Save filtered BAM file list
    print(f"\n8. Saving filtered BAM file list to: {OUTPUT_BAM_PATH}")
    with open(OUTPUT_BAM_PATH, 'w') as f:
        for bam_file in filtered_bam_files:
            f.write(bam_file + '\n')
    
    # Save filtered GC file list
    print(f"\n9. Saving filtered GC file list to: {OUTPUT_GC_PATH}")
    with open(OUTPUT_GC_PATH, 'w') as f:
        for gc_file in filtered_gc_files:
            f.write(gc_file + '\n')
    
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE!")
    print("=" * 60)
    print(f"\nFiltered files created:")
    print(f"  - BAM: {OUTPUT_BAM_PATH} ({len(filtered_bam_files)} files)")
    print(f"  - GC:  {OUTPUT_GC_PATH} ({len(filtered_gc_files)} files)")
    print(f"\nOriginal files kept at:")
    print(f"  - BAM: {INPUT_BAM_PATH} ({len(bam_files)} files)")
    print(f"  - GC:  {INPUT_GC_PATH} ({len(gc_files)} files)")
    print("\nTo use filtered files in slurm/extract_features.sh, update the paths:")
    print("  -I input/cris_bam_filtered.txt --gc_file input/cris_gc_filtered.txt")
    print()

if __name__ == "__main__":
    main()
