#!/bin/bash

# Specify the directory containing the .bam files
wig_directory="/tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/wigs"
ploidy="c(2)"

# Check if the directory exists
if [ ! -d "$wig_directory" ]; then
  echo "Error: Directory not found!"
  exit 1
fi

# Loop through .bam files and call hmmcopy readCounter
for wig_file in "$wig_directory"/*.wig; do
  if [ -f "$wig_file" ]; then
    echo "Processing $wig_file"

    # Get the filename without the path
    file_name=$(basename "$wig_file")

    # Remove the extension
    id="${file_name%.wig}"

    # Add ploidy to id
    id="${id}"


    results_directory="results/ichorCNA/${id}"
    echo "Creating directory $results_directory"
    mkdir -p "$results_directory"

    Rscript /tudelft.net/staff-umbrella/KWFcfDNA/emc/tools/ichorCNA/scripts/runIchorCNA.R --id $id --WIG $wig_file \
    --gcWig /tudelft.net/staff-umbrella/KWFcfDNA/emc/tools/ichorCNA/inst/extdata/gc_hg38_1000kb.wig --mapWig /tudelft.net/staff-umbrella/KWFcfDNA/emc/tools/ichorCNA/inst/extdata/map_hg38_1000kb.wig \
    --centromere /tudelft.net/staff-umbrella/KWFcfDNA/emc/tools/ichorCNA/inst/extdata/GRCh38.GCA_000001405.2_centromere_acen.txt --normalPanel /home/ubartu/frags/data/emc/pon_ichorCNA_HBD_shallow_seq_median.rds \
    --chrTrain 'c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22)' \
    --outDir $results_directory --genomeBuild 'hg38' --estimateScPrevalence FALSE --ploidy $ploidy

    echo "Writing to $results_directory"
    echo "Done."

  else
    echo "No .wig files found in the specified directory."
    exit 1
  fi
done
