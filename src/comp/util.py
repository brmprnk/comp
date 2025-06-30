import numpy as np


def get_filtered_alignments(bam, args, chrom=None, start=None, end=None):
    """
    Fetch alignments from a BAM file with optional filtering by chromosome and position.
    """

    # From GCparagon:
    # 4 (0x4): The read is unmapped.
    # 8 (0x8): The read's mate is unmapped.
    # 256 (0x100): The alignment is not primary.
    #              (This filters out secondary alignments, which can occur when a read maps to multiple locations).
    # 512 (0x200): The read failed quality control checks.
    # 1024 (0x400): The read is a PCR or optical duplicate.
    # 2048 (0x800): The alignment is supplementary.
    #              (This is another type of non-primary alignment, often used for chimeric reads).
    exclude_flags = np.uint32(3852)  # = 256 + 2048 + 512 + 1024 + 4 + 8
    exclude_flags_binary = bin(exclude_flags)

    # Fetch reads overlapping the region, or genome-wide if no region is specified
    try:
        filtered_alignments = filter(
            lambda a: a.is_paired
            and bin(~np.uint32(a.flag) & exclude_flags) == exclude_flags_binary
            and a.mapping_quality >= 5
            and (args.min_frag_len <= abs(a.template_length) <= args.max_frag_len)
            and not a.is_unmapped,
            bam.fetch(
                contig=chrom if chrom is not None else None,
                start=(start - args.max_frag_len) if start is not None else None,
                stop=(end + args.max_frag_len) if end is not None else None,
            ),
        )
    except Exception as e:
        print(f"Something went wrong while fetching reads for locus {chrom}:{start}-{end}")
        print(e)
        return None

    return filtered_alignments
