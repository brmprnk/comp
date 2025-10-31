import pandas as pd


def get_gc_content(sequence, round=False):
    """Calculate the GC content of a given sequence."""
    sequence = str(sequence).upper()  # Ensure the sequence is a string and in uppercase
    if not sequence or len(sequence) == 0:
        return 0
    gc_count = sequence.count("G") + sequence.count("C")

    if round:
        # Return gc count as rounded percentage (for GCFix)
        return int((gc_count / len(sequence)) * 100) if len(sequence) > 0 else 0
    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0


def load_gc_matrix(gc_file, min_frag_len, max_frag_len):
    """Load the GC matrix from a file."""
    gc_matrix = pd.read_csv(gc_file)
    # Set the index to be in range of args.min_frag_len to args.max_frag_len
    gc_matrix["frag_lengths"] = range(51, 400 + 1)
    gc_matrix.set_index("frag_lengths", inplace=True)
    # Filter the matrix to only include rows within the specified fragment length range
    return gc_matrix[(gc_matrix.index >= min_frag_len) & (gc_matrix.index <= max_frag_len)]


def count_kmers(sequence, k=3):
    """Count the frequency of each k-mer in a given sequence."""
    sequence = str(sequence).upper()  # Ensure the sequence is a string and in uppercase
    kmer_counts = {}

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if kmer not in kmer_counts:
            kmer_counts[kmer] = 0
        kmer_counts[kmer] += 1

    return kmer_counts
