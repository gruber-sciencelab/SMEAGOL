import numpy as np
import pandas as pd
from ushuffle import shuffle, set_seed

# Biopython imports
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Smeagol imports
from smeagol.io import read_fasta, write_fasta


def _equals(x, y, eps=1e-4):
    """Function to test whether two values or lists/arrays
    are equal with some tolerance.

    Args:
        x, y: float, int, list or np.array objects
        eps: tolerance value

    Returns:
        True if x and y are equal within the tolerance limit, False otherwise

    """
    if type(x) == type(y) == np.ndarray:
        assert x.shape == y.shape
        return np.all(abs(x - y) < (eps * x.size))
    elif type(x) == type(y) == list:
        assert len(x) == len(y)
        return np.all(abs(x - y) < (eps * len(x)))
    else:
        return abs(x - y) < eps


def shuffle_records(records, simN, simK, out_file=None, seed=1):
    """Function to shuffle sequences.

    Args:
        records (list): list of seqRecord objects
        simN (int): Number of times to shuffle
        simK (int): k-mer frequency to conserve
        out_file (str): Path to output file (optional)

    Returns:
        shuf_records (list): list of shuffled sequences
        Writes shuf_records to file if out_file provided.

    """
    shuf_records = []

    if seed is not None:
        set_seed(seed)

    # Shuffle each record
    for record in records:
        for n in range(simN):
            new_seq = shuffle(str.encode(record.seq.__str__()), simK).decode()
            new_seq = SeqRecord(Seq(new_seq),
                                id="background_seq_{0:d}".format(n))
            new_seq.name = record.id
            shuf_records.append(new_seq)
    print(
        "Shuffled {} sequence(s) {} times while conserving \
        k-mer frequency for k = {}.".format(
            len(records), simN, simK
        )
    )

    # Write to file
    if out_file is not None:
        write_fasta(shuf_records, out_file)

    return shuf_records


def get_Seq(seq):
    """Function to convert a sequence into a Seq object.

    Args:
        seq (str, Seq or SeqRecord): sequence

    Returns:
        Seq object

    """
    if type(seq) == str:
        return Seq(seq)
    elif type(seq) == SeqRecord:
        return seq.seq
    elif type(seq) == Seq:
        return seq
    else:
        raise TypeError(
            "Input sequence must be a string, \
         Seqrecord or Seq object."
        )


def read_bg_seqs(file, records, simN):
    """Function to read background sequences from FASTA file.

    Args:
        file (str): Path to fasta (or fasta.gz) file
        records (list): Original un-shuffled records
        simN (int): Number of shuffles

    Returns:
        list of seqrecord objects

    """
    # Read background sequences
    bg = read_fasta(file)
    # Check number of sequences
    assert len(bg) == len(records) * simN
    # Check sequence length and fix name
    for i, bg_record in enumerate(bg):
        matching_record = records[i // simN]
        assert len(bg_record) == len(
            matching_record
        ), "length of shuffled sequence does not match original sequence."
        bg_record.name = matching_record.name
    return bg


def _get_tiling_windows_over_record(record, width, shift):
    """Function to get tiling windows over a sequence.

    Args:
        record (SeqRecord): SeqRecord object
        width (int): width of tiling windows
        shift (int): shift between successive windows

    Returns:
        windows (pd.DataFrame): windows covering input sequence

    """
    # Get start positions
    starts = list(range(0, len(record), shift))
    # Get end positions
    ends = [np.min([x + width, len(record)]) for x in starts]
    # Add sequence ID
    idxs = np.tile(record.id, len(starts))
    # Combine
    windows = pd.DataFrame({"id": idxs, "start": starts, "end": ends})
    return windows


def get_tiling_windows_over_genome(genome, width, shift=None):
    """Function to get tiling windows over a genome.

    Args:
        genome (list): list of SeqRecord objects
        width (int): width of tiling windows
        shift (int): shift between successive windows. By default the
                     same as width.

    Returns:
        windows (pd.DataFrame): windows covering all sequences in genome

    """
    if shift is None:
        shift = width
    if len(genome) == 1:
        windows = _get_tiling_windows_over_record(genome[0], width, shift)
    else:
        windows = pd.concat(
            [_get_tiling_windows_over_record(
                record, width, shift) for record in genome]
        )
    return windows


def get_site_seq(site, seqs):
    """Function to extract the sequence for a binding site.

    Args:
        site: dataframe containing name, start and end.
        seqs (list / str): list of seqrecord objects or fasta file.
    Returns:
        result (str): sequence of binding site
    """
    if type(seqs) == str:
        seqs = read_fasta(seqs)
    seq = [x for x in seqs if x.name == site["name"]][0]
    return str(seq.seq[site.start:site.end])
