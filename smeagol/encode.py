# General imports
import numpy as np
from collections import defaultdict

# SMEAGOL imports
from .io import read_fasta
from .utils import get_Seq

# Dictionaries

one_hot_dict = {
    "Z": [0, 0, 0, 0],
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "U": [0, 0, 0, 1],
    "N": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    "W": [1 / 2, 0, 0, 1 / 2],
    "S": [0, 1 / 2, 1 / 2, 0],
    "M": [1 / 2, 1 / 2, 0, 0],
    "K": [0, 0, 1 / 2, 1 / 2],
    "R": [1 / 2, 0, 1 / 2, 0],
    "Y": [0, 1 / 2, 0, 1 / 2],
    "B": [0, 1 / 3, 1 / 3, 1 / 3],
    "D": [1 / 3, 0, 1 / 3, 1 / 3],
    "H": [1 / 3, 1 / 3, 0, 1 / 3],
    "V": [1 / 3, 1 / 3, 1 / 3, 0],
}

sense_complement_dict = {
    "+": "-",
    "-": "+",
}

bases = list(one_hot_dict.keys())
base_one_hot = list(one_hot_dict.values())

integer_encoding_dict = {}
for i, base in enumerate(bases):
    integer_encoding_dict[base] = i


# Sequence encoding


def integer_encode(seq, rc=False):
    """Function to encode a nucleic acid sequence as a sequence of integers.

    Args:
      seq (str, Seq or SeqRecord object): Nucleic acid sequence to encode.
      Allowed characters are A, C, G, T, U, Z, N, W, S, M, K, R, Y, B, D, H, V.
      rc (bool): If True, reverse complement the sequence before encoding

    Returns:
      result (np.array): Numpy array containing the integer encoded sequence.
      The shape is (L, ) where L is the length of the sequence.

    """
    # Convert to Seq object
    seq = get_Seq(seq)
    # Reverse complement
    if rc:
        seq = seq.reverse_complement()
    # Encode
    result = np.array([integer_encoding_dict[base] for base in seq],
                      dtype="float32")
    return result


def one_hot_encode(seq, rc=False):
    """Function to one-hot encode a nucleic acid sequence.

    Args:
      seq (str, Seq or SeqRecord object): Nucleic acid sequence to encode.
      Allowed characters are A, C, G, T, U, Z, N, W, S, M, K, R, Y, B, D, H, V.
      rc (bool): If True, reverse complement the sequence before encoding

    Returns:
      result (np.array): Numpy array containing the one-hot encoded sequence.
      The shape is (L, 4) where L is the length of the sequence.

    """
    # Reverse complement
    if rc:
        seq = get_Seq(seq).reverse_complement()
    # Encode
    result = np.vstack(np.array([one_hot_dict[base] for base in seq],
                                dtype="float32"))
    return result


class SeqEncoding:
    """This class encodes a single nucleic acid sequence, or a
    set of sequences, all of which must have the same length.
    Genomic sequences used to initialize the class must also
    have the same sense.

    Attributes:
        len (int): Length of the sequences
        ids (np.array): Numpy array containing the IDs of all sequences.
        names (np.array):Numpy array containing the names of all sequences.
        seqs (np.array): Numpy array containing the integer encoded sequences.
        senses (np.array): Numpy array containing the senses ('+' or '-') of
                           all sequences.

    """

    def __init__(self, records, rcomp="none", sense=None):
        """
        Args:
            records (str or SeqRecord): Either a SeqRecord object
                                        or the path to a fasta file
                                        containing sequences.
            rcomp (str): Either 'none', 'both', or 'only'. If 'none',
                         only the supplied sequences are stored. If
                         'both', both the original sequences and their
                         reverse complement sequences are stored. If
                         'only', only the reverse complement sequences
                         are stored.
            sense (str): Sense of the input sequences in records.
                         Either '+' or '-'.
        """
        if type(records) == "str":
            records = read_fasta(records)
        self._check_equal_lens(records)
        self.len = len(records[0].seq)
        self.ids = np.array([record.id for record in records])
        self.names = np.array([record.name for record in records])
        self.seqs = np.empty(shape=(0, self.len))
        self.senses = []
        assert rcomp in [
            "none",
            "both",
            "only",
        ], "rcomp should be 'none', 'only' or 'both'."
        if rcomp == "none":
            self.seqs = np.vstack(
                [integer_encode(record, rc=False) for record in records]
            )
            self.senses = np.array([sense] * len(records))
        if rcomp == "only":
            self.seqs = np.vstack(
                [integer_encode(record, rc=True) for record in records]
            )
            self.senses = np.array([
                sense_complement_dict[sense]] * len(records))
        if rcomp == "both":
            self.seqs = np.vstack(
                [
                    [integer_encode(record, rc=False) for record in records],
                    [integer_encode(record, rc=True) for record in records],
                ]
            )
            self.senses = np.concatenate(
                [[sense] * len(records),
                 [sense_complement_dict[sense]] * len(records)]
            )
            self.ids = np.tile(self.ids, 2)
            self.names = np.tile(self.names, 2)

    def _check_equal_lens(self, records):
        """
        Checks that all sequences have the same length.

        Args:
            records (SeqRecord): SeqRecord object

        Raises:
            ValueError: if sequences have unequal length.
        """
        if len(records) > 1:
            lens = np.unique([len(record.seq) for record in records])
            if len(lens) != 1:
                raise ValueError(
                    "Cannot encode - sequences have unequal length!")


class SeqGroups:
    """This class encodes one or more groups of equal-length nucleic
    acid sequences. Sequences in different groups may have different
    lengths.
    """

    def __init__(self, records, rcomp="none", sense=None, group_by=None):
        """
        Args:
            records (list / str): list of seqrecord objects or FASTA file
            rcomp (str): 'only' to encode the sequence reverse complements,
                         'both' to encode the reverse complements as well
                         as original sequences, or 'none'.
            sense (str): sense of sequences, '+' or '-'.
            group_by (str): An attribute by which to group the sequences.
                            If None, each sequence will be a separate group.

        """
        if type(records) == str:
            records = read_fasta(records)
        if (group_by is not None) and (len(records)) > 1:
            records = self._group_by(records, group_by)
            self.seqs = [
                SeqEncoding(record, sense=sense,
                            rcomp=rcomp) for record in records
            ]
        else:
            self.seqs = [
                SeqEncoding([record], sense=sense,
                            rcomp=rcomp) for record in records
            ]

    def _group_by(self, records, key):
        """
        Group sequences by a common attribute.
        """
        records_dict = defaultdict(list)
        for record in records:
            records_dict[record.__getattribute__(key)].append(record)
        return records_dict.values()
