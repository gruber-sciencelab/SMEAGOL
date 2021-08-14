# General imports
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict

# Biopython imports
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# SMEAGOL imports
from .io import read_fasta

# Dictionaries

one_hot_dict = {
    'Z': [0, 0, 0, 0],
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [1/4, 1/4, 1/4, 1/4],
    'W': [1/2, 0, 0, 1/2],
    'S': [0, 1/2, 1/2, 0],
    'M': [1/2, 1/2, 0, 0],
    'K': [0, 0, 1/2, 1/2],
    'R': [1/2, 0, 1/2, 0],
    'Y': [0, 1/2, 0, 1/2], 
    'B': [0, 1/3, 1/3, 1/3],
    'D': [1/3, 0, 1/3, 1/3],
    'H': [1/3, 1/3, 0, 1/3],
    'V': [1/3, 1/3, 1/3, 0]
}

sense_complement_dict = {
    '+':'-', 
    '-':'+',
}

bases = list(one_hot_dict.keys())
integer_encoding_dict = {}
for i, base in enumerate(bases):
    integer_encoding_dict[base] = i

    
# Sequence encoding

def integer_encode(record, rcomp=False):
    """Function to integer encode a DNA sequence.
    
    Args:
      seq (seqrecord): seqrecord object containing a sequence of length L
      rcomp (bool): reverse complement the sequence before encoding
    
    Returns:
      result (np.array): array containing integer encoded sequence. Shape (1, L, 1)
    
    """
    # Reverse complement
    if rcomp:
        seq = record.seq.reverse_complement()
    else:
        seq = record.seq
    # Encode
    result = [integer_encoding_dict[base] for base in seq]
    return result  


class SeqEncoding:
    """Encodes a single DNA sequence, or a set of DNA sequences all of which have the same length and sense.
    
    Args:
        records (list / str): list of seqrecord objects or FASTA file
        rcomp (str): 'only' to encode the sequence reverse complement, or 'both' 
                     to encode the reverse complement as well as original sequence
        sense (str): sense of sequence(s), '+' or '-'.
      
    Raises:
        ValueError: if sequences have unequal length.
    
    """
    def __init__(self, records, rcomp=None, sense=None):
        if type(records) == 'str':
            records = read_fasta(records)
        self.check_equal_lens(records)
        self.len = len(records[0].seq)
        self.ids = np.array([record.id for record in records])
        self.names = np.array([record.name for record in records])
        self.seqs = np.empty(shape=(0, self.len))
        self.senses = []
        assert rcomp in [None, 'both', 'only'], "rcomp should be 'only' or 'both', or else None."
        if rcomp != 'only':
            self.seqs = np.vstack([self.seqs, [integer_encode(record, rcomp=False) for record in records]])
            self.senses = np.concatenate([self.senses, [sense]*len(records)])
        if rcomp is not None:
            self.seqs = np.vstack([self.seqs, [integer_encode(record, rcomp=True) for record in records]])
            self.senses = np.concatenate([self.senses, [sense_complement_dict[sense]]*len(records)])
        if rcomp == 'both':    
            self.ids = np.tile(self.ids, 2)
            self.names = np.tile(self.names, 2)
    def check_equal_lens(self, records):
        if len(records) > 1:
            lens = np.unique([len(record.seq) for record in records])
            if len(lens) != 1:
                raise ValueError("Cannot encode - sequences have unequal length!")

    
class SeqGroups:
    """Encodes one or more groupings of equal-length sequences. 
       Sequences in different groupings may have different lengths.
    
    Args:
        records (list / str): list of seqrecord objects or FASTA file
        rcomp (str): 'only' to encode the sequence reverse complements, or 'both' to encode the reverse
                     complements as well as original sequences
        sense (str): sense of sequences, '+' or '-'.
        group_by (str): key by which to group sequences. If None, each sequence will be a separate group.
    
    """
    def __init__(self, records, rcomp=None, sense=None, group_by=None):
        if type(records) == str:
            records = read_fasta(records)
        if (group_by is not None) and (len(records)) > 1:
            records = self.group_by(records, group_by)
            self.seqs = [SeqEncoding(record, sense=sense, rcomp=rcomp) for record in records]
        else:
            self.seqs = [SeqEncoding([record], sense=sense, rcomp=rcomp) for record in records]
    def group_by(self, records, key):
        records_dict = defaultdict(list)
        for record in records:
            records_dict[record.__getattribute__(key)].append(record)
        return records_dict.values()
