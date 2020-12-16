# General imports
import numpy as np
import pandas as pd
import os
import itertools

# Biopython imports
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Biasaway imports
from biasaway.utils import GC, dinuc_count, IUPAC_DINUC
from ushuffle import shuffle, set_seed

# I/O imports
import h5py
import gzip
from mimetypes import guess_type
from functools import partial
from .fastaio import write_fasta


# PPM/PWM analysis

def entropy(probs):
    """
    Function to calculate entropy of a PPM or column of a PPM.
    
    Inputs:
    probs: Array containing probability values
    
    Returns:
    result: Entropy value
    """
    result = -np.sum(probs*np.log2(probs))
    return result


def avg_entropy(prob_arr):
    """
    Function to calculate average entropy over columns of a PPM.
    
    Inputs:
    prob_arr: Numpy array containing PPM probability values
    
    Returns:
    result: Average entropy value
    """
    result = entropy(prob_arr)/np.shape(prob_arr)[0]
    return result 

    
def position_wise_ic(prob_arr, axis=1):
    """
    Function to calculate information content of each column in a PPM.
    
    Inputs:
    prob_arr: Numpy array containing PPM probability values
    
    Returns:
    result: Numpy array containing information content of each column in prob_arr.
    """
    position_wise_entropy = np.apply_along_axis(entropy, axis=axis, arr=prob_arr)
    result = 2 - position_wise_entropy
    return result


def trim_ppm(prob_arr, frac_threshold):
    """
    Function to trim non-informative columns from ends of a PPM.
    
    Inputs:
    prob_arr: Numpy array containing PPM probability values
    frac_threshold: threshold (0-1) to filter out non-informative columns.
    
    Returns:
    Numpy array containing trimmed PPM.
    """
    pos_ic = position_wise_ic(prob_arr, axis=1)
    to_trim = (pos_ic/np.mean(pos_ic)) < frac_threshold
    positions = list(range(prob_arr.shape[0]))
    assert len(to_trim) == len(positions)

    # Trim from start
    while to_trim[0]:
        positions = positions[1:]
        to_trim = to_trim[1:]

    # Trim from end
    while to_trim[-1]:
        positions = positions[:-1]
        to_trim = to_trim[:-1]
    
    return prob_arr[positions,:]


# Shuffling

def shuffle_records(records, simN, simK, out_file=None):
    """
    Function to shuffle sequences.
    
    Inputs:
    records: list of seqRecord objects
    simN: Number of times to shuffle
    simK: k-mer frequency to conserve
    out_file: Path to output file (optional)
    
    Returns:
        shuf_records: list of shuffled sequences
        writes shuf_records to out_file if provided. 
    """
    # Shuffle
    shuf_records = []
    for n in range(0, simN):
        shuf = 1
        for record in records:
            new_seq = shuffle(str.encode(record.seq.__str__()), simK).decode()
            new_seq = SeqRecord(Seq(new_seq),id="background_seq_{0:d}".format(shuf))
            new_seq.name = record.id
            shuf_records.append(new_seq)                
            shuf += 1
    print('Shuffled ' + str(len(records)) + ' input sequence(s) ' + str(simN) + ' times while conserving k-mer frequency for k = ' + str(simK))
    # Write
    if out_file is not None:
        write_fasta(shuf_records, out_file)
    return shuf_records
