import gzip
from mimetypes import guess_type
from functools import partial
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# Biopython imports
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Biasaway imports
from biasaway.utils import GC, dinuc_count, IUPAC_DINUC
from ushuffle import shuffle, set_seed


def read_fasta(file):
    """Function to read sequences from a fasta or fasta.gz file
    
    Args:
        file (str): path to file
    
    Returns:
        records (list): list of seqRecords, one for each sequence in the fasta file.
    
    """
    records = []
    
    # check whether file is compressed
    encoding = guess_type(file)[1]
    _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
    
    # Read sequences
    with _open(file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            records.append(record)
        print('Read ' + str(len(records)) + ' records from ' + file)
    
    return records


def write_fasta(records, file, gz=True):
    """Function to write sequences to a fasta or fasta.gz file.
    
    Params:
        records (list): list of seqRecord objects to write.
        file (str): path to file
        gz (bool): whether output file is compressed.
    
    Returns:
        Writes records to the file
    """
    # Open file
    if gz:
        with gzip.open(file, "wt") as output_handle:
            for record in records:
                SeqIO.write(record, output_handle, "fasta")
    else:
        with open(file, "w") as output_handle:
            for record in records:
                SeqIO.write(record, output_handle, "fasta")
    print('Wrote ' + str(len(records)) + ' shuffled sequences to ' + file)
    

def shuffle_records(records, simN, simK, out_file=None):
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
    # Shuffle
    shuf_records = []
    for record in records:
        shuf = 1
        for n in range(0, simN):
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


def read_pms_from_file(file, value_col='probs', lengths=False, transpose=False):
    """Function to read position matrices from a fasta-like file in Attract format.
    
    Args:
        pm_file (str): file containing PMs
        value_col (str): name for column containing PM values
        lengths (bool): lengths are provided in the file
        transpose (bool): transpose the matrix
    
    Returns:
        pandas dataframe containing PMs
    
    """
    # Read file
    pms = list(open(file, 'r'))
    pms = [x.strip().split('\t') for x in pms]
    
    # Get matrix start and end positions
    starts = np.where([x[0].startswith(">") for x in pms])[0]
    assert starts[0] == 0
    ends = np.append(starts[1:], len(pms))
    
    # Get matrix IDs and values
    pm_ids = [l[0].strip('>') for l in pms if l[0].startswith(">")]
    if lengths:
        lens = np.array([l[1] for l in pms if l[0].startswith(">")]).astype('int')
        assert np.all(lens == ends - starts - 1)
    pms = [pms[start+1:end] for start, end in zip(starts, ends)]
    if transpose:
        pms = [np.transpose(np.array(x).astype('float')) for x in pms]
    else:
        pms = [np.array(x).astype('float') for x in pms]
    
    # Make dataframe
    return pd.DataFrame({'Matrix_id':pm_ids, value_col:pms})


def read_pms_from_dir(dirname, value_col='probs', transpose=False):
    """Function to read position matrices from a directory with separate files for each PM.
    
    Args:
        pm_file (str): file containing PMs
        value_col (str): name for column containing PM values
        transpose (bool): transpose the matrix
    
    Returns:
        pandas dataframe containing PMs
    
    """
    files = os.listdir(dirname)
    pm_ids = []
    pms = []
    # Read individual files
    for file in files:
        pm_ids.append(file.split("_")[0])
        pm = np.loadtxt(os.path.join(dirname, file))
        if transpose:
            pm = np.transpose(pm)
        pms.append(pm)
    
    # Make dataframe
    return pd.DataFrame({'Matrix_id':pm_ids, value_col:pms})