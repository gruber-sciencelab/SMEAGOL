# General imports
import gzip
from mimetypes import guess_type
from functools import partial
import numpy as np
import pandas as pd
import os

# Biopython imports
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Smeagol imports
from .matrices import check_pfm, check_pwm, check_ppm


def read_fasta(file):
    """Function to read sequences from a fasta or fasta.gz file
    
    Args:
        file (str): path to file
    
    Returns:
        records (list): list of seqRecords, one for each sequence in the fasta file.
    
    """
    records = []
    
    # check whether the file is compressed
    encoding = guess_type(file)[1]
    
    # Function to open the file
    _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
    
    # Open the file and read sequences
    with _open(file) as input_handle:
        for record in SeqIO.parse(input_handle, 'fasta'):
            records.append(record)
        print('Read ' + str(len(records)) + ' records from ' + file)
    
    return records


def write_fasta(records, file, gz=True):
    """Function to write sequences to a fasta or fasta.gz file.
    
    Params:
        records (list): list of seqRecord objects to write.
        file (str): path to file
    
    Returns:
        Writes records to the file
    """
    
    # Function to open the file
    _open = partial(gzip.open, mode='wt') if file.endswith('.gz') else partial(open, mode='wt')
    
    # Open the file and write sequences
    with _open(file) as output_handle:
        for record in records:
            SeqIO.write(record, output_handle, "fasta")
    print('Wrote ' + str(len(records)) + ' sequences to ' + file)


def read_pms_from_file(file, matrix_type='PPM', check_lens=False, transpose=False, delimiter='\t'):
    """Function to read position matrices from a FASTA-like file in Attract format.
    
    The input file is expected to follow the format:
    
    >Matrix_1_ID
    matrix values
    >Matrix_2_ID
    matrix values
    
    Matrices are by default expected to be in the position x base format, i.e. one row per position and 
    one column per base (A, C, G, T). If the matrices are instead in the base x position format, set
    `transpose=True`.
    
    Optionally, the ID rows may also contain a second field indicating the length of the
    matrix, for example:
    
    >Matrix_1_ID<tab>7
    
    The `pwm.txt` file downloaded from the Attract database follows this format. In this case, you can
    set `check_lens=True` to confirm that the loaded PWMs match the expected lengths.
        
    Args:
        pm_file (str): file containing PMs
        matrix_type (str): PWM, PPM (default) or PFM
        check_lens (bool): check that PWM lengths match the lengths provided in the file
        transpose (bool): transpose the matrices
        delimiter (str): The string used to separate values in the file
    
    Returns:
        pandas dataframe containing PMs
    
    """
    # Read file
    pms = list(open(file, 'r'))
    
    # Split tab-separated fields
    pms = [x.strip().split(delimiter) for x in pms]
    
    # Get matrix start and end positions
    starts = np.where([x[0].startswith(">") for x in pms])[0]
    assert starts[0] == 0 # Check that the first line of the file starts with >
    ends = np.append(starts[1:], len(pms))
    
    # Get matrix IDs and values
    pm_ids = [l[0].strip('>') for l in pms if l[0].startswith(">")]
    
    # Check that matrix lengths match values supplied in the file
    if check_lens:
        lens = np.array([l[1] for l in pms if l[0].startswith(">")]).astype('int')
        assert np.all(lens == ends - starts - 1)
    
    # Separate the PMs
    pms = [pms[start+1:end] for start, end in zip(starts, ends)]
    
    # Convert to numpy arrays
    pms = [np.array(pm, dtype='float32') for pm in pms]
    
    # Transpose each PM
    if transpose:
        pms = [np.transpose(pm) for pm in pms]
        
    # Check arrays
    for pm in pms:
        if matrix_type == 'PFM':
            check_pfm(pm)
        elif matrix_type == 'PWM':
            check_pwm(pm)
        elif matrix_type == 'PPM':
            check_ppm(pm, warn=True)
        else:
            raise ValueError('matrix_type should be one of: PWM, PPM, PFM.')
    
    # Name the column to contain values
    if matrix_type == 'PFM':
        value_col = 'freqs'
    elif matrix_type == 'PWM':
        value_col = 'weights'
    else:
        value_col = 'probs'

    # Make dataframe
    return pd.DataFrame({'Matrix_id':pm_ids, value_col:pms})


def read_pms_from_dir(dirname, matrix_type='PPM', transpose=False):
    """Function to read position matrices from a directory with separate files for each PM.
    
    The input directory is expected to contain individual files each of which represents a
    separate matrix. 
    
    The file name should be the name of the PWM followed by an extension. If `_` is included in
    the name, the characters after the `_` will be dropped.
    
    Matrices are by default expected to be in the position x base format, i.e. one row per position and 
    one column per base (A, C, G, T). If the matrices are instead in the base x position format, set
    `transpose=True`. 
    
    Args:
        dirname (str): folder containing PMs in individual files
        matrix_type (str): PWM, PPM (default) or PFM
        transpose (bool): transpose the matrix
    
    Returns:
        pandas dataframe containing PMs
    
    """
    pm_ids = []
    pms = []
    
    # List files
    files = sorted(os.listdir(dirname))
    
    # Read individual files
    for file in files:
        pm_ids.append(os.path.splitext(file)[0].split("_")[0])
        pm = np.loadtxt(os.path.join(dirname, file))
        pms.append(pm)
        
    # Transpose each PM
    if transpose:
        pms = [np.transpose(x) for x in pms]
        
    # Check arrays
    for pm in pms:
        if matrix_type == 'PFM':
            check_pfm(pm)
        elif matrix_type == 'PWM':
            check_pwm(pm)
        elif matrix_type == 'PPM':
            check_ppm(pm)
        else:
            raise ValueError('matrix_type should be one of: PWM, PPM, PFM.')
    
    # Name the column to contain values
    if matrix_type == 'PFM':
        value_col = 'freqs'
    elif matrix_type == 'PWM':
        value_col = 'weights'
    else:
        value_col = 'probs'
    
    # Make dataframe
    return pd.DataFrame({'Matrix_id':pm_ids, value_col:pms})