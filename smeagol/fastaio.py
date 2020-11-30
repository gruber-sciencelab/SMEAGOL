import gzip
from mimetypes import guess_type
from functools import partial

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def read_fasta(file):
    """
    Function to read sequence
s from a fasta or fasta.gz file
    
    Params:
    file: path to file
    
    Returns:
    records: list of seqRecords, one for each sequence in the fasta file.
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


def write_fasta(records, file):
    """
    Function to write sequences to a fasta.gz file.
    
    Params:
    records: list of seqRecord objects to write.
    file: path to file
    
    Returns:
    Writes records to the file
    """
    # Open file
    with gzip.open(file, "wt") as output_handle:
        # Write
        for record in records:
            SeqIO.write(record, output_handle, "fasta")
    print('Wrote ' + str(len(records)) + ' shuffled sequences to ' + file)