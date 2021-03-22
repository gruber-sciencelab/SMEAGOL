import gzip
from mimetypes import guess_type
from functools import partial

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
