# General imports
import gzip
from mimetypes import guess_type
from functools import partial
import numpy as np
import pandas as pd
import os

# Biopython imports
from Bio import SeqIO

# Smeagol imports
from smeagol.matrices import check_pfm, check_pwm, check_ppm, pfm_to_ppm, ppm_to_pwm

# Pyjaspar
from pyjaspar import jaspardb
import re

def read_fasta(file):
    """Function to read sequences from a fasta or fasta.gz file.

    Args:
        file (str): path to file

    Returns:
        records (list): list of seqRecord objects, one for each
        sequence in the fasta file.

    """
    records = []

    # check whether the file is compressed
    encoding = guess_type(file)[1]

    # Function to open the file
    _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open

    # Open the file and read sequences
    with _open(file) as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            records.append(record)
        print("Read " + str(len(records)) + " records from " + file)

    return records


def write_fasta(records, file, gz=True):
    """Function to write sequences to a fasta or fasta.gz file.

    Params:
        records (list): list of seqRecord objects to write.
        file (str): path to file
        gz (bool): If true, compress the output file with gzip.

    Returns:
        Writes records to the file
    """

    # Function to open the file
    _open = (
        partial(gzip.open, mode="wt")
        if file.endswith(".gz")
        else partial(open, mode="wt")
    )

    # Open the file and write sequences
    with _open(file) as output_handle:
        for record in records:
            SeqIO.write(record, output_handle, "fasta")
    print("Wrote " + str(len(records)) + " sequences to " + file)


def read_pms_from_file(
    file, matrix_type="PPM", check_lens=False, transpose=False,
    delimiter="\t"
):
    """Function to read position matrices from a FASTA-like file.

    The input file is expected to follow the format:

    >Matrix_1_ID
    matrix values
    >Matrix_2_ID
    matrix values

    Matrices are by default expected to be in the position x base
    format, i.e. one row per position and one column per base
    (A, C, G, T). If the matrices are instead in the base x position
    format, set `transpose=True`.

    Optionally, the ID rows may also contain a second field indicating
    the length of the matrix, for example:

    >Matrix_1_ID<tab>7

    The `pwm.txt` file downloaded from the Attract database follows
    this format. In this case, you can set `check_lens=True` to
    confirm that the loaded PWMs match the expected lengths.

    Args:
        pm_file (str): path to the file containing PMs
        matrix_type (str): 'PWM', 'PPM' (default) or 'PFM'
        check_lens (bool): check that the matrix lengths match
                           the lengths provided in the file
        transpose (bool): transpose the matrices
        delimiter (str): The string used to separate values in the file

    Returns:
        pandas dataframe containing PMs

    """
    # Read file
    pms = list(open(file, "r"))

    # Split tab-separated fields
    pms = [x.strip().split(delimiter) for x in pms]

    # Get matrix start and end positions
    starts = np.where([x[0].startswith(">") for x in pms])[0]
    # Check that the first line of the file starts with >
    assert starts[0] == 0
    ends = np.append(starts[1:], len(pms))

    # Get matrix IDs and values
    pm_ids = [l[0].strip(">") for l in pms if l[0].startswith(">")]

    # Check that matrix lengths match values supplied in the file
    if check_lens:
        lens = np.array([
            pm[1] for pm in pms if pm[0].startswith(">")]).astype("int")
        assert np.all(lens == ends - starts - 1)

    # Separate the PMs
    pms = [pms[start + 1:end] for start, end in zip(starts, ends)]

    # Convert to numpy arrays
    pms = [np.array(pm, dtype="float32") for pm in pms]

    # Transpose each PM
    if transpose:
        pms = [np.transpose(pm) for pm in pms]

    # Check arrays
    for pm in pms:
        if matrix_type == "PFM":
            check_pfm(pm, warn=True)
        elif matrix_type == "PWM":
            check_pwm(pm)
        elif matrix_type == "PPM":
            check_ppm(pm, warn=True)
        else:
            raise ValueError("matrix_type should be one of: PWM, PPM, PFM.")

    # Name the column to contain values
    if matrix_type == "PFM":
        value_col = "freqs"
    elif matrix_type == "PWM":
        value_col = "weights"
    else:
        value_col = "probs"

    # Make dataframe
    return pd.DataFrame({"Matrix_id": pm_ids, value_col: pms})


def read_pms_from_dir(dirname, matrix_type="PPM", transpose=False):
    """Function to read position matrices from a directory with
    separate files for each PM.

    The input directory is expected to contain individual files
    each of which represents a separate matrix.

    The file name should be the name of the PWM followed by
    an extension. If `_` is included in the name, the characters
    after the `_` will be dropped.

    Matrices are by default expected to be in the position x base
    format, i.e. one row per position and one column per base
    (A, C, G, T). If the matrices are instead in the base x
    position format, set `transpose=True`.

    Args:
        dirname (str): path to the folder containing PMs in
                       individual files
        matrix_type (str): 'PWM', 'PPM' (default) or 'PFM'
        transpose (bool): If true, transpose the matrix

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
        if matrix_type == "PFM":
            check_pfm(pm, warn=True)
        elif matrix_type == "PWM":
            check_pwm(pm)
        elif matrix_type == "PPM":
            check_ppm(pm, warn=True)
        else:
            raise ValueError("matrix_type should be one of: PWM, PPM, PFM.")

    # Name the column to contain values
    if matrix_type == "PFM":
        value_col = "freqs"
    elif matrix_type == "PWM":
        value_col = "weights"
    else:
        value_col = "probs"

    # Make dataframe
    return pd.DataFrame({"Matrix_id": pm_ids, value_col: pms})


def _download_rbpdb(species="H.sapiens"):
    """Function to download all motifs and metadata from RBPDB.
       Downloads version 1.3.1.

    Args:
        species (str): 'H.sapiens', 'M.musculus', 'D.melanogaster'
                       or 'C.elegans'.

    Returns:
        Motifs are downloaded into the 'motifs/rbpdb/{species}'
        directory.

    """
    download_dir = os.path.join("motifs", "rbpdb", species)
    print(f"Downloading RBPDB version 1.3.1 for species {species} \
          into {download_dir}")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

        # Replace species name with code
        rbpdb_species_codes = {
            "H.sapiens": "human",
            "M.musculus": "mouse",
            "D.melanogaster": "fly",
            "C.elegans": "worm",
        }
        species_code = rbpdb_species_codes[species]

        meta_file = "RBPDB_v1.3.1_{}_2012-11-21_TDT".format(species_code)
        mat_file = "matrices_{}".format(species_code)
        url = "http://rbpdb.ccbr.utoronto.ca/downloads"

        # Get remote paths
        meta_remote_path = os.path.join(url, meta_file + ".zip")
        mat_remote_path = os.path.join(url, mat_file + ".zip")
        meta_local_path = os.path.join(download_dir, meta_file + ".zip")
        mat_local_path = os.path.join(download_dir, mat_file + ".zip")

        # Download
        os.system("wget -P {} {}".format(download_dir, meta_remote_path))
        os.system("wget -P {} {}".format(download_dir, mat_remote_path))
        os.system("unzip {} -d {}".format(meta_local_path, download_dir))
        os.system("unzip {} -d {}".format(mat_local_path, download_dir))

        # Remove zipped files
        os.remove(meta_local_path)
        os.remove(mat_local_path)
        print("Done")
    else:
        print(f"Folder {download_dir} already exists.")


def load_rbpdb(species="H.sapiens", matrix_type="PWM"):
    """Function to download all motifs and metadata from RBPDB
    and load them as a pandas DF. Downloads version 1.3.1.

    Args:
        species (str): 'H.sapiens', 'M.musculus', 'D.melanogaster'
                        or 'C.elegans'.
        matrix_type (str): 'PWM' or 'PFM'

    Returns:
        df (pandas df): contains matrices

    """
    # Replace species name with code
    rbpdb_species_codes = {
        "H.sapiens": "human",
        "M.musculus": "mouse",
        "D.melanogaster": "fly",
        "C.elegans": "worm",
    }
    species_code = rbpdb_species_codes[species]

    # Download
    _download_rbpdb(species=species)

    # Set paths
    download_dir = os.path.join("motifs", "rbpdb", species)
    if matrix_type == "PWM":
        mat_dir = os.path.join(download_dir, "PWMDir")
    elif matrix_type == "PFM":
        mat_dir = os.path.join(download_dir, "PFMDir")
    else:
        raise ValueError("matrix_type must be PFM or PWM.")

    # Read matrices
    df = read_pms_from_dir(mat_dir, matrix_type=matrix_type, transpose=True)

    # Read metadata
    rbp = pd.read_csv(
        os.path.join(
            download_dir, "RBPDB_v1.3.1_proteins_{}_2012-11-21.tdt".format(
                species_code)
        ),
        header=None,
        sep="\t",
        usecols=(0, 4),
        names=("Prot_id", "Gene_name"),
        dtype="str",
    )
    rbp_pwm = pd.read_csv(
        os.path.join(
            download_dir, "RBPDB_v1.3.1_protExp_{}_2012-11-21.tdt".format(
                species_code)
        ),
        header=None,
        sep="\t",
        usecols=(0, 1),
        names=("Prot_id", "Matrix_id"),
        dtype="str",
    )

    # Merge
    rbp = rbp.merge(rbp_pwm, on="Prot_id")[["Matrix_id", "Gene_name"]]
    df = df.merge(rbp, on="Matrix_id")

    return df


def _download_attract():
    """Function to download all motifs and metadata from ATtRACT.

    Returns:
        Motifs are downloaded into the 'motifs/attract' directory.

    """
    download_dir = "motifs/attract"
    remote_path = "https://attract.cnic.es/attract/static/ATtRACT.zip"
    local_path = os.path.join(download_dir, "ATtRACT.zip")
    print(f"Downloading ATtRACT db into {download_dir}")
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
        os.system("wget  -P " + download_dir + " " + remote_path)
        os.system("unzip " + local_path + " -d " + download_dir)
    else:
        print(f"Folder {download_dir} already exists.")


def load_attract():
    """Function to download all motifs and metadata from ATtRACT
    and load them as a pandas DF.

    Returns:
        df (pandas df): contains matrices

    """
    _download_attract()
    download_dir = "motifs/attract"
    df = pd.read_csv(os.path.join(download_dir, "ATtRACT_db.txt"), sep="\t")
    ppms = read_pms_from_file(os.path.join(download_dir, "pwm.txt"),
                              check_lens=True)
    df = df.merge(ppms, on="Matrix_id")
    return df


def _download_smeagol_PWMset():
    """Function to download the curated set of motifs used in the
    SMEAGOL paper.

    Returns:
        df (pandas df): contains matrices

    """
    download_dir = "motifs/smeagol_datasets"
    remote_path = "https://github.com/gruber-sciencelab/" + \
                  "VirusHostInteractionAtlas/blob/main/DATA/PWMs/" + \
                  "attract_rbpdb_encode_filtered_human_pwms.h5?raw=true"
    local_path = os.path.join(
        download_dir, "attract_rbpdb_encode_filtered_human_pwms.h5?raw=true"
    )
    local_dest_path = os.path.join(
        download_dir, "attract_rbpdb_encode_filtered_human_pwms.h5"
    )
    print(f"Downloading custom PWM set into {download_dir}")
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
        os.system("wget  -P " + download_dir + " " + remote_path)
        os.rename(local_path, local_dest_path)
    else:
        print(f"Folder {download_dir} already exists.")


def load_smeagol_PWMset(dataset="representative"):
    """Function to download the curated set of motifs used in
    the SMEAGOL paper and load them as a pandas dataframe.

    Args:
        dataset: 'full' or 'representative'.

    Returns:
        df (pandas df): contains matrices

    """
    _download_smeagol_PWMset()
    download_dir = "motifs/smeagol_datasets"
    h5_path = os.path.join(download_dir,
                           "attract_rbpdb_encode_filtered_human_pwms.h5")
    df = pd.read_hdf(h5_path, "data")
    if dataset == "representative":
        df = df[df.representative].reset_index(drop=True)
    return df

def fetch_jaspar_PFMs(download_path='motifs/JASPAR2022', species='H.sapiens'):
    """Function to fetch transcription factor (TF) binding profiles stored as position frequency matrices (PFMs) from the 9th release JASPAR20022 and write them as text files.

    Args:
        species (str): 'H.sapiens': 9606, 'M.musculus': 10090, 'D.melanogaster': 7227 or 'C.elegans': 6239.
        
    Returns:
        Text files with motif matrix IDs, motif name and position frequency matrices (PFMs)
    
    """
    # Set path 
    download_dir = os.path.join(download_path, species.replace('.', '_'), 'PFMs')
    # Provie Jaspar db release
    jdb_obj = jaspardb(release='JASPAR2022')
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        # Select motifs based on species
        if species == 'H.sapiens':
            motifs = jdb_obj.fetch_motifs(
            collection = 'CORE',
            species = ['9606'])
            print(f"You selected to fetch {species} PFMs and {len(motifs)} were retrieved in total from JASPAR20022 database")
        elif species == 'M.musculus':
            motifs = jdb_obj.fetch_motifs(
            collection = 'CORE',
            species = ['10090'])
            print(f"You selected to fetch {species} PFMs and {len(motifs)} were retrieved in total from JASPAR20022 database")
        elif species == 'D.melanogaster':
            motifs = jdb_obj.fetch_motifs(
            collection = 'CORE',
            species = ['7227'])
            print(f"You selected to fetch {species} PFMs and {len(motifs)} were retrieved in total from JASPAR20022 database")
        elif species == 'C.elegans':
            motifs = jdb_obj.fetch_motifs(
            collection = 'CORE',
            species = ['6239'])
            print(f"You selected to fetch {species} PFMs and {len(motifs)} were retrieved in total from JASPAR20022 database")
        else:
            print(f"You selected to fetch {species} PFMs that are not supported or there is a typo in species name")
        # Write each motif as a separate file
        for motif in motifs:
            # Replace '::' and '-' in motif names with underscore ('_')
            motif.name = motif.name.replace('::', '_').replace('-', '_')
            # Replace '.' in motif matrix ids with underscore ('_')
            motif.matrix_id = motif.matrix_id.replace('.', '_')
            # Write each motif as a separate file
            with open(os.path.join(download_dir, f"{motif.name}_{motif.matrix_id}"), 'a') as myfile:
                myfile.write(">")
                myfile.write(motif.matrix_id)
                myfile.write(" ")
                myfile.write(motif.name)
                myfile.write("\n")
                myfile.write(str(motif.counts))  
    else:
        print(f"Folder {download_dir} already exists.")


def pfm_to_motevo_format(working_dir='motifs/JASPAR2022', species = 'H.sapiens', geneID_matrixID_delim = '_MA'):
    """Function to convert JASPAR pfms to ppms in Motevo format (PPM * 100) 

    Args:
        species (str): 'H.sapiens': 9606, 'M.musculus': 10090, 'D.melanogaster': 7227 or 'C.elegans': 6239.
        
    Returns:
        Text files with motif matrix IDs, motif name and position frequency matrices (PPMs) in Motevo format
        Table with Gene_name, Matrix_id, PFMs, PPMs, and PPMs in Motevo format
    
    """
    # Set the paths to downladed Jaspar PFMs (download_dir) and writing dir for PPMs in Motevo format  
    download_dir = os.path.join(working_dir, species.replace('.', '_'), 'PFMs')
    writing_dir = os.path.join(working_dir, species.replace('.', '_'), 'PPMs_Motevo')
    # Create an empty object to count PPMs with fractional values
    pfms_with_fractional_values = []
    ppms_not_valid = []
    # Create empty df
    df = pd.DataFrame()
    if not os.path.exists(writing_dir):
        os.makedirs(writing_dir)
        # List PFM files
        files = sorted(os.listdir(download_dir))
        if len(files) > 1:
            # Print how many files will be converted
            print (f"There are {len(files)} files to convert. I start my job")
            for file in files:
                # Open and read files
                open_file = open(os.path.join(download_dir, file), "r")
                text_data = open_file.read().split("\n")
                text_data = list(filter(None, text_data))
                print(f"checking file '{file}' for conversion")
                # Check each element of the list and its position: the header ">", numeration and 
                # nucleotides 'A', 'C', 'G', 'T'.
                if text_data[0].startswith(">") and " ".join(text_data[1].split()).startswith("0") \
                and text_data[2].startswith("A") and text_data[3].startswith("C") and \
                text_data[4].startswith("G") and text_data[5].startswith("T"):
                    print("1st line: header is valid")
                    print("2nd line: numeration starts from 0: OK")
                    print("3rd line: starts with 'A': OK")
                    print("4th line: starts with 'C': OK")
                    print("5th line: starts with 'G': OK")
                    print("6th line: starts with 'T': OK")
                    print("file is checked, format is valid")
                else:
                    raise ValueError ("PFM is not valid. You should visually inspect this motif. I quit :(")
                # Extract values for 'A', 'C', 'G', 'T' at each position and edit format for numpy array
                list_of_values = []
                for i in text_data:
                    if any(i.startswith(x) for x in ["A", "C", "G", "T"]):
                        list_of_values.append(re.sub("[A:]|[C:]|[G:]|[T:]", "", i).split())
                list_of_values
                # Convert list to a data frame (df)
                ACGT_df = pd.DataFrame(list_of_values)
                # Convert df into a numpy array
                ACGT_array = pd.DataFrame.to_numpy(pd.DataFrame(ACGT_df))
                # Transpose numpy array to bring it to the format for the function 'pfm_to_ppm'
                ACGT_array_transposed = np.transpose(ACGT_array)
                # Convert characters into numbers
                ACGT_array_transposed = ACGT_array_transposed.astype(np.float)
                # Check if array contains fractional values (as a rule must not)
                try:
                    check_pfm(ACGT_array_transposed)
                except ValueError:
                    print(f"Oh no! Input array contains fractional values. I will convert into integers \
                    but keep in mind, '{file}' PFM may have some problems")
                    pfms_with_fractional_values.append(file)
                    ACGT_array_transposed=ACGT_array_transposed.round(decimals=0, out=None)
                # Run pfm_to_ppm function on prepared array; add pseudocount 0.1/100
                # This is required because of Motevo format: pfm values are multiplied by 100
                ACGT_PPM = pfm_to_ppm(ACGT_array_transposed, pseudocount = 0.1/100)
                # Check if ppm is valid
                try:
                    check_ppm(ACGT_PPM)
                except ValueError:
                    print(f"Oh no! converted ppms may not be valid. I will report them")
                    ppms_not_valid.append(file)
                # Append data about each motif to a data frame
                df = df.append({
                    'Matrix_id': file[file.index(geneID_matrixID_delim):].replace('_','', 1),
                    'Gene_name': file[:file.index(geneID_matrixID_delim)],
                    'PFM': ACGT_array_transposed,
                    'PPM': ACGT_PPM,
                    'PPM_Motevo': ACGT_PPM*100,
                    'weights': ppm_to_pwm(ACGT_PPM)},
                    ignore_index= True)
                # Write files
                with open(os.path.join(writing_dir, file), 'a') as myfile:
                    myfile.write("//")
                    myfile.write("\n")
                    myfile.write("NA ")
                    myfile.write(file)
                    myfile.write("\n")
                    myfile.write("\t")
                    myfile.write("A")
                    myfile.write("\t")
                    myfile.write("C")
                    myfile.write("\t")
                    myfile.write("G")
                    myfile.write("\t")
                    myfile.write("T")
                    myfile.write("\n")
                    for i, line in enumerate((ACGT_PPM*100).round(decimals=3, out=None)):
                        line_fin = str(i+1).zfill(2) + "\t" + np.array2string(line).strip("[]")
                        line_fin = " ".join(line_fin.split())
                        myfile.write(line_fin.replace(' ', '\t'))
                        myfile.write("\n")
                    myfile.write("//")
        else: print(f"There are no files for conversion, check your input folder")
    else: print(f"Folder {writing_dir} already exists.")
    print(f"I'm done. The following PFMs can be problematic: {pfms_with_fractional_values}, \
    there is/are {len(pfms_with_fractional_values)} of them. There are {len(ppms_not_valid)} ppms found to be not valid")
    return df