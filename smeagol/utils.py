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

# Stats imports
from sklearn.cluster import AgglomerativeClustering

# Viz imports
from .visualization import plot_pwm_similarity


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


# metrics

def cos_sim(a, b):
    """
    Function to calculate cosine similarity between two vectors.
    """
    return (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))   


def row_wise_equal_matrix_similarity(X, Y, metric='cosine'):
    """
    Function to calculate per-row similarity between two equal-sized matrices.
    
    Inputs:
        X, Y: two numpy arrays with same shape
        metric: 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
    """
    assert X.shape == Y.shape
    if metric == 'corr':
        row_wise_sims = [np.corrcoef(X[i], Y[i])[0,1] for i in range(X.shape[0])]
    elif metric == 'cosine':
        row_wise_sims = [cos_sim(X[i], Y[i]) for i in range(X.shape[0])]
    return row_wise_sims
    

def matrix_similarity(X, Y, metric='cosine', min_overlap=None, pad=False):
    """
    Function to calculate similarity between two position matrices.
    
    Inputs:
        X, Y: two position matrices as numpy arrays.
        metric: 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        min_overlap: minimum overlap allowed between matrices
        pad: if one matrix is shorter, count 0 similarity between empty positions
        
    Returns:
        similarity value
    """
    # X should be the shorter matrix
    sims = []
    Ly = len(Y)
    Lx = len(X)
    if Ly < Lx:
        X_orig = X
        X = Y
        Y = X_orig
    Ly = len(Y)
    Lx = len(X)
    
    # Set minimum allowed overlap
    if min_overlap is None:
        if Ly - Lx >= 3:
            min_overlap = Lx - 1
        else:
            min_overlap = int(np.ceil(Lx/2))

    # Slide matrices to try different alignments
    if (Lx == Ly) and ((Lx % 2)==1):
        aln_starts = range(min_overlap - Lx, Ly - min_overlap + 1)
    else:
        aln_starts = range(min_overlap - Lx, Ly - min_overlap)
    for i in aln_starts:
        Y_i = Y[max(i, 0):min(Ly, Lx + i), : ]
        X_i = X[max(-i, 0):min(Lx, Ly - i), :]
        w = min(Ly, Lx + i) - max(i, 0)
        row_wise_sims = row_wise_equal_matrix_similarity(X_i, Y_i, metric=metric)
        # Extend alignment with zeros (optional)
        if (pad) and (w < Ly):
            row_wise_sims.extend([0]*(Ly-w))
        sims.append(np.mean(row_wise_sims))

    # Return the highest similarity value across all alignments.
    return max(sims)


def pairwise_similarities(mats, metric='cosine', pad=False):
    """
    Function to calculate all pairwise similarities between a list of position matrices.
    
    """
    # Get all pairwise combinations
    combins = list(itertools.combinations(range(len(mats)), 2))
    
    # Calculate pairwise similarities between all matrices
    sims = np.zeros(shape=(len(mats), len(mats)))
    for i, j in combins:
        sims[i, j] = matrix_similarity(mats[i], mats[j], metric=metric, pad=pad)
        sims[j, i] = sims[i, j]
    for i in range(len(mats)):
        sims[i, i] = 1
    print("Minimum pairwise similarity: " + str(np.min(sims)))
    return sims


def choose_representative_mat(pwms, sims=None, metric='cosine', maximize='median', pad=False, 
                              weight_col='probs'):
    """
    Function to choose a representative position matrix from a group.
    """
    mats = list(pwms[weight_col].values)
    ids = list(pwms.Matrix_id)
    if sims is None:
        sims = pairwise_similarities(mats, metric=metric, pad=pad)
    if len(mats)==2:
        # Choose matrix with lowest entropy
        entropies = [avg_entropy(np.exp2(mat)/4) for mat in mats]
        sel_mat = np.argmin(entropies)
    elif len(mats)>2:
        # Choose matrix closest to all
        if maximize == 'mean':
            sel_mat = np.argmax(np.mean(sims, axis=0))
        elif maximize == 'median':
            sel_mat = np.argmax(np.median(sims, axis=0))
    else:
        sel_mat = 0
    return ids[sel_mat]

    
def choose_cluster_representative_mats(pwms, sims=None, clusters=None, metric='cosine', 
                               maximize='median', pad=False, weight_col='probs'):
    """
    Function to choose a representative position matrix from each cluster.
    """
    representatives = []
    cluster_ids = np.unique(clusters)
    c_sims = None
    for cluster_id in cluster_ids:
        in_cluster = (clusters==cluster_id)
        if np.sum(in_cluster) > 1:
            mat_ids = np.array(pwms.Matrix_id)[in_cluster]
            mats = pwms[pwms.Matrix_id.isin(mat_ids)]
            if sims is not None:
                c_sims = sims[in_cluster, :][:, in_cluster]
            sel_mat = choose_representative_mat(mats, sims=c_sims, metric=metric, 
                                                maximize=maximize, pad=pad, weight_col=weight_col)
            representatives.append(sel_mat)
    return representatives


def cluster_pwms(pwms, n_clusters, sims=None, weight_col='probs', perplexity=4, plot=True, 
                 metric='cosine', pad=False, output='reps'):
    """
    Function to cluster position matrices.
    """
    if sims is None:
        sims = pairwise_similarities(list(pwms[weight_col]), metric=metric, pad=pad)
    cluster_ids = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', 
                                          distance_threshold=None, linkage='complete').fit(1-sims).labels_
    if plot:
        cmap = {3:'purple', 2:'green', 1:'blue', 0:'red', -1:'orange'}
        plot_pwm_similarity(sims, pwms.Matrix_id, perplexity=perplexity, clusters=cluster_ids, cmap=cmap)
    if output=='reps':
        reps = choose_cluster_representative_mats(pwms, sims=sims, clusters=cluster_ids, metric=metric, 
                               maximize='median', pad=pad, weight_col=weight_col)
        print("Representatives: " + str(reps))
        return reps
    elif output == 'clusters':
        return cluster_ids
