# General imports
import numpy as np
import pandas as pd
import os
import itertools

# Biopython imports
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# I/O imports
import h5py
import gzip
from mimetypes import guess_type
from functools import partial

# Stats imports
from sklearn.cluster import AgglomerativeClustering


# PPM/PWM analysis

def check_ppm(probs):
    """Function to check validity of a PPM.
    
    Args:
        probs (np.array): Array containing probability values
        
    Raises:
        ValueError: if probs is an invalid PPM.
        
    """
    if (np.min(probs) < 0) or (np.max(probs) > 1):
        raise ValueError('Values are not within the range [0,1].')
    if probs.shape[1] != 4:
        raise ValueError('Input array does not have 4 columns.')


# Metrics
def entropy(probs):
    """Function to calculate entropy of a PPM or column of a PPM.
    
    Args:
        probs (np.array): Array containing probability values
    
    Returns:
        result (float): Entropy value

    """
    #check_ppm(probs)
    result = -np.sum(probs*np.log2(probs))
    return result

    
def position_wise_ic(probs, axis=1):
    """Function to calculate information content of each column in a PPM.
    
    Args:
        probs (np.array): array containing PPM probability values
    
    Returns:
        result (np.array): information content of each column in probs.
        
    """
    check_ppm(probs)
    position_wise_entropy = np.apply_along_axis(entropy, axis=axis, arr=probs)
    result = 2 - position_wise_entropy
    return result


def ppm_to_pwm(probs):
    """Function to convert PPM to PWM.
    
    Args:
        probs (np.array): array containing PPM probability values
    
    Returns:
        Numpy array containing PWM.
        
    """
    return np.log2(probs/0.25)    


# Manipulation of position matrices

def trim_ppm(probs, frac_threshold):
    """Function to trim non-informative columns from ends of a PPM.
    
    Args:
        probs (np.array): array containing PPM probability values
        frac_threshold (float): threshold (0-1) to filter out non-informative columns.
    
    Returns:
        result (np.array): array containing trimmed PPM.
    """
    pos_ic = position_wise_ic(probs, axis=1)
    to_trim = (pos_ic/np.mean(pos_ic)) < frac_threshold
    positions = list(range(probs.shape[0]))
    assert len(to_trim) == len(positions)

    # Trim from start
    while to_trim[0]:
        positions = positions[1:]
        to_trim = to_trim[1:]

    # Trim from end
    while to_trim[-1]:
        positions = positions[:-1]
        to_trim = to_trim[:-1]
    
    result = probs[positions,:]
    return result


# Similarity and clustering

def cos_sim(a, b):
    """Function to calculate cosine similarity between two vectors.
    
    Args:
        a, b (np.array): 1-D arrays.
        
    Returns:
        result (float): cosine similarity between a and b.
        
    """
    result = (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))
    return result   


def matrix_correlation(X, Y):
    """Function to calculate per-position similarity between two equal-sized matrices.
    
    Args:
        X, Y (np.array): two numpy arrays with same shape
        
    Returns:
        corr (float): correlation between X and Y
        
    Raises:
        ValueError: if X and Y do not have equal shapes.
        
    """
    if X.shape != Y.shape:
        raise ValueError('inputs do not have equal shapes.')
    corr = np.corrcoef(np.concatenate(X), np.concatenate(Y))[0,1]
    return corr
    

def ncorr(X, Y, min_overlap=None):
    """Function to calculate normalized correlation between two position matrices.
    
    Inputs:
        X, Y (np.array): two position matrices.
        min_overlap (int): minimum overlap allowed between matrices
        
    Returns:
        result (float): normalized correlation value
    """
    # X should be the shorter matrix
    ncorrs = []
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
        min_overlap = 3

    # Slide matrices to try different alignments
    if (Lx == Ly) and ((Lx % 2)==1):
        aln_starts = range(min_overlap - Lx, Ly - min_overlap + 1)
    else:
        aln_starts = range(min_overlap - Lx, Ly - min_overlap)
    for i in aln_starts:
        Y_start = max(i, 0) # if i>0, cut Y from the left
        Y_end = min(Ly, i + Lx) # if i+Lx exceeds Ly, cut alignment at Ly
        X_start = max(-i, 0) # if i<0, cut X positions that don't align to Y
        X_end = min(Lx, Ly - i) # trim columns of X that don't align to Y on the right
        Y_i = Y[Y_start : Y_end, : ]
        X_i = X[X_start : X_end, :]
        w = Y_end - Y_start # no. of overlapping positions
        corr = matrix_correlation(X_i, Y_i)
        W = Lx + Ly - w
        ncorr = corr * w/W
        ncorrs.append(ncorr)

    # Return the highest similarity value across all alignments.
    result = max(ncorrs)
    return result


def pairwise_ncorrs(mats):
    """Function to calculate all pairwise normalized correlations between a list of position matrices.
    
    Args: 
        mats (list): list of position matrices.
        
    Returns:
        ncorrs (np.array): All pairwise normalized correlations between matrices in mats.
    
    """
    # Get all pairwise combinations
    combins = list(itertools.combinations(range(len(mats)), 2))
    
    # Calculate pairwise similarities between all matrices
    sims = np.zeros(shape=(len(mats), len(mats)))
    for i, j in combins:
        sims[i, j] = ncorr(mats[i], mats[j])
        sims[j, i] = sims[i, j]
    for i in range(len(mats)):
        sims[i, i] = 1
    # return
    return sims


def choose_representative_mat(df, sims=None, maximize='median', weight_col='weight'):
    """Function to choose a representative position matrix from a group.
    
    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in pwms
        maximize (str): 'mean' or 'median'. Metric  to choose representative matrix.
        weight_col(str): column in pwms that contains matrix values.
        
    Returns:
        result (list): IDs for the selected representative matrices.
        
    """
    mats = list(df[weight_col].values)
    ids = list(df.Matrix_id)
    if sims is None:
        sims = pairwise_ncorrs(mats)
    if len(mats)==2:
        # Choose matrix with lowest entropy
        entropies = [entropy(np.exp2(mat)/4) for mat in mats]
        sel_mat = np.argmin(entropies)
    elif len(mats)>2:
        # Choose matrix closest to all
        if maximize == 'mean':
            sel_mat = np.argmax(np.mean(sims, axis=0))
        elif maximize == 'median':
            sel_mat = np.argmax(np.median(sims, axis=0))
    else:
        sel_mat = 0
    # Final result
    result = ids[sel_mat]
    return result

    
def choose_cluster_representative_mats(df, sims=None, clusters=None, 
                               maximize='median', weight_col='weight'):
    """Function to choose a representative position matrix from each cluster.
        
    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in pwms
        clusters (list): cluster assignments for each PWM.
        maximize (str): 'mean' or 'median'. Metric  to choose representative matrix.
        weight_col(str): column in pwms that contains matrix values.
        
    Returns:
        representatives (list): IDs for the selected representative matrices.
        
    """
    representatives = []
    cluster_ids = np.unique(clusters)
    c_sims = None
    # Get pairwise similarities within each cluster
    for cluster_id in cluster_ids:
        in_cluster = (clusters==cluster_id)
        mat_ids = np.array(df.Matrix_id)[in_cluster]
        mats = df[df.Matrix_id.isin(mat_ids)]
        if sims is not None:
            c_sims = sims[in_cluster, :][:, in_cluster]
        # Choose representative matrix within cluster
        sel_mat = choose_representative_mat(mats, sims=c_sims, maximize=maximize, 
                                            weight_col=weight_col)
        representatives.append(sel_mat)
    return representatives


def cluster_pwms(df, n_clusters, sims=None, weight_col='weight'):
    """Function to cluster position matrices.
            
    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        n_clusters (int): Number of clusters
        sims (np.array): pairwise similarities between all PWMs in pwms
        weight_col(str): column in pwms that contains matrix values.
        
    Returns:
        result: dictionary containing cluster labels, representative matrix IDs, and
                minimum pairwise similarity within each cluster      
        
    """
    if sims is None:
        sims = pairwise_ncorrs(list(df[weight_col]))
    cluster_ids = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', 
                                          distance_threshold=None, linkage='complete').fit(2-sims).labels_
    reps = choose_cluster_representative_mats(df, sims=sims, clusters=cluster_ids, 
                               maximize='median', weight_col=weight_col)
    min_ncorrs = [np.min(sims[cluster_ids==i, :][:, cluster_ids==i]) for i in range(n_clusters)]
    result = {'clusters':cluster_ids, 'reps':reps, 'min_ncorr': min_ncorrs}
    return result
