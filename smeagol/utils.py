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

# Viz imports
from .visualize import plot_pwm_similarity


# PPM/PWM analysis

# Check

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
    check_ppm(probs)
    result = -np.sum(probs*np.log2(probs))
    return result


def avg_entropy(probs):
    """Function to calculate average entropy over columns of a PPM.
    
    Args:
        probs (np.array): array containing PPM probability values
    
    Returns:
        result (float): Average entropy value
    
    """
    check_ppm(probs)
    result = entropy(probs)/np.shape(probs)[0]
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


def row_wise_equal_matrix_similarity(X, Y, metric='cosine'):
    """Function to calculate per-position similarity between two equal-sized matrices.
    
    Args:
        X, Y (np.array): two numpy arrays with same shape
        metric (str): 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        
    Returns:
        row_wise_sims (list): row (position)-wise similarity between X and Y
        
    Raises:
        ValueError: if X and Y do not have equal shapes.
        
    """
    if X.shape != Y.shape:
        raise ValueError('inputs do not have equal shapes.')
    if metric == 'corr':
        row_wise_sims = [np.corrcoef(X[i], Y[i])[0,1] for i in range(X.shape[0])]
    elif metric == 'cosine':
        row_wise_sims = [cos_sim(X[i], Y[i]) for i in range(X.shape[0])]
    return row_wise_sims
    

def matrix_similarity(X, Y, metric='cosine', min_overlap=None, pad=False):
    """Function to calculate similarity between two position matrices.
    
    Inputs:
        X, Y (np.array): two position matrices.
        metric (str): 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        min_overlap (int): minimum overlap allowed between matrices
        pad (bool): if one matrix is shorter, count 0 similarity between empty positions
        
    Returns:
        result (float): similarity value
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
        min_overlap = Lx - 2

        #if Ly - Lx >= 3:
            #min_overlap = Lx - 1
        #else:
            #min_overlap = int(np.ceil(Lx/2))

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
    result = max(sims)
    return result


def pairwise_similarities(mats, metric='cosine', pad=False):
    """Function to calculate all pairwise similarities between a list of position matrices.
    
    Args: 
        mats (list): list of position matrices.
        metric (str): 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        pad (bool): if one matrix is shorter, count 0 similarity between empty positions
        
    Returns:
        sims (np.array): All pairwise similarities between matrices in mats.
    
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
    # return
    return sims


def choose_representative_mat(df, sims=None, metric='cosine', maximize='median', pad=False, 
                              weight_col='probs', pm_type='ppm'):
    """Function to choose a representative position matrix from a group.
    
    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in pwms
        metric (str): 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        maximize (str): 'mean' or 'median'. Metric  to choose representative matrix.
        pad (bool): if one matrix is shorter, count 0 similarity between empty positions
        weight_col(str): column in pwms that contains matrix values.
        pm_type (str): 'ppm' or 'pwm'
        
    Returns:
        result (list): IDs for the selected representative matrices.
        
    """
    mats = list(df[weight_col].values)
    ids = list(df.Matrix_id)
    if sims is None:
        sims = pairwise_similarities(mats, metric=metric, pad=pad)
    if len(mats)==2:
        # Choose matrix with lowest entropy
        if pm_type == 'ppm':
            entropies = [avg_entropy(mat) for mat in mats]
        elif pm_type == 'pwm':
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
    # Final result
    result = ids[sel_mat]
    return result

    
def choose_cluster_representative_mats(df, sims=None, clusters=None, metric='cosine', 
                               maximize='median', pad=False, weight_col='probs', pm_type='ppm'):
    """Function to choose a representative position matrix from each cluster.
        
    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in pwms
        clusters (list): cluster assignments for each PWM.
        metric (str): 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        maximize (str): 'mean' or 'median'. Metric  to choose representative matrix.
        pad (bool): if one matrix is shorter, count 0 similarity between empty positions
        weight_col(str): column in pwms that contains matrix values.
        pm_type (str): 'ppm' or 'pwm'
        
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
        sel_mat = choose_representative_mat(mats, sims=c_sims, metric=metric, 
                                            maximize=maximize, pad=pad, weight_col=weight_col,
                                            pm_type=pm_type)
        representatives.append(sel_mat)
    return representatives


def cluster_pwms(df, n_clusters, sims=None, weight_col='probs', perplexity=4, plot=True, 
                 metric='cosine', pad=False, output='reps', pm_type='ppm'):
    """Function to cluster position matrices.
            
    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        n_clusters (int): Number of clusters
        sims (np.array): pairwise similarities between all PWMs in pwms
        weight_col(str): column in pwms that contains matrix values.
        perplexity (int): parameter for t-SNE plot.
        plot (bool): whether to show t-SNE plot.
        metric (str): 'cosine' (Cosine similarity) or 'corr' (Pearson correlation)
        pad (bool): if one matrix is shorter, count 0 similarity between empty positions
        output (str): 'reps' (representative matrices for each cluster) or 'clusters' (cluster IDs).
        pm_type (str): 'ppm' or 'pwm'
        
    Returns:
        reps (list): IDs for the selected representative matrices.
        
        
    """
    if sims is None:
        sims = pairwise_similarities(list(df[weight_col]), metric=metric, pad=pad)
    cluster_ids = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', 
                                          distance_threshold=None, linkage='complete').fit(1-sims).labels_
    if plot:
        cmap = {3:'purple', 2:'green', 1:'blue', 0:'red', -1:'orange'}
        plot_pwm_similarity(sims, df.Matrix_id, perplexity=perplexity, clusters=cluster_ids, cmap=cmap)
    if output=='reps':
        reps = choose_cluster_representative_mats(df, sims=sims, clusters=cluster_ids, metric=metric, 
                               maximize='median', pad=pad, weight_col=weight_col, pm_type=pm_type)
        print("Representatives: " + str(reps))
        return reps
    elif output == 'clusters':
        return cluster_ids
