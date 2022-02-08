# General imports
import numpy as np
import pandas as pd
from itertools import combinations
import warnings

# Stats imports
from sklearn.cluster import AgglomerativeClustering


# Functions to check matrices

def check_ppm(probs, warn=False, eps=1e-3):
    """Function to check validity of a PPM.

    Args:
        probs (np.array): Array containing probability values

    Raises:
        ValueError: if probs is an invalid PPM.

    """
    if (np.min(probs) < (0 - eps)) or (np.max(probs) > (1 + eps)):
        raise ValueError('Values are not within the range [0,1].')
    rowsums = np.sum(probs, axis=1)
    for s in rowsums:
        if (s < (1 - eps)) or (s > (1 + eps)):
            if warn:
                warnings.warn('Rows do not all sum to 1. Check the values.')
            else:
                raise ValueError('Rows do not all sum to 1.')
    if probs.shape[1] != 4:
        raise ValueError('Input array does not have 4 columns.')


def check_pfm(freqs, warn=False):
    """Function to check validity of a PFM.

    Args:
        freqs (np.array): Array containing frequency values

    Raise:
        ValueError: if freqs is an invalid PFM.
    """
    if freqs.shape[1] != 4:
        raise ValueError('Input array does not have 4 columns.')
    if np.min(freqs) < 0:
        raise ValueError('Input array contains values less than 0.')
    if np.any(freqs % 1 != 0):
        if warn:
            warnings.warn('Input array contains fractional values.')
        else:
            raise ValueError('Input array contains fractional values.')


def check_pwm(weights):
    """Function to check validity of a PWM.

    Args:
        weights (np.array): Array containing PWM weights

    Raise:
        ValueError: if weights is an invalid PWM.
    """
    if weights.shape[1] != 4:
        raise ValueError('Input array does not have 4 columns.')


# Functions to calculate matrix properties

def entropy(probs):
    """Function to calculate entropy of a PPM or column of a PPM.

    Args:
        probs (np.array): Array containing probability values

    Returns:
        result (float): Entropy value

    """
    result = -np.sum(probs*np.log2(probs))
    return result


def position_wise_ic(probs):
    """Function to calculate information content of each position in a PPM.

    Args:
        probs (np.array): array containing PPM probability values

    Returns:
        result (np.array): information content of each column in probs.

    """
    check_ppm(probs)
    position_wise_entropy = np.apply_along_axis(entropy, axis=1, arr=probs)
    result = 2 - position_wise_entropy
    return result


# Functions to convert matrix types

def ppm_to_pwm(probs):
    """Function to convert PPM to PWM.

    Args:
        probs (np.array): array containing PPM probability values

    Returns:
        Numpy array containing PWM.

    """
    check_ppm(probs)
    return np.log2(probs/0.25)


def pfm_to_ppm(freqs, pseudocount=0.1):
    """Function to convert PFM to PPM.

    Args:
        freqs (np.array): array containing PFM values
        pseudocount (float): pseudocount to add

    Returns:
        Numpy array containing PPM.

    """
    check_pfm(freqs, warn=False)
    freqs = freqs + (pseudocount/4)
    return normalize_pm(freqs)


def pwm_to_ppm(weights):
    return np.exp2(weights)/4


# Functions to manipulate matrices

def normalize_pm(pm):
    """Function to normalize position matrix so that all rows sum to 1.

    Args:
        pm (np.array): Array containing probability values

    Return:
        np.array with all rows normalized to sum to 1.
    """
    return np.array([x / np.sum(x) for x in pm])


def trim_ppm(probs, frac_threshold):
    """Function to trim non-informative columns from ends of a PPM.

    Args:
        probs (np.array): array containing PPM probability values
        frac_threshold (float): threshold between 0 and 1. The mean information content of all columns in
                                the matrix will be calculated and any continuous columns in the beginning and start
                                of the matrix that have information content less than frac_threshold * mean IC will
                                be dropped.

    Returns:
        result (np.array): array containing trimmed PPM.
    """
    # Checks
    check_ppm(probs)
    assert (frac_threshold >= 0) and (frac_threshold <= 1)

    # Calculate information content
    pos_ic = position_wise_ic(probs)

    # Identify positions to trim
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


# Functions to calculate similarity between matrices

def cos_sim(a, b):
    """Function to calculate cosine similarity between two vectors.

    Args:
        a, b (np.array): 1-D arrays.

    Returns:
        result (float): cosine similarity between a and b.

    """
    result = (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    return result


def matrix_correlation(X, Y):
    """Function to calculate correlation between two equal-sized matrices.

    Args:
        X, Y (np.array): two numpy arrays with same shape

    Returns:
        corr (float): correlation between X and Y

    Raises:
        ValueError: if X and Y do not have equal shapes.

    """
    # Check shapes
    if X.shape != Y.shape:
        raise ValueError('Inputs do not have equal shapes.')

    # Calculate correlation
    corr = np.corrcoef(np.concatenate(X), np.concatenate(Y))[0,1]

    return corr


def order_pms(X, Y):
    """Function to order two matrices by width.

    Inputs:
        X, Y (np.array): two position matrices.

    Returns:
        X, Y (np.array): two position matrices ordered by width.
    """
    if len(Y) < len(X):
        X, Y = Y, X
    return X, Y


def align_pms(X, Y, min_overlap=None):
    """Function to calculate normalized correlation between two position matrices.

    Inputs:
        X, Y (np.array): two position matrices.
        min_overlap (int): minimum overlap allowed between matrices

    Returns:
        result: tuples containing matrix start and end positions corresponding to each alignment.
    """

    X, Y = order_pms(X, Y)
    Lx = len(X)
    Ly = len(Y)

    # Set minimum allowed overlap
    if min_overlap is not None:
        assert type(min_overlap) == int
        assert (min_overlap > 0) & (min_overlap <= Lx)
    else:
        min_overlap = min(3, Lx)

    # Identify different possible alignments of the two matrices
    aln_starts = range(min_overlap - Lx, Ly - min_overlap + 1)

    # Identify the aligned portions of the matrices
    X_starts = [max(-i, 0) for i in aln_starts] # if i<0, cut X positions that don't align to Y
    X_ends = [min(Lx, Ly - i) for i in aln_starts] # trim columns of X that don't align to Y on the right
    Y_starts = [max(i, 0) for i in aln_starts] # if i>0, cut Y from the left
    Y_ends = [min(Ly, i + Lx) for i in aln_starts] # if i+Lx exceeds Ly, cut alignment at Ly

    X_w = np.unique([e - s for e, s in zip(X_ends, X_starts)])
    Y_w = np.unique([e - s for e, s in zip(Y_ends, Y_starts)])
    assert np.all(X_w == Y_w)
    assert np.all(X_w >= min_overlap)

    result = zip(X_starts, X_ends, Y_starts, Y_ends)
    return result


def ncorr(X, Y, min_overlap=None):
    """Function to calculate normalized correlation between two position matrices.

    Inputs:
        X, Y (np.array): two position matrices.
        min_overlap (int): minimum overlap allowed between matrices

    Returns:
        result (float): normalized correlation value
    """
    ncorrs = []

    X, Y = order_pms(X, Y)
    alignments = align_pms(X, Y, min_overlap=None)

    for X_start, X_end, Y_start, Y_end in alignments:

        # Get portions of matrices that are aligned
        Y_i = Y[Y_start : Y_end, : ]
        X_i = X[X_start : X_end, :]

        # Calculate the length of the alignment
        w = Y_end - Y_start

        # Calculate the correlation
        corr = matrix_correlation(X_i, Y_i)

        # Normalize the correlation
        W = len(X) + len(Y) - w
        ncorr = corr * w/W
        ncorrs.append(ncorr)

    # Return the highest normalized correlation value across all alignments.
    result = max(ncorrs)
    return result


def pairwise_ncorrs(mats):
    """Function to calculate all pairwise normalized correlations between a list of position matrices.

    Args:
        mats (list): list of position matrices.

    Returns:
        ncorrs (np.array): All pairwise normalized correlations between matrices in mats.

    """
    # Get all pairwise combinations of matrices
    combins = list(combinations(range(len(mats)), 2))

    # Calculate pairwise similarities between all matrices
    sims = np.zeros(shape=(len(mats), len(mats)))
    for i, j in combins:
        if i != j:
            sims[i, j] = sims[j, i] = ncorr(mats[i], mats[j])

    # Set diagonal values to 1
    for i in range(len(mats)):
        sims[i, i] = 1

    # return
    return sims


# Functions to cluster matrices and choose representatives

def choose_representative_pm(df, sims=None, maximize='median', weight_col='weights', matrix_type='PWM'):
    """Function to choose a representative position matrix from a group.

    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in df
        maximize (str): 'mean' or 'median'. Metric  to choose representative matrix.
        weight_col(str): column in df that contains matrix values.
        matrix_type (str): PPM or PWM

    Returns:
        result (list): IDs for the selected representative matrices.

    """
    mats = list(df[weight_col].values)
    ids = list(df.Matrix_id)

    # Get pairwise similarities
    if sims is None:
        sims = pairwise_ncorrs(mats)

    if len(mats)==2:
        # Between two matrices, choose the one with lowest entropy
        if matrix_type == 'PPM':
            entropies = [entropy(mat) for mat in mats]
        elif matrix_type == 'PWM':
            entropies = [entropy(pwm_to_ppm(mat)) for mat in mats]
        else:
            raise ValueError("matrix_type should be PPM or PWM.")
        sel_mat = np.argmin(entropies)
    elif len(mats) > 2:
        # Otherwise, choose the matrix closest to other matrices
        if maximize == 'mean':
            sel_mat = np.argmax(np.mean(sims, axis=0))
        elif maximize == 'median':
            sel_mat = np.argmax(np.median(sims, axis=0))
    else:
        sel_mat = 0

    # Final result
    result = ids[sel_mat]
    return result


def choose_cluster_representative_pms(df, sims=None, clusters=None, maximize='median',
                                       weight_col='weights', matrix_type='PWM'):
    """Function to choose a representative position matrix from each cluster.

    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in pwms
        clusters (list): cluster assignments for each PWM.
        maximize (str): 'mean' or 'median'. Metric  to choose representative matrix.
        weight_col(str): column in pwms that contains matrix values.
        matrix_type (str): PPM or PWM

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
        sel_mat = choose_representative_pm(mats, sims=c_sims, maximize=maximize,
                                            weight_col=weight_col, matrix_type=matrix_type)
        representatives.append(sel_mat)
    return representatives


def cluster_pms(df, n_clusters, sims=None, weight_col='weights'):
    """Function to cluster position matrices.

    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        n_clusters (int): Number of clusters
        sims (np.array): pairwise similarities between all matrices in pwms
        weight_col(str): column in pwms that contains matrix values.

    Returns:
        result: dictionary containing cluster labels, representative matrix IDs, and
                minimum pairwise similarity within each cluster

    """
    # Get pairwise similarities between PMs
    if sims is None:
        sims = pairwise_ncorrs(list(df[weight_col]))

    # Cluster using agglomerative clustering
    cluster_ids = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                          distance_threshold=None, linkage='complete').fit(2-sims).labels_

    # Choose representative matrix from each cluster
    reps = choose_cluster_representative_pms(df, sims=sims, clusters=cluster_ids,
                               maximize='median', weight_col=weight_col)
    min_ncorrs = [np.min(sims[cluster_ids==i, :][:, cluster_ids==i]) for i in range(n_clusters)]
    result = {'clusters':cluster_ids, 'reps':reps, 'min_ncorr': min_ncorrs}
    return result
