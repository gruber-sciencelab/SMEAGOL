# General imports
import numpy as np
from itertools import combinations
import warnings

# Stats imports
from sklearn.cluster import AgglomerativeClustering


# Functions to check matrices


def check_ppm(probs, warn=False, eps=1e-2):
    """Function to check the validity of a PPM.
    A valid PPM should have all values within the range [0,1],
    all rows summing to 1 (within a tolerance of eps), and exactly
    4 columns.

    Args:
        probs (np.array): Numpy array containing probability
                          values. The array should be 2-dimensional
                          and have 4 columns.

    Raises:
        ValueError: if probs is an invalid PPM.

    """
    if (np.min(probs) < (0 - eps)) or (np.max(probs) > (1 + eps)):
        raise ValueError("Values are not within the range [0,1].")
    rowsums = np.sum(probs, axis=1)
    for s in rowsums:
        if (s < (1 - eps)) or (s > (1 + eps)):
            if warn:
                warnings.warn("Rows do not all sum to 1. Check the values.")
            else:
                raise ValueError("Rows do not all sum to 1.")
    if probs.shape[1] != 4:
        raise ValueError("Input array does not have 4 columns.")


def check_pfm(freqs, warn=False):
    """Function to check the validity of a PFM.
    A valid PFM should have non-negative integer values and exactly
    4 columns.

    Args:
        freqs (np.array): Numpy array containing frequency values.
                          The array should be 2-dimensional and have
                          4 columns.

    Raise:
        ValueError: if freqs is an invalid PFM.
    """
    if freqs.shape[1] != 4:
        raise ValueError("Input array does not have 4 columns.")
    if np.min(freqs) < 0:
        raise ValueError("Input array contains values less than 0.")
    if np.any(freqs % 1 != 0):
        if warn:
            warnings.warn("Input array contains fractional values.")
        else:
            raise ValueError("Input array contains fractional values.")


def check_pwm(weights):
    """Function to check the validity of a PWM.
    A valid PWM should have exactly 4 columns.

    Args:
        weights (np.array): Numpy array containing PWM weights.
                            The array should be 2-dimensional and have
                            4 columns.

    Raise:
        ValueError: if weights is an invalid PWM.
    """
    if weights.shape[1] != 4:
        raise ValueError("Input array does not have 4 columns.")


# Functions to calculate matrix properties


def entropy(probs):
    """Function to calculate the entropy of a PPM or column
    of a PPM. Entropy is calculated using the formula
    `H = -sum(p*log2(p))`.

    Args:
        probs (np.array): Numpy array containing probability values.

    Returns:
        result (float): Entropy value

    """
    result = -np.sum(probs * np.log2(probs))
    return result


def position_wise_ic(probs):
    """Function to calculate the information content (IC) of
    each position in a PPM. The IC for a specific position is
    calculated using the formula: IC = 2 + sum(p*log2(p)),
    assuming a background probability of 0.25 for each of the 4 bases.

    Args:
        probs (np.array): Numpy array containing PPM probability values.

    Returns:
        result (np.array): Numpy array containing the calculated
                           information content of each column in probs.

    """
    check_ppm(probs)
    position_wise_entropy = np.apply_along_axis(entropy, axis=1, arr=probs)
    result = 2 - position_wise_entropy
    return result


# Functions to convert matrix types


def ppm_to_pwm(probs):
    """Function to convert a valid PPM into a PWM, using
    the formula: PWM = log2(PPM/B), where the background
    probability B is set to 0.25.

    Args:
        probs (np.array): Numpy array containing PPM probability values

    Returns:
        Numpy array containing the PWM. The shape of this
        array is (L, 4) where L is the PWM length.

    """
    check_ppm(probs)
    return np.log2(probs / 0.25)


def pfm_to_ppm(freqs, pseudocount=0.1):
    """Function to convert a valid PFM into a PPM. The matrix
    is normalized so that every position sums to 1, thus
    converting from a matrix of frequencies to a matrix of
    probabilities. To avoid zeros, a value of pseudocount/4
    is first added to each position before normalization.

    Args:
        freqs (np.array): array containing PFM values
        pseudocount (float): pseudocount to add

    Returns:
        Numpy array containing PPM.

    """
    check_pfm(freqs, warn=False)
    freqs = freqs + (pseudocount / 4)
    return normalize_pm(freqs)


def pwm_to_ppm(weights):
    return np.exp2(weights) / 4


# Functions to manipulate matrices


def normalize_pm(pm):
    """Function to normalize a position matrix so that all rows
    sum to 1.

    Args:
        pm (np.array): Numpy array containing probability values

    Return:
        Numpy array with all rows normalized to sum to 1.
    """
    return np.array([x / np.sum(x) for x in pm])


def trim_ppm(probs, frac_threshold):
    """Function to trim non-informative positions from the ends
    of a PPM. See `position_wise_ic` for how information content
    of each position is calculated.

    Args:
        probs (np.array): Numpy array containing PPM probability values
        frac_threshold (float): threshold between 0 and 1. The
                                mean information content (IC) of all
                                columns in the matrix will be calculated
                                and any continuous columns in the
                                beginning and end of the matrix that
                                have IC < frac_threshold * mean IC will
                                be dropped.

    Returns:
        result (np.array): Numpy array containing trimmed PPM.
    """
    # Checks
    check_ppm(probs)
    assert (frac_threshold >= 0) and (frac_threshold <= 1)

    # Calculate information content
    pos_ic = position_wise_ic(probs)

    # Identify positions to trim
    to_trim = (pos_ic / np.mean(pos_ic)) < frac_threshold
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

    result = probs[positions, :]
    return result


# Functions to calculate similarity between matrices


def cos_sim(a, b):
    """Function to calculate the cosine similarity between
    two vectors.

    Args:
        a, b (np.array): 1-D arrays.

    Returns:
        result (float): cosine similarity between a and b.

    """
    result = (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    return result


def matrix_correlation(X, Y):
    """Function to calculate the correlation between two
    equal-sized matrices. The values in each matrix are
    unrolled into a 1-D array and the Pearson correlation
    of the two 1-D arrays is returned.

    Args:
        X, Y (np.array): two numpy arrays with same shape

    Returns:
        corr (float): Pearson correlation between X and Y
                      (unrolled into 1-D arrays)

    Raises:
        ValueError: if X and Y do not have equal shapes.

    """
    # Check shapes
    if X.shape != Y.shape:
        raise ValueError("Inputs do not have equal shapes.")

    # Calculate correlation
    corr = np.corrcoef(np.concatenate(X), np.concatenate(Y))[0, 1]

    return corr


def order_pms(X, Y):
    """Function to order two matrices by width.

    Inputs:
        X, Y (np.array): two position matrices.

    Returns:
        X, Y (np.array): two position matrices ordered
        by width, X being smaller.
    """
    if len(Y) < len(X):
        X, Y = Y, X
    return X, Y


def align_pms(X, Y, min_overlap=None):
    """Function to generate all possible overlaps between two
    position matrices.

    Inputs:
        X, Y (np.array): two position matrices.
        min_overlap (int): minimum overlap allowed between matrices

    Returns:
        result: tuples containing matrix start and end positions
                corresponding to each possible alignment of the
                two matrices.
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
    X_starts = [
        max(-i, 0) for i in aln_starts
    ]  # if i<0, cut X positions that don't align to Y
    X_ends = [
        min(Lx, Ly - i) for i in aln_starts
    ]  # trim columns of X that don't align to Y on the right
    # if i>0, cut Y from the left
    Y_starts = [max(i, 0) for i in aln_starts]
    Y_ends = [
        min(Ly, i + Lx) for i in aln_starts
    ]  # if i+Lx exceeds Ly, cut alignment at Ly

    X_w = np.unique([e - s for e, s in zip(X_ends, X_starts)])
    Y_w = np.unique([e - s for e, s in zip(Y_ends, Y_starts)])
    assert np.all(X_w == Y_w)
    assert np.all(X_w >= min_overlap)

    result = zip(X_starts, X_ends, Y_starts, Y_ends)
    return result


def ncorr(X, Y, min_overlap=None):
    """Function to calculate the normalized Pearson correlation
    between two position matrices, as defined in
    doi: 10.1093/nar/gkx314.

    Inputs:
        X, Y (np.array): two position matrices.
        min_overlap (int): minimum overlap allowed between matrices

    Returns:
        result (float): normalized Pearson correlation value
    """
    ncorrs = []

    X, Y = order_pms(X, Y)
    alignments = align_pms(X, Y, min_overlap=None)

    for X_start, X_end, Y_start, Y_end in alignments:

        # Get portions of matrices that are aligned
        Y_i = Y[Y_start:Y_end, :]
        X_i = X[X_start:X_end, :]

        # Calculate the length of the alignment
        w = Y_end - Y_start

        # Calculate the correlation
        corr = matrix_correlation(X_i, Y_i)

        # Normalize the correlation
        W = len(X) + len(Y) - w
        ncorr = corr * w / W
        ncorrs.append(ncorr)

    # Return the highest normalized correlation value
    # across all alignments.
    result = max(ncorrs)
    return result


def pairwise_ncorrs(mats):
    """Function to calculate all pairwise normalized Pearson
    correlations between a list of position matrices.

    Args:
        mats (list): a list of position matrices.

    Returns:
        ncorrs (np.array): All pairwise normalized Pearson
        correlations between matrices in mats.

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


def choose_representative_pm(
    df, sims=None, maximize="median", weight_col="weights",
    matrix_type="PWM"
):
    """Function to choose a representative position matrix
    from a group. If 2 matrices are supplied, the one with
    lower entropy is chosen. For larger sets of matrices, the
    matrix with the highest mean (or median) normalized Pearson
    correlation to all other matrices is chosen.

    Args:
        df (pandas df): Dataframe containing position matrix
                        values and IDs. The ID column should be
                        named 'Matrix_id'.
        sims (np.array): pairwise similarities between all PWMs in df
        maximize (str): 'mean' or 'median'. Metric to choose
                        representative matrix.
        weight_col(str): the column in df that contains matrix values.
        matrix_type (str): PPM or PWM

    Returns:
        result (list): IDs for the selected representative matrices.

    """
    mats = list(df[weight_col].values)
    ids = list(df.Matrix_id)

    # Get pairwise similarities
    if sims is None:
        sims = pairwise_ncorrs(mats)

    if len(mats) == 2:
        # Between two matrices, choose the one with lowest entropy
        if matrix_type == "PPM":
            entropies = [entropy(mat) for mat in mats]
        elif matrix_type == "PWM":
            entropies = [entropy(pwm_to_ppm(mat)) for mat in mats]
        else:
            raise ValueError("matrix_type should be PPM or PWM.")
        sel_mat = np.argmin(entropies)
    elif len(mats) > 2:
        # Otherwise, choose the matrix closest to other matrices
        if maximize == "mean":
            sel_mat = np.argmax(np.mean(sims, axis=0))
        elif maximize == "median":
            sel_mat = np.argmax(np.median(sims, axis=0))
    else:
        sel_mat = 0

    # Final result
    result = ids[sel_mat]
    return result


def choose_cluster_representative_pms(
    df,
    sims=None,
    clusters=None,
    maximize="median",
    weight_col="weights",
    matrix_type="PWM",
):
    """Function to choose a representative position matrix
    from each cluster. This performs `choose_representative_pm`
    to choose a representative matrix within each supplied cluster.

    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        sims (np.array): pairwise similarities between all PWMs in pwms
        clusters (list): cluster assignments for each PWM.
        maximize (str): 'mean' or 'median'. Metric  to choose
                        representative matrix.
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
        in_cluster = clusters == cluster_id
        mat_ids = np.array(df.Matrix_id)[in_cluster]
        mats = df[df.Matrix_id.isin(mat_ids)]
        if sims is not None:
            c_sims = sims[in_cluster, :][:, in_cluster]
        # Choose representative matrix within cluster
        sel_mat = choose_representative_pm(
            mats,
            sims=c_sims,
            maximize=maximize,
            weight_col=weight_col,
            matrix_type=matrix_type,
        )
        representatives.append(sel_mat)
    return representatives


def cluster_pms(df, n_clusters, sims=None, weight_col="weights"):
    """Function to cluster position matrices. A distance matrix
    between the matrices is computed using the normalized Pearson
    correlation metric and agglomerative clustering is used to
    find clusters. `choose_representative_pm` is called to
    identify a representative matrix from each cluster.

    Args:
        df (pandas df): Dataframe containing position matrix values and IDs.
        n_clusters (int): Number of clusters
        sims (np.array): pairwise similarities between all matrices in pwms
        weight_col(str): column in pwms that contains matrix values.

    Returns:
        result: dictionary containing cluster labels, representative
                matrix IDs, and minimum pairwise similarity within each
                cluster

    """
    # Get pairwise similarities between PMs
    if sims is None:
        sims = pairwise_ncorrs(list(df[weight_col]))

    # Cluster using agglomerative clustering
    cluster_ids = (
        AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            distance_threshold=None,
            linkage="complete",
        )
        .fit(2 - sims)
        .labels_
    )

    # Choose representative matrix from each cluster
    reps = choose_cluster_representative_pms(
        df, sims=sims, clusters=cluster_ids, maximize="median",
        weight_col=weight_col
    )
    min_ncorrs = [
        np.min(sims[cluster_ids == i, :][:, cluster_ids == i])
        for i in range(n_clusters)
    ]
    result = {"clusters": cluster_ids, "reps": reps, "min_ncorr": min_ncorrs}
    return result
