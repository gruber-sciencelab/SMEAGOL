import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import itertools


def predict(encoding, model, threshold, score=False, method="fast"):
    """Prediction by scanning an encoded sequence with a convolutional model.
    
    Args:
        encoding (SeqEncoding): SeqEncoding object
        model (model): Prameterized convolutional model
        threshold (float): fraction of maximum score to use as binding site detection threshold
        score (bool): Output binding site scores as well as positions
        method (str): 'fast' (default) or 'lowmem' (slower but uses less memory)
        
    Returns:
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
        scores (np.array): score for each potential binding site
        
    """
    assert (threshold >= 0) & (threshold <= 1) 
    thresholds = threshold * model.max_scores
    predictions = model.predict(encoding.seqs)
    if method == "lowmem" and isinstance(predictions, list):
        thresholded, scores = threshold_lowmem(predictions, thresholds, score)
    else:
        thresholded, scores = threshold_fast(encoding, predictions, thresholds, score)
    return thresholded, scores


def threshold_fast(encoding, predictions, thresholds, score=False):
    """Default function to threshold model predictions.
    
    Args:
        encoding (SeqEncoding): SeqEncoding object.
        predictions (np.array): model output.
        thresholds (np.array): score cutoff for each PWM in model
        score (np.array): output score for each potential binding site
        
    Returns:
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
        scores (np.array): score for each potential binding site
    """
    scores = None
    # Concatenate predictions from each set of PWMs
    if isinstance(predictions, list):
        predictions = [np.pad(x, ((0,0), (0, encoding.len - x.shape[1]), (0,0)), 
            mode='constant', constant_values=-1) for x in predictions]
        predictions = np.concatenate(predictions, axis=2)
    # Threshold predictions
    thresholded = np.where(predictions > thresholds)
    # Combine site locations with scores
    if score:
        scores = predictions[thresholded]
    return thresholded, scores


def threshold_lowmem(predictions, thresholds, score=False):
    """Thresholding function for high-memory jobs (e.g. long genomes).
    
    Args:
        predictions (np.array): model output.
        thresholds (np.array): score cutoff for each PWM in model
        score (np.array): output score for each potential binding site
        
    Returns:
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
        scores (np.array): score for each potential binding site
        
    """
    scores = None
    # Split predictions from each PWM
    predictions = [np.split(x, x.shape[2],2) for x in predictions]
    predictions = list(itertools.chain.from_iterable(predictions))
    # Threshold predictions
    thresholded = [[], []]
    for i,p in enumerate(predictions):
        x = np.where(p >= thresholds[i])
        thresholded[0].append(x[0])
        thresholded[1].append(x[1])
    if score:
        scores = np.concatenate([predictions[i][:,:,0][thresholded[0][i], thresholded[1][i]] for i in range(len(predictions))])
    # Add back PWM ID
    thresholded.append([[i] * thresholded[0][i].shape[0] for i in range(len(predictions))])
    # Concatenate results
    thresholded = [np.concatenate(x).astype(int) for x in thresholded]
    return thresholded, scores


def locate_sites(encoding, model, thresholded, scores=None):
    """ Function to locate potential binding sites given model predictions.
    
    Args:
        encoding (SeqEncoding): SeqEncoding object
        model (model): parameterized convolutional model
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
        scores (np.array): score for each potential binding site
      
    Returns:
        sites (pd.DataFrame): dataframe containing information on each potential binding site.
     
    """
    seq_idx = thresholded[0]
    pos_idx = thresholded[1]
    pwm_idx = thresholded[2]
    sites = pd.DataFrame({'id': encoding.ids[seq_idx],
                              'name':encoding.names[seq_idx],
                              'sense':encoding.senses[seq_idx],
                              'start': pos_idx,
                              'Matrix_id': model.Matrix_ids[pwm_idx]})
    sites['width'] = model.widths[pwm_idx]
    sites['end'] = sites['start'] + sites['width']
    if scores is not None:
        sites['score'] = scores
        sites['max_score'] = model.max_scores[pwm_idx]
        sites['frac_score'] = sites['score']/sites['max_score']
    sites = sites[sites.end <= encoding.len].reset_index(drop=True)
    return sites


def bin_sites_by_score(encoding, model, thresholded, scores, bins):
    """ Function to locate potential binding sites and bin them based on their score.
    
    Args:
        encoding (SeqEncoding): SeqEncoding object
        model (model): parameterized convolutional model
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
        scores (np.array): score for each potential binding site.
        bins (list): list of fractional values (0 to 1) to bin scores.
      
    Returns:
        sites (pd.DataFrame): dataframe containing information on each potential binding site.
     
    """
    seq_idx = thresholded[0]
    pwm_idx = thresholded[2]
    frac_scores = scores/model.max_scores[pwm_idx]
    binned_scores = pd.cut(frac_scores, np.concatenate([bins, [1]]))
    result = pd.crosstab(index = model.Matrix_ids[pwm_idx], 
                         columns = [encoding.ids[seq_idx], 
                                    encoding.names[seq_idx], 
                                    encoding.senses[seq_idx], 
                                    binned_scores])
    result = result.melt(ignore_index=False).reset_index()
    result.columns = ['Matrix_id', 'id', 'name', 'sense', 'bin', 'num']
    return result

                       
def count_sites(encoding, model, thresholded):
    """Function to count the number of matches per PWM.
        
    Args:
        encoding (SeqEncoding): SeqEncoding object
        model (model): parameterized convolutional model
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
      
    Returns:
        result (pd.DataFrame): dataframe containing number of matches per PWM.
     
    """
    seq_idx = thresholded[0]
    if len(seq_idx) == 0:
        return pd.DataFrame({'Matrix_id': [], 'id': [], 'name': [], 'sense': [], 
                            'num': []})
    else:
        pwm_idx = thresholded[2]
        result = pd.crosstab(index = model.Matrix_ids[pwm_idx], 
                         columns = [encoding.ids[seq_idx], 
                                    encoding.names[seq_idx], 
                                    encoding.senses[seq_idx]])
        result = result.melt(ignore_index=False).reset_index()
        result.columns = ['Matrix_id', 'id', 'name', 'sense', 'num']
        return result


def find_sites_seq(encoding, model, threshold, sites=False, binned_counts=False, total_counts=False, stats=False, score=False, method="fast"):
    """Function to predict binding sites on encoded sequence(s).
    
    Args:
        encoding (SeqEncoding): object of class SeqEncoding
        model (model): class PWMModel
        threshold (float or np.arange): threshold (from 0 to 1) to identify binding sites OR np.arange (with binned_counts=True).
        sites (bool): output binding site locations
        binned_counts (bool): output binned counts of binding sites per PWM
        total_counts (bool): output total count of binding sites per PWM
        stats (bool): output mean and standard deviation of the count of binding sites per PWM
        score (bool): output scores for binding sites
        method (str): prediction function, "fast" or "highmem"
    
    Returns: 
        output (dict): dictionary containing specified outputs.
        
    """
    if binned_counts:
        score = True
    if sites:
        score = True
    if binned_counts:
        assert type(threshold) == np.ndarray
        thresholded, scores = predict(encoding, model, min(threshold), score, method)
    else:
        thresholded, scores = predict(encoding, model, threshold, score, method)
    output = {}
    if sites:
        output['sites'] = locate_sites(encoding, model, thresholded, scores)
    if binned_counts:
        output['binned_counts'] = bin_sites_by_score(encoding, model, thresholded, scores, threshold)
    if (total_counts or stats):
        total = count_sites(encoding, model, thresholded)
    if total_counts:
        output['total_counts'] = total
    if stats:
        stats = total.groupby(['Matrix_id', 'sense', 'name']).agg([len, np.mean, np.std]).reset_index()
        stats.columns = ['Matrix_id', 'sense', 'name', 'len', 'avg', 'sd'] 
        output['stats'] = stats

    return output


def find_sites_multiseq(encoding, model, threshold, sites=False, binned_counts=False, total_counts=False, stats=False, score=False, combine_seqs=False, sep_ids=False, method="fast"):
    """Function to predict binding sites on class MultiSeqEncoding.
    
    Args:
        encoding (MultiSeqEncoding): object of class MultiSeqEncoding
        model (model): class PWMModel
        threshold (float or np.arange): threshold (from 0 to 1) to identify binding sites OR np.arange (with binned_counts=True).
        sites (bool): output binding site locations
        binned_counts (bool): output binned counts of binding sites per PWM
        total_counts (bool): output total count of binding sites per PWM
        stats (bool): output mean and standard deviation of the count of binding sites per PWM
        score (bool): output scores for binding sites
        combine_seqs (bool): combine outputs for multiple sequence groups into a single dataframe
        sep_ids (bool): separate outputs by sequence ID.
        method (str): prediction function, "fast" or "highmem"
    
    Returns: 
        output (dict): dictionary containing specified outputs.
        
    """
    # Find binding sites per sequence or group of sequences
    if combine_seqs and stats:
        output_per_seq = [find_sites_seq(seq, model, threshold, sites, binned_counts, total_counts=True, stats=False, score=score, method=method) for seq in encoding.seqs]
    else:
        output_per_seq = [find_sites_seq(seq, model, threshold, sites, binned_counts, total_counts, stats, score, method) for seq in encoding.seqs]
    # Concatenate binding sites
    output = {}
    for key in output_per_seq[0].keys():
        output[key] = pd.concat([x[key] for x in output_per_seq]).reset_index(drop=True)
    # Combine binding sites
    if combine_seqs:
        if sep_ids:
            if 'total_counts' in output.keys():
                output['total_counts'] = output['total_counts'].groupby(['Matrix_id', 'sense', 'id']).num.sum().reset_index()
            if 'binned_counts' in output.keys():
                output['binned_counts'] = output['binned_counts'].groupby(['Matrix_id', 'sense', 'bin', 'id']).num.sum().reset_index()
        else:
            if 'total_counts' in output.keys():
                output['total_counts'] = output['total_counts'].groupby(['Matrix_id', 'sense']).num.sum().reset_index()
            if 'binned_counts' in output.keys():
                output['binned_counts'] = output['binned_counts'].groupby(['Matrix_id', 'sense', 'bin']).num.sum().reset_index()
        # Calculate stats
        if stats:
            stats = output['total_counts'].groupby(['Matrix_id', 'sense']).agg([len, np.mean, np.std]).reset_index()
            stats.columns = ['Matrix_id', 'sense', 'len', 'avg', 'sd'] 
            output['stats'] = stats 
            if not total_counts:
                del output['total_counts']
    
    return output
