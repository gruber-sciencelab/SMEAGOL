import pandas as pd
import numpy as np
import itertools


def predict(encoding, model, threshold, score=False):
    """Prediction by scanning an encoded sequence with a convolutional model.
    
    Args:
        encoding (SeqEncoding): SeqEncoding object
        model (model): Prameterized convolutional model
        threshold (float): fraction of maximum score to use as binding site detection threshold
        score (bool): Output binding site scores as well as positions
        
    Returns:
        thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
        scores (np.array): score for each potential binding site
        
    """
    # Get threshold for each PWM
    assert (threshold >= 0) & (threshold <= 1) 
    thresholds = threshold * model.max_scores
    # Inference using convolutional model
    predictions = model.predict(encoding.seqs)
    # Threshold predictions
    thresholded = np.where(predictions > thresholds)
    # Combine site locations with scores
    if score:
        scores = predictions[thresholded]
    else:
        scores = None
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
        return pd.DataFrame({'Matrix_id': [], 'width': [], 'id': [], 'name': [], 'sense': [], 
                            'num': []})
    else:
        pwm_idx = thresholded[2]
        result = pd.crosstab(index = [model.Matrix_ids[pwm_idx], model.widths[pwm_idx]], 
                         columns = [encoding.ids[seq_idx], 
                                    encoding.names[seq_idx], 
                                    encoding.senses[seq_idx]])
        result = result.melt(ignore_index=False).reset_index()
        result.columns = ['Matrix_id', 'width', 'id', 'name', 'sense', 'num']
        return result


def find_sites_seq(encoding, model, threshold, sites=False, binned_counts=False, total_counts=False, stats=False, score=False):
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
    
    Returns: 
        output (dict): dictionary containing specified outputs.
        
    """
    if binned_counts:
        score = True
    if sites:
        score = True
    if binned_counts:
        assert type(threshold) == np.ndarray
        thresholded, scores = predict(encoding, model, min(threshold), score)
    else:
        thresholded, scores = predict(encoding, model, threshold, score)
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


def find_sites_multiseq(encoding, model, threshold, sites=False, binned_counts=False, total_counts=False, stats=False, score=False, combine_seqs=False, sep_ids=False):
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
    
    Returns: 
        output (dict): dictionary containing specified outputs.
        
    """
    # Find binding sites per sequence or group of sequences
    if combine_seqs and stats:
        output_per_seq = [find_sites_seq(seq, model, threshold, sites, binned_counts, total_counts=True, stats=False, score=score) for seq in encoding.seqs]
    else:
        output_per_seq = [find_sites_seq(seq, model, threshold, sites, binned_counts, total_counts, stats, score) for seq in encoding.seqs]
    # Concatenate binding sites
    output = {}
    for key in output_per_seq[0].keys():
        output[key] = pd.concat([x[key] for x in output_per_seq]).reset_index(drop=True)
    # Combine binding sites
    if combine_seqs:
        if sep_ids:
            if 'total_counts' in output.keys():
                output['total_counts'] = output['total_counts'].groupby(['Matrix_id', 'width', 'sense', 'id']).num.sum().reset_index()
            if 'binned_counts' in output.keys():
                output['binned_counts'] = output['binned_counts'].groupby(['Matrix_id', 'width', 'sense', 'bin', 'id']).num.sum().reset_index()
        else:
            if 'total_counts' in output.keys():
                output['total_counts'] = output['total_counts'].groupby(['Matrix_id', 'width', 'sense']).num.sum().reset_index()
            if 'binned_counts' in output.keys():
                output['binned_counts'] = output['binned_counts'].groupby(['Matrix_id', 'width', 'sense', 'bin']).num.sum().reset_index()
        # Calculate stats
        if stats:
            stats = output['total_counts'].groupby(['Matrix_id', 'width', 'sense']).agg([len, np.mean, np.std]).reset_index()
            stats.columns = ['Matrix_id', 'width', 'sense', 'len', 'avg', 'sd'] 
            output['stats'] = stats 
            if not total_counts:
                del output['total_counts']
    
    return output