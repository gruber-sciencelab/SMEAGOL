import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import itertools


def predict_fast(encoding, predictions, thresholds, score=False):
    """
    Default prediction function.
    encoding: integer-encoded sequence.
    predictions: model output.
    thresholds: cutoff for each PWM in model
    score: output score for each match
    """
    scores = None
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


def predict_highmem(predictions, thresholds, score=False):
    """
    Prediction function for high-memory jobs (e.g. long genomes).
    predictions: model output.
    thresholds: cutoff for each PWM in model
    score: output score for each match
    """
    scores = None
    predictions = [np.split(x, x.shape[2],2) for x in predictions]
    predictions = list(itertools.chain.from_iterable(predictions))
    thresholded = [[], []]
    for i,p in enumerate(predictions):
        x = np.where(p >= thresholds[i])
        thresholded[0].append(x[0])
        thresholded[1].append(x[1])
    if score:
        scores = np.concatenate([predictions[i][:,:,0][thresholded[0][i], thresholded[1][i]] for i in range(len(predictions))])
    thresholded.append([[i] * thresholded[0][i].shape[0] for i in range(len(predictions))])
    thresholded = [np.concatenate(x).astype(int) for x in thresholded]
    return thresholded, scores


def predict(encoding, model, threshold, score=False, method="fast"):
    thresholds = threshold*model.max_scores
    predictions = model.predict(encoding.encoded)
    if method == "highmem" and isinstance(predictions, list):
        thresholded, scores = predict_highmem(predictions, thresholds, score)
    else:
        thresholded, scores = predict_fast(encoding, predictions, thresholds, score)
    return thresholded, scores


def locate_sites(encoding, model, thresholded, scores=None):
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
    seq_idx = thresholded[0]
    pwm_idx = thresholded[2]
    frac_scores = scores/model.max_scores[pwm_idx]
    result = pd.crosstab(index = model.Matrix_ids[pwm_idx], 
                         columns = [encoding.ids[seq_idx], 
                                    encoding.names[seq_idx], 
                                    encoding.senses[seq_idx], 
                                    bins[np.digitize(frac_scores, bins)-1]])
    result = result.melt(ignore_index=False).reset_index()
    result.columns = ['Matrix_id', 'id', 'name', 'sense', 'bin', 'num']
    return result

                       
def count_sites(encoding, model, thresholded):
    seq_idx = thresholded[0]
    pwm_idx = thresholded[2]
    result = pd.crosstab(index = model.Matrix_ids[pwm_idx], 
                         columns = [encoding.ids[seq_idx], 
                                    encoding.names[seq_idx], 
                                    encoding.senses[seq_idx]])
    result = result.melt(ignore_index=False).reset_index()
    result.columns = ['Matrix_id', 'id', 'name', 'sense', 'num']
    return result


def find_sites_seq(encoding, model, threshold, sites=False, binned_counts=False, total_counts=False, stats=False, score=False, method="fast"):
    """
    Function to predict binding sites on sequence(s).
    
    Inputs:
        encoding: class MultiSeqEncoding or SeqEncoding
        model: class PWMModel
        threshold: threshold (from 0 to 1) to identify binding sites OR np.arange (with binned_counts=True).
        sites: output binding site locations
        binned_counts: output binned counts of binding sites per PWM
        total_counts: output total count of binding sites per PWM
        stats: output mean and standard deviation of the count of binding sites per PWM
        score: output scores for binding sites
        method: prediction function, "fast" or "highmem"
    
    Returns: 
        output: dictionary containing specified outputs.
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


def find_sites_multiseq(encodings, model, threshold, sites=False, binned_counts=False, total_counts=False, stats=False, score=False, combine_seqs=False, sep_ids=False, method="fast"):
    """
    Function to predict binding site on class MultiSeqEncoding.
    
    """
    # Find binding sites
    if combine_seqs and stats:
        output_per_seq = [find_sites_seq(seq, model, threshold, sites, binned_counts, total_counts=True, stats=False, score=score, method=method) for seq in encodings.seqs]
    else:
        output_per_seq = [find_sites_seq(seq, model, threshold, sites, binned_counts, total_counts, stats, score, method) for seq in encodings.seqs]
    # Concatenate
    output = {}
    for key in output_per_seq[0].keys():
        output[key] = pd.concat([x[key] for x in output_per_seq]).reset_index(drop=True)
    # Combine
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
