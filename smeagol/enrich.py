# General imports
import numpy as np
import pandas as pd
import itertools

# Stats imports
import scipy.stats as stats
import statsmodels.stats.multitest as multitest

# Smeagol imports
from .utils import shuffle_records
from .encode import MultiSeqEncoding
from .scan import find_sites_multiseq


def enrich_over_shuffled(real_counts, shuf_stats, background='binomial', seqlen=None):
    """Function to calculate enrichment of binding sites in real vs. shuffled genomes
    
    Args:
        real_counts (pd.DataFrame): counts of binding sites in real genome
        shuf_stats (pd.DataFrame): statistics for binding sites across multiple shuffled genomes
        background (str): 'binomial' or 'normal'
        seqlen (int): length of input sequence. Only needed if background='binomial'
        
    Returns:
        enr_full (pd.DataFrame): dataframe containing FDR-corrected p-values for enrichment of each PWM.  
        
    """
    # Compare counts on real and shuffled genomes for each PWM
    if 'name' in shuf_stats.columns:
        enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'width', 'sense', 'name'], how='outer')
    else:
        enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'width', 'sense'], how='outer')
    # If 0 sites are present in real genome, include
    enr['num'] = enr['num'].fillna(0)
    # If 0 sites are present in shuffled genomes, set a minimum of 1 site
    num_shuf = enr['len'][0]
    enr.loc[enr.avg==0, 'avg'] = 1/num_shuf
    # Calculate normal z-score
    if (background == 'normal') or (background == 'both'):
        enr.loc[enr.avg==0, 'sd'] = np.std([1] + [0]*(num_shuf - 1))
        enr['z'] = (enr.num - enr.avg)/enr.sd
        enr['z'] = enr.z.replace([-np.inf], -10)
    if (background == 'binomial') or (background == 'both'):
        assert seqlen is not None
        enr['adj_len'] = seqlen - enr['width'] + 1
    # Calculate p-value
    if background == 'normal':
        enr['p'] = stats.norm.sf(abs(enr.z))*2
    elif background == 'binomial':
        enr['p'] = enr.apply(lambda x:stats.binom_test(x['num'], x['adj_len'], x['avg']/x['adj_len'], alternative='two-sided'), axis=1)
    elif background == 'both':
        enr['pnorm'] = stats.norm.sf(abs(enr.z))*2
        enr['pbinom'] = enr.apply(lambda x:stats.binom_test(x['num'], x['adj_len'], x['avg']/x['adj_len'], alternative='two-sided'), axis=1)
    enr_full = pd.DataFrame()
    # FDR correction per molecule
    for sense in pd.unique(enr.sense):
        enr_x = enr[enr.sense == sense].copy()
        if (background == 'normal') or (background == 'binomial'):
            enr_x['fdr'] = multitest.fdrcorrection(enr_x.p)[1]
        elif background == 'both':
            enr_x['fdr_norm'] = multitest.fdrcorrection(enr_x.pnorm)[1]  
            enr_x['fdr_binom'] = multitest.fdrcorrection(enr_x.pbinom)[1]
        enr_full = pd.concat([enr_full, enr_x])
    # Sort and index final results
    if (background == 'normal') or (background == 'binomial'):
        enr_full.sort_values(['sense', 'fdr'], inplace=True)
    elif background == 'both':
        enr_full.sort_values(['sense', 'fdr_binom'], inplace=True)
    enr_full.reset_index(inplace=True, drop=True)
    enr_full.drop(columns='adj_len', inplace=True)
    return enr_full


def enrich_in_genome(records, model, simN, simK, rcomp, sense, threshold, background='binomial', verbose=False, combine_seqs=True):
    """Function to shuffle sequence(s) and calculate enrichment of PWMs in sequence(s) relative to the shuffled background.
        
    Args:
        records (list): list of seqrecord objects
        model (PWMModel): parameterized convolutional model
        simN (int): number of shuffles
        simK (int): k-mer frequency to conserve while shuffling
        rcomp (bool): calculate enrichment in reverse complement as well as original sequence
        sense (str): '+' or '-'        
        background (str): 'binomial' or 'normal'
        verbose (bool): output all information
        combine_seqs (bool): combine outputs for multiple sequences into single dataframe
        
    Returns:
        results (dict): dictionary containing results.  
        
    """
    # Encode sequence
    encoded = MultiSeqEncoding(records, rcomp=rcomp, sense=sense)
    # Find sites on real genome
    real_preds = find_sites_multiseq(encoded, model, threshold, sites=True, total_counts=True, combine_seqs=combine_seqs)
    # Shuffle genome
    shuf = shuffle_records(records, simN, simK)
    # Encode shuffled genomes
    encoded_shuffled = MultiSeqEncoding(shuf, sense=sense, rcomp=rcomp, group_by_name=True)
    # Count sites on shuffled genomes
    shuf_preds = find_sites_multiseq(encoded_shuffled, model, threshold, total_counts=verbose, stats=True, combine_seqs=combine_seqs, sep_ids=True)
    # Calculate binding site enrichment
    if background == 'normal':
        enr = enrich_over_shuffled(real_preds['total_counts'], shuf_preds['stats'], background=background)
    elif background == 'binomial' or background == 'both':
        seqlen = sum([len(x) for x in records])
        enr = enrich_over_shuffled(real_preds['total_counts'], shuf_preds['stats'], background=background, seqlen=seqlen)
    # Combine results
    results = {'enrichment': enr, 
               'real_sites': real_preds['sites'], 
               'real_counts': real_preds['total_counts'], 
               'shuf_stats': shuf_preds['stats']}
    if verbose:
        results['shuf_counts'] = shuf_preds['total_counts']
        results['shuf_seqs'] = shuf
    return results


def examine_thresholds(records, model, simN, simK, rcomp, sense, min_threshold, verbose=False, combine_seqs=True):
    """Function to compare the number of binding sites at various thresholds.
            
    Args:
        records (list): list of seqrecord objects
        model (PWMModel): parameterized convolutional model
        simN (int): number of shuffles
        simK (int): k-mer frequency to conserve while shuffling
        rcomp (bool): calculate enrichment in reverse complement as well as original sequence
        sense (str): '+' or '-'        
        min_threshold (float): minimum threshold for a binding site (0 to 1)
        verbose (bool): output all information
        combine_seqs (bool): combine outputs for multiple sequences into single dataframe
        
    Returns:
        results (dict): dictionary containing results. 
    """
    encoded = MultiSeqEncoding(records, rcomp=rcomp, sense=sense)
    shuf = shuffle_records(records, simN, simK)
    encoded_shuffled = MultiSeqEncoding(shuf, sense=sense, rcomp=rcomp, group_by_name=True)
    thresholds = np.arange(min_threshold, 1.0, 0.1)
    real_binned = find_sites_multiseq(encoded, model, thresholds, binned_counts=True, combine_seqs=combine_seqs)['binned_counts']
    shuf_binned = find_sites_multiseq(encoded_shuffled, model, thresholds, binned_counts=True, combine_seqs=combine_seqs, sep_ids=True)['binned_counts']
    results = {'real_binned':real_binned, 'shuf_binned': shuf_binned}
    if verbose:
        results['shuf_seqs'] = shuf
    return results


# Analysis in windows

def get_tiling_windows_over_record(record, width, shift):
    """Function to get tiling windows over a sequence.
    
    Args:
        record (SeqRecord): SeqRecord object
        width (int): width of tiling windows
        shift (int): shift between successive windows
    
    Returns:
        windows (pd.DataFrame): windows covering input sequence
        
    """
    # Get start positions
    starts = list(range(0, len(record), shift))
    # Get end positions
    ends = [np.min([x + width, len(record)]) for x in starts]
    # Add sequence ID
    idxs = np.tile(record.id, len(starts))
    # Combine
    windows = pd.DataFrame({'id':idxs, 'start':starts, 'end':ends})
    return windows


def get_tiling_windows_over_genome(genome, width, shift=None):
    """Function to get tiling windows over a genome.
        
    Args:
        genome (list): list of SeqRecord objects
        width (int): width of tiling windows
        shift (int): shift between successive windows. By default the same as width.
    
    Returns:
        windows (pd.DataFrame): windows covering all sequences in genome
        
    """
    if shift is None:
        shift = width
    if len(genome) == 1:
        windows = get_tiling_windows_over_record(genome[0], width, shift)
    else:
        windows = pd.concat([get_tiling_windows_over_record(record, width, shift) for record in genome])
    return windows


def count_in_window(window, sites, matrix_id):
    """Function to calculate count of PWMs in subsequence.
    
    Args:
        window (pd.DataFrame): dataframe with columns id, start, end
        sites (pd.DataFrame): dataframe containing locations of binding sites
        matrix_id (str): selected PWM
    
    Returns:
        count (int): Number of binding sites for given PWM in selected window.

    """
    count = len(sites[(sites.Matrix_id==matrix_id) & (sites.id==window.id) & (sites.start >= window.start) & (sites.start < window.end)])
    return count


def enrich_in_window(window, sites, genome, matrix_id):
    """Function to calculate enrichment of PWMs in subsequence relative to total sequence.
    
    Args:
        window (pd.DataFrame): dataframe with columns id, start, end
        sites (pd.DataFrame): dataframe containing locations of binding sites
        genome (list): list of SeqRecord objects
        matrix_id (str): selected PWM
        
    Returns:
        result (pd.DataFrame): Dataframe containing result of enrichment analysis in selected window.
    
    """
    result = window.copy()
    # Get sites for required matrix
    matrix_sites = sites[sites.Matrix_id == matrix_id]
    # Get matrix width
    width = matrix_sites.width[0]
    # Correct window length and genome length
    effective_window_len = window.end - window.start - width + 1
    effective_genome_len = sum([len(x) for x in genome]) - width + 1
    # Calculate number of successes and failures in selected region
    result['len'] = effective_window_len
    result['count'] = count_in_window(window, sites, matrix_id)
    count_neg = result['len'] - result['count']
    # Calculate number of successes and failures in whole genome
    result['tot_count'] = len(matrix_sites)
    tot_neg = effective_genome_len - result.tot_count
    # Expected value
    result['expected'] = (result.tot_count * result.len) / effective_genome_len
    # Fisher's exact test
    odds, p = stats.fisher_exact([[result['count'], count_neg], [result.tot_count, tot_neg]], alternative='two-sided')
    result['odds'] = odds
    result['p'] = p
    return result


def count_in_sliding_windows(sites, genome, matrix_id, width, shift=None):
    """Function to count binding sites in sliding windows across a genome.
     
    Args:
        sites (pd.DataFrame): dataframe containing locations of binding sites
        genome (list): list of SeqRecord objects
        matrix_id (str): selected PWM
        width (int): width of tiling windows
        shift (int): shift between successive windows
    
    Returns:
        results (pd.DataFrame): dataframe containing number of binding sites per window
        
    """
    windows = get_tiling_windows_over_genome(genome, width, shift)
    windows['count'] = windows.apply(count_in_window, axis=1, args=(sites, genome, matrix_id))
    windows.reset_index(inplace=True, drop=True)
    return windows


def enrich_in_sliding_windows(sites, genome, matrix_id, width, shift=None):
    """Function to test enrichment of binding sites in sliding windows across a genome.
     
    Args:
        sites (pd.DataFrame): dataframe containing locations of binding sites
        genome (list): list of SeqRecord objects
        matrix_id (str): selected PWM
        width (int): width of tiling windows
        shift (int): shift between successive windows
    
    Returns:
        windows (pd.DataFrame): results of enrichment analysis in windows covering all sequences in genome
        
    """
    # Get sliding windows across genome
    windows = get_tiling_windows_over_genome(genome, width, shift)
    # Perform enrichment in each window
    windows = windows.apply(enrich_in_window, axis=1, args=(sites, genome, matrix_id)).reset_index(drop=True)
    # Overall FDR correction
    windows['padj'] = multitest.fdrcorrection(windows.p)[1]
    return windows
