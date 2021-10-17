# General imports
import numpy as np
import pandas as pd
import itertools

# Stats imports
import scipy.stats as stats
import statsmodels.stats.multitest as multitest

# Smeagol imports
from .utils import shuffle_records
from .encode import SeqGroups
from .scan import scan_sequences, get_tiling_windows_over_genome, count_in_sliding_windows, count_in_window


def enrich_over_shuffled(real_counts, shuf_stats, background='binomial', records=None):
    """Function to calculate enrichment of binding sites in real vs. shuffled genomes
    
    Args:
        real_counts (pd.DataFrame): counts of binding sites in real genome
        shuf_stats (pd.DataFrame): statistics for binding sites across multiple shuffled genomes
        background (str): 'binomial' or 'normal'
        records (SeqGroups): SeqGroups object. Only needed if background='binomial'
        
    Returns:
        enr_full (pd.DataFrame): dataframe containing FDR-corrected p-values for enrichment of each PWM.  
        
    """
    # Compare counts on real and shuffled genomes for each PWM
    if 'name' in shuf_stats.columns:
        enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'width', 'sense', 'name'], how='outer')
    else:
        enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'width', 'sense'], how='outer')

    # If 0 sites are present in real genome, include the entry
    enr['num'] = enr['num'].fillna(0)

    # If 0 sites are present in shuffled genomes, set a minimum of 1 site
    num_shuf = enr['len'][0]
    enr.loc[enr.avg==0, 'avg'] = 1/num_shuf

    # Calculate normal z-score
    if (background == 'normal') or (background == 'both'):
        enr.loc[enr.avg==0, 'sd'] = np.std([1] + [0]*(num_shuf - 1))
        enr['z'] = (enr.num - enr.avg)/enr.sd
        enr['z'] = enr.z.replace([-np.inf], -10)

    # Calculate p-value
    if background == 'normal':
        enr['pnorm'] = stats.norm.sf(abs(enr.z))*2
    else:
        enr['adj_len'] = sum([len(record) - enr.width + 1 for record in records])
        enr['p'] = enr.apply(lambda x:stats.binom_test(x['num'], x['adj_len'], x['avg']/x['adj_len'], alternative='two-sided'), axis=1)
    enr_full = pd.DataFrame()

    # FDR correction per sense
    for sense in pd.unique(enr.sense):
        enr_x = enr[enr.sense == sense].copy()
        if background == 'normal':
            enr_x['fdr_norm'] = multitest.fdrcorrection(enr_x.pnorm)[1]  
        else:
            enr_x['fdr'] = multitest.fdrcorrection(enr_x.p)[1]
        enr_full = pd.concat([enr_full, enr_x])

    # Sort and index final results
    if background == 'normal':
        enr_full.sort_values(['sense', 'fdr_binom'], inplace=True)
    else:
        enr_full.sort_values(['sense', 'fdr'], inplace=True)
    enr_full.reset_index(inplace=True, drop=True)

    return enr_full


def enrich_in_genome(records, model, simN, simK, rcomp, threshold, sense='+', background='binomial', verbose=False, combine_seqs=True, seq_batch=0):
    """Function to shuffle sequence(s) and calculate enrichment of PWMs in sequence(s) relative to the shuffled background.
        
    Args:
        records (list): list of seqrecord objects
        model (PWMModel): parameterized convolutional model
        simN (int): number of shuffles
        simK (int): k-mer frequency to conserve while shuffling
        rcomp (str): 'only' to encode the sequence reverse complements, 'both' to encode the reverse
                     complements as well as original sequences, or 'none'.
        sense (str): '+' or '-'        
        background (str): 'binomial' or 'normal'
        combine_seqs (bool): combine outputs for all sequence groups into single dataframe
        seq_batch (int): number of shuffled sequences to scan at a time. If 0, scan all.
        
    Returns:
        results (dict): dictionary containing results.  
        
    """
    # Find sites on real genome
    real_preds = scan_sequences(records, model, threshold, sense, rcomp, outputs=['sites', 'counts'], score=False, combine_seqs=combine_seqs)

    # Shuffle genome
    shuf = shuffle_records(records, simN, simK)
    
    # Count sites on shuffled genomes
    shuf_preds = scan_sequences(shuf, model, threshold, sense, rcomp, outputs=['counts', 'stats'], group_by='name', combine_seqs=combine_seqs, sep_ids=True, seq_batch=seq_batch)
        
    # Calculate binding site enrichment
    enr = enrich_over_shuffled(real_preds['counts'], shuf_preds['stats'], background=background, records=records)

    # Combine results
    results = {'enrichment': enr, 
               'real_sites': real_preds['sites'], 
               'real_counts': real_preds['counts'], 
               'shuf_counts': shuf_preds['counts'],
               'shuf_stats': shuf_preds['stats']}
    if verbose:
        
        results['shuf_seqs'] = shuf
    return results


def examine_thresholds(records, model, simN, simK, rcomp, min_threshold, sense='+', verbose=False, combine_seqs=True):
    """Function to compare the number of binding sites at various thresholds.
            
    Args:
        records (list): list of seqrecord objects
        model (PWMModel): parameterized convolutional model
        simN (int): number of shuffles
        simK (int): k-mer frequency to conserve while shuffling
        rcomp (str): 'only' to encode the sequence reverse complements, 'both' to encode the reverse
                     complements as well as original sequences, or 'none'.
        sense (str): '+' or '-'        
        min_threshold (float): minimum threshold for a binding site (0 to 1)
        verbose (bool): output all information
        combine_seqs (bool): combine outputs for multiple sequences into single dataframe
        
    Returns:
        results (dict): dictionary containing results. 
    """
    thresholds = np.arange(min_threshold, 1.0, 0.1)
    
    # scan real sequence
    real_binned = scan_sequences(records, model, thresholds, sense, rcomp, outputs=['binned_counts'], combine_seqs=combine_seqs)['binned_counts']
    
    # shuffle
    shuf = shuffle_records(records, simN, simK)
    
    # scan shuffled sequence
    shuf_binned = scan_sequences(shuf, model, thresholds, sense, rcomp, outputs=['binned_counts'], combine_seqs=combine_seqs, group_by='name', sep_ids=True)['binned_counts']
    
    results = {'real_binned':real_binned, 'shuf_binned': shuf_binned}
    if verbose:
        results['shuf_seqs'] = shuf
    return results


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
    matrix_sites = sites[sites.Matrix_id == matrix_id].reset_index(drop=True)
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
