# General imports
import numpy as np
import pandas as pd
import itertools

# Stats imports
import scipy.stats as stats
import statsmodels.stats.multitest as multitest

# Smeagol imports
from .fastaio import write_fasta
from .encoding import MultiSeqEncoding
from .inference import find_sites_multiseq
from .utils import shuffle_records



def enrich_over_shuffled(real_counts, shuf_stats, background='binomial', seqlen=None):
    """
    Function to calculate enrichment of binding sites in real vs. shuffled genomes
    
    Inputs:
        real_counts: counts of binding sites in real genome
        shuf_stats: statistics for binding sites across multiple shuffled genomes
        
    Returns:
        enr: DF containing FDR-corrected p-values for enrichment of each PWM.  
    """
    # Compare counts on real and shuffled genomes
    if 'name' in shuf_stats.columns:
        enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'sense', 'name'])
    else:
        enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'sense'])
    enr = enr[(enr.num > 0) | (enr.avg > 0)].copy().reset_index(drop=True)
    # Calculate normal z-score
    if (background == 'normal') or (background == 'both'):
        enr['z'] = (enr.num - enr.avg)/enr.sd
        enr['z'] = enr.z.replace([-np.inf], -100)
    if (background == 'binomial') or (background == 'both'):
        assert seqlen is not None
    # Calculate p-value
    if background == 'normal':
        enr['p'] = stats.norm.sf(abs(enr.z))*2
    elif background == 'binomial':
        enr['p'] = enr.apply(lambda x:stats.binom_test(x['num'], seqlen, x['avg']/seqlen, alternative='two-sided'), axis=1)
    elif background == 'both':
        enr['pnorm'] = stats.norm.sf(abs(enr.z))*2
        enr['pbinom'] = enr.apply(lambda x:stats.binom_test(x['num'], seqlen, x['avg']/seqlen, alternative='two-sided'), axis=1)
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
    return enr_full


def enrich_in_genome(genome, model, simN, simK, rcomp, genome_sense, threshold, background='binomial', verbose=False, combine_seqs=True, method='fast'):
    """
    Function to calculate enrichment of PWMs in a sequence relative to a shuffled background.
    """
    # Encode genome
    encoded_genome = MultiSeqEncoding(genome, rcomp=rcomp, sense=genome_sense)
    # Find sites on real genome
    real_preds = find_sites_multiseq(encoded_genome, model, threshold, sites=True, total_counts=True, combine_seqs=combine_seqs)
    real_sites = real_preds['sites']
    real_counts = real_preds['total_counts']
    # Shuffle genome
    shuf_genome = shuffle_records(genome, simN, simK)
    # Encode shuffled genomes
    encoded_shuffled = MultiSeqEncoding(shuf_genome, sense=genome_sense, rcomp=rcomp, group_by_name=True)
    # Count sites on shuffled genomes
    if verbose:
        shuf_preds = find_sites_multiseq(encoded_shuffled, model, threshold, total_counts=True, stats=True, combine_seqs=combine_seqs, sep_ids=True, method=method)
        shuf_counts = shuf_preds['total_counts']
    else:
        shuf_preds = find_sites_multiseq(encoded_shuffled, model, threshold, stats=True, 
                                         combine_seqs=combine_seqs, sep_ids=True, method=method)
    shuf_stats = shuf_preds['stats']
    # Calculate binding site enrichment
    if background == 'normal':
        enr = enrich_over_shuffled(real_counts, shuf_stats, background=background)
    elif background == 'binomial' or background == 'both':
        seqlen = sum([len(x) for x in genome])
        enr = enrich_over_shuffled(real_counts, shuf_stats, background=background, seqlen=seqlen)
    results = {'enrichment':enr, 'real_sites':real_sites, 'real_counts':real_counts, 'shuf_stats': shuf_stats}
    if verbose:
        results['shuf_counts'] = shuf_counts
        results['shuf_genome'] = shuf_genome
    return results


def enrich_in_window(real_sites, genome, sel_id, sel_start, sel_end):
    """
    Function to calculate enrichment of PWMs in subsequence relative to total sequence.
    """
    # Calculate total number of successes and failures over genome
    tot_len = sum([len(x) for x in genome])
    tot_count = len(real_sites)
    tot_neg = tot_len - tot_count
    # Calculate number of successes and failures in selected region
    window_len = sel_end - sel_start
    sel_count = len(real_sites[(real_sites.id==sel_id) & (real_sites.start>=sel_start) & (real_sites.end<=sel_end)])
    sel_neg = window_len - sel_count
    # Expected value
    exp = (tot_count*window_len)/tot_len
    # Fisher's exact test
    odds, p = stats.fisher_exact([[sel_count, sel_neg], [tot_count, tot_neg]], alternative='two-sided')
    # Make combined dataframe
    #result = pd.DataFrame()
    result = {'start': [sel_start],
              'end': [sel_end],
              'count': [sel_count],
              'tot_count': [tot_count],
              'expected': [exp],
              'odds': [odds],
              'p': [p]}
    result = pd.DataFrame.from_dict(result)
    return result


def get_tiling_windows_over_record(record, width, shift):
    """
    Function to get tiling windows over a sequence.
    """
    starts = list(range(0, len(record), shift))
    ends = [np.min([x + width, len(record)]) for x in starts]
    idxs = np.tile(record.id, len(starts))
    return [x for x in zip(idxs, starts, ends)]


def get_tiling_windows_over_genome(genome, width, shift):
    """
    Function to get tiling windows over a genome.
    """
    if len(genome) == 1:
        windows = get_tiling_windows_over_record(genome[0], width, shift)
    else:
        windows = [get_tiling_windows_over_record(record, width, shift) for record in genome]
        windows = list(itertools.chain.from_iterable(windows))
    return windows


def enrich_in_sliding_windows(real_sites, genome, width, shift):
    """
    Function to test enrichment of binding sites in sliding windows across a genome.
    """
    # Get sliding windows across genome
    windows = get_tiling_windows_over_genome(genome, width, shift)
    results = pd.DataFrame()
    # Perform enrichment in each window
    for window in windows:
        result = enrich_in_window(real_sites, genome, window[0], window[1], window[2])
        result['id'] = window[0]
        results = pd.concat([results, result])
    # Overall FDR correction
    results['padj'] = multitest.fdrcorrection(results.p)[1]
    results.reset_index(inplace=True, drop=True)
    return results


def examine_thresholds(genome, model, simN, simK, rcomp, sense, min_threshold, verbose=False, combine_seqs=True):
    """
    Function to compare number of binding sites at various thresholds.
    """
    encoded_genome = MultiSeqEncoding(genome, rcomp=rcomp, sense=sense)
    shuf_genome = shuffle_records(genome, simN, simK)
    encoded_shuffled = MultiSeqEncoding(shuf_genome, sense=sense, rcomp=rcomp, group_by_name=True)
    thresholds = np.arange(min_threshold, 1.0, 0.1)
    real_binned = find_sites_multiseq(encoded_genome, model, thresholds, binned_counts=True, combine_seqs=combine_seqs)['binned_counts']
    shuf_binned = find_sites_multiseq(encoded_shuffled, model, thresholds, binned_counts=True, combine_seqs=combine_seqs, sep_ids=True)['binned_counts']
    results = {'real_binned':real_binned, 'shuf_binned': shuf_binned}
    if verbose:
        results['shuf_genome'] = shuf_genome
    return results
