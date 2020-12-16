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
    enr = real_counts.merge(shuf_stats, on=['Matrix_id', 'sense'])
    enr = enr[(enr.num > 0) | (enr.avg > 0)].copy().reset_index(drop=True)
    if background == 'normal':
        enr['z'] = (enr.num - enr.avg)/enr.sd
        enr['z'] = enr.z.replace([-np.inf], -100)
        enr['p'] = stats.norm.sf(abs(enr.z))*2
    elif background == 'binomial':
        assert seqlen is not None
        enr['p'] = enr.apply(lambda x:stats.binom(seqlen, x['avg']/seqlen).pmf(x['num']), axis=1)
    enr_full = pd.DataFrame()
    for sense in pd.unique(enr.sense):
        enr_x = enr[enr.sense == sense].copy()
        enr_x['fdr'] = multitest.fdrcorrection(enr_x.p)[1]
        enr_full = pd.concat([enr_full, enr_x])
    enr_full.sort_values(['sense', 'fdr'], inplace=True)
    return enr_full


def enrich_in_genome(genome, model, simN, simK, rcomp, sense, threshold, background='binomial', verbose=False, combine_seqs=True):
    """
    Function to calculate enrichment of PWMs in a sequence relative to a shuffled background.
    """
    # Encode genome
    encoded_genome = MultiSeqEncoding(genome, rcomp=rcomp, sense=sense)
    # Find sites on real genome
    real_preds = find_sites_multiseq(encoded_genome, model, threshold, sites=True, total_counts=True, combine_seqs=combine_seqs)
    real_sites = real_preds['sites']
    real_counts = real_preds['total_counts']
    # Shuffle genome
    shuf_genome = shuffle_records(genome, simN, simK)
    # Encode shuffled genomes
    encoded_shuffled = MultiSeqEncoding(shuf_genome, sense=sense, rcomp=rcomp, group_by_name=True)
    # Count sites on shuffled genomes
    if verbose:
        shuf_preds = find_sites_multiseq(encoded_shuffled, model, threshold, total_counts=True, stats=True, combine_seqs=combine_seqs, sep_ids=True)
        shuf_counts = shuf_preds['total_counts']
    else:
        shuf_preds = find_sites_multiseq(encoded_shuffled, model, threshold, stats=True, 
                                         combine_seqs=combine_seqs, sep_ids=True)
    shuf_stats = shuf_preds['stats']
    # Calculate binding site enrichment
    if background == 'normal':
        enr = enrich_over_shuffled(real_counts, shuf_stats, background=background)
    elif background == 'binomial':
        seqlen = sum([len(x) for x in genome])
        enr = enrich_over_shuffled(real_counts, shuf_stats, background=background, seqlen=seqlen)
    results = {'enrichment':enr, 'real_sites':real_sites, 'real_counts':real_counts, 'shuf_stats': shuf_stats}
    if verbose:
        results['shuf_counts'] = shuf_counts
    return results


def enrich_in_window(real_sites, genome, sel_id, sel_start, sel_end):
    """
    Function to calculate enrichment of PWMs in subsequence relative to total sequence.
    """
    # Calculate number of successes and failures in selected region
    sel_counts = real_sites.loc[((real_sites.id==sel_id) & (real_sites.start>=sel_start) & (real_sites.end<=sel_end))]
    if len(sel_counts) > 0:
        sel_counts = sel_counts[['Matrix_id', 'sense']].value_counts().reset_index()
        sel_counts.columns = ['Matrix_id', 'sense', 'count']
        # Calculate total number of successes and failures over genome
        tot_counts = real_sites[['Matrix_id', 'sense']].value_counts().reset_index()
        tot_counts.columns = ['Matrix_id', 'sense', 'count_total']
        tot_len = sum([len(x) for x in genome])
        tot_counts['neg_total'] = tot_len - tot_counts['count_total']
        # Make combined dataframe
        sel_counts['id'] = sel_id
        sel_counts['start'] = sel_start
        sel_counts['end'] = sel_end
        sel_counts['neg'] = (sel_end - sel_start) - sel_counts['count']
        sel_counts = sel_counts.merge(tot_counts, on=['Matrix_id', 'sense'])
        # Fisher's exact test
        def fisher_test_window(x):
            odds, p = stats.fisher_exact([[x['count'], x['neg']], [x['count_total'], x['neg_total']]], alternative='two-sided')
            x['odds'] = odds
            x['p'] = p
            return x
        sel_counts = sel_counts.apply(fisher_test_window, axis = 1)
        sel_counts['padj'] = multitest.fdrcorrection(sel_counts.p)[1]
        return sel_counts


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
        results = pd.concat([results, result])
    # Overall FDR correction
    results['padj'] = multitest.fdrcorrection(results.p)[1]
    return results