import pandas as pd
import numpy as np
from .scan import score_site, _score_base
from .utils import get_site_seq


def _variant_effect_on_site(site, variants):
    """Function to predict the effects of variants on a single binding site.

    Args:
        sites (pd.DataFrame): dataframe containing information on the binding
                              site to analyze. Contains columns 'name',
                              'start' and 'end'
        variants (pd.DataFrame): dataframe containing information on each
                                 variant to analyze. Contains columns 'name',
                                 'pos', 'alt'.
    Returns:
        site_variants (pd.DataFrame): variants that intersect with the
                                      site and their predicted effect.

    """
    # Get variants that overlap with site
    site_variants = variants[
        (variants["name"] == site["name"])
        & (variants.pos >= site.start)
        & (variants.pos < site.end)
    ]
    if len(site_variants) > 0:
        site_variants = site_variants.merge(pd.DataFrame(site).transpose())
        # Get score of each variant
        site_variants["relative_pos"] = site_variants.pos - site.start
        site_variants["alt_base_score"] = site_variants.apply(
            lambda x: _score_base(x.weights, x.alt, x.relative_pos), axis=1
        )
        site_variants["variant_score"] = site_variants.apply(
            lambda x: np.sum(np.delete(x.ref_scores, x.relative_pos)), axis=1
        )
        site_variants["variant_score"] = (
            site_variants.variant_score + site_variants.alt_base_score
        ) / site_variants.max_score
        site_variants.drop(
            columns=["alt_base_score", "weights", "ref_scores",
                     "relative_pos"],
            inplace=True,
        )

    return site_variants


def variant_effect_on_sites(sites, variants, seqs, pwms):
    """Function to predict variant effect on sites.

    Args:
       sites (pd.DataFrame): dataframe containing information on each
                             binding site to analyze. Contains columns 'name'
                             (sequence name), 'matrix_id' (ID of the PWM that
                             matched to the site), 'start' (site position) and
                             'end' (site end).
       variants (pd.DataFrame): dataframe containing information on each
                                variant to analyze. Contains columns 'name'
                                (sequence name), 'pos' (variant position),
                                and 'alt' (alternate base at variant position).
       seqs (list / str): list of seqrecord objects or fasta file.
       pwms (pd.DataFrame): dataframe containing information on PWMs.
                            Contains columns 'matrix_id' and 'weights'.
    Returns:
       effects (pd.DataFrame): dataframe containing information on
                               variant effects.

    """
    # Copy sites
    all_sites = sites.copy()

    # Get site sequences
    all_sites["seq"] = all_sites.apply(lambda x: get_site_seq(x, seqs), axis=1)

    # Get PWM
    all_sites["weights"] = [
        pwms.weights[
            pwms.Matrix_id == x].values[0] for x in all_sites.Matrix_id
    ]

    # Get max scores
    all_sites["max_score"] = [
        np.sum(np.max(x, axis=1)) for x in all_sites.weights]

    # Score sites
    all_sites["ref_scores"] = all_sites.apply(
        lambda x: score_site(x.weights, x.seq, position_wise=True), axis=1
    )
    if "score" not in all_sites.columns:
        all_sites["score"] = all_sites.apply(
            lambda x: np.sum(x.ref_scores) / x.max_score, axis=1
        )

    # Get and concatenate results for all sites
    variant_effects = pd.concat(
        list(all_sites.apply(lambda x: _variant_effect_on_site(x, variants),
                             axis=1))
    )
    variant_effects.reset_index(drop=True, inplace=True)

    return variant_effects
