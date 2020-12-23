import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import weblogo as wl
import IPython.display as ipd
import seqlogo
import scipy.stats as stats
from sklearn import manifold


def plot_pwm(pwm_df, Matrix_id, height=15):
    """
    Function to plot PWM
    
    Inputs:
        pwm_df: DF containing cols weight, Matrix_id
        Matrix_id: ID of PWM to plot
    
    Returns:
        Plots PWM
    """
    weights = pwm_df.weight.values[pwm_df.Matrix_id==Matrix_id]
    pm = np.exp2(weights[0])/4
    pm = seqlogo.Ppm(pm/np.expand_dims(np.sum(pm, axis=1),1))
    options = wl.LogoOptions(unit_name = 'bits', color_scheme = wl.std_color_schemes['classic'], 
                         show_fineprint = False, stack_width = height)
    options.logo_title = Matrix_id
    out = wl.formatters['png'](pm, wl.LogoFormat(pm, options))
    return ipd.Image(out)


def plot_ppm(ppm_df, Matrix_id, height=15):
    """
    Function to plot PPM
    
    Inputs:
        ppm_df: DF containing cols probs, Matrix_id
        Matrix_id: ID of PWM to plot
    
    Returns:
        Plots PPM
    """
    probs = ppm_df.probs.values[ppm_df.Matrix_id==Matrix_id]
    pm = seqlogo.Ppm(probs[0])
    options = wl.LogoOptions(unit_name = 'bits', color_scheme = wl.std_color_schemes['classic'], 
                         show_fineprint = False, stack_width = height)
    options.logo_title = Matrix_id
    out = wl.formatters['png'](pm, wl.LogoFormat(pm, options))
    return ipd.Image(out)


def plot_binned_count_dist(real_preds, Matrix_id, sense, shuf_preds=None, rounding=3, file_path=None):
    """
    Function to plot distribution of (fractional) binding site scores in real vs. shuffled genomes
    
    Inputs:
        real_preds: DF containing Matrix_id, sense, bin
        shuf_preds: DF containing Matrix_id, sense, bin
        Matrix_id: ID of PWM for which to plot distribution
        sense: sense of strand for which to plot distribution. If not given, both strands are used.
    
    Returns:
        Plot of binding site counts binned by score.
    """
    real_selected = real_preds[(real_preds.sense==sense) & (real_preds.Matrix_id==Matrix_id)].copy()
    real_selected.bin = np.round(real_selected.bin, rounding)
    if shuf_preds is not None:
        shuf_selected = shuf_preds[(shuf_preds.sense==sense) & (shuf_preds.Matrix_id==Matrix_id)].copy()
        shuf_selected.bin = np.round(shuf_selected.bin, rounding)
        shuf_selected['sequence'] = 'shuffled'
        real_selected['sequence'] = 'real'
        selected = pd.concat([real_selected, shuf_selected])
        plt.figure(figsize=(12, 4))
        sns.barplot(x="bin", y="num", hue="sequence", data=selected).set_title(Matrix_id)
        if file_path is not None:
            plt.savefig(file_path, facecolor='white')
    else:
        plt.figure(figsize=(12, 4))
        sns.barplot(x="bin", y="num", data=real_selected).set_title(Matrix_id)
        if file_path is not None:
            plt.savefig(file_path, facecolor='white')


def plot_background(shuf_counts, real_counts, Matrix_ids, genome_len=None, background='binomial', figsize=(17,8), ncols=4, file_path=None):
    """
    Function to plot the distribution of background counts.
    """
    ncols = min(len(Matrix_ids), ncols)
    nrows=int(np.ceil(len(Matrix_ids)/ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()
    for i, matrix in enumerate(Matrix_ids):
        # Get shuffled counts
        shuf_nums = shuf_counts[(shuf_counts.Matrix_id==matrix)].num
        
        # Create count table
        cttable = shuf_nums.value_counts().sort_index()/len(shuf_nums)
        
        # Plot count table
        axs[i].bar(cttable.index, cttable)
        
        if background == 'binomial':
            binom_p = shuf_nums.mean()/genome_len
            preds = stats.binom(genome_len, binom_p).pmf(cttable.index)
            axs[i].plot(cttable.index, preds, c='orange')
        elif background == 'normal':
            preds = stats.norm.pdf(cttable.index, shuf_nums.mean(), shuf_nums.std())
            axs[i].plot(cttable.index, preds, c='orange')
        if background == 'both':
            binom_p = shuf_nums.mean()/genome_len
            binom_preds = stats.binom(genome_len, binom_p).pmf(cttable.index)
            norm_preds = stats.norm.pdf(cttable.index, shuf_nums.mean(), shuf_nums.std())
            axs[i].plot(cttable.index, binom_preds, c='orange')
            axs[i].plot(cttable.index, norm_preds, c='green')
        
        # Add real counts
        real_num = real_counts[(real_counts.Matrix_id==matrix)].num.values
        axs[i].axvline(real_num, color='red')
        axs[i].set_title(matrix)

    plt.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, facecolor='white')
    else:
        plt.show()
        

def plot_pwm_similarity(sims, labels, perplexity=5, clusters=None, cmap=None):
    coords = manifold.TSNE(n_components=2, metric="precomputed", perplexity=perplexity).fit(1-sims).embedding_
    if cmap is not None:
        plt.scatter(coords[:, 0], coords[:, 1], marker = 'o', c=pd.Series(clusters).map(cmap))
    else:
        plt.scatter(coords[:, 0], coords[:, 1], marker = 'o')
    for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
        plt.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', alpha = 0.2),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()