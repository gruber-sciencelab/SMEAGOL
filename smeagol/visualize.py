# General imports
import numpy as np
import pandas as pd

# Viz imports
from matplotlib import pyplot as plt
import seaborn as sns
import weblogo as wl
import IPython.display as ipd
import seqlogo

# Stats imports
import scipy.stats as stats
from sklearn import manifold


def plot_pwm(pwm_df, Matrix_id, height=15):
    """Function to plot sequence logo from PWM
    
    Args:
        pwm_df (pd.DataFrame): Dataframe containing cols weight, Matrix_id
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
    Function to plot sequence logo from PPM
    
    Args:
        ppm_df (pd.DataFrame): Dataframe containing cols probs, Matrix_id
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
    """Function to plot distribution of (fractional) binding site scores in real vs. shuffled genomes
    
    Args:
        real_preds (pd.DataFrame): DF containing Matrix_id, sense, bin
        shuf_preds (pd.DataFrame): DF containing Matrix_id, sense, bin
        Matrix_id (str): ID of PWM for which to plot distribution
        sense (str): sense of strand for which to plot distribution. If not given, both strands are used.
        rounding (int): number of digits to round bin thresholds
        file_path (str): path to save figure
    
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


def plot_background(shuf_counts, real_counts, Matrix_ids, genome_len=None, 
                    background='binomial', figsize=(17,8), ncols=4, file_path=None):
    """Function to plot the distribution of background counts.
    
    Args:
        shuf_counts (pd.DataFrame): Dataframe with motif counts in shuffled sequences. 
        real_counts (pd.DataFrame): Dataframe with motif counts in real sequence.
        Matrix_ids (list): IDs of PWMs to plot
        genome_len (int): total length of genome. Not needed if background = 'normal'.
        background (str): 'binomial', 'normal' or 'both'
        figsize (int): total figure size
        ncols (int): number of columns for figure panels
        file_path (str): path to save figure.
    
    Returns:
        Plot of motif distribution in real vs. background sequences.
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
    """Function to visualize a group of PWMs using t-SNE.
    
    Args:
        sims (np.array): Pairwise similarity matrix for PWMs.
        labels (list): PWM IDs.
        perplexity (float): parameter for t-SNE
        clusters (list): cluster IDs to label points
        cmap (dict): dictionary mapping cluster IDs to colors
    
    Returns:
        t-SNE plot of PWMs
        
    """
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


def sliding_window_count_plot(df, title, cols=3, aspect=1.4, file_path=None):
    """Function to visualize site counts using sliding windows.
    
    Args:
        df (pd.DataFrame): A pandas dataframe that contains the data to plot.
        title (str): The title to be used for the plot.
        cols (int): number of columns for plot
        aspect (float): width/height
        file_path (str): The path to the file into which the plot should be written.
    
    Returns:
        Sliding window plot. 
    
    """

    # Make sure plt is closed before starting a new one
    plt.close()
    
    # Get window labels and order
    dfc = df.copy()
    dfc['window'] = df.apply(lambda row: str(row.start)+'-'+str(row.end), axis=1)
    dfc['group_order'] = np.concatenate([list(range(x)) for x in df.id.value_counts()])
    
    with sns.plotting_context("notebook", font_scale=1):
        g = sns.catplot(x='window', y="count", row_order='group_order', col="id", data=dfc, kind="bar", 
                    height=4, aspect=aspect, col_wrap=cols, sharex=False, sharey=False, color='blue')
        g.set_xticklabels(rotation=90)
        plt.suptitle(title)
        plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, transparent=False, facecolor='white', dpi=300)


def sliding_window_enrichment_plot(sliding_window_df,x_var,y_var,xticklabels,title,file_path=None):
    """Function to visualize enrichment / depletion analysis using sliding windows.
    
    Args:
        sliding_window_df (pd.DataFrame): A pandas dataframe that contains the data to plot.
        x_var (str): The name of the column that should be used as x-axis. 
        y_var (str): The name of the column that should be used as y-axis.
        xticklabels (str): The name of the column that contains the x-axis labels.
        title (str): The title to be used for the plot.
        file_path (str): The path to the file into which the plot should be written.
    
    Returns:
        Sliding window plot. 
    
    """

    # Make sure plt is closed before starting a new one
    plt.close()
    
    # Set the size of the plot
    plt.figure(figsize=(12, 5))
    
    #sig_sorted = sliding_window_df.sort_values(y_var)['sig']

    barplot = sns.barplot(x=x_var, y=y_var, data=sliding_window_df, color='blue')
    barplot.set_xticklabels(labels=sliding_window_df[xticklabels], rotation=90)
    barplot.get_xticklabels()
    barplot.set(xlabel="Viral genome region (start - end)[nt]", ylabel = "Site counts")
    barplot.set_title(title)

    for p, sig in zip(barplot.patches, sliding_window_df.sig):
        if sig == True:
            barplot.text(p.get_x() + p.get_width() / 2., p.get_height(), 
                         '*', ha='center')

    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, transparent=False, facecolor='white', dpi=300)
