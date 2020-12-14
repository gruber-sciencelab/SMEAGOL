import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import seqlogo


def plot_pwm(pwm_df, Matrix_id):
    """
    Function to plot PWM
    
    Inputs:
        pwm_df: DF containing cols weight, Matrix_id
        Matrix_id: ID of PWM to plot
    
    Returns:
        Plots PWM
    """
    weights = pwm_df.weight.values[pwm_df.Matrix_id==Matrix_id]
    ppm = seqlogo.Ppm(np.exp2(weights[0])/4)
    return seqlogo.seqlogo(ppm, format='png', size='small')


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


def plot_background(shuf_counts, real_counts, Matrix_ids, file_path=None, figsize=(14,7)):
    fig, axs = plt.subplots(nrows=int(np.ceil(len(Matrix_ids)/5)), ncols=5, figsize=figsize)
    axs = axs.flatten()
    for i, matrix in enumerate(Matrix_ids):
        shuf_nums = shuf_counts[(shuf_counts.Matrix_id==matrix)].num
        real_num = real_counts[(real_counts.Matrix_id==matrix)].num.values
        axs[i].hist(shuf_nums)
        axs[i].axvline(real_num, color='red')
        axs[i].set_title(matrix)

    plt.tight_layout()
    plt.show()
    if file_path is not None:
        plt.savefig(file_path, facecolor='white')