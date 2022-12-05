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


def pwm_logo(weights, title="", height=15):
    """Function to visualize the sequence logo of a PWM.

    Args:
        weights (np.array): array containing PWM weight values
        title (str): Title of the logo.
        height (int): Height of the plot

    Returns:
        PWM logo

    """
    pm = np.exp2(weights) / 4
    pm = seqlogo.Ppm(pm / np.expand_dims(np.sum(pm, axis=1), 1))
    options = wl.LogoOptions(
        unit_name="bits",
        color_scheme=wl.std_color_schemes["classic"],
        show_fineprint=False,
        stack_width=height,
    )
    options.logo_title = title
    out = wl.formatters["png"](pm, wl.LogoFormat(pm, options))
    return ipd.Image(out)


def plot_pwm(pwm_df, Matrix_id, height=15):
    """Function to plot sequence logo from PWM

    Args:
        pwm_df (pd.DataFrame): Dataframe containing cols weight, Matrix_id
        Matrix_id: ID of PWM to plot (will be used as logo title)
        height (int): Height of the plot

    Returns:
        Plots PWM

    """
    weights = pwm_df.weights.values[pwm_df.Matrix_id == Matrix_id]
    logo = pwm_logo(weights=weights[0], title=Matrix_id, height=height)

    return logo


def plot_ppm(ppm_df, Matrix_id, height=15):
    """
    Function to plot sequence logo from PPM

    Args:
        ppm_df (pd.DataFrame): Dataframe containing cols probs, Matrix_id
        Matrix_id: ID of PWM to plot

    Returns:
        Plots PPM

    """
    probs = ppm_df.probs.values[ppm_df.Matrix_id == Matrix_id]
    pm = seqlogo.Ppm(probs[0])
    options = wl.LogoOptions(
        unit_name="bits",
        color_scheme=wl.std_color_schemes["classic"],
        show_fineprint=False,
        stack_width=height,
    )
    options.logo_title = Matrix_id
    out = wl.formatters["png"](pm, wl.LogoFormat(pm, options))
    return ipd.Image(out)


def plot_binned_count_dist(
    real_preds, Matrix_id, sense, shuf_preds=None, rounding=3, file_path=None
):
    """Function to plot distribution of (fractional) binding site scores
    in real vs. shuffled genomes

    Args:
        real_preds (pd.DataFrame): DF containing Matrix_id, sense, bin
        shuf_preds (pd.DataFrame): DF containing Matrix_id, sense, bin
        Matrix_id (str): ID of PWM for which to plot distribution
        sense (str): sense of strand for which to plot distribution.
                     If not given, both strands are used.
        file_path (str): path to save figure

    Returns:
        Plot of binding site counts binned by score.

    """
    real_selected = real_preds[
        (real_preds.sense == sense) & (real_preds.Matrix_id == Matrix_id)
    ].copy()
    if shuf_preds is not None:
        shuf_selected = shuf_preds[
            (shuf_preds.sense == sense) & (shuf_preds.Matrix_id == Matrix_id)
        ].copy()
        shuf_selected["sequence"] = "shuffled"
        real_selected["sequence"] = "real"
        selected = pd.concat([real_selected, shuf_selected])
        plt.figure(figsize=(12, 4))
        sns.barplot(x="bin", y="num", hue="sequence", data=selected).set_title(
            Matrix_id
        )
        if file_path is not None:
            plt.savefig(file_path, facecolor="white")
    else:
        plt.figure(figsize=(12, 4))
        sns.barplot(x="bin", y="num", data=real_selected).set_title(Matrix_id)
        if file_path is not None:
            plt.savefig(file_path, facecolor="white")


def plot_background(
    enrichment_result, Matrix_ids, sense, figsize=(17, 8), ncols=4,
    file_path=None
):
    """Function to plot the distribution of background counts.

    Args:
        enrichment_result (dict): Dictionary with result from enrich_in_genome
        Matrix_ids (list): IDs of PWMs to plot
        sense (str): '+' or '-'
        figsize (tuple): total figure size
        ncols (int): number of columns for figure panels
        file_path (str): path to save figure.

    Returns:
        Plot of motif distribution in real vs. background sequences.
    """
    # Setup figure
    ncols = min(len(Matrix_ids), ncols)
    nrows = int(np.ceil(len(Matrix_ids) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    # Subset dataframes
    shuf_counts = enrichment_result["shuf_counts"]
    shuf_counts = shuf_counts[shuf_counts.sense == sense]
    enr = enrichment_result["enrichment"]
    enr = enr[enr.sense == sense]
    real_counts = enrichment_result["real_counts"]
    real_counts = real_counts[real_counts.sense == sense]

    for i, matrix in enumerate(Matrix_ids):
        # Get shuffled counts
        shuf_nums = shuf_counts[(shuf_counts.Matrix_id == matrix)].num

        # Create count table
        cttable = shuf_nums.value_counts().sort_index() / len(shuf_nums)

        # Plot count table
        axs[i].bar(cttable.index, cttable)

        if "p" in enr.columns:
            enr_curr = enr[(enr.Matrix_id == matrix)]
            adj_len = enr_curr.adj_len.values[0]
            binom_p = shuf_nums.mean() / adj_len
            preds = stats.binom(adj_len, binom_p).pmf(cttable.index)
            axs[i].plot(cttable.index, preds, c="orange")
        if "pnorm" in enr.columns:
            preds = stats.norm.pdf(cttable.index, shuf_nums.mean(),
                                   shuf_nums.std())
            axs[i].plot(cttable.index, preds, c="green")

        # Add real counts
        real_num = real_counts[(real_counts.Matrix_id == matrix)].num.values
        axs[i].axvline(real_num, color="red")
        axs[i].set_title(matrix)

    plt.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, facecolor="white")
    else:
        plt.show()


def sliding_window_count_plot(df, title, cols=3, aspect=1.4, file_path=None):
    """Function to visualize site counts using sliding windows.

    Args:
        df (pd.DataFrame): A pandas dataframe that contains the data to plot.
        title (str): The title to be used for the plot.
        cols (int): number of columns for plot
        aspect (float): width/height
        file_path (str): The path to the file into which the plot
                         should be written.

    Returns:
        Sliding window plot.

    """

    # Make sure plt is closed before starting a new one
    plt.close()

    # Get window labels and order
    dfc = df.copy()
    dfc["window"] = df.apply(lambda row: str(row.start) + "-" + str(row.end),
                             axis=1)
    dfc["group_order"] = np.concatenate([
        list(range(x)) for x in df.id.value_counts()])

    with sns.plotting_context("notebook", font_scale=1):
        g = sns.catplot(
            x="window",
            y="count",
            row_order="group_order",
            col="id",
            data=dfc,
            kind="bar",
            height=4,
            aspect=aspect,
            col_wrap=cols,
            sharex=False,
            sharey=False,
            color="blue",
        )
        g.set_xticklabels(rotation=90)
        plt.suptitle(title)
        plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, transparent=False, facecolor="white", dpi=300)


def sliding_window_enrichment_plot(
    sliding_window_df,
    x_var,
    y_var,
    xticklabels,
    title,
    fig_size=(12, 5),
    file_path=None,
):
    """Function to visualize enrichment / depletion analysis using sliding
    windows.

    Args:
        sliding_window_df (pd.DataFrame): A pandas dataframe that contains
                                          the data to plot.
        x_var (str): The name of the column that should be used as x-axis.
        y_var (str): The name of the column that should be used as y-axis.
        xticklabels (str): The name of the column that contains the x-axis
                           labels.
        title (str): The title to be used for the plot.
        file_path (str): The path to the file into which the plot should be
                         written.

    Returns:
        Sliding window plot.

    """

    # Make sure plt is closed before starting a new one
    plt.close()

    # Set the size of the plot
    plt.figure(figsize=fig_size)

    # sig_sorted = sliding_window_df.sort_values(y_var)['sig']

    barplot = sns.barplot(x=x_var, y=y_var, data=sliding_window_df,
                          color="blue")
    barplot.set_xticklabels(labels=sliding_window_df[xticklabels], rotation=90)
    barplot.get_xticklabels()
    barplot.set(xlabel="Viral genome region (start - end)[nt]",
                ylabel="Site counts")
    barplot.set_title(title)

    for p, sig in zip(barplot.patches, sliding_window_df.sig):
        if sig is True:
            barplot.text(
                p.get_x() + p.get_width() / 2.0, p.get_height(), "*",
                ha="center"
            )

    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, transparent=False, facecolor="white", dpi=300)


def plot_clustermap(
    foldchanges,
    pvalues,
    threshold=0.05,
    row_cluster=True,
    dendogram=True,
    file_path=None,
):
    """Simple function to plot clustermap of foldchanges and pwms
    (without annotation); function will filter foldchanges whenever
    pvalues are significant.

    Args:
        foldchanges (np.array): matrix of foldchanges for enrichment and
                                depletion of pwms
        pvalues (np.array): matrix of pvalues for enrichment of the same pwms
        threshold (float): threshold for pvalues. default 0.05
        row_cluster (bool): Cluster rows. Default True
        dendogram (bool): Plot dendograms. Default True
        file_path (str): The path to the file into which the plot
                         should be written.

    Returns:
        heatmap plot.
    """
    np.seterr(divide="ignore")

    # Create log2FC tables based on foldchanges and significant pvalues
    log2fc = np.log2(foldchanges)
    log2fc = log2fc.fillna(0)
    log2fc[pvalues > threshold] = 0

    # Transpose (species as rows and columns as pwms)
    plot_df = log2fc.transpose()

    # Parameters for plot
    cbar_kws = {"extend": "both"}
    cmap = plt.get_cmap(sns.diverging_palette(240, 0, l=30, s=100,
                                              as_cmap=True))
    cmap.set_under("navy")
    cmap.set_over("darkred")
    height = len(plot_df) * 0.25

    # Create clustermap plot and save pdf
    sns_plot = sns.clustermap(
        plot_df,
        cmap=cmap,
        cbar_kws=cbar_kws,
        row_cluster=row_cluster,
        xticklabels=1,
        yticklabels=1,
        figsize=(35, height),
        center=0,
    )

    # Show dendogram
    sns_plot.ax_col_dendrogram.set_visible(dendogram)
    sns_plot.ax_row_dendrogram.set_visible(dendogram)

    plt.show()
    if file_path is not None:
        sns_plot.savefig(file_path, dpi=300)


def plot_clustermap_annot(
    foldchanges,
    pvalues,
    annot,
    column,
    threshold=0.05,
    group_by=None,
    drop_zeros=False,
    plot=None,
    min_occ=3,
    cluster=True,
    col_cluster=True,
    row_cluster=True,
    dendogram=True,
    col_dendogram=True,
    row_dendogram=True,
    file_path=None,
):
    """Function to plot clustermap of foldchanges and pwms, with annotation
    and more parameters

    Args:
        foldchanges (np.array): matrix of foldchanges for enrichment and
                                depletion of pwms
        pvalues (np.array): matrix of pvalues for enrichment of the same pwms
        annot (pd.DataFrame): annotation table with information on each column
                              of pwm matrix
        column (str):  which column contains the name to be used as labels on
                       the plot
        threshold (float): threshold for pvalues. default 0.05
        group_by (str): extra column in annot table that groups the occurrences
                       (ex. Genus, Family, Kingdom, Host, etc) (Default: None).
                       If no group_by function was provided but column
                       parameter is "Species", we try to cluster species by
                       "Genus" using the first word of Species.
        drop_zeros (bool): if True, drops all columns and rows in which all
                           values are zero (Default: False)
        plot (str): 'sep' if you wish to separate the plots according to
                    the groups provided in group_by (Default: None)
        min_occ (int): min number of occurrences within a group
                       (defined by group_by) to plot separately if plot="sep"
        cluster (bool): Clusters both columns and rows (Default: True)
        col_cluster (bool): Cluster columns (Default: True)
        row_cluster (bool): Cluster rows (Default: True)
        dendogram (bool): Plot both row and col dendograms (Default: True)
        row_dendogram (bool): Plot row dendograms (Default: True)
        col_dendogram (bool): Plot column dendograms (Default: True)
        file_path (str): The path to the file into which the plot
                         should be written.

    Returns:
        annotated heatmap plot.

    """

    np.seterr(divide="ignore")

    # Create log2FC tables based on foldchanges and significant pvalues
    log2fc = np.log2(foldchanges)
    log2fc = log2fc.fillna(0)
    log2fc[pvalues > threshold] = 0

    # Drop columns and rows with only zeros:
    if drop_zeros is True:
        log2fc = log2fc.loc[(log2fc.sum(axis=1) != 0),
                            (log2fc.sum(axis=0) != 0)]

    # Merge log2fc with annotation table
    plot_df = pd.merge(
        annot, log2fc.transpose(), how="right", left_index=True,
        right_index=True
    )

    flag = 0  # Flag to check if user defined "group_by"

    # If user did not define "group_by" for annotation, and column is Species,
    # try to create a group 'Genus' based on the first word of Species
    if group_by is None:
        flag = 1  # Change flag to not include legend later
        # unless user defines the column
        if column == "Species":
            # Create column as genus
            plot_df["Genus"] = plot_df["Species"].str.split(" ").str[0]
            group_by = "Genus"
        else:
            group_by = column

    # Order dataframe by "group_by" (or column if "group_by" does not exist)
    plot_df = plot_df.sort_values(by=group_by)

    # Create colors for annotation in "group_by"
    # (or column if "group_by" does not exist)
    lut = dict(
        zip(
            plot_df[group_by].unique(), sns.hls_palette(
                len(plot_df[group_by].unique()))
        )
    )
    row_colors = plot_df[group_by].map(lut)

    # Parameters for plot
    cbar_kws = {"extend": "both"}
    cmap = plt.get_cmap(sns.diverging_palette(240, 0, l=30, s=100,
                                              as_cmap=True))
    cmap.set_under("navy")
    cmap.set_over("darkred")
    min = len(annot.columns)
    max = len(plot_df.columns)
    height = len(plot_df) * 0.25  # height of figure depends on how many rows
    width = max * 0.2  # width of figure depends on how many columns

    # Clustering on or off
    if cluster is False:
        row_cluster = False
        col_cluster = False

    # Create clustermap plot
    sns_plot = sns.clustermap(
        plot_df.iloc[:, min:max],
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        cmap=cmap,
        cbar_kws=cbar_kws,
        xticklabels=1,
        yticklabels=plot_df[column],
        row_colors=row_colors,
        figsize=(width, height),
        center=0,
    )

    # Show dendogram
    if dendogram is False:
        col_dendogram = False
        row_dendogram = False
    sns_plot.ax_col_dendrogram.set_visible(col_dendogram)
    sns_plot.ax_row_dendrogram.set_visible(row_dendogram)

    # If user defined "group_by" for annotation, include legend
    if flag == 0:
        from matplotlib.patches import Patch

        handles1 = [Patch(facecolor=lut[name]) for name in lut]
        plt.legend(
            handles1,
            lut,
            title=group_by,
            bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure,
            loc="upper right",
        )

    # Plot and save pdf
    plt.show()
    sns_plot.savefig(file_path)

    # If you want separate plots for "group_by" annotation include plot = "sep"
    # row_cluster parameter was not included here, default is True
    if plot == "sep":
        k = len(plot_df[group_by].unique())
        for i in range(0, k):
            col2 = plot_df[group_by].unique()[i]
            df = plot_df.loc[plot_df[group_by] == col2]
            if len(df) > min_occ:
                height = len(df) * 0.5
                sns_col2_plot = sns.clustermap(
                    df.iloc[:, min:max],
                    cmap=cmap,
                    cbar_kws=cbar_kws,
                    xticklabels=1,
                    yticklabels=df[column],
                    figsize=(width, height),
                    row_colors=row_colors,
                    center=0,
                )
                plt.show()
                sns_col2_plot.savefig(col2 + "_Significant_Log2FC.pdf")
