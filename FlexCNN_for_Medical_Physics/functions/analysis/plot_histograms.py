import pandas as pd
import matplotlib.pyplot as plt


def plot_hist_1D(ax, dataframe, title, x_label, y_label, column_1, column_2, xlim, ylim, 
                 bins=400, alpha=0.5, titlesize=13, fontsize=12
):
    """
    Plot overlaid histograms for two dataframe columns within specified limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    dataframe : pd.DataFrame
        Source data.
    title, x_label, y_label : str
        Text for plot labelling.
    column_1, column_2 : str
        Column names to plot.
    xlim, ylim : tuple(float, float)
        Axis limits for x and y.
    bins : int
        Number of histogram bins (default 400).
    alpha : float
        Bar transparency for overlaid histograms.
    titlesize : int
        Title font size (default 13).
    fontsize : int
        Axis label and tick font size (default 12).

    Notes
    -----
    - Rows outside xlim are filtered out for both columns before plotting.
    - Mutates the provided `ax` with histogram artists.
    - Assumes columns contain numeric data.
    """
    df = dataframe.copy()
    df = df[df[column_1] > xlim[0]]
    df = df[df[column_1] < xlim[1]]
    df = df[df[column_2] > xlim[0]]
    df = df[df[column_2] < xlim[1]]

    df[[column_1, column_2]].plot.hist(
        xlim=xlim,
        ylim=ylim,
        bins=bins,
        alpha=alpha,
        ax=ax,
        fontsize=fontsize
    )
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)


def plot_hist_2D(ax, dataframe, title, x_label, y_label,
    x_column, y_column,
    xlim=(0, 1), ylim=(0, 1), gridsize=None,
    titlesize=13, fontsize=12, ticksize=10
):
    """
    Plot a hexbin (2D histogram) for two dataframe columns and draw diagonal line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    dataframe : pd.DataFrame
        Source data.
    title, x_label, y_label : str
        Plot labelling.
    x_column, y_column : str
        Column names for the x and y axes.
    xlim, ylim : tuple(float, float)
        Axis limits (defaults (0,1)).
    gridsize : int or None
        Hexbin resolution; None lets pandas choose defaults.
    titlesize : int
        Title font size (default 13).
    fontsize : int
        Axis label font size (default 12).
    ticksize : int
        Tick label font size (default 10).

    Notes
    -----
    - Filters rows outside provided limits before plotting.
    - Always draws a diagonal reference line (y = x mapped through limits).
    - Assumes numeric columns.
    """
    df = dataframe.copy()
    df = df[df[x_column] > xlim[0]]
    df = df[df[x_column] < xlim[1]]
    df = df[df[y_column] > ylim[0]]
    df = df[df[y_column] < ylim[1]]

    df.plot.hexbin(
        ax=ax,
        x=x_column,
        y=y_column,
        xlim=xlim,
        ylim=ylim,
        gridsize=gridsize,
        fontsize=ticksize
    )
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.plot(xlim, ylim, linestyle='--', color='gray')