"""Plot lightcones, coeval cubes, etc."""

from types import Callable

import matplotlib.pyplot as plt
from astropy import units as un


def plot_3d_box(
    box: un.Quantity,
    box_length: un.Quantity,
    zaxis: un.Quantity | None = None,
    x_axis: int = 0,
    transform: Callable | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | list[plt.Axes] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    colors: list | None = None,
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    log: list[bool] | None = None,
    labels: list | None = None,
    smooth: float | bool = False,
    leg_kwargs: dict | None = None,
):
    """3D plot of the lightcone or coeval.

    Parameters
    ----------
    box : un.Quantity
        The lightcone or coeval box.
    box_length : un.Quantity
        The length of the box in cMpc.
    zaxis : un.Quantity, optional
        The z-axis values, optional for coeval box.
        For a lightcone, these are the redshifts of the slices
        and are mandatory.
    trasform : Callable, optional
        A function to transform the data before plotting.
    x_axis : int, optional
        The axis to plot along. Default is 0 (x-axis).
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If None, a new figure will be created.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    cmap : str, optional
        The colormap to use for the plot.
    norm : matplotlib.colors.Normalize, optional
        The normalization to use for the plot.
    log : bool, optional
        If True, use logarithmic scaling for the color map.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """


def plot_2d_slice(
    box: un.Quantity,
    box_length: un.Quantity,
    zaxis: un.Quantity | None = None,
    x_axis: int = 0,
    transform: Callable | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | list[plt.Axes] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    colors: list | None = None,
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    log: list[bool] | None = None,
    labels: list | None = None,
    smooth: float | bool = False,
    leg_kwargs: dict | None = None,
):
    """Plot a 2D slice of the lightcone.

    Parameters
    ----------
    lc : np.ndarray
        The lightcone data.
    zs : np.ndarray
        The redshift values.
    box_length : astropy.units.Quantity
        The length of the box in Mpc.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If None, a new figure will be created.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    cmap : str, optional
        The colormap to use for the plot.
    norm : matplotlib.colors.Normalize, optional
        The normalization to use for the plot.
    log : bool, optional
        If True, use logarithmic scaling for the color map.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """


def plot_1d_pdf(
    box: un.Quantity,
    box_length: un.Quantity,
    transform: Callable | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | list[plt.Axes] | None = None,
    titles: str | None = None,
    colors: list | None = None,
    fontsize: float | None = 16,
    log: list[bool] | None = None,
    labels: list | None = None,
    leg_kwargs: dict | None = None,
):
    """Plot the pixel distribution function: a histogram of the lc or coeval box.

    Parameters
    ----------
    box : un.Quantity
        The lightcone or coeval box.
    box_length : un.Quantity
        The length of the box in cMpc.





    """


def lightcone_sliceplot():
    return plot_2d_slice()


def coeval_sliceplot():
    return plot_2d_slice()


def plot_global_history(
    box: un.Quantity,
    box_length: un.Quantity,
    transform: Callable | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | list[plt.Axes] | None = None,
    titles: str | None = None,
    colors: list | None = None,
    fontsize: float | None = 16,
    log: list[bool] | None = None,
    labels: list | None = None,
    leg_kwargs: dict | None = None,
):
    """Plot the global history of the lightcone or coeval box.

    Parameters
    ----------
    box : un.Quantity
        The lightcone or coeval box.
    box_length : un.Quantity
        The length of the box in cMpc.
    transform : Callable, optional
        A function to transform the data before plotting.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new axes will be created.
    titles : str, optional
        The title of the plot.
    colors : list, optional
        The colors to use for the plot.
    fontsize : float, optional
        The font size to use for the plot.
    log : list[bool], optional
        If True, use logarithmic scaling for all axes.
    labels : list, optional
        The labels to use for the plot.
    leg_kwargs : dict, optional
        Additional keyword arguments to pass to the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
