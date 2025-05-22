"""Plotting functions for 1D and 2D power spectra."""

import warnings

import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from tuesday.core import CylindricalPS, SphericalPS
from tuesday.core.units import validatePS as validate


def plot_1d_power_spectrum(
    power_spectrum: SphericalPS,*,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: list | None = None,
    log: list[bool] | None = False,
    fontsize: float | None = 16,
    label: list | None = None,
    smooth: float | bool = False,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 1D power spectrum.

    Parameters
    ----------
    power_spectrum : SphericalPS
        Instance of the SphericalPS class.
    ax : plt.Axes, optional
        Axes object to plot on. If None, a new axes is created.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    color : str, optional
        Color of the PS line in the plot.
    log : list[bool], optional
        List of booleans to set the x and y axes to log scale.
    fontsize : float, optional
        Font size for the plot labels.
    label : list, optional
        Label for the PS.
    smooth : float, optional
        Standard deviation for Gaussian smoothing.
        If True, uses a standard deviation of 1.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    """
    rcParams.update({"font.size": fontsize})
    wavemodes = power_spectrum.k
    power_spectrum = power_spectrum.ps
    if color is None:
        color = "C0"
    if xlabel is None:
        xlabel = f"k [{wavemodes.unit:latex_inline}]"

    if ylabel is None:
        ylabel = f"[{power_spectrum.unit:latex_inline}]"
        if power_spectrum.unit == un.mK**2:
            ylabel = r"$\Delta^2_{21} \,$" + ylabel
        elif power_spectrum.unit == un.dimensionless_unscaled:
            ylabel = r"$\Delta^2_{21}$"
        else:
            ylabel = r"$P(k) \,$" + ylabel
    if smooth:
        power_spectrum = gaussian_filter(power_spectrum, sigma=smooth)
    ax.plot(wavemodes, power_spectrum, color=color, label=label)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")
    if label is not None:
        ax.legend(**legend_kwargs)
    return ax


def plot_2d_power_spectrum(
    power_spectrum: CylindricalPS,*,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    log: list[bool] | None = False,
    smooth: float | bool = False,
    cbar: bool | None = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D power spectrum.

    Parameters
    ----------
    power_spectrum : CylindricalPS
        Instance of the CylindricalPS class.
    axs : plt.Axes | list[plt.Axes], optional
        Axes object(s) to plot on. If None, new axes are created.
    title : str, optional
        Title(s) of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    clabel : str, optional
        Label for the colorbar.
    cmap : str, optional
        Colormap for the plot.
    fontsize : float, optional
        Font size for the plot labels.
    vmin : float, optional
        Minimum value for the color scale.
    vmax : float, optional
        Maximum value for the color scale.
    log : list[bool], optional
        List of booleans to set the kperp, kpar, and PS axes to log scale.
    smooth : float, optional
        Standard deviation for Gaussian smoothing.
        Default is False, if True, uses a standard deviation of 1.
    """
    rcParams.update({"font.size": fontsize})
    kperp = power_spectrum.kperp
    kpar = power_spectrum.kpar
    power_spectrum = power_spectrum.ps

    if xlabel is None:
        xlabel = r"k$_\perp \,$" + f"[{kperp.unit:latex_inline}]"

    if ylabel is None:
        ylabel = r"k$_\parallel \,$" + f"[{kpar.unit:latex_inline}]"

    if clabel is None:
        clabel = f"[{power_spectrum.unit:latex_inline}]"
        if power_spectrum.unit == un.mK**2:
            clabel = r"$\Delta^2_{21} \,$" + clabel
        elif power_spectrum.unit == un.dimensionless_unscaled:
            clabel = r"$\Delta^2_{21}$"
        else:
            clabel = r"$P(k) \,$" + clabel

    cmap_kwargs = {}
    if vmin is None:
        if log[2]:
            cmap_kwargs["vmin"] = np.nanpercentile(np.log10(power_spectrum.value), 5)
        else:
            cmap_kwargs["vmin"] = np.nanpercentile(power_spectrum.value, 5)
    if vmax is None:
        if log[2]:
            cmap_kwargs["vmax"] = np.nanpercentile(np.log10(power_spectrum.value), 95)
        else:
            cmap_kwargs["vmax"] = np.nanpercentile(power_spectrum.value, 95)
    if log[2]:
        cmap_kwargs = {}
        cmap_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if smooth:
        unit = power_spectrum.unit
        power_spectrum = gaussian_filter(power_spectrum, sigma=smooth) * unit
    mask = np.isnan(np.nanmean(power_spectrum, axis=-1))
    power_spectrum = power_spectrum[~mask]
    kperp = kperp[~mask]
    im = ax.pcolormesh(
        kperp.value,
        kpar.value,
        power_spectrum.value.T,
        cmap=cmap,
        **cmap_kwargs,
    )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    if cbar:
        plt.colorbar(im, label=clabel)
    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")

    return ax


def plot_power_spectrum(
    power_spectrum: SphericalPS | CylindricalPS,*,
    ax: plt.Axes | list[plt.Axes] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    color: list | None = None,
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    logx: bool | None = False,
    logy: bool | None = False,
    logc: bool | None = False,
    cbar: bool | None = True,
    label: str | None = None,
    smooth: float | bool = False,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a power spectrum.

    Parameters
    ----------
    power_spectrum : np.ndarray
        Power spectrum array.
    k : un.Quantity, optional
        Wavemodes corresponding to the spherical power spectrum.
        Mandatory to plot a 1D power spectrum.
    kperp : un.Quantity, optional
        kperp wavemodes of the cylindrical power spectrum.
        Mandatory to plot a 2D power spectrum.
    kpar : un.Quantity, optional
        kpar wavemodes of the cylindrical power spectrum.
        Mandatory to plot a 2D power spectrum.
    ax : plt.Axes | list[plt.Axes], optional
        Axes object(s) to plot on. If None, new axes are created.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    clabel : str, optional
        Label for the colorbar.
    cmap : str, optional
        Colormap for the plot.
    colors : list, optional
        List of colors for each line in the plot.
    fontsize : float, optional
        Font size for the plot labels.
    vmin : float, optional
        Minimum value for the color scale.
    vmax : float, optional
        Maximum value for the color scale.
    log : list[bool], optional
        List of booleans to set the axes to log scale.
    labels : list, optional
        List of labels for each line in the plot.
    smooth : bool or float, optional
        Standard deviation for Gaussian smoothing.
        If True, uses a standard deviation of 1.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    """
    if isinstance(smooth, bool) and smooth:
        smooth = 1.0
    validate(power_spectrum)
    if isinstance(power_spectrum, SphericalPS):
        if legend_kwargs is None:
            legend_kwargs = {}
        if power_spectrum.ps.ndim > 1:
            raise ValueError("Plot one 1D PS at a time.")
        if ax is None:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=(7, 6), sharey=True, sharex=True
            )
        ax = plot_1d_power_spectrum(
            power_spectrum,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            fontsize=fontsize,
            log=[logx, logy],
            label=label,
            smooth=smooth,
            legend_kwargs=legend_kwargs,
        )
    elif isinstance(power_spectrum, CylindricalPS):
        if label is not None or legend_kwargs is not None:
            warnings.warn(
                "Cylindrical PS plots do not support labels and legends.", stacklevel=2
            )
        if ax is None:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=(7, 6), sharey=True, sharex=True
            )
            cbar = True
        else:
            fig = ax.get_figure()
            if len(fig.get_axes()) > 1:
                cbar = False

        ax = plot_2d_power_spectrum(
            power_spectrum,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            clabel=clabel,
            cmap=cmap,
            fontsize=fontsize,
            vmin=vmin,
            vmax=vmax,
            log=[logx, logy, logc],
            smooth=smooth,
            cbar=cbar,
        )
    else:
        raise ValueError("Input must be SphericalPS or CylindricalPS object instances.")
    return ax
