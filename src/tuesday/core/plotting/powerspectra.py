"""Plotting functions for 1D and 2D power spectra."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from ..summaries import CylindricalPS, SphericalPS


def plot_1d_power_spectrum_k(
    power_spectrum: SphericalPS,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: list | None = None,
    log: list[bool] | None = False,
    fontsize: float | None = 16,
    legend: str | None = None,
    smooth: float | bool = False,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D power spectrum vs wave mode.

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
    legend : str, optional
        Legend label for the PS.
    smooth : float, optional
        Standard deviation for Gaussian smoothing.
        If True, uses a standard deviation of 1.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    """
    if not isinstance(power_spectrum, SphericalPS):
        raise ValueError(
            "power_spectrum must be a SphericalPS object,"
            f" got {type(power_spectrum)} instead."
        )
    rcParams.update({"font.size": fontsize})
    wavemodes = power_spectrum.kcenters
    is_deltasq = power_spectrum.is_deltasq
    power_spectrum = power_spectrum.ps

    if color is None:
        color = "C0"
    if xlabel is None:
        xlabel = f"k [{wavemodes.unit:latex_inline}]"

    if ylabel is None:
        ylabel = f"[{power_spectrum.unit:latex_inline}]"
        ylabel = r"$\Delta^2_{21} \,$" + ylabel if is_deltasq else r"$P(k) \,$" + ylabel
    if smooth:
        power_spectrum = gaussian_filter(power_spectrum, sigma=smooth)
    ax.plot(wavemodes, power_spectrum, color=color, label=legend)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")
    if legend is not None:
        ax.legend(**legend_kwargs)
    return ax


def plot_1d_power_spectrum_z(
    power_spectra: list[SphericalPS],
    at_k: float,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = "Redshift",
    ylabel: str | None = None,
    color: list | None = "C0",
    log: list[bool] | None = False,
    fontsize: float | None = 16,
    legend: str | None = None,
    smooth: float | bool = False,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1D power spectra as a function of redshift at a given scale.

    Parameters
    ----------
    power_spectrum : list[SphericalPS]
        List of instances of the SphericalPS class.
    at_k : float
        If provided, plots the 1D power spectrum at a specific k value.
        The k value is assumed to be in the same unit
        as the k in the SphericalPS instance wavemodes.
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
    legend : str, optional
        Legend label for the PS.
    smooth : float, optional
        Standard deviation for Gaussian smoothing.
        If True, uses a standard deviation of 1.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    """
    for i in range(len(power_spectra)):
        if not isinstance(power_spectra[i], SphericalPS):
            raise ValueError(
                "power_spectrum must be a SphericalPS object or a list of "
                "SphericalPS objects,"
                f" got {type(power_spectra[i])} instead."
            )

    rcParams.update({"font.size": fontsize})

    is_deltasq = power_spectra[0].is_deltasq

    xaxis = [ps.redshift for ps in power_spectra]

    kbins = np.abs(power_spectra[0].kcenters.value - at_k)
    kbins[np.isnan(kbins)] = np.inf  # Avoid NaNs
    at_k = np.argmin(kbins)

    if ylabel is None:
        ylabel = f"[{power_spectra[0].ps.unit:latex_inline}]"
        ylabel = r"$\Delta^2_{21} \,$" + ylabel if is_deltasq else r"$P(k) \,$" + ylabel
    psvals = []
    for power_spectrum in power_spectra:
        if smooth:
            ps = gaussian_filter(power_spectrum.ps, sigma=smooth)
        else:
            ps = power_spectrum.ps.value
        psvals.append(ps[at_k])
    ax.plot(xaxis, psvals, color=color, label=legend)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")
    if legend is not None:
        ax.legend(**legend_kwargs)
    return ax


def plot_2d_power_spectrum(
    power_spectrum: CylindricalPS,
    *,
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
    if not isinstance(power_spectrum, CylindricalPS):
        raise ValueError(
            "power_spectrum must be a CylindricalPS object,"
            f" got {type(power_spectrum)} instead."
        )
    rcParams.update({"font.size": fontsize})
    kperp = power_spectrum.kperp
    kpar = power_spectrum.kpar
    is_deltasq = power_spectrum.is_deltasq
    power_spectrum = power_spectrum.ps

    if xlabel is None:
        xlabel = r"k$_\perp \,$" + f"[{kperp.unit:latex_inline}]"

    if ylabel is None:
        ylabel = r"k$_\parallel \,$" + f"[{kpar.unit:latex_inline}]"

    if clabel is None:
        clabel = f"[{power_spectrum.unit:latex_inline}]"
        clabel = r"$\Delta^2_{21} \,$" + clabel if is_deltasq else r"$P(k) \,$" + clabel
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
    power_spectrum: SphericalPS | CylindricalPS | list[SphericalPS],
    *,
    ax: plt.Axes | list[plt.Axes] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    at_k: float | int | None = None,
    cmap: str | None = "viridis",
    color: list | None = None,
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    logx: bool | None = False,
    logy: bool | None = False,
    logc: bool | None = False,
    cbar: bool | None = True,
    legend: str | None = None,
    smooth: float | bool = False,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a power spectrum.

    Parameters
    ----------
    power_spectrum : CylindricalPS | SphericalPS | list[SphericalPS]
        Instance of the CylindricalPS class, or instance or
        list of instances of the or SphericalPS class.
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
    at_k : float | int, optional
        If provided, plots the 1D power spectrum at a specific k value.
        If int, it is interpreted as the index of the k value.
        If float, it is interpreted as the k value itself
        in the same unit as the k in the SphericalPS instance wavemodes.
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
    logx : bool, optional
        Whether to set the x-axis to log scale.
    logy : bool, optional
        Whether to set the y-axis to log scale.
    logc : bool, optional
        Whether to set the color-axis to log scale.
    legend : str, optional
        Legend label for the 1D PS.
    smooth : bool or float, optional
        Standard deviation for Gaussian smoothing.
        If True, uses a standard deviation of 1.
    legend_kwargs : dict, optional
        Keyword arguments for the legend on the 1D PS plot.
    """
    if isinstance(smooth, bool) and smooth:
        smooth = 1.0
    if isinstance(power_spectrum, SphericalPS):
        if legend_kwargs is None:
            legend_kwargs = {}
        if ax is None:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=(7, 6), sharey=True, sharex=True
            )
        ax = plot_1d_power_spectrum_k(
            power_spectrum,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            fontsize=fontsize,
            log=[logx, logy],
            legend=legend,
            smooth=smooth,
            legend_kwargs=legend_kwargs,
        )
    elif hasattr(power_spectrum, "__len__") and np.all(
        [isinstance(ps, SphericalPS) for ps in power_spectrum]
    ):
        if legend_kwargs is None:
            legend_kwargs = {}
        ax = plot_1d_power_spectrum_z(
            power_spectrum,
            at_k,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            fontsize=fontsize,
            log=[logx, logy],
            legend=legend,
            smooth=smooth,
            legend_kwargs=legend_kwargs,
        )
    elif isinstance(power_spectrum, CylindricalPS):
        if legend is not None or legend_kwargs is not None:
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
        raise ValueError(
            "Input must be SphericalPS or CylindricalPS objects,"
            f"got {type(power_spectrum)} instead."
        )
    return ax
