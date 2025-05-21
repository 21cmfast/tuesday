"""Plotting functions for 1D and 2D power spectra."""

import warnings

import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology.units import littleh
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


def plot_1d_power_spectrum(  # noqa: C901
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    fig: plt.Figure | None = None,
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
    wavemodes : un.Quantity
        Wavemodes corresponding to the power spectrum.
        Accepted units are 1/Mpc or h/Mpc.
    power_spectrum : un.Quantity
        Power spectrum array of shape [Nsamples, Nwavemodes] or [Nwavemodes].
        There are six accepted units: mK^2 Mpc^3, mK^2 Mpc^3/h^3, mK^2,
        or dimensionless instead of mK^2.
    fig : plt.Figure, optional
        Figure object to plot on. If None, a new figure is created.
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
    if color is None:
        color = "C0"
    if xlabel is None:
        if wavemodes.unit == 1 / un.Mpc:
            xlabel = r"$k \, [\rm{Mpc}^{-1}]$"
        elif wavemodes.unit == littleh / un.Mpc:
            xlabel = r"$k \, [h \, \rm{Mpc}^{-1}]$"
        else:
            raise ValueError("Wavemodes must be in units of 1/Mpc or h/Mpc.")
    if ylabel is None:
        if power_spectrum.unit == un.mK**2 * un.Mpc**3:
            ylabel = r"$P(k) \, [\rm{mK}^2 \, \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2 * un.Mpc**3 / littleh**3:
            ylabel = r"$P(k) \, [\rm{mK}^2 \, h^{-3} \, \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2:
            ylabel = r"$\Delta^2_{21} \, [\rm{mK}^2]$"
        elif power_spectrum.unit == un.Mpc**3:
            ylabel = r"$P(k) \, [\rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.Mpc**3 / littleh**3:
            ylabel = r"$P(k) \, [h^{-3} \, \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.dimensionless_unscaled:
            ylabel = r"$\Delta^2_{21}$"
        else:
            raise ValueError(
                "Accepted PS units: mK^2 Mpc^3, mK^2 Mpc^3/h^3, mK^2 or dimless."
            )

    if (isinstance(smooth, bool) and smooth) or (
        isinstance(smooth, float) and smooth > 0
    ):
        if isinstance(smooth, bool):
            smooth = 1.0
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
    return fig, ax


def plot_2d_power_spectrum(  # noqa: C901
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    fig: plt.Figure | None = None,
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
    wavemodes : un.Quantity
        Wavemodes [kperp, kpar].
        Accepted units are 1/Mpc or h/Mpc.
    power_spectrum : un.Quantity
        Power spectrum array of shape [Nsamples, Nkperp, Nkpar]
        or [Nkperp, Nkpar].
        There are six accepted units: mK^2 Mpc^3, mK^2 Mpc^3/h^3, mK^2,
        or dimensionless instead of mK^2.
    fig : plt.Figure, optional
        Figure object to plot on. If None, a new figure is created.
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
        If True, uses a standard deviation of 1.
    """
    rcParams.update({"font.size": fontsize})
    kperp = wavemodes[0]
    kpar = wavemodes[1]
    if xlabel is None:
        if kperp.unit == 1 / un.Mpc:
            xlabel = r"$k_\perp \, [\rm{Mpc}^{-1}]$"
        elif kperp.unit == littleh / un.Mpc:
            xlabel = r"$k_\perp \, [h \, \rm{Mpc}^{-1}]$"
        else:
            raise ValueError("kperp must be in units of 1/Mpc or h/Mpc.")

    if ylabel is None:
        if kpar.unit == 1 / un.Mpc:
            ylabel = r"$k_\parallel \, [\rm{Mpc}^{-1}]$"
        elif kpar.unit == littleh / un.Mpc:
            ylabel = r"$k_\parallel \, [h \, \rm{Mpc}^{-1}]$"
        else:
            raise ValueError("kpar must be in units of 1/Mpc or h/Mpc.")

    if clabel is None:
        if power_spectrum.unit == un.mK**2 * un.Mpc**3:
            clabel = r"$P(k) \, [\rm{mK}^2 \, \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2 * un.Mpc**3 / littleh**3:
            clabel = r"$P(k) \, [\rm{mK}^2 \, h^{-3} \, \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2:
            clabel = r"$\Delta^2_{21} \, [\rm{mK}^2]$"
        elif power_spectrum.unit == un.Mpc**3:
            clabel = r"$P(k) \, [\rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.Mpc**3 / littleh**3:
            clabel = r"$P(k) \, [h^{-3} \, \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.dimensionless_unscaled:
            clabel = r"$\Delta^2_{21}$"
        else:
            raise ValueError(
                "Accepted PS units: mK^2 Mpc^3, mK^2 Mpc^3/h^3, mK^2 or dimless."
            )
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
    if (isinstance(smooth, bool) and smooth) or (
        isinstance(smooth, float) and smooth > 0
    ):
        if isinstance(smooth, bool):
            smooth = 1.0
            unit = power_spectrum.unit
        power_spectrum = gaussian_filter(power_spectrum, sigma=smooth) * unit
    
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

    return fig, ax


def plot_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    fig: plt.Figure | None = None,
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
    log: list[bool] | None = False,
    label: list | None = None,
    smooth: float | bool = False,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the power spectrum.

    Parameters
    ----------
    wavemodes : np.ndarray
        Wavemodes corresponding to the power spectrum.
    power_spectrum : np.ndarray
        Power spectrum values.
    fig : plt.Figure, optional
        Figure object to plot on. If None, a new figure is created.
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
    power_spectrum = power_spectrum.squeeze()
    if isinstance(label, str):
        label = [label]
    if not np.iterable(log):
        log = [log]
    if (hasattr(wavemodes, "ndim") and wavemodes.ndim == 1) or len(wavemodes) == 1:
        if legend_kwargs is None:
            legend_kwargs = {}
        if len(log) == 1:
            log = [log[0], log[0]]
        if power_spectrum.ndim > 1:
            raise ValueError("Plot one 1D PS at a time.")
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if ax is None:
            ax = fig.get_axes()[0]
        fig, ax = plot_1d_power_spectrum(
            wavemodes,
            power_spectrum,
            fig=fig,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            fontsize=fontsize,
            log=log,
            label=label,
            smooth=smooth,
            legend_kwargs=legend_kwargs,
        )
    elif (hasattr(wavemodes, "ndim") and wavemodes.ndim == 2) or len(wavemodes) == 2:
        if label is not None or legend_kwargs is not None:
            warnings.warn(
                "Cylindrical PS plots do not support labels and legends.", stacklevel=2
            )
        if len(log) == 1:
            log = [log[0], log[0], True]
        if len(log) == 2:
            log = [log[0], log[1], True]
        if power_spectrum.ndim > 2:
            raise ValueError("Plot one 2D PS at a time.")
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots(
                    nrows=1, ncols=1, figsize=(7, 6), sharey=True, sharex=True
                )
                cbar = True
            else:
                fig = ax.get_figure()
                if len(fig.get_axes()) > 1:
                    cbar = False

        if ax is None:
            ax = fig.get_axes()[0]
            if len(fig.get_axes()) > 1:
                cbar = False
            else:
                cbar = True
        fig, ax = plot_2d_power_spectrum(
            wavemodes,
            power_spectrum,
            fig=fig,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            clabel=clabel,
            cmap=cmap,
            fontsize=fontsize,
            vmin=vmin,
            vmax=vmax,
            log=log,
            smooth=smooth,
            cbar=cbar,
        )
    else:
        raise ValueError("Wavemodes must be 1D or 2D arrays.")
    return fig, ax
