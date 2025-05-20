"""Plotting functions for 1D and 2D power spectra."""

import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology.units import littleh
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter


def plot_1d_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    fig: plt.Figure | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: list | None = None,
    log: list[bool] = None,
    fontsize: float | None = 16,
    labels: list | None = None,
    smooth: float | bool = False,
    leg_kwargs: dict = {},
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 1D power spectrum.

    Parameters
    ----------
    wavemodes : np.ndarray
        Wavemodes corresponding to the power spectrum.
    power_spectrum : np.ndarray
        Power spectrum values.
    fig : plt.Figure, optional
        Figure object to plot on. If None, a new figure is created.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    colors : list, optional
        List of colors for each line in the plot.
    log : list[bool], optional
        List of booleans to set the x and y axes to log scale.
    fontsize : float, optional
        Font size for the plot labels.
    labels : list, optional
        List of labels for each line in the plot.
    smooth : float, optional
        Standard deviation for Gaussian smoothing. If True, uses a standard deviation of 1.
    leg_kwargs : dict, optional
        Keyword arguments for the legend.

    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = fig.get_axes()[0]
    rcParams.update({"font.size": fontsize})
    if log is None:
        log = [True, True]
    if power_spectrum.ndim == 1:
        power_spectrum = np.expand_dims(power_spectrum, axis=0)
    n = power_spectrum.shape[0]
    if colors is None:
        colors = [f"C{i}" for i in range(n)]
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
        else:
            raise ValueError(
                "Accepted power spectrum units: mK^2 Mpc^3, mK^2 Mpc^3/h^3 or mK^2."
            )
    if (isinstance(smooth, bool) and smooth) or (
        isinstance(smooth, float) and smooth > 0
    ):
        if isinstance(smooth, bool):
            smooth = 1.0
        power_spectrum = gaussian_filter(power_spectrum, sigma=smooth)
    for i in range(n):
        ax.plot(
            wavemodes,
            power_spectrum[i],
            color=colors[i],
            label=labels[i] if labels is not None else None,
        )
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")
    if labels is not None:
        ax.legend(**leg_kwargs)
    return fig, ax


def plot_2d_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    fig: plt.Figure | None = None,
    title: list[str] | str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    log: list[bool] = None,
    labels: list | str | None = None,
    smooth: float | bool = False,
    leg_kwargs: dict = {},
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D power spectrum.

    Parameters
    ----------
    wavemodes : np.ndarray
        Wavemodes corresponding to the power spectrum.
        wavemodes[0] is kperp, wavemodes[1] is kpar.
    power_spectrum : np.ndarray
        Power spectrum values.
    fig : plt.Figure, optional
        Figure object to plot on. If None, a new figure is created.
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
    labels : list, optional
        Label for the plot legend.
    smooth : float, optional
        Standard deviation for Gaussian smoothing. If True, uses a standard deviation of 1.
    leg_kwargs : dict, optional
        Keyword arguments for the legend.
    """
    if power_spectrum.ndim == 2:
        power_spectrum = np.expand_dims(power_spectrum, axis=0)
    n = power_spectrum.shape[0]
    if fig is None:
        fig, axs = plt.subplots(
            nrows=1, ncols=n, figsize=(7 * n, 6), sharey=True, sharex=True
        )
        if n == 1:
            axs = [axs]
    else:
        axs = fig.get_axes()
    if isinstance(labels, str):
        labels = [labels] * n
    rcParams.update({"font.size": fontsize})
    if log is None:
        log = [True, True, False]
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
        else:
            raise ValueError(
                "Accepted power spectrum units: mK^2 Mpc^3, mK^2 Mpc^3/h^3 or mK^2."
            )
    if log[2]:
        power_spectrum = np.log10(power_spectrum)
        clabel = r"$\log_{10}$ " + clabel
    if vmin is None:
        vmin = np.percentile(power_spectrum.value, 5)
    if vmax is None:
        vmax = np.percentile(power_spectrum.value, 95)
    if title is not None and isinstance(title, str):
        axs[0].set_title(title, fontsize=fontsize)
    axs[0].set_ylabel(ylabel, fontsize=fontsize)
    if (isinstance(smooth, bool) and smooth) or (
        isinstance(smooth, float) and smooth > 0
    ):
        if isinstance(smooth, bool):
            smooth = 1.0
            unit = power_spectrum.unit
        power_spectrum = gaussian_filter(power_spectrum, sigma=smooth) * unit
    for i in range(n):
        im = axs[i].pcolormesh(
            kperp.value,
            kpar.value,
            power_spectrum[i].value.T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label=labels[i] if labels is not None else None,
        )
        if title is not None and isinstance(title, list):
            axs[i].set_title(title[i], fontsize=fontsize)
        axs[i].set_xlabel(xlabel, fontsize=fontsize)
    plt.colorbar(im, label=clabel)
    if log[0]:
        axs[0].set_xscale("log")
    if log[1]:
        axs[0].set_yscale("log")
    if labels is not None:
        axs[0].legend(**leg_kwargs)
    return fig, axs


def plot_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    fig: plt.Figure | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    colors: list | None = None,
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    log: list[bool] = None,
    labels: list | None = None,
    smooth: float | bool = False,
    leg_kwargs: dict = {},
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the power spectrum.

    Parameters
    ----------
    wavemodes : np.ndarray
        Wavemodes corresponding to the power spectrum.
    power_spectrum : np.ndarray
        Power spectrum values.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    """
    if (hasattr(wavemodes, "ndim") and wavemodes.ndim == 1) or len(wavemodes) == 1:
        fig, ax = plot_1d_power_spectrum(
            wavemodes,
            power_spectrum,
            fig=fig,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=colors,
            fontsize=fontsize,
            log=log,
            labels=labels,
            smooth=smooth,
            leg_kwargs=leg_kwargs,
        )
    elif (hasattr(wavemodes, "ndim") and wavemodes.ndim == 2) or len(wavemodes) == 2:
        fig, ax = plot_2d_power_spectrum(
            wavemodes,
            power_spectrum,
            fig=fig,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            clabel=clabel,
            cmap=cmap,
            fontsize=fontsize,
            vmin=vmin,
            vmax=vmax,
            log=log,
            labels=labels,
            smooth=smooth,
            leg_kwargs=leg_kwargs,
        )
    else:
        raise ValueError("Wavemodes must be 1D or 2D arrays.")
    return fig, ax
