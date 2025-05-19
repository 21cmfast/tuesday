import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology.units import littleh
from matplotlib import rcParams


def plot_1d_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: list | None = None,
    fontsize: float | None = 16,
    save_path: str | None = None,
) -> None:
    """
    Plot a 1D power spectrum.

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
    colors : list, optional
        List of colors for each line in the plot.
    save_path : str, optional
        Path to save the plot. If None, the plot will be shown instead.
    """
    plt.figure(figsize=(8, 6))
    rcParams.update({"font.size": fontsize})
    N = power_spectrum.shape[0]
    if colors is None:
        colors = ["k"] * N
    if xlabel is None:
        if wavemodes.unit == 1 / un.Mpc:
            xlabel = r"$k \, [\rm{Mpc}^{-1}]$"
        elif wavemodes.unit == littleh / un.Mpc:
            xlabel = r"$k \, [h \, \rm{Mpc}^{-1}]$"
        else:
            raise ValueError("Wavemodes must be in units of 1/Mpc or h/Mpc.")
    if ylabel is None:
        if power_spectrum.unit == un.mK**2 * un.Mpc**3:
            ylabel = r"$P(k) \, [\rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2 * un.Mpc**3 / littleh**3:
            ylabel = r"$P(k) \, [h^{-3} \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2:
            ylabel = r"$\Delta^2_{21} [\rm{mK}^2]$"
        else:
            raise ValueError(
                "Power spectrum must be in units of mK^2 * Mpc^3 or mK^2 * Mpc^3 / h^3 or mK^2."
            )
    for i in range(N):
        plt.plot(wavemodes, power_spectrum[i], color=colors[i])
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if np.all(np.diff(wavemodes) != np.diff(wavemodes)[0]):
        plt.loglog()
    else:
        plt.yscale("log")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_2d_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot a 2D power spectrum.

    Parameters
    ----------
    wavemodes : np.ndarray
        Wavemodes corresponding to the power spectrum.
        wavemodes[0] is kperp, wavemodes[1] is kpar.
    power_spectrum : np.ndarray
        Power spectrum values.
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
    fontsize : float, optional
        Font size for the plot labels.
    vmin : float, optional
        Minimum value for the color scale.
    vmax : float, optional
        Maximum value for the color scale.
    save_path : str, optional
        Path to save the plot. If None, the plot will be shown instead.
    """
    plt.figure(figsize=(8, 6))
    rcParams.update({"font.size": fontsize})
    kperp = wavemodes[0]
    kpar = wavemodes[1]
    if xlabel is None:
        if kperp.unit == 1 / un.Mpc or kperp.unit == littleh / un.Mpc:
            xlabel = r"$k \, [h \, Mpc^{-1}]$"
        else:
            raise ValueError("kperp must be in units of 1/Mpc or h/Mpc.")

    if ylabel is None:
        if kpar.unit == 1 / un.Mpc or kpar.unit == littleh / un.Mpc:
            ylabel = r"$k \, [h \, Mpc^{-1}]$"
        else:
            raise ValueError("kpar must be in units of 1/Mpc or h/Mpc.")

    if clabel is None:
        if power_spectrum.unit == un.mK**2 * un.Mpc**3:
            clabel = r"$P(k) \, [\rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2 * un.Mpc**3 / littleh**3:
            clabel = r"$P(k) \, [h^{-3} \rm{Mpc}^{3}]$"
        elif power_spectrum.unit == un.mK**2:
            clabel = r"$\Delta^2_{21} [\rm{mK}^2]$"
        else:
            raise ValueError(
                "Power spectrum must be in units of mK^2 * Mpc^3 or mK^2 * Mpc^3 / h^3 or mK^2."
            )
    if vmin is None:
        vmin = np.percentile(power_spectrum, 5)
    if vmax is None:
        vmax = np.percentile(power_spectrum, 95)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.pcolormesh(kperp, kpar, power_spectrum.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=clabel)
    if np.all(np.diff(kperp) != np.diff(kperp)[0]):
        plt.xscale("log")
    if np.all(np.diff(kpar) != np.diff(kpar)[0]):
        plt.yscale("log")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_power_spectrum(
    wavemodes: un.Quantity,
    power_spectrum: un.Quantity,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str | None = "viridis",
    colors: list | None = None,
    fontsize: float | None = 16,
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
) -> None:
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
    save_path : str, optional
        Path to save the plot. If None, the plot will be shown instead.
    """
    if wavemodes.ndim == 1:
        plot_1d_power_spectrum(
            wavemodes,
            power_spectrum,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=colors,
            fontsize=fontsize,
            save_path=save_path,
        )
    elif wavemodes.ndim == 2:
        plot_2d_power_spectrum(
            wavemodes,
            power_spectrum,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            clabel=clabel,
            cmap=cmap,
            fontsize=fontsize,
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
        )
    else:
        raise ValueError("Wavemodes must be 1D or 2D arrays.")
