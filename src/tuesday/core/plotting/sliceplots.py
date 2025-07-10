"""Module for LC and coeval sliceplots."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from matplotlib import colormaps, colors, rcParams
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from ..units import validate

try:
    eor_colour = colors.LinearSegmentedColormap.from_list(
        "eor",
        [
            (0, "white"),
            (0.21, "yellow"),
            (0.42, "orange"),
            (0.63, "red"),
            (0.86, "black"),
            (0.9, "blue"),
            (1, "cyan"),
        ],
    )

    colormaps.register(cmap=eor_colour)
except ValueError:
    # If the colormap already exists, we can ignore this error.
    pass


def _plot_slice(
    img_slice: un.Quantity,
    xaxis: un.Quantity,
    yaxis: un.Quantity,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    fontsize: float | None = 16,
    log: tuple[bool, bool, bool] = (False, False, False),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot a 2D slice of the data."""
    validate(yaxis, "length")
    rcParams.update({"font.size": fontsize})
    if xaxis.unit.physical_type != "dimensionless":
        validate(xaxis, "length")
    if ax is None:
        _, ax = plt.subplots()
    cmap_kwargs = {}
    if vmin is None:
        if log[2]:
            cmap_kwargs["vmin"] = np.nanpercentile(np.log10(img_slice.value), 5)
        else:
            cmap_kwargs["vmin"] = np.nanpercentile(img_slice.value, 5)
    else:
        cmap_kwargs["vmin"] = vmin
    if vmax is None:
        if log[2]:
            cmap_kwargs["vmax"] = np.nanpercentile(np.log10(img_slice.value), 95)
        else:
            cmap_kwargs["vmax"] = np.nanpercentile(img_slice.value, 95)
    else:
        cmap_kwargs["vmax"] = vmax
    if log[2]:
        cmap_kwargs = {}
        cmap_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(
        xaxis.value,
        yaxis.value,
        img_slice.value.T,
        cmap=cmap,
        shading="auto",
        **cmap_kwargs,
    )

    if log[0]:
        ax.set_xscale("log")
    if log[1]:
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.colorbar(im, ax=ax, label=clabel)

    return ax


def lc2slice_x(
    zmin: float | None = None,
    zmax: float | None = None,
    idx: int | None = 0,
) -> un.Quantity:
    """Get the slice index for a given redshift range."""

    def slice_index(
        box: un.Quantity, redshift: np.ndarray | un.Quantity
    ) -> un.Quantity:
        """Get the slice index for a given redshift range."""
        idx_min = 0 if zmin is None else np.argmin(np.abs(redshift - zmin))
        idx_max = (
            box.shape[-1] if zmax is None else np.argmin(np.abs(redshift - zmax)) + 1
        )

        return box[idx, :, idx_min:idx_max]

    return slice_index


def lc2slice_y(
    zmin: float | None = None,
    zmax: float | None = None,
    idx: int | None = 0,
) -> un.Quantity:
    """Get the slice index for a given redshift range."""

    def slice_index(
        box: un.Quantity, redshift: np.ndarray | un.Quantity
    ) -> un.Quantity:
        """Get the slice index for a given redshift range."""
        idx_min = 0 if zmin is None else np.argmin(np.abs(redshift - zmin))
        idx_max = (
            box.shape[-1] if zmax is None else np.argmin(np.abs(redshift - zmax)) + 1
        )

        return box[:, idx, idx_min:idx_max]

    return slice_index


def coeval2slice_x(
    idx: int | None = 0,
) -> un.Quantity:
    """Slice the box along the x-axis."""

    def slice_index(box: un.Quantity) -> un.Quantity:
        """Slice the box along the x-axis."""
        return box[idx, :, :]

    return slice_index


def coeval2slice_y(
    idx: int | None = 0,
) -> un.Quantity:
    """Slice the box along the y-axis."""

    def slice_index(box: un.Quantity) -> un.Quantity:
        """Slice the box along the y-axis."""
        return box[:, idx, :]

    return slice_index


def coeval2slice_z(
    idx: int | None = 0,
) -> un.Quantity:
    """Slice the box along the z-axis."""

    def slice_index(box: un.Quantity) -> un.Quantity:
        """Slice the box along the z-axis."""
        return box[:, :, idx]

    return slice_index


def plot_redshift_slice(
    lightcone: un.Quantity,
    box_length: un.Quantity,
    redshift: np.ndarray | un.Quantity,
    *,
    fontsize: float | None = 16,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str = "eor",
    logx: bool = False,
    logy: bool = False,
    logc: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    smooth: bool | float = False,
    transform2slice: Callable | None = None,
) -> plt.Axes:
    """Plot a slice from a lightcone of shape (N_x, N_y, N_redshifts).

    Parameters
    ----------
    lightcone : un.Quantity
        The lightcone data to plot with shape (N_x, N_y, N_redshifts).
    box_length : un.Quantity
        The length of the box.
    redshift : np.ndarray | un.Quantity
        The redshift values corresponding to the lightcone.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    clabel : str, optional
        The label for the colorbar.
    cmap : str, optional
        The colormap to use for the plot.
    logx : bool, optional
        Whether to use a logarithmic scale for the x-axis.
    logy : bool, optional
        Whether to use a logarithmic scale for the y-axis.
    logc : bool, optional
        Whether to use a logarithmic scale for the colorbar.
    zmin : float, optional
        The minimum redshift of the lightcone.
    zmax : float, optional
        The maximum redshift of the lightcone.
    vmin : float, optional
        The minimum value for the color scale.
    vmax : float, optional
        The maximum value for the color scale.
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    smooth : bool | float, optional
        Whether to apply Gaussian smoothing to the lightcone data.
        If True, a default sigma of 1.0 will be used.
        If a float, it will be used as the sigma for the Gaussian filter.


    """
    validate(box_length, "length")
    rcParams.update({"font.size": fontsize})
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 4))
    if transform2slice is not None:
        lightcone = transform2slice(lightcone, redshift)
    else:
        lightcone = lc2slice_x(zmin=zmin, zmax=zmax, idx=0)(lightcone, redshift)
    if smooth:
        if isinstance(smooth, bool):
            smooth = 1.0
        lightcone = gaussian_filter(lightcone.value, sigma=smooth) * lightcone.unit
    yaxis = np.linspace(0, box_length, lightcone.shape[0])
    if not isinstance(redshift, un.Quantity):
        redshift = redshift * un.dimensionless_unscaled

    if clabel is None:
        if lightcone.unit.physical_type == un.get_physical_type("temperature"):
            clabel = "Brightness Temperature " + f" [{lightcone.unit:latex_inline}]"
        elif lightcone.unit.is_equivalent(un.dimensionless_unscaled):
            clabel = "Density Contrast"
        else:
            clabel = (
                f"{lightcone.unit.physical_type} " + f" [{lightcone.unit:latex_inline}]"
            )
    if vmin is None and vmax is None:
        if logc:
            vmin = np.nanpercentile(np.log10(lightcone.value), 5)
        else:
            vmin = np.nanpercentile(lightcone.value, 5)
        if cmap.lower() == "eor":
            vmax = -1.0 * vmin / 0.86 + vmin
    return _plot_slice(
        lightcone.T,
        redshift,
        yaxis,
        vmin=vmin,
        vmax=vmax,
        log=[logx, logy, logc],
        title=title,
        xlabel="Redshift" if xlabel is None else xlabel,
        ylabel=f"Distance [{box_length.unit:latex_inline}]"
        if ylabel is None
        else ylabel,
        clabel=clabel,
        cmap=cmap,
        ax=ax,
    )


def plot_coeval_slice(
    coeval: un.Quantity,
    box_length: un.Quantity,
    *,
    fontsize: float | None = 16,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    cmap: str = "viridis",
    logx: bool = False,
    logy: bool = False,
    logc: bool = False,
    idx: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    smooth: bool | float = False,
    transform2slice: Callable | None = None,
    v_x: un.Quantity | None = None,
    v_y: un.Quantity | None = None,
    quiver_label: str | bool = False,
    quiver_kwargs: dict | None = None,
    quiver_label_kwargs: dict | None = None,
    quiver_decimate_factor: int = 1,
) -> plt.Axes:
    """Plot a slice from a coeval of shape (Nx, Ny, N redshifts).

    Parameters
    ----------
    coeval : un.Quantity
        The coeval data cube with shape (Nx, Ny, N redshifts).
    box_length : un.Quantity
        The length of the box.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    clabel : str, optional
        The label for the colorbar.
    cmap : str, optional
        The colormap to use for the plot.
    logx : bool, optional
        Whether to use a logarithmic scale for the x-axis.
    logy : bool, optional
        Whether to use a logarithmic scale for the y-axis.
    logc : bool, optional
        Whether to use a logarithmic scale for the colorbar.
    idx : int, optional
        The index of the slice to plot along the z-axis.
        Default is 0.
    vmin : float, optional
        The minimum value for the color scale.
    vmax : float, optional
        The maximum value for the color scale.
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    smooth : bool | float, optional
        Whether to apply Gaussian smoothing to the coeval data.
        If True, a default sigma of 1.0 will be used.
        If a float, it will be used as the sigma for the Gaussian filter.
    transform2slice : Callable, optional
        A function to transform the coeval data into a slice.
        If None, the default slicing function will be used.
    v_x : un.Quantity, optional
        The x-component of the velocity field to
        plot as a vector field on top of the slice plot.
        This is a 2D array with shape (Nx, Ny).
    v_y : un.Quantity, optional
        The y-component of the velocity field to
        plot as a vector field on top of the slice plot.
        This is a 2D array with shape (Nx, Ny).
    quiver_label : str | bool, optional
        The label for the quiver plot that appears on the
        top right corner right outside of the plot area.
        If True, a default label will be put,
        assuming the velocity is being plotted.
        If False, no label will be added.
    quiver_kwargs : dict, optional
        Additional keyword arguments for the quiver plot,
        such as arrow color, width, etc.
        See `matplotlib.pyplot.quiver` for more details.
    quiver_label_kwargs : dict, optional
        Additional keyword arguments for the quiver label,
        such as color, angle, etc.
        See `matplotlib.pyplot.quiverkey` for more details.
    quiver_decimate_factor : int, optional
        The factor by which to decimate the vector field for plotting.
        This is useful for reducing the number of arrows in the quiver plot
        to avoid cluttering the plot. Default is 1 (no decimation).

    Returns
    -------
    plt.Axes
        The axes with the coeval slice plot.
    """
    validate(box_length, "length")
    rcParams.update({"font.size": fontsize})
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if transform2slice is not None:
        coeval = transform2slice(coeval)
    else:
        coeval = coeval2slice_z(idx=idx)(coeval)
    if smooth:
        if isinstance(smooth, bool):
            smooth = 1.0
        coeval = gaussian_filter(coeval.value, sigma=smooth) * coeval.unit
    xaxis = np.linspace(0, box_length, coeval.shape[0])
    yaxis = np.linspace(0, box_length, coeval.shape[1])

    if clabel is None:
        if coeval.unit.physical_type == un.get_physical_type("temperature"):
            clabel = "Brightness Temperature " + f" [{coeval.unit:latex_inline}]"
        elif coeval.unit.is_equivalent(un.dimensionless_unscaled):
            clabel = "Density Contrast"
        else:
            clabel = f"{coeval.unit.physical_type} " + f" [{coeval.unit:latex_inline}]"
    ax = _plot_slice(
        coeval,
        xaxis,
        yaxis,
        vmin=vmin,
        vmax=vmax,
        log=[logx, logy, logc],
        title=title,
        xlabel=f"Distance [{box_length.unit:latex_inline}]"
        if xlabel is None
        else xlabel,
        ylabel=f"Distance [{box_length.unit:latex_inline}]"
        if ylabel is None
        else ylabel,
        clabel=clabel,
        cmap=cmap,
        ax=ax,
    )
    if v_x is not None and v_y is not None:
        if quiver_kwargs is None:
            quiver_kwargs = {
                "color": "k",
                "width": 0.006,
                "headwidth": 4,
            }

        if quiver_label:
            quiver_label = "Velocity " + f"[{v_x.unit:latex_inline}]"
        if quiver_label_kwargs is None:
            quiver_label_kwargs = {
                "labelpos": "E",
                "coordinates": "figure",
            }

        axq = ax.quiver(
            quiver_kwargs.pop("X", xaxis.value[::quiver_decimate_factor]),
            quiver_kwargs.pop("Y", yaxis.value[::quiver_decimate_factor]),
            quiver_kwargs.pop(
                "U", v_x.value[::quiver_decimate_factor, ::quiver_decimate_factor]
            ),
            quiver_kwargs.pop(
                "V", v_y.value[::quiver_decimate_factor, ::quiver_decimate_factor]
            ),
            **quiver_kwargs,
        )
        if isinstance(quiver_label, str):
            ax.quiverkey(
                axq,
                quiver_label_kwargs.pop("X", 0.9),
                quiver_label_kwargs.pop("Y", 0.9),
                quiver_label_kwargs.pop("U", 1.0),
                quiver_label,
                **quiver_label_kwargs,
            )
    return ax


def plot_pdf(
    box: un.Quantity,
    *,
    fontsize: float | None = 16,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    logx: bool = False,
    ax: plt.Axes | None = None,
    smooth: bool | float = False,
    hist_kwargs,
) -> plt.Axes:
    """Plot a pxiel distribution function (PDF) of the box.

    Parameters
    ----------
    box : un.Quantity
        The box data to plot.
    fontsize : float, optional
        The font size for the plot.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    logx : bool, optional
        Whether to use a logarithmic scale for the x-axis.
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    smooth : bool | float, optional
        Whether to apply Gaussian smoothing to the box data.
        If True, a default sigma of 1.0 will be used.
        If a float, it will be used as the sigma for the Gaussian filter.

    Returns
    -------
    plt.Axes
        The axes with the PDF plot.


    """
    rcParams.update({"font.size": fontsize})

    if smooth:
        if isinstance(smooth, bool):
            smooth = 1.0
        box = gaussian_filter(box.value, sigma=smooth) * box.unit
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.hist(
        box.value.flatten(),
        **hist_kwargs,
    )
    if xlabel is None:
        if box.unit.physical_type == un.get_physical_type("temperature"):
            xlabel = "Brightness Temperature " + f" [{box.unit:latex_inline}]"
        elif box.unit.is_equivalent(un.dimensionless_unscaled):
            xlabel = "Density Contrast"
        else:
            xlabel = f"{box.unit.physical_type} " + f" [{box.unit:latex_inline}]"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts" if ylabel is None else ylabel)
    if title is not None:
        ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    return ax
