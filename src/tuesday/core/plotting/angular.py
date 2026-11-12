"""Plots of angular lightcones.

An angular lightcone holds a field along a set of directions on the sky, one column per
comoving distance, so it has shape ``(N_pixels, N_slices)``.
:func:`plot_angular_lightcone_slice` shows such a lightcone along a path on the sky,
e.g. an arc of the equator, as a wedge of its true shape, and :func:`plot_healpix_map`
shows one slice of a lightcone whose directions are the pixels of a HEALPix grid, for
the full sky or a patch. By default both use the colour map of 21cmFAST's lightcone
plots (:func:`eor_colormap`); with ``dark_mode=True`` they also have its dark
background.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from astropy_healpix import HEALPix
from matplotlib import rcParams
from matplotlib.colors import Colormap, LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MaxNLocator

from ..lightcones.healpix import _check_order, _check_pixels, pixel_side
from .style import with_dark_mode

#: The fraction of tuesday's "eor" colour map at which it is black.
_EOR_BLACK = 0.86


def _check_quantity(values: un.Quantity, name: str) -> None:
    if not isinstance(values, un.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity")


def _colour_label(unit: un.UnitBase) -> str:
    """Return the default label of the colour bar."""
    if unit.physical_type == un.get_physical_type("temperature"):
        return rf"$T_{{\rm B}}$ [{unit:latex_inline}]"
    if unit.is_equivalent(un.dimensionless_unscaled):
        return "Density Contrast"
    return f"{unit.physical_type}  [{unit:latex_inline}]"


def eor_colormap(vmin: float, vmax: float) -> LinearSegmentedColormap:
    """Return the colour map of 21cmFAST's lightcone plots, for limits vmin and vmax.

    Zero is black. Negative values (absorption) go from maroon through orange and
    yellow to white at ``vmin``, and positive values (emission) from blue to sky blue
    at ``vmax``, as in the angular lightcone plots of 21cmFAST's lightcone tutorial.

    Parameters
    ----------
    vmin : float
        The lowest value of the colour scale.
    vmax : float
        The highest value of the colour scale.

    Returns
    -------
    LinearSegmentedColormap
        The colour map, to use with ``vmin`` and ``vmax`` as the limits.
    """
    if not vmin < vmax:
        raise ValueError("vmin must be less than vmax")
    zero = min(max(-vmin / (vmax - vmin), 0.0), 1.0)
    negative = ["white", "yellow", "orange", "maroon"]
    positive = ["#1f77b4", "skyblue"]
    colours = [(zero * i / 4, c) for i, c in enumerate(negative)] if zero > 0 else []
    colours.append((zero, "black"))
    if zero < 1:
        colours += [
            (zero + (1 - zero) * (i + 1) / 2, c) for i, c in enumerate(positive)
        ]
    return LinearSegmentedColormap.from_list(f"EoR-{vmin:g}-{vmax:g}", colours)


def _colour_scale(
    values: np.ndarray,
    *,
    cmap: str | Colormap | None,
    logc: bool,
    vmin: float | None,
    vmax: float | None,
) -> dict:
    """Return the keywords that set the colours of ``pcolormesh``.

    With ``cmap=None``, the colour map is :func:`eor_colormap` and the limits that are
    not given are the lowest and highest finite values, extended to zero if they do
    not include it. With a named colour map, they
    are the 5th and 95th percentiles of the finite values (the positive ones for a
    logarithmic scale), and with the "eor" map, no limits given and a negative lower
    limit, the upper limit puts zero at the black of the map, as for rectilinear
    lightcones.
    """
    finite = values[np.isfinite(values)]
    if logc:
        finite = finite[finite > 0]
    if finite.size == 0:
        raise ValueError("there are no values to plot")
    if cmap is None:
        # The limits include zero, which is black.
        low, high = min(float(finite.min()), 0.0), max(float(finite.max()), 0.0)
        if vmin is None and vmax is None and low == high:
            low, high = low - 0.5, high + 0.5
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
        if logc:
            return {"cmap": "viridis", "norm": LogNorm(vmin=vmin, vmax=vmax)}
        return {"cmap": eor_colormap(vmin, vmax), "vmin": vmin, "vmax": vmax}
    eor_limits = (
        vmin is None
        and vmax is None
        and isinstance(cmap, str)
        and cmap.lower() == "eor"
    )
    if vmin is None:
        vmin = float(np.percentile(finite, 5))
    if vmax is None:
        vmax = (
            vmin - vmin / _EOR_BLACK
            if eor_limits and vmin < 0 and not logc
            else float(np.percentile(finite, 95))
        )
    if logc:
        return {"cmap": cmap, "norm": LogNorm(vmin=vmin, vmax=vmax)}
    return {"cmap": cmap, "vmin": vmin, "vmax": vmax}


@contextmanager
def _plot_context(fontsize: float | None) -> Iterator[None]:
    """Apply the font size of one plot, and restore it afterwards.

    A backend that starts while plotting (e.g. Jupyter's inline backend, at the first
    figure of a session) turns interactive mode on; it stays on afterwards, so that
    later figures are shown.
    """
    if fontsize is None:
        yield
        return
    with plt.rc_context({"font.size": fontsize}):
        try:
            yield
        finally:
            interactive = rcParams["interactive"]
    rcParams["interactive"] = interactive


def _edges(centres: np.ndarray) -> np.ndarray:
    """Return the edges of cells with the given (increasing) centres."""
    if centres.size == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate(
        [[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]]
    )


@with_dark_mode
def plot_angular_lightcone_slice(
    lightcone: un.Quantity,
    angle: un.Quantity,
    distance: un.Quantity,
    *,
    redshift: np.ndarray | un.Quantity | None = None,
    box_length: un.Quantity | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    grid: bool = True,
    fontsize: float | None = 16,
    title: str | None = None,
    rlabel: str | None = None,
    clabel: str | None = None,
    cmap: str | Colormap | None = None,
    logc: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot an angular lightcone along a path on the sky, with its shape.

    The lines of sight follow a path on the sky, such as an arc of the equator, and
    ``angle`` is their angle along it. Each value is drawn at its polar angle and
    comoving distance from the observer, so the plot shows the lightcone as a wedge
    of its true shape, as the angular lightcone plots of 21cmFAST's lightcone
    tutorial do. By default it also has their colour map (:func:`eor_colormap`), with
    zero in black, and with ``dark_mode=True`` their dark background.
    The distances are labelled with their redshifts if these are given.

    Parameters
    ----------
    lightcone : un.Quantity
        The angular lightcone, of shape (N_pixels, N_slices).
    angle : un.Quantity
        The angle of each line of sight along its path, in increasing order and
        within 360 degrees, e.g. the longitude of directions on the equator.
    distance : un.Quantity
        The comoving distance of each slice, in increasing order.
    redshift : np.ndarray | un.Quantity, optional
        The redshift of each slice. With it, the distances are labelled with their
        redshifts, and ``zmin`` and ``zmax`` can be given.
    box_length : un.Quantity, optional
        The side of the simulation box. With it, the grid marks the boxes that tile
        the lightcone: radial lines one box apart at the nearest distance, and arcs
        at multiples of the box length.
    zmin : float, optional
        The lowest redshift to plot; the slice nearest to it is the first one plotted.
    zmax : float, optional
        The highest redshift to plot; the slice nearest to it is the last one plotted.
    grid : bool, optional
        Whether to draw the grid lines. Their labels are drawn either way.
    fontsize : float, optional
        The font size of the plot. None uses the current font size.
    title : str, optional
        The title of the plot.
    rlabel : str, optional
        The label of the distances, "Redshift" or the comoving distance by default.
    clabel : str, optional
        The label for the colour bar.
    cmap : str | Colormap, optional
        The colour map. By default, :func:`eor_colormap` for the limits.
    logc : bool, optional
        Whether to use a logarithmic colour scale.
    vmin : float, optional
        The lowest value of the colour scale. By default, the lowest value or zero,
        whichever is lower, or the 5th percentile with a named colour map.
    vmax : float, optional
        The highest value of the colour scale. By default, the highest value or zero,
        whichever is higher, or the 95th percentile with a named colour map.
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    dark_mode : bool, optional
        Whether to draw the plot in dark mode (matplotlib's "dark_background"
        style), as 21cmFAST's lightcone plots are. By default, as set by
        :func:`~tuesday.core.plotting.set_dark_mode`, which is off until it is set.

    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    _check_quantity(lightcone, "lightcone")
    _check_quantity(angle, "angle")
    _check_quantity(distance, "distance")
    if lightcone.ndim != 2:
        raise ValueError("lightcone must have shape (N_pixels, N_slices)")
    theta = angle.to_value(un.rad)
    if (
        theta.shape != lightcone.shape[:1]
        or np.any(np.diff(theta) <= 0)
        or theta[-1] - theta[0] > 2 * np.pi
    ):
        raise ValueError(
            "angle must have one increasing value per line of sight, within 360 degrees"
        )
    radius = distance.to_value(un.Mpc)
    if radius.shape != lightcone.shape[1:] or np.any(np.diff(radius) <= 0):
        raise ValueError("distance must have one increasing value per slice")
    if redshift is not None:
        redshift = np.asarray(un.Quantity(redshift, un.dimensionless_unscaled).value)
        if redshift.shape != radius.shape:
            raise ValueError("redshift must have one value per slice")
    elif zmin is not None or zmax is not None:
        raise ValueError("zmin and zmax need the redshift of each slice")

    first = 0 if zmin is None else int(np.argmin(np.abs(redshift - zmin)))
    last = radius.size - 1 if zmax is None else int(np.argmin(np.abs(redshift - zmax)))
    lightcone = lightcone[:, first : last + 1]
    radius = radius[first : last + 1]
    if redshift is not None:
        redshift = redshift[first : last + 1]

    # The wedge: the edges of the cells in angle and distance, in the plane.
    t_edges, r_edges = _edges(theta), _edges(radius)
    t_mesh, r_mesh = np.meshgrid(t_edges, r_edges)

    # Grid lines.
    if box_length is not None:
        box = box_length.to_value(un.Mpc)
        step = box / radius[0]
        t_grid = np.arange(np.ceil(theta[0] / step), np.floor(theta[-1] / step) + 1)
        t_grid = t_grid * step
        r_grid = np.arange(np.ceil(radius[0] / box), np.floor(radius[-1] / box) + 1)
        r_grid = r_grid * box
        # Angles one box apart are not round; label them to a tenth of the step.
        decimals = max(0, int(np.ceil(-np.log10(np.degrees(step) / 10))))
        t_format = f"{{:.{decimals}f}}°"
    else:
        t_grid = np.radians(
            MaxNLocator(nbins=4).tick_values(
                np.degrees(theta[0]), np.degrees(theta[-1])
            )
        )
        r_grid = MaxNLocator(nbins=4).tick_values(radius[0], radius[-1])
        t_format = "{:g}°"
    t_grid = t_grid[(t_grid >= t_edges[0]) & (t_grid <= t_edges[-1])]
    r_grid = r_grid[(r_grid >= r_edges[0]) & (r_grid <= r_edges[-1])]
    if redshift is not None:
        r_labels = [f"{z:.1f}" for z in np.interp(r_grid, radius, redshift)]
    else:
        r_labels = [f"{r:.0f}" for r in r_grid]
    if rlabel is None:
        rlabel = "Redshift" if redshift is not None else "Comoving distance [Mpc]"

    with _plot_context(fontsize):
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 10), layout="constrained")
        colours = _colour_scale(
            lightcone.value, cmap=cmap, logc=logc, vmin=vmin, vmax=vmax
        )
        im = ax.pcolormesh(
            r_mesh * np.cos(t_mesh),
            r_mesh * np.sin(t_mesh),
            lightcone.value.T,
            shading="flat",
            rasterized=True,
            **colours,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()

        line = rcParams["axes.edgecolor"]
        arc = np.linspace(t_edges[0], t_edges[-1], 361)
        if grid:
            for t in t_grid:
                ax.plot(
                    r_edges[[0, -1]] * np.cos(t),
                    r_edges[[0, -1]] * np.sin(t),
                    color=line,
                    lw=0.8,
                )
            for r in r_grid:
                ax.plot(r * np.cos(arc), r * np.sin(arc), color=line, lw=0.8)
        outline_t = np.concatenate([arc, arc[::-1], arc[:1]])
        outline_r = np.concatenate(
            [np.full(arc.size, r_edges[-1]), np.full(arc.size, r_edges[0]), r_edges[:1]]
        )
        ax.plot(
            outline_r * np.cos(outline_t), outline_r * np.sin(outline_t), color=line
        )

        # Labels of the distances below the first edge, and of the angles beyond the
        # far arc, offset in points along the outward directions.
        below = np.array([np.sin(t_edges[0]), -np.cos(t_edges[0])])
        rotation = np.degrees(t_edges[0])
        for r, label in zip(r_grid, r_labels, strict=True):
            ax.annotate(
                label,
                (r * np.cos(t_edges[0]), r * np.sin(t_edges[0])),
                xytext=tuple(14 * below),
                textcoords="offset points",
                ha="center",
                va="center",
                rotation=rotation,
            )
        middle = 0.5 * (r_edges[0] + r_edges[-1])
        ax.annotate(
            rlabel,
            (middle * np.cos(t_edges[0]), middle * np.sin(t_edges[0])),
            xytext=tuple(40 * below),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=rotation,
        )
        for t in t_grid:
            ax.annotate(
                t_format.format(np.degrees(t)),
                (r_edges[-1] * np.cos(t), r_edges[-1] * np.sin(t)),
                xytext=(18 * np.cos(t), 18 * np.sin(t)),
                textcoords="offset points",
                ha="center",
                va="center",
            )
        if title is not None:
            ax.set_title(title)
        plt.colorbar(
            im,
            ax=ax,
            label=_colour_label(lightcone.unit) if clabel is None else clabel,
        )
    return ax


@with_dark_mode
def plot_healpix_map(
    values: un.Quantity,
    nside: int,
    *,
    pixels: Sequence[int] | np.ndarray | None = None,
    order: Literal["ring", "nested"] = "ring",
    projection: Literal["mollweide", "cartesian"] | None = None,
    n_grid: int = 720,
    fontsize: float | None = 16,
    title: str | None = None,
    clabel: str | None = None,
    cmap: str | Colormap | None = None,
    logc: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a HEALPix map, e.g. one slice of an angular lightcone on a HEALPix grid.

    The map is drawn on a grid of longitudes and latitudes, each point taking the value
    of the pixel that contains it, so the pixels keep their shapes. The longitude
    increases to the right. Points outside the given pixels are left blank. In the
    Mollweide projection, the grid lines are 30 degrees apart in longitude and 15
    degrees in latitude, and only the latitudes are labelled. By default the colours
    are those of :func:`plot_angular_lightcone_slice`.

    Parameters
    ----------
    values : un.Quantity
        The value of each pixel, of shape (N_pixels,): all ``12 * nside**2`` pixels in
        ``order``, or those of ``pixels``, in the same order. For a lightcone
        ``lc`` of shape (N_pixels, N_redshifts), ``lc[:, i]`` is its i-th slice.
    nside : int
        The Nside of the grid, a power of 2.
    pixels : Sequence[int] | np.ndarray, optional
        The distinct indices (in ``order``) of the pixels, for part of the sky. All
        pixels by default.
    order : {"ring", "nested"}, optional
        The ordering of the pixels.
    projection : {"mollweide", "cartesian"}, optional
        "mollweide" shows the full sky, with longitudes from -180 to 180 degrees.
        "cartesian" shows the longitude (between -180 and 180 degrees) and latitude
        in degrees around the pixels, for a patch. By default "mollweide" for the
        full sky and "cartesian" with ``pixels``.
    n_grid : int, optional
        The number of points of the grid along the longitude; the latitude gets the
        number that keeps the points about square on the sky.
    fontsize : float, optional
        The font size of the plot. None uses the current font size.
    title : str, optional
        The title of the plot.
    clabel : str, optional
        The label for the colour bar.
    cmap : str | Colormap, optional
        The colour map. By default, :func:`eor_colormap` for the limits.
    logc : bool, optional
        Whether to use a logarithmic colour scale.
    vmin : float, optional
        The lowest value of the colour scale. By default, the lowest value or zero,
        whichever is lower, or the 5th percentile with a named colour map.
    vmax : float, optional
        The highest value of the colour scale. By default, the highest value or zero,
        whichever is higher, or the 95th percentile with a named colour map.
    ax : plt.Axes, optional
        The axes to plot on, with the "mollweide" projection for that projection. If
        None, a new figure and axes will be created.
    dark_mode : bool, optional
        Whether to draw the plot in dark mode (matplotlib's "dark_background"
        style), as 21cmFAST's lightcone plots are. By default, as set by
        :func:`~tuesday.core.plotting.set_dark_mode`, which is off until it is set.

    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    _check_quantity(values, "values")
    _check_order(order)
    grid = HEALPix(nside=nside, order=order)
    pixels = (
        np.arange(grid.npix) if pixels is None else _check_pixels(pixels, grid.npix)
    )
    if values.shape != pixels.shape:
        raise ValueError(f"values must have one value for each of {pixels.size} pixels")
    if projection is None:
        projection = "mollweide" if pixels.size == grid.npix else "cartesian"
    if projection not in ("mollweide", "cartesian"):
        raise ValueError(
            f"projection must be 'mollweide' or 'cartesian', not {projection!r}"
        )
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")

    if projection == "mollweide":
        lon = np.linspace(-np.pi, np.pi, n_grid)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_grid // 2 + 1)
        aspect = None
    else:
        plon, plat = grid.healpix_to_lonlat(pixels)
        plon, plat = plon.to_value(un.rad), plat.to_value(un.rad)
        # Longitudes relative to the middle of the patch, which may cross zero.
        lon0 = np.angle(np.mean(np.exp(1j * plon)))
        rel = (plon - lon0 + np.pi) % (2 * np.pi) - np.pi
        pad = pixel_side(nside).to_value(un.rad)
        lat_lo = max(plat.min() - pad, -np.pi / 2)
        lat_hi = min(plat.max() + pad, np.pi / 2)
        squeeze = np.cos(np.clip(0.5 * (lat_lo + lat_hi), -1.5, 1.5))
        lon = lon0 + np.linspace(
            rel.min() - pad / squeeze, rel.max() + pad / squeeze, n_grid
        )
        width = (lon[-1] - lon[0]) * squeeze
        n_lat = max(2, round(n_grid * (lat_hi - lat_lo) / width))
        lat = np.linspace(lat_lo, lat_hi, n_lat)
        aspect = 1 / squeeze
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    found = grid.lonlat_to_healpix((lon_grid % (2 * np.pi)) * un.rad, lat_grid * un.rad)
    by_pixel = np.argsort(pixels)
    where = np.searchsorted(pixels[by_pixel], found).clip(max=pixels.size - 1)
    inside = pixels[by_pixel][where] == found
    image = np.where(inside, values.value[by_pixel][where], np.nan)

    if ax is not None and (ax.name == "mollweide") != (projection == "mollweide"):
        raise ValueError(
            "ax must have the 'mollweide' projection for projection='mollweide', "
            "and no projection otherwise"
        )
    with _plot_context(fontsize):
        if ax is None:
            figsize = (10, 6) if projection == "mollweide" else (6, 5)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(
                projection="mollweide" if projection == "mollweide" else None
            )
        colours = _colour_scale(
            values.value, cmap=cmap, logc=logc, vmin=vmin, vmax=vmax
        )
        if projection == "mollweide":
            im = ax.pcolormesh(lon, lat, image, shading="auto", **colours)
            ax.grid(True)
            # Longitude labels would cover the map; the grid lines are 30 degrees
            # apart.
            ax.set_xticklabels([])
            colorbar = {"orientation": "horizontal", "shrink": 0.6, "pad": 0.05}
        else:
            im = ax.pcolormesh(
                np.degrees(lon), np.degrees(lat), image, shading="auto", **colours
            )
            ax.set_aspect(aspect)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")
            colorbar = {}
        if title is not None:
            ax.set_title(title)
        plt.colorbar(
            im,
            ax=ax,
            label=_colour_label(values.unit) if clabel is None else clabel,
            **colorbar,
        )
    return ax
