"""Tests of the plots of angular lightcones in core/plotting/angular.py."""

import astropy.units as un
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy_healpix import HEALPix
from matplotlib import rcParams
from matplotlib.colors import LogNorm, to_rgb, to_rgba

from tuesday.core import eor_colormap, plot_angular_lightcone_slice, plot_healpix_map
from tuesday.core.lightcones import disc_pixels
from tuesday.core.plotting.angular import _plot_context


@pytest.fixture
def arc():
    """A lightcone along an arc of 64 lines of sight and 40 slices.

    Returns the lightcone, the angles, the distances and the redshifts.
    """
    rng = np.random.default_rng(1)
    lightcone = (rng.random((64, 40)) * 60 - 40) * un.mK
    angle = np.linspace(0, 20, 64) * un.deg
    distance = np.linspace(8000, 10000, 40) * un.Mpc
    return lightcone, angle, distance, np.linspace(6, 12, 40)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_eor_colormap():
    """Zero is black, and the ends are white and sky blue."""
    cmap = eor_colormap(-120, 30)
    np.testing.assert_allclose(cmap(120 / 150)[:3], (0, 0, 0), atol=0.01)
    np.testing.assert_allclose(cmap(0.0)[:3], (1, 1, 1), atol=1e-12)
    np.testing.assert_allclose(cmap(1.0)[:3], to_rgb("skyblue"), atol=1e-12)
    # Only emission, or only absorption.
    np.testing.assert_allclose(eor_colormap(0, 30)(0.0)[:3], (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(eor_colormap(-5, -1)(1.0)[:3], (0, 0, 0), atol=1e-12)
    with pytest.raises(ValueError, match="less than"):
        eor_colormap(1, 1)


def test_angular_slice_shape(arc):
    """Each value is drawn at its angle and distance, in a wedge of the true shape."""
    lc, angle, distance, _ = arc
    ax = plot_angular_lightcone_slice(lc, angle, distance, title="arc")
    assert ax.get_title() == "arc"
    mesh = ax.collections[0]
    np.testing.assert_array_equal(mesh.get_array(), lc.value.T)
    corners = mesh.get_coordinates()
    assert corners.shape == (41, 65, 2)
    radius = np.hypot(corners[..., 0], corners[..., 1])
    theta = np.degrees(np.arctan2(corners[..., 1], corners[..., 0]))
    # The corners lie half a cell beyond the centres.
    np.testing.assert_allclose(
        radius[:, 0], np.linspace(7974.36, 10025.64, 41), atol=0.01
    )
    np.testing.assert_allclose(theta[0, [0, -1]], [-10 / 63, 20 + 10 / 63], atol=1e-9)
    assert ax.get_aspect() == 1
    # Distances labelled with comoving distances, and the default colours.
    assert "Comoving distance [Mpc]" in _texts(ax)
    assert ax.figure.axes[-1].get_ylabel() == "$T_{\\rm B}$ [$\\mathrm{mK}$]"
    assert (mesh.norm.vmin, mesh.norm.vmax) == (lc.value.min(), lc.value.max())
    zero = -mesh.norm.vmin / (mesh.norm.vmax - mesh.norm.vmin)
    np.testing.assert_allclose(mesh.cmap(zero)[:3], (0, 0, 0), atol=0.01)


def test_angular_slice_labels_and_cut(arc):
    """Redshift labels, the grid of boxes and the cut in redshift."""
    lc, angle, distance, z = arc
    ax = plot_angular_lightcone_slice(
        lc, angle, distance, redshift=z, zmin=7, zmax=11, grid=False
    )
    first, last = np.argmin(np.abs(z - 7)), np.argmin(np.abs(z - 11))
    mesh = ax.collections[0]
    np.testing.assert_array_equal(mesh.get_array(), lc[:, first : last + 1].value.T)
    texts = _texts(ax)
    assert "Redshift" in texts
    labels = [float(t) for t in texts if t.replace(".", "").isdigit()]
    assert labels
    assert all(z[first] <= label <= z[last] for label in labels)
    # Without grid lines, only the outline is drawn.
    assert len(ax.lines) == 1

    # Radial lines one box apart at the nearest distance (3.6 degrees), labelled to
    # a tenth of that, and arcs at multiples of the box.
    box = 500 * un.Mpc
    ax = plot_angular_lightcone_slice(lc, angle, distance, redshift=z, box_length=box)
    step = np.degrees(500 / 8000)
    angles = [t for t in _texts(ax) if t.endswith("°")]
    assert angles == [f"{a:.1f}°" for a in np.arange(6) * step]
    radii = np.arange(8000, 10001, 500)
    assert len(ax.lines) == radii.size + len(angles) + 1
    expected = [f"{v:.1f}" for v in np.interp(radii, distance.value, z)]
    assert [t for t in _texts(ax) if t in expected] == expected


def test_angular_slice_options(arc):
    """A named colour map, a log scale, dark mode, given axes and labels."""
    lc, angle, distance, _ = arc
    mesh = plot_angular_lightcone_slice(
        lc, angle, distance, cmap="viridis"
    ).collections[0]
    assert mesh.norm.vmin == pytest.approx(np.percentile(lc.value, 5), rel=1e-12)
    assert mesh.norm.vmax == pytest.approx(np.percentile(lc.value, 95), rel=1e-12)

    positive = np.abs(lc) + 1 * un.mK
    mesh = plot_angular_lightcone_slice(
        positive, angle, distance, logc=True, vmin=1, vmax=30
    ).collections[0]
    assert isinstance(mesh.norm, LogNorm)
    assert (mesh.norm.vmin, mesh.norm.vmax) == (1, 30)

    rcParams["font.size"] = 10
    _, ax = plt.subplots()
    out = plot_angular_lightcone_slice(
        lc.value * un.dimensionless_unscaled,
        angle,
        distance,
        ax=ax,
        fontsize=20,
        rlabel="d",
        clabel="x_HI",
    )
    assert out is ax
    assert "d" in _texts(ax)
    assert ax.figure.axes[-1].yaxis.label.get_fontsize() == 20
    assert ax.figure.axes[-1].get_ylabel() == "x_HI"
    # The font size applies to this plot only.
    assert rcParams["font.size"] == 10

    # A white background by default, and a black one in dark mode.
    ax = plot_angular_lightcone_slice(lc, angle, distance)
    assert to_rgba(ax.figure.get_facecolor()) == to_rgba("white")
    ax = plot_angular_lightcone_slice(lc, angle, distance, dark_mode=True)
    assert to_rgba(ax.figure.get_facecolor()) == to_rgba("black")
    ax = plot_healpix_map(np.arange(48) * un.mK, 2, dark_mode=True)
    assert to_rgba(ax.figure.get_facecolor()) == to_rgba("black")
    assert rcParams["figure.facecolor"] == "white"


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"lightcone": "values"}, TypeError),
        ({"lightcone": "3d"}, ValueError),
        ({"angle": np.linspace(0, 1, 64)}, TypeError),
        ({"angle": np.linspace(0, 20, 63) * un.deg}, ValueError),
        ({"angle": np.linspace(20, 0, 64) * un.deg}, ValueError),
        ({"angle": np.linspace(0, 400, 64) * un.deg}, ValueError),
        ({"distance": np.linspace(1, 2, 39) * un.Mpc}, ValueError),
        ({"redshift": np.ones(39)}, ValueError),
        ({"zmin": 7}, ValueError),
    ],
)
def test_angular_slice_errors(arc, change, error):
    """Inputs of the wrong type, shape or order raise errors."""
    lc, angle, distance, _ = arc
    kwargs = {"lightcone": lc, "angle": angle, "distance": distance, **change}
    if isinstance(kwargs["lightcone"], str):
        kwargs["lightcone"] = {"values": lc.value, "3d": lc[:, None, :]}[
            kwargs["lightcone"]
        ]
    with pytest.raises(error):
        plot_angular_lightcone_slice(**kwargs)


@pytest.mark.parametrize("order", ["ring", "nested"])
def test_healpix_full_sky(order):
    """A full-sky map in the Mollweide projection shows the value of each pixel."""
    nside = 4
    grid = HEALPix(nside=nside, order=order)
    values = np.arange(grid.npix) * un.mK
    ax = plot_healpix_map(values, nside, order=order, n_grid=90, title="sky")
    assert ax.name == "mollweide"
    assert ax.get_title() == "sky"
    image = np.ma.filled(ax.collections[0].get_array(), np.nan)
    assert np.all(np.isfinite(image))
    # Each point shows the pixel that contains it.
    lon = np.linspace(-np.pi, np.pi, 90)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 46)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    pixel = grid.lonlat_to_healpix((lon_grid % (2 * np.pi)) * un.rad, lat_grid * un.rad)
    np.testing.assert_array_equal(image, pixel)
    assert len(np.unique(image)) == grid.npix


def test_healpix_patch_across_zero_longitude():
    """A patch is shown in longitude and latitude, around the patch."""
    nside = 64
    patch = disc_pixels(nside, 0.0 * un.rad, 0.5 * un.rad, 3 * un.deg)
    # Values in a shuffled order of the pixels.
    shuffled = np.random.default_rng(2).permutation(patch)
    ax = plot_healpix_map(shuffled * un.dimensionless_unscaled, nside, pixels=shuffled)
    assert ax.name == "rectilinear"
    assert ax.get_xlabel() == "Longitude [deg]"
    assert ax.get_ylabel() == "Latitude [deg]"
    image = np.ma.filled(ax.collections[0].get_array(), np.nan)
    shown = image[np.isfinite(image)].astype(int)
    assert set(shown) <= set(patch)
    assert len(set(shown)) > 0.9 * patch.size
    assert np.any(~np.isfinite(image))
    xmin, xmax = ax.get_xlim()
    assert -6 < xmin < -3
    assert 3 < xmax < 6


def test_healpix_default_colours():
    """The default limits include zero, which is black."""
    values = (np.arange(48) + 5.0) * un.mK
    mesh = plot_healpix_map(values, 2).collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (0, 52)
    np.testing.assert_allclose(mesh.cmap(0.0)[:3], (0, 0, 0), atol=1e-12)
    mesh = plot_healpix_map(-values, 2).collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (-52, 0)
    np.testing.assert_allclose(mesh.cmap(1.0)[:3], (0, 0, 0), atol=1e-12)
    mesh = plot_healpix_map(np.zeros(48) * un.mK, 2).collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (-0.5, 0.5)


def test_healpix_on_given_axes():
    """Given axes must match the projection."""
    nside = 2
    values = np.ones(48) * un.mK
    fig = plt.figure()
    moll = fig.add_subplot(1, 2, 1, projection="mollweide")
    assert plot_healpix_map(values, nside, ax=moll, clabel="T") is moll
    plain = fig.add_subplot(1, 2, 2)
    with pytest.raises(ValueError, match="mollweide"):
        plot_healpix_map(values, nside, ax=plain)
    patch = np.array([0, 1, 2])
    assert plot_healpix_map(values[:3], nside, pixels=patch, ax=plain) is plain
    with pytest.raises(ValueError, match="mollweide"):
        plot_healpix_map(values[:3], nside, pixels=patch, ax=moll)


@pytest.mark.parametrize(
    ("values", "kwargs", "error"),
    [
        (np.ones(48), {}, TypeError),
        (np.ones(47) * un.mK, {}, ValueError),
        (np.ones(3) * un.mK, {"pixels": [0, 1, 1]}, ValueError),
        (np.ones(48) * un.mK, {"order": "nest"}, ValueError),
        (np.ones(48) * un.mK, {"projection": "gnomonic"}, ValueError),
        (np.ones(48) * un.mK, {"n_grid": 1}, ValueError),
        (-np.ones(48) * un.mK, {"logc": True}, ValueError),
    ],
)
def test_healpix_errors(values, kwargs, error):
    """Inputs of the wrong type, shape or option raise errors."""
    with pytest.raises(error):
        plot_healpix_map(values, 2, **kwargs)


def test_plot_context_keeps_interactive_mode():
    """The font size is restored, but not interactive mode if a backend set it."""
    was_interactive = matplotlib.is_interactive()
    matplotlib.interactive(False)
    size = rcParams["font.size"]
    try:
        with _plot_context(size + 3):
            assert rcParams["font.size"] == size + 3
            matplotlib.interactive(True)
        assert matplotlib.is_interactive()
        assert rcParams["font.size"] == size
    finally:
        matplotlib.interactive(was_interactive)
