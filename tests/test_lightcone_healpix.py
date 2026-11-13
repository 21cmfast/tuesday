"""Tests of the HEALPix helpers for angular lightcones."""

import astropy.units as un
import numpy as np
import pytest
from astropy.coordinates import angular_separation
from astropy_healpix import HEALPix, nside_to_pixel_resolution

from tuesday.core.lightcones import (
    degrade,
    disc_pixels,
    nside_for_cell_size,
    subpixel_lonlat,
    subsample_level_for_cell_size,
)


def _pixel_size(nside: int) -> float:
    return nside_to_pixel_resolution(nside).to_value(un.rad)


def test_nside_for_cell_size():
    """The smallest power of 2 whose pixels are no larger than a cell."""
    assert nside_for_cell_size(1.5 * un.Mpc, 11773 * un.Mpc) == 8192
    for cell, distance in [(1.5, 11773), (3.0, 5000), (1.0, 2.0)]:
        nside = nside_for_cell_size(cell * un.Mpc, distance * un.Mpc)
        angle = cell / distance
        assert _pixel_size(nside) <= angle
        assert nside == 1 or _pixel_size(nside // 2) > angle
    assert nside_for_cell_size(1 * un.Gpc, 1 * un.Mpc) == 1
    with pytest.raises(ValueError, match="positive"):
        nside_for_cell_size(0 * un.Mpc, 1 * un.Mpc)


def test_subsample_level_for_cell_size():
    """Sub-pixels no larger than a cell; 0 if the pixels already are."""
    assert subsample_level_for_cell_size(2048, 1.5 * un.Mpc, 11773 * un.Mpc) == 2
    assert subsample_level_for_cell_size(8192, 1.5 * un.Mpc, 11773 * un.Mpc) == 0
    assert subsample_level_for_cell_size(16384, 1.5 * un.Mpc, 11773 * un.Mpc) == 0
    with pytest.raises(ValueError, match="nside must be positive"):
        subsample_level_for_cell_size(0, 1.5 * un.Mpc, 11773 * un.Mpc)


@pytest.mark.parametrize("order", ["ring", "nested"])
def test_subpixel_lonlat(order):
    """Level 0 gives the pixel centres; level k the sub-pixels inside each pixel."""
    nside = 4
    grid = HEALPix(nside=nside, order=order)
    lon, lat = subpixel_lonlat(nside, order=order)
    centre_lon, centre_lat = grid.healpix_to_lonlat(np.arange(grid.npix))
    assert lon.shape == (1, grid.npix)
    np.testing.assert_allclose(lon[0], centre_lon, rtol=1e-14)
    np.testing.assert_allclose(lat[0], centre_lat, rtol=1e-14)

    lon, lat = subpixel_lonlat(nside, order=order, subsample_level=2)
    assert lon.shape == (16, grid.npix)
    # every sub-pixel lies in its pixel, and the sub-pixels are distinct
    parent = grid.lonlat_to_healpix(lon, lat)
    np.testing.assert_array_equal(
        parent, np.broadcast_to(np.arange(grid.npix), (16, grid.npix))
    )
    fine = HEALPix(nside=4 * nside, order="nested")
    assert np.unique(fine.lonlat_to_healpix(lon, lat)).size == fine.npix
    assert np.all((lon >= 0) & (lon < 2 * np.pi * un.rad))


def test_subpixel_lonlat_pixels():
    """A subset of pixels gives the same directions as the full sky."""
    pixels = np.array([5, 0, 100, 37])
    full = subpixel_lonlat(8, subsample_level=1)
    part = subpixel_lonlat(8, subsample_level=1, pixels=pixels)
    for a, b in zip(part, full, strict=True):
        np.testing.assert_array_equal(a, b[:, pixels])
    with pytest.raises(ValueError, match="distinct pixel indices"):
        subpixel_lonlat(8, pixels=[1, 1])
    with pytest.raises(ValueError, match="distinct pixel indices"):
        subpixel_lonlat(8, pixels=[768])
    with pytest.raises(ValueError, match="order must be"):
        subpixel_lonlat(8, order="RING")
    with pytest.raises(ValueError, match="non-negative"):
        subpixel_lonlat(8, subsample_level=-1)
    # astropy-healpix supports only powers of 2, also in ring order
    with pytest.raises(ValueError, match="power of two"):
        subpixel_lonlat(6, order="ring")


@pytest.mark.parametrize("order", ["ring", "nested"])
def test_disc_pixels(order):
    """Exactly the pixels whose centres are within the radius."""
    nside = 16
    grid = HEALPix(nside=nside, order=order)
    lon0, lat0, radius = 3.0 * un.rad, -0.4 * un.rad, 10 * un.deg
    pixels = disc_pixels(nside, lon0, lat0, radius, order=order)
    lon, lat = grid.healpix_to_lonlat(np.arange(grid.npix))
    inside = np.nonzero(angular_separation(lon, lat, lon0, lat0) <= radius)[0]
    np.testing.assert_array_equal(pixels, inside)
    # about the area of the disc
    area = 2 * np.pi * (1 - np.cos(radius)) * un.sr
    assert abs(pixels.size * grid.pixel_area - area) < 0.1 * area
    with pytest.raises(ValueError, match="radius must be positive"):
        disc_pixels(nside, lon0, lat0, 0 * un.deg)


@pytest.mark.parametrize("order", ["ring", "nested"])
def test_degrade(order):
    """Each coarse pixel is the mean of the fine pixels that tile it."""
    nside, nside_out = 8, 2
    fine = HEALPix(nside=nside, order=order)
    coarse = HEALPix(nside=nside_out, order=order)
    rng = np.random.default_rng(4)
    maps = rng.normal(size=(fine.npix, 3))

    # the parent of each fine pixel, by position
    lon, lat = fine.healpix_to_lonlat(np.arange(fine.npix))
    parent = coarse.lonlat_to_healpix(lon, lat)
    expected = np.array([maps[parent == p].mean(axis=0) for p in range(coarse.npix)])
    np.testing.assert_allclose(
        degrade(maps, nside_out, order=order), expected, rtol=1e-12
    )
    np.testing.assert_array_equal(degrade(maps, nside, order=order), maps)

    with pytest.raises(ValueError, match="full-sky"):
        degrade(maps[:-1], nside_out, order=order)
    with pytest.raises(ValueError, match="power of 2"):
        degrade(maps, 3, order=order)
    with pytest.raises(ValueError, match="power of 2"):
        degrade(maps, 16, order=order)
