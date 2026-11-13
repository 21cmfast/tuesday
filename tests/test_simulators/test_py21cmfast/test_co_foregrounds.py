"""Tests of the 21cmFAST wrappers for the CO interloper foreground."""

import numpy as np
import py21cmfast as p21c
import pytest
from astropy import units as un
from py21cmfast.drivers.lightcone import LightCone

from tuesday.core.postprocessing.co_foregrounds import (
    DensityLightcone,
    co_foreground_brightness,
)
from tuesday.simulators.py21cmfast.postprocessing.co_foregrounds import (
    add_co_foregrounds,
    build_density_lightcone,
)


def density_inputs() -> p21c.InputParameters:
    """Small inputs (3 Mpc cells, 10 cells per side) with linear density evolution."""
    return p21c.InputParameters.from_template(
        "simple", random_seed=2
    ).evolve_input_structs(
        BOX_LEN=30,
        HII_DIM=10,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        PERTURB_ALGORITHM="LINEAR",
        # Only the density is used; keep the box valid for the reionisation options.
        R_BUBBLE_MAX=5,
    )


@pytest.fixture(scope="module")
def density_lc() -> DensityLightcone:
    """A density lightcone of five 30 Mpc boxes, out to ~150 Mpc (z = 0.0345)."""
    return build_density_lightcone(density_inputs(), max_redshift=0.0345)


def test_density_lightcone_geometry(density_lc):
    assert density_lc.cell_size == 3 * un.Mpc
    assert density_lc.density.shape[:2] == (10, 10)
    # Slices are one cell apart, starting half a cell from the observer.
    np.testing.assert_allclose(
        density_lc.distances.to_value(un.Mpc),
        1.5 + 3 * np.arange(density_lc.density.shape[-1]),
        rtol=1e-12,
    )
    assert density_lc.density.shape[-1] >= 40


def test_density_lightcone_rotates_the_boxes(density_lc):
    # Each box of 10 slices is the same initial conditions, evolved linearly to the
    # redshift of the box centre, and rotated or flipped in the transverse plane
    # with a period of four boxes. Undoing the transformation must leave the first
    # box times a growth-factor ratio.
    boxes = [density_lc.density[..., 10 * b : 10 * (b + 1)] for b in range(4)]
    undo = [
        lambda box: box,
        lambda box: np.rot90(box, k=-1, axes=(0, 1)),
        lambda box: np.flip(box, axis=0),
        lambda box: np.flip(np.rot90(box, k=-1, axes=(0, 1)), axis=1),
    ]
    # 21cmFAST floors the density contrast at -1, which linear evolution reaches in
    # many cells this close to z = 0; those cells do not scale with the growth.
    unfloored = boxes[0] > -0.99
    for b in range(1, 4):
        restored = undo[b](boxes[b])
        ratio = (restored / boxes[0])[unfloored]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-4)
        assert not np.allclose(boxes[b], boxes[0] * ratio[0], rtol=1e-3)


def make_cii_lightcone() -> LightCone:
    """A small [CII] lightcone at z ~ 6.1-6.9 with a uniform brightness."""
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=[7.0, 6.5, 6.0]
    ).evolve_input_structs(BOX_LEN=50, HII_DIM=10, DIM=20, PERTURB_ALGORITHM="LINEAR")
    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=6.1,
        max_redshift=6.9,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp",),
    )
    shape = (10, 10, len(lightconer.lc_distances))
    return LightCone(
        lightcone_distances=lightconer.lc_distances,
        inputs=inputs,
        lightcones={"cii_brightness": np.full(shape, 100.0)},
    )


def uniform_density() -> DensityLightcone:
    """No density fluctuations, out to beyond the CO 6-5 redshift (~1.9)."""
    distances = np.arange(2.5, 5500, 5) * un.Mpc
    # Same pixels and cell size as the [CII] lightcone, as the research code maps
    # density cells to pixels one to one.
    return DensityLightcone(
        density=np.zeros((10, 10, distances.size)),
        distances=distances,
        cell_size=5 * un.Mpc,
    )


def test_add_co_foregrounds_matches_core():
    lc = make_cii_lightcone()
    density = uniform_density()
    out = add_co_foregrounds(lc, density, seed=5)
    expected = co_foreground_brightness(
        lc.lightcone_distances,
        10,
        5 * un.Mpc,
        lc.cosmo_params.cosmo,
        density,
        seed=5,
    )

    np.testing.assert_allclose(out.lightcones["co_brightness"], expected.value)
    np.testing.assert_allclose(
        out.lightcones["cii_co_brightness"], 100.0 + expected.value, rtol=1e-12
    )
    assert np.all(out.lightcones["co_brightness"] >= 0)


def test_add_co_foregrounds_does_not_mutate_input():
    lc = make_cii_lightcone()
    original = lc.lightcones["cii_brightness"]
    out = add_co_foregrounds(lc, uniform_density(), seed=0)

    assert set(lc.lightcones) == {"cii_brightness"}
    assert out is not lc
    assert out.lightcones["cii_brightness"] is original


def test_add_co_foregrounds_seed():
    lc = make_cii_lightcone()
    a = add_co_foregrounds(lc, uniform_density(), seed=1).lightcones["co_brightness"]
    b = add_co_foregrounds(lc, uniform_density(), seed=1).lightcones["co_brightness"]
    c = add_co_foregrounds(lc, uniform_density(), seed=2).lightcones["co_brightness"]

    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_add_co_foregrounds_needs_cii_brightness():
    lc = make_cii_lightcone()
    lc.lightcones.pop("cii_brightness")
    with pytest.raises(KeyError, match="cii_brightness"):
        add_co_foregrounds(lc, uniform_density(), seed=0)
