"""Tests of the CO interloper foreground of [CII] line-intensity maps."""

import csv
from pathlib import Path

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck18, z_at_value

from tuesday.core.postprocessing import co_foregrounds
from tuesday.core.postprocessing.co_foregrounds import (
    CO_TRANSITIONS,
    DensityLightcone,
    bias_weights,
    co_bias,
    co_foreground_brightness,
    co_luminosity_function,
    co_rest_frequency,
    mean_co_brightness,
)
from tuesday.core.postprocessing.line_intensities import NU_CII_REST

TABLE = Path(co_foregrounds.__file__).parent / "data" / "co_schechter_popping16.csv"
LOG_X = np.linspace(-3, 1, 100)
# [CII] slices around z = 6.5, where CO 3-2 to 6-5 are interlopers.
CELL = 1.5 * un.Mpc
DISTANCES = Planck18.comoving_distance(6.5) + np.arange(6) * CELL


def table_row(transition: str, redshift: float) -> dict[str, float]:
    """Read one row of the Popping et al. (2016) table, independently of the module."""
    with TABLE.open() as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        row = next(
            r
            for r in rows
            if r["transition"] == transition and float(r["redshift"]) == redshift
        )
    return {k: float(v) for k, v in row.items() if k != "transition"}


def schechter(transition: str, redshift: float) -> np.ndarray:
    """Number density per dex of L/L* on LOG_X, from the table row at a redshift."""
    p = table_row(transition, redshift)
    x = 10**LOG_X
    return np.log(10) * 10 ** p["log_phi_star"] * x ** (p["alpha"] + 1) * np.exp(-x)


def uniform_density(
    max_distance: un.Quantity, n_cells: int = 16, cell: un.Quantity = CELL
) -> DensityLightcone:
    """A density lightcone with no fluctuations, from 0 to ``max_distance``."""
    distances = np.arange(cell.value / 2, max_distance.to_value(un.Mpc), cell.value)
    return DensityLightcone(
        density=np.zeros((n_cells, n_cells, distances.size), dtype=np.float32),
        distances=distances * un.Mpc,
        cell_size=cell,
    )


# --- Luminosity function and bias ---------------------------------------------------


def test_transitions_and_rest_frequencies():
    assert tuple(f"CO J={j + 1}-{j}" for j in range(6)) == CO_TRANSITIONS
    assert co_rest_frequency("CO J=4-3") == 461.04 * un.GHz
    with pytest.raises(ValueError, match="Unknown CO transition"):
        co_rest_frequency("CO J=9-8")


@pytest.mark.parametrize("transition", CO_TRANSITIONS)
@pytest.mark.parametrize("redshift", [0, 1, 2, 4, 6])
def test_luminosity_function_at_tabulated_redshifts(transition: str, redshift: int):
    # At a tabulated redshift, the fit of that redshift is used (not a neighbour's).
    phi, log_l_star = co_luminosity_function(transition, redshift, LOG_X)

    np.testing.assert_allclose(phi, schechter(transition, redshift), rtol=1e-12)
    assert log_l_star == table_row(transition, redshift)["log_L_star"]


def test_luminosity_function_interpolation():
    phi, log_l_star = co_luminosity_function("CO J=3-2", 1.5, LOG_X)

    expected = (schechter("CO J=3-2", 1) + schechter("CO J=3-2", 2)) / 2
    np.testing.assert_allclose(phi, expected, rtol=1e-12)
    np.testing.assert_allclose(
        log_l_star,
        (
            table_row("CO J=3-2", 1)["log_L_star"]
            + table_row("CO J=3-2", 2)["log_L_star"]
        )
        / 2,
        rtol=1e-12,
    )


def test_luminosity_function_held_above_last_redshift():
    phi7, log_l_star7 = co_luminosity_function("CO J=5-4", 7.0, LOG_X)
    phi6, log_l_star6 = co_luminosity_function("CO J=5-4", 6.0, LOG_X)

    np.testing.assert_array_equal(phi7, phi6)
    assert log_l_star7 == log_l_star6


def test_luminosity_function_negative_redshift_raises():
    with pytest.raises(ValueError, match="redshift"):
        co_luminosity_function("CO J=1-0", -0.1, LOG_X)


def test_co_bias():
    assert co_bias(1.0) == pytest.approx(1.49, rel=1e-12)
    assert co_bias(0.75) == pytest.approx((1.19 + 1.49) / 2, rel=1e-12)
    # Held constant outside the tabulated range.
    assert co_bias(0.1) == pytest.approx(1.19, rel=1e-12)
    assert co_bias(5.0) == pytest.approx(2.81, rel=1e-12)


def test_bias_weights_are_clipped():
    # Strong fluctuations, as for 1.5 Mpc cells at low redshift: 1 + b delta < 0
    # in many cells. As in the research code, the weights are clipped at zero and
    # not renormalised, so their mean exceeds one.
    delta = np.random.default_rng(0).normal(0, 2, size=(64, 64))
    weights = bias_weights(delta, bias=1.5)
    raw = 1 + 1.5 * delta

    assert (raw < 0).mean() > 0.2
    np.testing.assert_array_equal(weights, np.clip(raw, 0, None))
    assert weights.mean() > 1.5


# --- Mean brightness ----------------------------------------------------------------


def direct_count(transition: str, distance: un.Quantity) -> un.Quantity:
    """Mean CO brightness of one [CII] slice, by counting emitters in its volume.

    The channel edges are the redshifts of the slice edges; at the CO redshift they
    bound a comoving shell, whose volume per steradian is integrated numerically.
    """
    nu_co = co_rest_frequency(transition)
    tol = {"ztol": 1e-12, "maxfun": 10_000}
    z_front = z_at_value(Planck18.comoving_distance, distance - CELL / 2, **tol)
    z_back = z_at_value(Planck18.comoving_distance, distance + CELL / 2, **tol)
    z = z_at_value(Planck18.comoving_distance, distance, **tol)
    nu_hi, nu_lo, nu_obs = (NU_CII_REST / (1 + zz) for zz in (z_front, z_back, z))
    z_co = (nu_co / nu_obs).value - 1
    chi = np.linspace(
        Planck18.comoving_distance((nu_co / nu_hi).value - 1),
        Planck18.comoving_distance((nu_co / nu_lo).value - 1),
        2001,
    )
    volume_per_sr = np.trapezoid(chi**2, chi) / un.sr

    phi, log_l_star = co_luminosity_function(transition, z_co, LOG_X)
    emissivity = (
        np.sum(phi * (LOG_X[1] - LOG_X[0]) * 10 ** (LOG_X + log_l_star))
        * un.Jy
        * un.km
        / un.s
        * un.Mpc**2
        / un.Mpc**3
    )
    d_lum = Planck18.luminosity_distance(z_co)
    flux = emissivity * volume_per_sr / (4 * np.pi * d_lum**2)
    return (flux * nu_obs / (const.c * (nu_hi - nu_lo))).to(un.Jy / un.sr)


def research_mean(transition: str, distances: un.Quantity) -> np.ndarray:
    """Mean CO brightness of each slice as in the research code, in Jy/sr.

    The emitters of one cell^3 of comoving volume, at the CO redshift, spread over
    the solid angle (cell / D_co)^2 and the observed width of the [CII] slice.
    """
    tol = {"ztol": 1e-12, "maxfun": 10_000}
    z_cii = np.array(
        [z_at_value(Planck18.comoving_distance, d, **tol).value for d in distances]
    )
    nu_obs = NU_CII_REST / (1 + z_cii)
    nu_co = co_rest_frequency(transition)
    lc_z_co = (nu_co / nu_obs).value - 1
    dz_co = np.gradient(lc_z_co)
    out = []
    for i, z_co in enumerate(lc_z_co):
        phi, log_l_star = co_luminosity_function(transition, z_co, LOG_X)
        luminosity = np.sum(phi * (LOG_X[1] - LOG_X[0]) * 10 ** (LOG_X + log_l_star))
        lum = luminosity * CELL.value**3 * un.Jy * un.km / un.s * un.Mpc**2
        d_co = Planck18.comoving_distance(z_co)
        d_lum = d_co * (1 + z_co)
        solid_angle = (CELL / d_co) ** 2 * un.sr
        delta_nu = nu_co / (1 + z_co) ** 2 * dz_co[i]
        b = lum * nu_obs[i] / (4 * np.pi * d_lum**2 * const.c * delta_nu * solid_angle)
        out.append(b.to_value(un.Jy / un.sr))
    return np.array(out)


@pytest.mark.parametrize("transition", ["CO J=3-2", "CO J=4-3", "CO J=5-4", "CO J=6-5"])
def test_mean_brightness_matches_research_formula(transition: str):
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))

    assert mean.unit == un.Jy / un.sr
    np.testing.assert_allclose(
        mean.value, research_mean(transition, DISTANCES), rtol=1e-6
    )


@pytest.mark.parametrize("transition", ["CO J=3-2", "CO J=4-3", "CO J=5-4", "CO J=6-5"])
def test_cell_volume_approximation(transition: str):
    # Documents the approximation of the research code: it counts the emitters of
    # one cell^3, but a [CII] slice one cell deep spans a depth of
    # cell * H(z_cii) (1 + z_co) / (H(z_co) (1 + z_cii)) at the CO redshift, i.e.
    # 1.6-1.8 cells. A direct count of the emitters is larger by this ratio.
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))
    distance = DISTANCES[2]
    z_cii = z_at_value(Planck18.comoving_distance, distance).value
    nu_obs = NU_CII_REST / (1 + z_cii)
    z_co = (co_rest_frequency(transition) / nu_obs).value - 1
    depth_ratio = (
        (Planck18.H(z_cii) / Planck18.H(z_co)).value * (1 + z_co) / (1 + z_cii)
    )

    assert 1.5 < depth_ratio < 1.9
    np.testing.assert_allclose(
        direct_count(transition, distance).value / mean[2].value, depth_ratio, rtol=1e-3
    )


def test_mean_brightness_needs_two_slices():
    with pytest.raises(ValueError, match="two slices"):
        mean_co_brightness(DISTANCES[:1], CELL, Planck18)


def test_lines_from_negative_redshift_are_skipped():
    # At z_cii = 6.5 (254 GHz), CO 1-0 and 2-1 would come from z < 0.
    mean = mean_co_brightness(
        DISTANCES, CELL, Planck18, transitions=("CO J=1-0", "CO J=2-1")
    )

    np.testing.assert_array_equal(mean.value, 0)


def test_mean_brightness_sums_transitions():
    total = mean_co_brightness(DISTANCES, CELL, Planck18)
    parts = sum(
        mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(t,))
        for t in CO_TRANSITIONS
    )

    np.testing.assert_allclose(total.value, parts.value, rtol=1e-12)


# --- Sampled maps -------------------------------------------------------------------


def test_sampled_mean_matches_expectation():
    # Large cells and only the faint, numerous emitters, so that the Poisson noise
    # of the map mean is below a per cent.
    cell = 10 * un.Mpc
    distances = Planck18.comoving_distance(6.5) + np.arange(4) * cell
    log_x = np.linspace(-3, -1, 50)
    density = uniform_density(Planck18.comoving_distance(2.2), n_cells=128, cell=cell)

    brightness = co_foreground_brightness(
        distances, 128, cell, Planck18, density, log_l_ratio=log_x, seed=1
    )
    expected = mean_co_brightness(distances, cell, Planck18, log_l_ratio=log_x)

    assert brightness.shape == (128, 128, 4)
    np.testing.assert_allclose(
        brightness.mean(axis=(0, 1)).value, expected.value, rtol=0.03
    )


def test_expected_map_of_uniform_density_is_the_mean():
    density = uniform_density(Planck18.comoving_distance(2.2))
    brightness = co_foreground_brightness(
        DISTANCES, 16, CELL, Planck18, density, poisson=False
    )
    mean = mean_co_brightness(DISTANCES, CELL, Planck18)

    np.testing.assert_allclose(
        brightness.value, np.broadcast_to(mean.value, brightness.shape), rtol=1e-10
    )


def test_density_cells_map_to_pixels():
    # As in the research code, density cell (j, k) of the slice nearest to the CO
    # distance gives the weight of pixel (j, k): a density wave along x appears in
    # the map with the same number of cells per period.
    transition = "CO J=4-3"
    amplitude, period = 0.2, 8
    n_pixels = 16
    distances = np.arange(0.75, 4000, 1.5) * un.Mpc
    wave = amplitude * np.sin(2 * np.pi * (np.arange(n_pixels) + 0.5) / period)
    density = DensityLightcone(
        density=np.broadcast_to(
            wave[:, None, None], (n_pixels, n_pixels, distances.size)
        ).copy(),
        distances=distances,
        cell_size=CELL,
    )
    brightness = co_foreground_brightness(
        DISTANCES,
        n_pixels,
        CELL,
        Planck18,
        density,
        transitions=(transition,),
        poisson=False,
    )
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))

    z_cii = z_at_value(Planck18.comoving_distance, DISTANCES[0]).value
    z_co = (co_rest_frequency(transition) * (1 + z_cii) / NU_CII_REST).value - 1
    expected = 1 + co_bias(z_co) * wave

    weights = (brightness[:, :, 0] / mean[0]).value
    # z_at_value's default tolerance changes the bias at the ~1e-10 level.
    np.testing.assert_allclose(weights[:, 0], expected, rtol=1e-8)
    np.testing.assert_allclose(
        weights, np.broadcast_to(weights[:, :1], weights.shape), rtol=1e-10
    )


def test_density_must_match_the_map():
    density = uniform_density(Planck18.comoving_distance(2.2), n_cells=8)
    with pytest.raises(ValueError, match="pixels"):
        co_foreground_brightness(DISTANCES, 16, CELL, Planck18, density, seed=0)

    density = uniform_density(Planck18.comoving_distance(2.2), cell=2 * un.Mpc)
    with pytest.raises(ValueError, match="cell size"):
        co_foreground_brightness(DISTANCES, 16, CELL, Planck18, density, seed=0)


def test_seed_is_reproducible():
    density = uniform_density(Planck18.comoving_distance(2.2))

    def draw(seed):
        return co_foreground_brightness(
            DISTANCES, 16, CELL, Planck18, density, seed=seed
        )

    np.testing.assert_array_equal(draw(3).value, draw(3).value)
    assert not np.allclose(draw(3).value, draw(4).value)


def test_density_must_cover_the_co_redshifts():
    # CO 6-5 from z_cii = 6.5 comes from z_co ~ 1.7, beyond this density lightcone.
    density = uniform_density(Planck18.comoving_distance(1.0))
    with pytest.raises(ValueError, match="density lightcone"):
        co_foreground_brightness(DISTANCES, 16, CELL, Planck18, density, seed=0)


def test_density_lightcone_validation():
    with pytest.raises(ValueError, match="distances"):
        DensityLightcone(
            density=np.zeros((4, 4, 3)),
            distances=np.arange(4) * un.Mpc,
            cell_size=1 * un.Mpc,
        )
