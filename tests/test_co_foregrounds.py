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
    # in many cells. The weights are clipped at zero, so their mean exceeds one;
    # co_foreground_brightness renormalises them.
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


def co_redshifts(transition: str, distances: un.Quantity) -> np.ndarray:
    """CO redshift of each [CII] slice, independently of the module."""
    tol = {"ztol": 1e-12, "maxfun": 10_000}
    z_cii = np.array(
        [z_at_value(Planck18.comoving_distance, d, **tol).value for d in distances]
    )
    return (co_rest_frequency(transition) * (1 + z_cii) / NU_CII_REST).value - 1


def channel_depth(transition: str, distances: un.Quantity) -> un.Quantity:
    """Comoving depth at the CO redshift of the channel of each [CII] slice."""
    z_co = co_redshifts(transition, distances)
    return (const.c / Planck18.H(z_co)).to(un.Mpc) * np.abs(np.gradient(z_co))


def luminosity_density_mean(transition: str, distances: un.Quantity) -> np.ndarray:
    """Mean CO brightness of each slice from the CO luminosity density, in Jy/sr.

    Emitters of luminosity density eps (Jy km/s Mpc^2 per Mpc^3) counted over the
    comoving depth of the channel give eps / (4 pi H(z_co) (1 + z_co)), whatever the
    cell size.
    """
    out = []
    for z_co in co_redshifts(transition, distances):
        phi, log_l_star = co_luminosity_function(transition, z_co, LOG_X)
        eps = (
            np.sum(phi * (LOG_X[1] - LOG_X[0]) * 10 ** (LOG_X + log_l_star))
            * un.Jy
            * un.km
            / un.s
            / un.Mpc
        )
        b = eps / (4 * np.pi * Planck18.H(z_co) * (1 + z_co)) / un.sr
        out.append(b.to_value(un.Jy / un.sr))
    return np.array(out)


@pytest.mark.parametrize("transition", ["CO J=3-2", "CO J=4-3", "CO J=5-4", "CO J=6-5"])
def test_mean_brightness_matches_luminosity_density(transition: str):
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))

    assert mean.unit == un.Jy / un.sr
    np.testing.assert_allclose(
        mean.value, luminosity_density_mean(transition, DISTANCES), rtol=1e-6
    )


@pytest.mark.parametrize("transition", ["CO J=3-2", "CO J=4-3", "CO J=5-4", "CO J=6-5"])
def test_mean_brightness_matches_direct_count(transition: str):
    # The emitters of a pixel are counted over the comoving depth of the channel at
    # the CO redshift, 1.6-1.8 cells for CO J=3-2 to 6-5 in front of [CII] at
    # z = 6.5, not over one cell: the mean equals a direct count of the emitters in
    # the volume of the channel.
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))
    depth = channel_depth(transition, DISTANCES)

    assert np.all((depth > 1.5 * CELL) & (depth < 1.9 * CELL))
    np.testing.assert_allclose(
        direct_count(transition, DISTANCES[2]).value, mean[2].value, rtol=1e-3
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
        DISTANCES, 16, CELL, Planck18, density, shot_noise=False
    )
    mean = mean_co_brightness(DISTANCES, CELL, Planck18)

    np.testing.assert_allclose(
        brightness.value, np.broadcast_to(mean.value, brightness.shape), rtol=1e-10
    )


def test_density_cells_map_to_pixels():
    # As in the research code, density cell (j, k) gives the weight of pixel
    # (j, k): a density wave along x appears in the map with the same number of
    # cells per period.
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
        shot_noise=False,
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


def wave_density(
    amplitude: float, n_cells: int, cell: un.Quantity, period: int = 8
) -> tuple[DensityLightcone, np.ndarray]:
    """A density wave along x, out to 4000 Mpc; also return the wave."""
    distances = np.arange(cell.value / 2, 4000, cell.value) * un.Mpc
    wave = amplitude * np.sin(2 * np.pi * (np.arange(n_cells) + 0.5) / period)
    density = DensityLightcone(
        density=np.broadcast_to(
            wave[:, None, None], (n_cells, n_cells, distances.size)
        ).copy(),
        distances=distances,
        cell_size=cell,
    )
    return density, wave


def first_slice_bias(transition: str, distances: un.Quantity) -> float:
    """CO bias at the CO redshift of the first [CII] slice."""
    z_cii = z_at_value(Planck18.comoving_distance, distances[0]).value
    z_co = (co_rest_frequency(transition) * (1 + z_cii) / NU_CII_REST).value - 1
    return co_bias(z_co)


def test_clustering_only_keeps_negative_weights():
    # Without shot noise, each pixel holds the expected brightness with the linear
    # weight 1 + b delta, which is neither clipped nor renormalised: it is negative
    # where b delta < -1.
    transition = "CO J=4-3"
    density, wave = wave_density(amplitude=1.0, n_cells=16, cell=CELL)
    expected = 1 + first_slice_bias(transition, DISTANCES) * wave
    assert (expected < 0).any()

    def clustering(seed):
        return co_foreground_brightness(
            DISTANCES,
            16,
            CELL,
            Planck18,
            density,
            transitions=(transition,),
            shot_noise=False,
            seed=seed,
        )

    brightness = clustering(seed=1)
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))
    weights = (brightness[:, :, 0] / mean[0]).value

    # z_at_value's default tolerance changes the bias at the ~1e-10 level.
    np.testing.assert_allclose(
        weights, np.broadcast_to(expected[:, None], weights.shape), rtol=1e-8
    )
    # No random draws: the seed does not matter.
    np.testing.assert_array_equal(brightness.value, clustering(seed=2).value)


def test_shot_noise_draws_from_renormalised_clipped_weights():
    # With shot noise, the counts are Poisson draws of mean <N> w, with
    # w = clip(1 + b delta, 0) renormalised to unit mean over the slice: pixels with
    # a negative 1 + b delta are empty, the map mean is <N>, and each pixel follows
    # w. Large cells and only the faint, numerous emitters keep the Poisson noise
    # low.
    transition = "CO J=4-3"
    cell = 10 * un.Mpc
    n_cells, period = 128, 8
    distances = Planck18.comoving_distance(6.5) + np.arange(4) * cell
    log_x = np.linspace(-3, -1, 50)
    density, wave = wave_density(amplitude=1.5, n_cells=n_cells, cell=cell)
    raw = 1 + first_slice_bias(transition, distances) * wave
    clipped = np.clip(raw, 0, None)

    kwargs = {"transitions": (transition,), "log_l_ratio": log_x}
    brightness = co_foreground_brightness(
        distances, n_cells, cell, Planck18, density, seed=1, **kwargs
    )[:, :, 0].value
    mean = mean_co_brightness(distances, cell, Planck18, **kwargs)[0].value
    # Mean over the pixels at the same phase of the wave.
    per_phase = (brightness.mean(axis=1) / mean).reshape(-1, period).mean(axis=0)

    assert (raw < 0).any()
    assert clipped.mean() > 1.1
    np.testing.assert_array_equal(brightness[raw < 0], 0)
    np.testing.assert_allclose(brightness.mean() / mean, 1, rtol=0.01)
    np.testing.assert_allclose(
        per_phase, (clipped / clipped.mean())[:period], rtol=0.03
    )


def test_density_is_averaged_over_the_channel_depth():
    # The density of a pixel is the mean of the slices whose centres lie within the
    # comoving depth of the channel at the CO redshift, chi_co -/+ depth / 2.
    transition = "CO J=4-3"
    distances = np.arange(0.75, 4000, 1.5) * un.Mpc
    along = np.random.default_rng(0).uniform(-0.3, 0.3, distances.size)
    density = DensityLightcone(
        density=np.broadcast_to(along, (4, 4, distances.size)).copy(),
        distances=distances,
        cell_size=CELL,
    )
    brightness = co_foreground_brightness(
        DISTANCES,
        4,
        CELL,
        Planck18,
        density,
        transitions=(transition,),
        shot_noise=False,
    )
    mean = mean_co_brightness(DISTANCES, CELL, Planck18, transitions=(transition,))

    z_co = co_redshifts(transition, DISTANCES)
    half = channel_depth(transition, DISTANCES) / 2
    chi_co = Planck18.comoving_distance(z_co)
    for i in range(len(DISTANCES)):
        inside = (distances >= chi_co[i] - half[i]) & (distances < chi_co[i] + half[i])
        assert 1 <= inside.sum() <= 2
        expected = 1 + co_bias(z_co[i]) * along[inside].mean()
        np.testing.assert_allclose(
            (brightness[..., i] / mean[i]).value, expected, rtol=1e-8
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


def test_depth_average():
    # Slices at 0.5, 1.5, ..., 9.5 Mpc, with density equal to the slice index.
    density = DensityLightcone(
        density=np.broadcast_to(np.arange(10.0), (2, 2, 10)).copy(),
        distances=(np.arange(10) + 0.5) * un.Mpc,
        cell_size=1 * un.Mpc,
    )

    # Centres in [3.5, 5.5): slices 3 and 4.
    np.testing.assert_array_equal(
        density.depth_average(4.5 * un.Mpc, 2 * un.Mpc), np.full((2, 2), 3.5)
    )
    # No centre in [4.6, 5.4): the nearest slice, 4.
    np.testing.assert_array_equal(
        density.depth_average(5.0 * un.Mpc, 0.8 * un.Mpc), np.full((2, 2), 4.0)
    )
    with pytest.raises(ValueError, match="density lightcone"):
        density.depth_average(20 * un.Mpc, 2 * un.Mpc)
