"""Tests of the fast angular lightconer for 21cmFAST."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from py21cmfast import AngularLightconer, InputParameters, run_lightcone
from py21cmfast.io.caching import OutputCache, RunCache
from scipy.spatial.transform import Rotation

from tuesday.core import lightcones
from tuesday.core.lightcones import CELLS_PER_SAMPLE
from tuesday.simulators.py21cmfast.lightconers import FastAngularLightconer
from tuesday.simulators.py21cmfast.lightcones import construct_lightcone_from_cache

QUANTITIES = ("brightness_temp", "density", "z_reion", "los_velocity")


def _ring(n: int, latitude: float = 0.1) -> dict[str, np.ndarray]:
    return {
        "latitude": np.full(n, latitude),
        "longitude": np.arange(n) * 2 * np.pi / n,
    }


def _lightconer_kwargs(inputs: InputParameters, **kw) -> dict:
    so = inputs.simulation_options
    return {
        "min_redshift": 6.1,
        "max_redshift": 6.9,
        "resolution": so.cell_size,
        "cosmo": inputs.cosmo_params.cosmo,
        "origin": np.full(3, so.HII_DIM / 2) * un.pixel,
        "quantities": QUANTITIES,
        **kw,
    }


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> SimpleNamespace:
    """A tiny cached run with 3D velocities, made with the fast lightconer.

    The run_lightcone call checks that the lightconer works in 21cmFAST's own loop.
    """
    inputs = InputParameters.from_template(
        ["latest-discrete", "size-tiny"],
        random_seed=1,
        node_redshifts=[7.0, 6.5, 6.0],
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    ).evolve_input_structs(
        BOX_LEN=30,
        HII_DIM=10,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        PERTURB_ALGORITHM="LINEAR",
        KEEP_3D_VELOCITIES=True,
    )
    cache_dir = tmp_path_factory.mktemp("cache")
    lightconer = FastAngularLightconer.between_redshifts(
        **_ring(32), tile_first=True, **_lightconer_kwargs(inputs)
    )
    lightcone = run_lightcone(
        lightconer=lightconer,
        inputs=inputs,
        cache=OutputCache(cache_dir),
        write=True,
        progressbar=False,
        include_dvdr_in_tau21=False,
        apply_rsds=False,
    )
    cache = RunCache.from_inputs(inputs, OutputCache(cache_dir))
    return SimpleNamespace(inputs=inputs, cache=cache, lightcone=lightcone)


def _record_ways(monkeypatch) -> list[str]:
    """Record which way each slice is made: "sample" first or "interpolate" first."""
    called = []
    for name in ("sample_then_interpolate", "interpolate_then_sample"):
        original = getattr(lightcones.slices, name)

        def spy(*args, _name=name, _original=original, **kw):
            called.append(_name.split("_")[0])
            return _original(*args, **kw)

        monkeypatch.setattr(lightcones.slices, name, spy)
    return called


def _assert_lightcones_close(result, reference, quantities, rtol=1e-6):
    for q in quantities:
        got = np.asarray(result.lightcones[q], dtype=float)
        want = np.asarray(reference.lightcones[q], dtype=float)
        assert got.shape == want.shape, q
        assert np.any(want != 0), q
        np.testing.assert_allclose(
            got, want, rtol=0, atol=rtol * np.max(np.abs(want)), err_msg=q
        )


def test_run_lightcone_matches_stock(run):
    """In run_lightcone, it gives what 21cmFAST's AngularLightconer gives."""
    stock = AngularLightconer.between_redshifts(
        **_ring(32), **_lightconer_kwargs(run.inputs)
    )
    reference = construct_lightcone_from_cache(run.cache, stock)
    _assert_lightcones_close(run.lightcone, reference, QUANTITIES)


ROTATION = Rotation.from_euler("zyx", [0.3, 0.2, 0.1])


@pytest.mark.parametrize(
    ("tile_first", "rotation", "dvdr_and_rsds"),
    [
        (True, None, False),
        (True, ROTATION, False),
        (False, ROTATION, False),
        (None, ROTATION, False),
        (True, ROTATION, True),
        (False, ROTATION, True),
    ],
)
def test_matches_stock(run, monkeypatch, tile_first, rotation, dvdr_and_rsds):
    """Sampling first or not, the lightcones equal 21cmFAST's, with dv/dr and RSDs.

    Every slice is made the way tile_first asks for, or, with tile_first=None, the
    way samples_first picks: sampling first, since the 10^3 boxes have more than 16
    cells for each of the 48 directions.
    """
    kw = _lightconer_kwargs(run.inputs, rotation=rotation)
    directions = _ring(48, latitude=-0.3)
    options = {
        "include_dvdr_in_tau21": dvdr_and_rsds,
        "apply_rsds": dvdr_and_rsds,
        "n_rsd_subcells": 2,
    }
    reference = construct_lightcone_from_cache(
        run.cache, AngularLightconer.between_redshifts(**directions, **kw), **options
    )
    ways = _record_ways(monkeypatch)
    result = construct_lightcone_from_cache(
        run.cache,
        FastAngularLightconer.between_redshifts(
            **directions, tile_first=tile_first, threads=2, chunk_size=7, **kw
        ),
        **options,
    )
    assert ways
    assert set(ways) == {"interpolate" if tile_first is False else "sample"}
    quantities = list(reference.lightcones)
    assert sorted(result.lightcones) == sorted(quantities)
    _assert_lightcones_close(result, reference, quantities)


@pytest.mark.parametrize("order", [0, 3])
def test_other_orders(run, order):
    """Order 0 can sample first; orders above 1 use 21cmFAST's path."""
    kw = _lightconer_kwargs(run.inputs, interpolation_order=order)
    reference = construct_lightcone_from_cache(
        run.cache, AngularLightconer.between_redshifts(**_ring(40), **kw)
    )
    lightconer = FastAngularLightconer.between_redshifts(
        **_ring(40), tile_first=True, **kw
    )
    assert lightconer.samples_first(10**3) is (order == 0)
    result = construct_lightcone_from_cache(run.cache, lightconer)
    _assert_lightcones_close(result, reference, QUANTITIES)


def test_subsamples(run):
    """Each pixel is the mean over its directions (sub-samples)."""
    n_pixels, n_subsamples = 24, 4
    rng = np.random.default_rng(5)
    latitude = rng.uniform(-1.2, 1.2, n_subsamples * n_pixels)
    longitude = rng.uniform(0, 2 * np.pi, n_subsamples * n_pixels)
    kw = _lightconer_kwargs(run.inputs)
    reference = construct_lightcone_from_cache(
        run.cache,
        AngularLightconer.between_redshifts(
            latitude=latitude, longitude=longitude, **kw
        ),
    )
    for q in reference.lightcones:
        values = np.asarray(reference.lightcones[q])
        reference.lightcones[q] = values.reshape(n_subsamples, n_pixels, -1).mean(
            axis=0
        )

    for tile_first in (True, False):
        lightconer = FastAngularLightconer.between_redshifts(
            latitude=latitude,
            longitude=longitude,
            n_subsamples=n_subsamples,
            tile_first=tile_first,
            **kw,
        )
        assert lightconer.n_pixels == n_pixels
        result = construct_lightcone_from_cache(run.cache, lightconer)
        assert result.lightcones["density"].shape[0] == n_pixels
        _assert_lightcones_close(result, reference, QUANTITIES)


def test_samples_first_rule():
    """Sample first when the box has more than CELLS_PER_SAMPLE cells per direction.

    tile_first overrides the choice, except for orders above 1.
    """
    distances = np.array([100.0]) * un.Mpc
    lightconer = FastAngularLightconer(**_ring(100), lc_distances=distances)
    assert lightconer.samples_first(100 * CELLS_PER_SAMPLE + 1)
    assert not lightconer.samples_first(100 * CELLS_PER_SAMPLE)
    for tile_first in (True, False):
        forced = FastAngularLightconer(
            **_ring(100), tile_first=tile_first, lc_distances=distances
        )
        assert forced.samples_first(1) is tile_first
        assert forced.samples_first(10**9) is tile_first
    cubic = FastAngularLightconer(
        **_ring(100), tile_first=True, interpolation_order=3, lc_distances=distances
    )
    assert not cubic.samples_first(10**9)


def test_validation():
    """n_subsamples must split the directions; threads and chunk_size positive."""
    distances = np.array([100.0]) * un.Mpc
    with pytest.raises(ValueError, match="cannot be split"):
        FastAngularLightconer(**_ring(10), n_subsamples=4, lc_distances=distances)
    with pytest.raises(ValueError, match="n_subsamples must be at least 1"):
        FastAngularLightconer(**_ring(10), n_subsamples=0, lc_distances=distances)
    with pytest.raises(ValueError, match="threads must be at least 1"):
        FastAngularLightconer(**_ring(10), threads=0, lc_distances=distances)
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        FastAngularLightconer(**_ring(10), chunk_size=0, lc_distances=distances)
