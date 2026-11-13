"""Tests of the fast angular lightconer for 21cmFAST."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from py21cmfast import AngularLightconer, InputParameters, run_lightcone
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.lightconers import Lightconer
from scipy.spatial.transform import Rotation

from tuesday.core.lightcones import CELLS_PER_SAMPLE, disc_pixels
from tuesday.simulators.py21cmfast import lightconers
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
    """Record which way each slice is made: "sample" first or "interpolate" first.

    Sampling first goes through tuesday's sample_then_interpolate; interpolating
    first through 21cmFAST's coeval_subselect.
    """
    called = []
    sample = lightconers.sample_then_interpolate

    def spy_sample(*args, **kw):
        called.append("sample")
        return sample(*args, **kw)

    subselect = Lightconer.coeval_subselect

    def spy_subselect(self, *args, **kw):
        called.append("interpolate")
        return subselect(self, *args, **kw)

    monkeypatch.setattr(lightconers, "sample_then_interpolate", spy_sample)
    monkeypatch.setattr(Lightconer, "coeval_subselect", spy_subselect)
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


def test_run_lightcone_matches_21cmfast(run):
    """In run_lightcone, it gives what 21cmFAST's AngularLightconer gives."""
    angular = AngularLightconer.between_redshifts(
        **_ring(32), **_lightconer_kwargs(run.inputs)
    )
    reference = construct_lightcone_from_cache(run.cache, angular)
    _assert_lightcones_close(run.lightcone, reference, QUANTITIES)


ROTATION = Rotation.from_euler("zyx", [0.3, 0.2, 0.1])


def _random_directions(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    return {
        "latitude": rng.uniform(-1.2, 1.2, n),
        "longitude": rng.uniform(0, 2 * np.pi, n),
    }


#: The lightconers compared with 21cmFAST's AngularLightconer, made from the keywords
#: of between_redshifts.
LIGHTCONERS = {
    "ring": lambda **kw: FastAngularLightconer.between_redshifts(
        **_ring(48, latitude=-0.3), **kw
    ),
    "subsamples": lambda **kw: FastAngularLightconer.between_redshifts(
        **_random_directions(4 * 24), n_subsamples=4, **kw
    ),
    "healpix-ring": lambda **kw: FastAngularLightconer.healpix(4, **kw),
    "healpix-nested": lambda **kw: FastAngularLightconer.healpix(
        4, order="nested", **kw
    ),
    "healpix-patch": lambda **kw: FastAngularLightconer.healpix(
        8,
        pixels=disc_pixels(8, 1.0 * un.rad, 0.3 * un.rad, 25 * un.deg),
        subsample_level=1,
        **kw,
    ),
}


@pytest.mark.parametrize("kind", LIGHTCONERS)
@pytest.mark.parametrize(
    ("tile_first", "rotation", "dvdr_and_rsds"),
    [
        (None, None, False),
        (True, ROTATION, False),
        (False, ROTATION, False),
        (True, ROTATION, True),
        (False, ROTATION, True),
    ],
)
def test_matches_21cmfast(run, monkeypatch, kind, tile_first, rotation, dvdr_and_rsds):
    """The lightcones equal 21cmFAST's AngularLightconer's at the same directions.

    A pixel with sub-samples equals the mean of 21cmFAST's lightcone over them. Every
    slice is made the way tile_first asks for, or, with tile_first=None, the way
    samples_first picks for the 10^3 boxes.
    """
    kw = _lightconer_kwargs(run.inputs, rotation=rotation)
    lightconer = LIGHTCONERS[kind](tile_first=tile_first, **kw)
    if dvdr_and_rsds and lightconer.n_subsamples > 1:
        pytest.skip("dv/dr and RSDs are applied to the averaged pixels")
    options = {
        "include_dvdr_in_tau21": dvdr_and_rsds,
        "apply_rsds": dvdr_and_rsds,
        "n_rsd_subcells": 2,
    }
    reference = construct_lightcone_from_cache(
        run.cache,
        AngularLightconer.between_redshifts(
            latitude=lightconer.latitude, longitude=lightconer.longitude, **kw
        ),
        **options,
    )
    for q, values in reference.lightcones.items():
        reference.lightcones[q] = np.reshape(
            values, (lightconer.n_subsamples, lightconer.n_pixels, -1)
        ).mean(axis=0)

    ways = _record_ways(monkeypatch)
    result = construct_lightcone_from_cache(run.cache, lightconer, **options)
    assert set(ways) == {"sample" if lightconer.samples_first(10**3) else "interpolate"}
    quantities = list(reference.lightcones)
    assert sorted(result.lightcones) == sorted(quantities)
    assert result.lightcones["density"].shape == (
        lightconer.n_pixels,
        len(lightconer.lc_redshifts),
    )
    _assert_lightcones_close(result, reference, quantities)


@pytest.mark.parametrize("order", [0, 3])
def test_other_orders(run, monkeypatch, order):
    """Orders 0 and 3 sample first when asked, and give 21cmFAST's lightcones.

    Without tile_first, order 3 interpolates first, which is cheaper for splines.
    """
    kw = _lightconer_kwargs(run.inputs, interpolation_order=order)
    reference = construct_lightcone_from_cache(
        run.cache, AngularLightconer.between_redshifts(**_ring(40), **kw)
    )
    default = FastAngularLightconer.between_redshifts(**_ring(40), **kw)
    assert default.samples_first(10**9) is (order == 0)
    lightconer = FastAngularLightconer.between_redshifts(
        **_ring(40), tile_first=True, **kw
    )
    ways = _record_ways(monkeypatch)
    result = construct_lightcone_from_cache(run.cache, lightconer)
    assert set(ways) == {"sample"}
    _assert_lightcones_close(result, reference, QUANTITIES)


def test_samples_first_rule():
    """Sample first when the box has more than CELLS_PER_SAMPLE cells per direction.

    Orders above 1 interpolate first, and tile_first overrides the choice.
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
        **_ring(100), interpolation_order=3, lc_distances=distances
    )
    assert not cubic.samples_first(10**9)
    cubic = FastAngularLightconer(
        **_ring(100), tile_first=True, interpolation_order=3, lc_distances=distances
    )
    assert cubic.samples_first(1)


def test_validation():
    """n_subsamples must split the directions."""
    distances = np.array([100.0]) * un.Mpc
    with pytest.raises(ValueError, match="cannot be split"):
        FastAngularLightconer(**_ring(10), n_subsamples=4, lc_distances=distances)
    with pytest.raises(ValueError, match="n_subsamples must be at least 1"):
        FastAngularLightconer(**_ring(10), n_subsamples=0, lc_distances=distances)


def test_slice_interpolator():
    """cosmotile's interpolator, with the sub-samples of each pixel, kept per slice."""
    distances = np.array([100.0]) * un.Mpc
    lightconer = FastAngularLightconer(
        **_ring(12), n_subsamples=3, lc_distances=distances
    )
    first = lightconer.slice_interpolator(50.0 * un.pixel)
    assert lightconer.slice_interpolator(50.0 * un.pixel) is first
    assert lightconer.slice_interpolator(51.0 * un.pixel) is not first
    assert first.keywords["coordinates"].shape == (3, 12)
    np.testing.assert_allclose(first.keywords["weights"], 1 / 3)


def test_healpix_without_redshifts():
    """Without min_redshift, the parameters go to the constructor."""
    lightconer = FastAngularLightconer.healpix(
        2, subsample_level=1, lc_distances=np.array([100.0, 101.0]) * un.Mpc
    )
    assert lightconer.longitude.size == 4 * 48
    assert lightconer.n_pixels == 48
    # min_redshift=None (e.g. an unset optional argument) also goes to the constructor
    same = FastAngularLightconer.healpix(
        2,
        subsample_level=1,
        min_redshift=None,
        lc_distances=np.array([100.0, 101.0]) * un.Mpc,
    )
    assert same == lightconer
