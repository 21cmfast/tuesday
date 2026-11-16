"""Tests of the angular lightconer of tuesday in the lightcone loops of 21cmFAST."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from py21cmfast import InputParameters, run_lightcone
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.lightconers import LightconerProtocol, context_from_inputs
from scipy.spatial.transform import Rotation

from tuesday.core.lightcones import CELLS_PER_SAMPLE, AngularLightconer, disc_pixels
from tuesday.core.lightcones import lightconers as core_lightconers
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
    """A tiny cached run with 3D velocities, made with tuesday's angular lightconer.

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
    lightconer = AngularLightconer.between_redshifts(
        **_ring(32), tile_first=True, **_lightconer_kwargs(inputs)
    )
    assert isinstance(lightconer, LightconerProtocol)
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
    shell_slice = core_lightconers.shell_slice

    def spy(*args, sample_first, **kw):
        called.append("sample" if sample_first else "interpolate")
        return shell_slice(*args, sample_first=sample_first, **kw)

    monkeypatch.setattr(core_lightconers, "shell_slice", spy)
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


def _reference(**kw) -> AngularLightconer:
    """The plain lightconer: no sub-samples, and the boxes interpolated first.

    This makes each slice the way the AngularLightconer of 21cmFAST v4.2 did.
    """
    return AngularLightconer.between_redshifts(tile_first=False, **kw)


def test_run_lightcone_matches_cache(run):
    """run_lightcone gives the lightcone that is made from the cache."""
    angular = _reference(**_ring(32), **_lightconer_kwargs(run.inputs))
    reference = construct_lightcone_from_cache(run.cache, angular)
    _assert_lightcones_close(run.lightcone, reference, QUANTITIES)


ROTATION = Rotation.from_euler("zyx", [0.3, 0.2, 0.1])


def _random_directions(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    return {
        "latitude": rng.uniform(-1.2, 1.2, n),
        "longitude": rng.uniform(0, 2 * np.pi, n),
    }


#: The lightconers compared with the plain one, made from the keywords of
#: between_redshifts.
LIGHTCONERS = {
    "ring": lambda **kw: AngularLightconer.between_redshifts(
        **_ring(48, latitude=-0.3), **kw
    ),
    "subsamples": lambda **kw: AngularLightconer.between_redshifts(
        **_random_directions(4 * 24), n_subsamples=4, **kw
    ),
    "healpix-ring": lambda **kw: AngularLightconer.healpix(4, **kw),
    "healpix-nested": lambda **kw: AngularLightconer.healpix(4, order="nested", **kw),
    "healpix-patch": lambda **kw: AngularLightconer.healpix(
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
def test_matches_plain(run, monkeypatch, kind, tile_first, rotation, dvdr_and_rsds):
    """The lightcones equal the plain lightconer's at the same directions.

    A pixel with sub-samples equals the mean of the plain lightcone over them. Every
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
        _reference(latitude=lightconer.latitude, longitude=lightconer.longitude, **kw),
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
    """Orders 0 and 3 sample first when asked, and give the plain lightcones.

    Without tile_first, order 3 interpolates first, which is cheaper for splines.
    """
    kw = _lightconer_kwargs(run.inputs, interpolation_order=order)
    reference = construct_lightcone_from_cache(run.cache, _reference(**_ring(40), **kw))
    default = AngularLightconer.between_redshifts(**_ring(40), **kw)
    assert default.samples_first(10**9) is (order == 0)
    lightconer = AngularLightconer.between_redshifts(**_ring(40), tile_first=True, **kw)
    ways = _record_ways(monkeypatch)
    result = construct_lightcone_from_cache(run.cache, lightconer)
    assert set(ways) == {"sample"}
    _assert_lightcones_close(result, reference, QUANTITIES)


def test_samples_first_rule():
    """Sample first when the box has more than CELLS_PER_SAMPLE cells per direction.

    Orders above 1 interpolate first, and tile_first overrides the choice.
    """
    distances = np.array([100.0]) * un.Mpc
    lightconer = AngularLightconer(**_ring(100), lc_distances=distances)
    assert lightconer.samples_first(100 * CELLS_PER_SAMPLE + 1)
    assert not lightconer.samples_first(100 * CELLS_PER_SAMPLE)
    for tile_first in (True, False):
        forced = AngularLightconer(
            **_ring(100), tile_first=tile_first, lc_distances=distances
        )
        assert forced.samples_first(1) is tile_first
        assert forced.samples_first(10**9) is tile_first
    cubic = AngularLightconer(
        **_ring(100), interpolation_order=3, lc_distances=distances
    )
    assert not cubic.samples_first(10**9)
    cubic = AngularLightconer(
        **_ring(100), tile_first=True, interpolation_order=3, lc_distances=distances
    )
    assert cubic.samples_first(1)


def test_validation():
    """n_subsamples must split the directions."""
    distances = np.array([100.0]) * un.Mpc
    with pytest.raises(ValueError, match="cannot be split"):
        AngularLightconer(**_ring(10), n_subsamples=4, lc_distances=distances)
    with pytest.raises(ValueError, match="n_subsamples must be at least 1"):
        AngularLightconer(**_ring(10), n_subsamples=0, lc_distances=distances)


def test_slice_interpolator(run, monkeypatch):
    """cosmotile's interpolator has the sub-samples, and is made once per slice."""
    distances = np.array([100.0]) * un.Mpc
    lightconer = AngularLightconer(**_ring(12), n_subsamples=3, lc_distances=distances)
    first = lightconer.slice_interpolator(50.0 * un.pixel)
    assert first.keywords["coordinates"].shape == (3, 12)
    np.testing.assert_allclose(first.keywords["weights"], 1 / 3)
    # A distance without units is in cells.
    same = lightconer.slice_interpolator(50.0)
    np.testing.assert_array_equal(
        same.keywords["coordinates"], first.keywords["coordinates"]
    )

    # In a run, the interpolator of a slice is shared by all its quantities.
    made = []
    make = AngularLightconer.slice_interpolator

    def counting(self, distance):
        made.append(float(distance))
        return make(self, distance)

    monkeypatch.setattr(AngularLightconer, "slice_interpolator", counting)
    lightconer = AngularLightconer.between_redshifts(
        **_ring(12), **_lightconer_kwargs(run.inputs)
    )
    result = construct_lightcone_from_cache(run.cache, lightconer)
    n_slices = result.lightcones["density"].shape[-1]
    assert len(lightconer.quantities) > 1
    assert len(made) == len(set(made)) == n_slices


def test_shape_and_fields(run):
    """What 21cmFAST asks of the lightconer before a run."""
    lightconer = AngularLightconer.healpix(
        2, subsample_level=1, **_lightconer_kwargs(run.inputs)
    )
    session = lightconer.start(context_from_inputs(run.inputs))
    assert session.shape == (48, len(lightconer.lc_distances))
    assert session.resume_depth == 1
    assert lightconer.required_fields() == (
        "brightness_temp",
        "density",
        "z_reion",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    )
    assert lightconer.los_velocity_rms_factor == 1.0

    other = context_from_inputs(
        run.inputs.evolve_input_structs(SIGMA_8=run.inputs.cosmo_params.SIGMA_8 * 1.1)
    )
    assert other.cosmo == run.inputs.cosmo_params.cosmo  # sigma8 is not in it
    wrong = AngularLightconer.healpix(
        2,
        **_lightconer_kwargs(
            run.inputs, cosmo=run.inputs.cosmo_params.cosmo.clone(H0=50)
        ),
    )
    with pytest.raises(ValueError, match="The cosmology of the lightconer"):
        wrong.start(context_from_inputs(run.inputs))


def test_fast_angular_lightconer_is_deprecated(run):
    """The old name still makes the same lightconer, with 21cmFAST's cosmology."""
    kw = _lightconer_kwargs(run.inputs)
    with pytest.warns(DeprecationWarning, match="FastAngularLightconer is deprecated"):
        old = FastAngularLightconer.between_redshifts(**_ring(16), n_subsamples=2, **kw)
    assert isinstance(old, AngularLightconer)
    new = AngularLightconer.between_redshifts(**_ring(16), n_subsamples=2, **kw)
    np.testing.assert_allclose(old.lc_distances, new.lc_distances)
    assert old.n_pixels == new.n_pixels == 8

    from py21cmfast.wrapper.inputs import Planck18

    with pytest.warns(DeprecationWarning, match="FastAngularLightconer"):
        default = FastAngularLightconer(
            **_ring(4), lc_distances=np.array([100.0]) * un.Mpc
        )
    assert default.cosmo is Planck18


def test_healpix_without_redshifts():
    """Without min_redshift, the parameters go to the constructor."""
    lightconer = AngularLightconer.healpix(
        2, subsample_level=1, lc_distances=np.array([100.0, 101.0]) * un.Mpc
    )
    assert lightconer.longitude.size == 4 * 48
    assert lightconer.n_pixels == 48
    # min_redshift=None (e.g. an unset optional argument) also goes to the constructor
    same = AngularLightconer.healpix(
        2,
        subsample_level=1,
        min_redshift=None,
        lc_distances=np.array([100.0, 101.0]) * un.Mpc,
    )
    assert same == lightconer
