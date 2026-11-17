"""Tests of the functions to create lightcones from output cache."""

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import attrs
import numpy as np
import pytest
from astropy import units as un
from py21cmfast import (
    AngularLightconer,
    InputParameters,
    RectilinearLightconer,
    run_lightcone,
)
from py21cmfast.drivers.lightcone import LightCone
from py21cmfast.io import h5
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.wrapper import outputs

from tuesday.simulators.py21cmfast.lightcones import (
    construct_lightcone_from_cache,
    construct_lightcone_from_coevals,
    construct_lightcone_from_filelist,
)


def create_mock_cache_output(cachedir: Path, ics: bool = False) -> RunCache:
    """Create a mock RunCache object with mock output data.

    This data is mocked up fast -- we don't even use 21cmfast here, just whack some
    random data in each of the required arrays, and write to disk.
    """
    cachedir.mkdir()
    inputs = InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=[10, 9, 8, 7, 6]
    ).evolve_input_structs(BOX_LEN=100, HII_DIM=50, DIM=100, KEEP_3D_VELOCITIES=True)
    cache = RunCache.from_inputs(inputs, cache=OutputCache(cachedir))

    for fldname, fld in attrs.asdict(cache, recurse=False).items():
        if isinstance(fld, dict):
            for z, fname in fld.items():
                o = getattr(outputs, fldname).new(redshift=z, inputs=inputs)
                o._init_arrays()

                # Go through each array and set it to be "computed" so we can trick
                # the writer into writing it out to file.
                for k, v in o.arrays.items():
                    setattr(o, k, v.with_value(o.get(k)))

                # Mock the primitive fields as well...
                for fld in o._struct.primitive_fields:
                    setattr(o, fld, 0.0)

                h5.write_output_to_hdf5(o, fname)
        elif fldname == "InitialConditions":
            o = outputs.InitialConditions.new(inputs=inputs)
            o._init_arrays()
            for k, v in o.arrays.items():
                setattr(o, k, v.with_value(o.get(k)))
            h5.write_output_to_hdf5(o, fld)

    return cache


def test_construct_rect_lightcone_from_cache(tmp_path: Path):
    """Test the construction of a lightcone from a cache."""

    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)

    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6.5,
        max_redshift=9.5,
        resolution=cache.inputs.simulation_options.BOX_LEN
        * un.Mpc
        / cache.inputs.simulation_options.HII_DIM,
        quantities=("density", "brightness_temp"),
    )

    lightcone = construct_lightcone_from_cache(
        cache,
        lightconer,
        global_quantities=("log10_mturn_acg", "log10_mturn_mcg", "brightness_temp"),
    )

    assert isinstance(lightcone, LightCone)

    # Check that the data is stored in the lightcone object
    assert lightcone.lightcones["brightness_temp"].shape == (
        50,
        50,
        len(lightconer.lc_redshifts),
    )
    assert lightcone.lightcones["density"].shape == (
        50,
        50,
        len(lightconer.lc_redshifts),
    )

    assert lightcone.global_quantities["log10_mturn_acg"].shape == (
        len(cache.inputs.node_redshifts),
    )
    assert lightcone.global_quantities["log10_mturn_mcg"].shape == (
        len(cache.inputs.node_redshifts),
    )
    assert lightcone.global_quantities["brightness_temp"].shape == (
        len(cache.inputs.node_redshifts),
    )


def test_construct_ang_lightcone_from_cache(tmp_path: Path):
    """Test the construction of a lightcone from a cache."""
    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)

    lightconer = AngularLightconer.like_rectilinear(
        simulation_options=cache.inputs.simulation_options,
        max_redshift=9.5,
        match_at_z=6.5,
        quantities=("density", "brightness_temp"),
    )

    lightcone = construct_lightcone_from_cache(cache, lightconer)

    assert isinstance(lightcone, LightCone)
    assert lightcone.lightcones["brightness_temp"].shape == (
        2500,
        len(lightconer.lc_redshifts),
    )
    assert lightcone.lightcones["density"].shape == (2500, len(lightconer.lc_redshifts))


def test_construct_rect_lightcone_from_filelist(tmp_path: Path):
    """Test the construction of a lightcone from a cache."""

    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)

    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6.5,
        max_redshift=9.5,
        resolution=cache.inputs.simulation_options.BOX_LEN
        * un.Mpc
        / cache.inputs.simulation_options.HII_DIM,
        quantities=("density",),
    )

    filelist = [
        cache.PerturbedField[z]
        for z in sorted(cache.inputs.node_redshifts, reverse=True)
    ]

    lightcone = construct_lightcone_from_filelist(filelist, lightconer)

    assert isinstance(lightcone, LightCone)

    # Check that the data is stored in the lightcone object
    assert lightcone.lightcones["density"].shape == (
        50,
        50,
        len(lightconer.lc_redshifts),
    )


def test_construct_ang_lightcone_from_filelist(tmp_path: Path):
    """Test the construction of a lightcone from a cache."""
    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)

    lightconer = AngularLightconer.like_rectilinear(
        simulation_options=cache.inputs.simulation_options,
        max_redshift=9.5,
        match_at_z=6.5,
        quantities=("density",),
    )

    filelist = [
        cache.PerturbedField[z]
        for z in sorted(cache.inputs.node_redshifts, reverse=True)
    ]

    lightcone = construct_lightcone_from_filelist(filelist, lightconer)

    assert isinstance(lightcone, LightCone)
    assert lightcone.lightcones["density"].shape == (2500, len(lightconer.lc_redshifts))


def test_exceptions(tmp_path: Path):
    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)
    cache.PerturbedField[cache.inputs.node_redshifts[-1]].unlink()
    lightconer = AngularLightconer.like_rectilinear(
        simulation_options=cache.inputs.simulation_options,
        max_redshift=9.6,
        match_at_z=6.0,
        quantities=("density", "brightness_temp"),
    )

    with pytest.raises(ValueError, match="The cache specified is not complete!"):
        construct_lightcone_from_cache(cache, lightconer)


class CoevalLike(SimpleNamespace):
    """A minimal coeval box that can be weakly referenced."""

    __slots__ = ("__weakref__",)
    __hash__ = object.__hash__


def density_coevals(cache: RunCache, alive: list | None = None):
    """Yield the cached density boxes as coeval-like objects, by decreasing redshift.

    If ``alive`` is given, the number of boxes still in memory when each box is
    created is appended to it.
    """
    boxes = weakref.WeakSet()
    for z in sorted(cache.inputs.node_redshifts, reverse=True):
        if alive is not None:
            gc.collect()
            alive.append(len(boxes))
        box = h5.read_output_struct(cache.PerturbedField[z])
        coeval = CoevalLike(
            redshift=box.redshift,
            simulation_options=box.inputs.simulation_options,
            cosmo_params=box.inputs.cosmo_params,
            density=box.get("density"),
        )
        boxes.add(coeval)
        yield coeval


@pytest.mark.parametrize("kind", ["rectilinear", "angular"])
def test_construct_lightcone_from_coevals_matches_filelist(tmp_path: Path, kind: str):
    cache = create_mock_cache_output(tmp_path / "cache")
    options = cache.inputs.simulation_options
    if kind == "rectilinear":
        lightconer = RectilinearLightconer.between_redshifts(
            min_redshift=6.5,
            max_redshift=9.5,
            resolution=options.BOX_LEN * un.Mpc / options.HII_DIM,
            quantities=("density",),
        )
    else:
        lightconer = AngularLightconer.like_rectilinear(
            simulation_options=options,
            max_redshift=9.5,
            match_at_z=6.5,
            quantities=("density",),
        )
    filelist = [
        cache.PerturbedField[z]
        for z in sorted(cache.inputs.node_redshifts, reverse=True)
    ]

    from_files = construct_lightcone_from_filelist(filelist, lightconer)
    from_coevals = construct_lightcone_from_coevals(
        density_coevals(cache), lightconer, cache.inputs
    )

    assert isinstance(from_coevals, LightCone)
    np.testing.assert_array_equal(
        from_coevals.lightcones["density"], from_files.lightcones["density"]
    )


def test_construct_lightcone_from_coevals_keeps_two_in_memory(tmp_path: Path):
    # Each box is interpolated with the previous one as soon as it is produced, so
    # at most the previous box is still alive when the next one is created.
    cache = create_mock_cache_output(tmp_path / "cache")
    options = cache.inputs.simulation_options
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6.5,
        max_redshift=9.5,
        resolution=options.BOX_LEN * un.Mpc / options.HII_DIM,
        quantities=("density",),
    )
    alive = []
    construct_lightcone_from_coevals(
        density_coevals(cache, alive), lightconer, cache.inputs
    )

    assert alive == [0, 1, 1, 1, 1]


def test_construct_lightcone_from_coevals_checks_redshift_range(tmp_path: Path):
    cache = create_mock_cache_output(tmp_path / "cache")
    options = cache.inputs.simulation_options
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6.5,
        max_redshift=11.0,
        resolution=options.BOX_LEN * un.Mpc / options.HII_DIM,
        quantities=("density",),
    )
    with pytest.raises(ValueError, match="not compatible"):
        construct_lightcone_from_coevals(
            density_coevals(cache), lightconer, cache.inputs
        )


# --- The dv/dr correction and RSDs, against a (tiny) 21cmFAST run --------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> SimpleNamespace:
    """Run a tiny 21cmFAST lightcone with and without dv/dr and RSDs; keep its cache."""
    cache_dir = tmp_path_factory.mktemp("cache")
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
    )
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6.1,
        max_redshift=6.9,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "density"),
    )
    references = {
        (dvdr, rsds): run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            cache=OutputCache(cache_dir),
            write=True,
            progressbar=False,
            include_dvdr_in_tau21=dvdr,
            apply_rsds=rsds,
            n_rsd_subcells=4,
        )
        for dvdr, rsds in [(False, False), (True, False), (False, True), (True, True)]
    }
    return SimpleNamespace(
        cache=RunCache.from_inputs(inputs, OutputCache(cache_dir)),
        lightconer=lightconer,
        references=references,
    )


@pytest.mark.parametrize(
    ("dvdr", "rsds"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_construct_lightcone_from_cache_matches_run_lightcone(
    tiny_run, dvdr: bool, rsds: bool
):
    # The dv/dr correction and the RSDs are applied as run_lightcone applies them.
    reference = tiny_run.references[(dvdr, rsds)]
    lightcone = construct_lightcone_from_cache(
        tiny_run.cache,
        tiny_run.lightconer,
        include_dvdr_in_tau21=dvdr,
        apply_rsds=rsds,
        n_rsd_subcells=4,
    )

    np.testing.assert_array_equal(
        lightcone.lightcone_distances, reference.lightcone_distances
    )
    assert set(lightcone.lightcones) == set(reference.lightcones)
    for name, values in reference.lightcones.items():
        np.testing.assert_allclose(lightcone.lightcones[name], values, rtol=1e-6)
    if dvdr:
        assert not np.allclose(
            reference.lightcones["brightness_temp"],
            tiny_run.references[(False, rsds)].lightcones["brightness_temp"],
        )
