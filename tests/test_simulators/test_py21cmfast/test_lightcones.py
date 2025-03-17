"""Tests of the functions to create lightcones from output cache."""

from pathlib import Path

import attrs
from astropy import units as un
from py21cmfast import AngularLightconer, InputParameters, RectilinearLightconer
from py21cmfast.drivers.lightcone import LightCone
from py21cmfast.io import h5
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.wrapper import outputs

from tuesday.simulators.py21cmfast.lightcones import construct_lightcone_from_cache


def create_mock_cache_output(cachedir: Path, ics: bool = False) -> RunCache:
    """Create a mock RunCache object with mock output data.

    This data is mocked up fast -- we don't even use 21cmfast here, just whack some
    random data in each of the required arrays, and write to disk.
    """
    cachedir.mkdir()
    inputs = InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=[10, 9, 8, 7, 6]
    ).evolve_input_structs(BOX_LEN=100, HII_DIM=50, DIM=100, APPLY_RSDS=False)
    cache = RunCache.from_inputs(inputs, cache=OutputCache(cachedir))

    for fldname, fld in attrs.asdict(cache, recurse=False).items():
        if isinstance(fld, dict):
            for z, fname in fld.items():
                o = getattr(outputs, fldname).new(redshift=z, inputs=inputs)
                o._init_arrays()

                # Go through each array and set it to be "computed" so we can trick
                # the writer into writing it out to file.
                for k, v in o.arrays.items():
                    setattr(o, k, v.with_value(v.value))

                # Mock the primitive fields as well...
                for fld in o.struct.primitive_fields:
                    setattr(o, fld, 0.0)

                h5.write_output_to_hdf5(o, fname)
        elif fldname == "InitialConditions":
            o = outputs.InitialConditions.new(inputs=inputs)
            o._init_arrays()
            for k, v in o.arrays.items():
                setattr(o, k, v.with_value(v.value))
            h5.write_output_to_hdf5(o, fld)

    return cache


def test_construct_rect_lightcone_from_cache(tmp_path: Path):
    """Test the construction of a lightcone from a cache."""

    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)

    lightconer = RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=6,
        max_redshift=10,
        resolution=cache.inputs.user_params.BOX_LEN
        * un.Mpc
        / cache.inputs.user_params.HII_DIM,
        quantities=("density", "brightness_temp"),
    )

    lightcone = construct_lightcone_from_cache(cache, lightconer)

    assert isinstance(lightcone, LightCone)

    # Check that the data is stored in the lightcone object
    assert lightcone.lightcones["brightness_temp"].shape == (50, 50, 607)
    assert lightcone.lightcones["density"].shape == (50, 50, 607)
    assert lightcone.global_quantities == {}


def test_construct_ang_lightcone_from_cache(tmp_path: Path):
    """Test the construction of a lightcone from a cache."""
    cachedir = tmp_path / "cache"
    cache = create_mock_cache_output(cachedir)

    lightconer = AngularLightconer.like_rectilinear(
        user_params=cache.inputs.user_params,
        max_redshift=10,
        match_at_z=6.0,
        quantities=("density", "brightness_temp"),
    )

    lightcone = construct_lightcone_from_cache(cache, lightconer)

    assert isinstance(lightcone, LightCone)
    assert lightcone.lightcones["brightness_temp"].shape == (2500, 607)
    assert lightcone.lightcones["density"].shape == (2500, 607)
    assert lightcone.global_quantities == {}
