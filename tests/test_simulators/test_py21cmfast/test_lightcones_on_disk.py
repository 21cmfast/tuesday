"""Tests of the lightcones made on disk from the cache of a run."""

import shutil
from pathlib import Path

import attrs
import numpy as np
import pytest
from py21cmfast import AngularLightconer, InputParameters, RectilinearLightconer
from py21cmfast.io import h5
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.wrapper import outputs
from scipy.spatial.transform import Rotation

from tuesday.core.lightcones import LightconeStore
from tuesday.simulators.py21cmfast.lightconers import healpix_lightconer
from tuesday.simulators.py21cmfast.lightcones import (
    CachedFields,
    construct_lightcone_from_cache,
    lightconer_metadata,
    make_lightcones_on_disk,
)

NODES = [10.0, 9.0, 8.0, 7.0, 6.0]


def random_cache(cachedir: Path) -> RunCache:
    """Write a cache of random boxes (no 21cmFAST run), for 5 node redshifts."""
    inputs = InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=NODES
    ).evolve_input_structs(BOX_LEN=48, HII_DIM=16, DIM=48, KEEP_3D_VELOCITIES=True)
    cache = RunCache.from_inputs(inputs, cache=OutputCache(cachedir))
    rng = np.random.default_rng(3)

    def write(box, path):
        box._init_arrays()
        for name, array in box.arrays.items():
            values = rng.normal(size=array.shape).astype(array.dtype)
            setattr(box, name, array.with_value(values))
        for name in box._struct.primitive_fields:
            setattr(box, name, 0.0)
        h5.write_output_to_hdf5(box, path)

    write(outputs.InitialConditions.new(inputs=inputs), cache.InitialConditions)
    for kind, files in attrs.asdict(cache, recurse=False).items():
        if isinstance(files, dict):
            for z, path in files.items():
                write(getattr(outputs, kind).new(redshift=z, inputs=inputs), path)
    return cache


def same_metadata(a: dict, b: dict) -> bool:
    """Whether two descriptions of lightconers are equal."""
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> RunCache:
    """A cache of random boxes."""
    return random_cache(tmp_path_factory.mktemp("cache") / "cache")


def lightconers(cache: RunCache) -> dict:
    """A rectilinear, an angular and a HEALPix lightconer, between z = 6.2 and 9.8."""
    options = cache.inputs.simulation_options
    between = {
        "min_redshift": 6.2,
        "max_redshift": 9.8,
        "resolution": options.cell_size,
    }
    return {
        "rect": RectilinearLightconer.between_redshifts(**between),
        "angular": AngularLightconer.like_rectilinear(
            simulation_options=options, max_redshift=9.5, match_at_z=6.5
        ),
        "healpix": healpix_lightconer(nside=2, subsample_level=1, **between),
    }


def test_cached_fields(cache):
    """Each field is read from its output struct, at each node redshift."""
    source = CachedFields(cache)
    assert source.node_redshifts == NODES
    assert source.kinds["density"] == "PerturbedField"
    assert source.kinds["brightness_temp"] == "BrightnessTemp"
    assert source.kinds["z_reion"] == "IonizedBox"
    coeval = cache.get_coeval_at_z(8.0)
    for field in ("density", "velocity_x", "brightness_temp", "neutral_fraction"):
        np.testing.assert_array_equal(source.read(8.0, field), getattr(coeval, field))
    assert source.has(8.0, ["density", "z_reion"])
    with pytest.raises(KeyError, match="not a coeval field"):
        source.has(8.0, ["Tk_box"])
    with pytest.raises(KeyError, match="not a node redshift"):
        source.read(8.5, "density")


@pytest.mark.parametrize("name", ["rect", "angular", "healpix"])
def test_matches_in_memory(cache, tmp_path, name):
    """The lightcones on disk are those made in memory from the cache."""
    lightconer = lightconers(cache)[name]
    quantities = ("density", "z_reion", "los_velocity")
    paths = make_lightcones_on_disk({name: lightconer}, quantities, cache, tmp_path)
    assert set(paths[name]) == set(quantities)
    reference = construct_lightcone_from_cache(
        cache,
        attrs.evolve(lightconer, quantities=quantities),
        global_quantities=("density", "z_reion"),
    )
    for q in quantities:
        assert paths[name][q] == tmp_path / name / f"{q}.h5"
        with LightconeStore.open(paths[name][q]) as store:
            assert store.complete
            assert store.done.all()
            np.testing.assert_array_equal(store.lightcone[()], reference.lightcones[q])
            np.testing.assert_allclose(
                store.distances, lightconer.lc_distances, rtol=1e-12
            )
            np.testing.assert_array_equal(store.node_redshifts, NODES)
            if q == "los_velocity":
                assert np.all(np.isnan(store.global_mean))
            else:
                # the means at the nodes that bound slices (here, all of them)
                np.testing.assert_allclose(
                    store.global_mean, reference.global_quantities[q], rtol=1e-6
                )
            assert same_metadata(store.metadata, lightconer_metadata(lightconer))


def test_los_velocity_alone(cache, tmp_path):
    """los_velocity alone is made, without a file for the field made with it."""
    lightconer = lightconers(cache)["healpix"]
    paths = make_lightcones_on_disk({"hp": lightconer}, "los_velocity", cache, tmp_path)
    assert paths == {"hp": {"los_velocity": tmp_path / "hp" / "los_velocity.h5"}}
    assert sorted(p.name for p in (tmp_path / "hp").iterdir()) == ["los_velocity.h5"]
    reference = construct_lightcone_from_cache(
        cache, attrs.evolve(lightconer, quantities=("density", "los_velocity"))
    )
    with LightconeStore.open(paths["hp"]["los_velocity"]) as store:
        np.testing.assert_array_equal(
            store.lightcone[()], reference.lightcones["los_velocity"]
        )


class CountingFields(CachedFields):
    """Cached fields that count the boxes read at each node redshift."""

    def __init__(self, cache):
        super().__init__(cache)
        self.reads = []

    def read(self, redshift, field):
        """Read one field, and count it."""
        self.reads.append((redshift, field))
        return super().read(redshift, field)


def test_partial_and_resume(cache, tmp_path):
    """Missing boxes stop the job, or are skipped; a later call fills the rest."""
    copy = random_cache(tmp_path / "cache")
    lcs = lightconers(copy)
    full = make_lightcones_on_disk(lcs, "density", copy, tmp_path / "full")

    # Move away the density at z = 8, as if the run had not reached it yet.
    moved = Path(copy.PerturbedField[8.0])
    moved.rename(moved.with_suffix(".away"))
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="partial=True"):
        make_lightcones_on_disk(lcs, "density", copy, out)
    assert not out.exists()

    source = CountingFields(copy)
    paths = make_lightcones_on_disk(lcs, "density", source, out, partial=True)
    assert {z for z, _ in source.reads} == {10.0, 9.0, 7.0, 6.0}
    for name, by_quantity in paths.items():
        with (
            LightconeStore.open(by_quantity["density"]) as store,
            LightconeStore.open(full[name]["density"]) as reference,
        ):
            np.testing.assert_array_equal(store.done, [True, False, False, True])
            assert not store.complete
            lc, ref = store.lightcone[()], reference.lightcone[()]
            filled = ~np.isnan(lc).all(axis=tuple(range(lc.ndim - 1)))
            np.testing.assert_array_equal(lc[..., filled], ref[..., filled])
            assert 0 < filled.sum() < filled.size
            assert np.isnan(store.global_mean[2])
            assert not np.isnan(store.global_mean[[0, 1, 3, 4]]).any()

    # Once the box is there, only the intervals left are filled.
    moved.with_suffix(".away").rename(moved)
    source = CountingFields(copy)
    make_lightcones_on_disk(lcs, "density", source, out)
    assert {z for z, _ in source.reads} == {9.0, 8.0, 7.0}
    for name, by_quantity in paths.items():
        with (
            LightconeStore.open(by_quantity["density"]) as store,
            LightconeStore.open(full[name]["density"]) as reference,
        ):
            assert store.complete
            np.testing.assert_array_equal(store.lightcone[()], reference.lightcone[()])

    # Nothing is left to do, and nothing is read.
    source = CountingFields(copy)
    make_lightcones_on_disk(lcs, "density", source, out)
    assert source.reads == []
    shutil.rmtree(tmp_path / "cache")


def test_other_lightconer(cache, tmp_path):
    """A file made with another lightconer is not resumed, unless clobbered."""
    lightconer = lightconers(cache)["healpix"]
    make_lightcones_on_disk({"hp": lightconer}, "density", cache, tmp_path)
    rotated = attrs.evolve(lightconer, rotation=Rotation.from_euler("z", 0.3))
    np.testing.assert_allclose(
        lightconer_metadata(rotated)["rotation"],
        Rotation.from_euler("z", 0.3).as_matrix(),
    )
    with pytest.raises(ValueError, match="metadata 'rotation'"):
        make_lightcones_on_disk({"hp": rotated}, "density", cache, tmp_path)
    paths = make_lightcones_on_disk(
        {"hp": rotated}, "density", cache, tmp_path, clobber=True
    )
    reference = construct_lightcone_from_cache(
        cache, attrs.evolve(rotated, quantities=("density",))
    )
    with LightconeStore.open(paths["hp"]["density"]) as store:
        assert same_metadata(store.metadata, lightconer_metadata(rotated))
        np.testing.assert_array_equal(
            store.lightcone[()], reference.lightcones["density"]
        )


def test_lightconer_metadata(cache):
    """The options that change the slices, with large arrays hashed."""
    lcs = lightconers(cache)
    rect = lightconer_metadata(lcs["rect"])
    assert rect == {
        "lightconer": "py21cmfast.lightconers.RectilinearLightconer",
        "interp_kinds": '{"z_reion": "mean_max"}',
        "index_offset": lcs["rect"].index_offset,
        "line_of_sight_axis": lcs["rect"].line_of_sight_axis,
    }
    hp = lightconer_metadata(lcs["healpix"])
    assert hp["lightconer"].endswith("FastAngularLightconer")
    assert hp["latitude"].startswith("sha256:")
    assert hp["n_subsamples"] == 4
    assert hp["rotation"] == "None"
    assert "tile_first" not in hp
    assert "threads" not in hp
    faster = attrs.evolve(lcs["healpix"], tile_first=True, threads=4)
    assert same_metadata(lightconer_metadata(faster), hp)
    other = healpix_lightconer(
        nside=2,
        subsample_level=1,
        order="nested",
        min_redshift=6.2,
        max_redshift=9.8,
        resolution=cache.inputs.simulation_options.cell_size,
    )
    assert lightconer_metadata(other)["latitude"] != hp["latitude"]


def test_errors(cache, tmp_path):
    """Quantities that are not fields, and lightcones beyond the nodes."""
    lcs = lightconers(cache)
    with pytest.raises(KeyError, match="not a coeval field"):
        make_lightcones_on_disk(lcs, "Tk_box", cache, tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        make_lightcones_on_disk(lcs, ["density", "density"], cache, tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        make_lightcones_on_disk(lcs, [], cache, tmp_path)
    far = RectilinearLightconer.between_redshifts(
        min_redshift=6.2,
        max_redshift=11.0,
        resolution=cache.inputs.simulation_options.cell_size,
    )
    with pytest.raises(ValueError, match="not compatible"):
        make_lightcones_on_disk({"far": far}, "density", cache, tmp_path)
    assert not any(tmp_path.iterdir())
