"""Tests of the lightcones made on disk from the cache of a run."""

import shutil
from pathlib import Path

import attrs
import numpy as np
import pytest
from py21cmfast import AngularLightconer, InputParameters, RectilinearLightconer
from py21cmfast.drivers.lightcone import AngularLightcone, LightCone
from py21cmfast.io import h5
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.wrapper import outputs
from py21cmfast.wrapper.classy_interface import run_classy
from scipy.spatial.transform import Rotation

from tuesday.core.lightcones import LightconeStore
from tuesday.simulators.py21cmfast.lightconers import FastAngularLightconer
from tuesday.simulators.py21cmfast.lightcones import (
    DVDR,
    CachedFields,
    apply_rsds_on_disk,
    construct_lightcone_from_cache,
    include_dvdr_on_disk,
    lightconer_metadata,
    link_lightcone_file,
    make_lightcones_on_disk,
    rsd_lightconer,
)

NODES = [10.0, 9.0, 8.0, 7.0, 6.0]


def random_cache(cachedir: Path, ts: bool = False) -> RunCache:
    """Write a cache of random boxes (no 21cmFAST run), for 5 node redshifts."""
    inputs = InputParameters.from_template(
        "simple",
        random_seed=1,
        node_redshifts=NODES,
        USE_TS_FLUCT=ts,
        Z_HEAT_MAX=NODES[0],
    ).evolve_input_structs(BOX_LEN=48, HII_DIM=16, DIM=48, KEEP_3D_VELOCITIES=True)
    cache = RunCache.from_inputs(inputs, cache=OutputCache(cachedir))
    rng = np.random.default_rng(3)

    def write(box, path):
        box._init_arrays()
        for name, array in box.arrays.items():
            values = rng.normal(size=array.shape)
            if name.startswith("velocity"):
                # about 3000 km/s (in Mpc/s), which moves the field by about a cell
                values *= 1e-16
            elif name == "tau_21":
                values = np.abs(values)
            values = values.astype(array.dtype)
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


@pytest.fixture(scope="module")
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
        "healpix": FastAngularLightconer.healpix(nside=2, subsample_level=1, **between),
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
def test_matches_in_memory(cache, lightconers, tmp_path, name):
    """The lightcones on disk are those made in memory from the cache."""
    lightconer = lightconers[name]
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


def test_los_velocity_alone(cache, lightconers, tmp_path):
    """los_velocity alone is made, without a file for the field made with it."""
    lightconer = lightconers["healpix"]
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


def test_partial_and_resume(lightconers, tmp_path):
    """Missing boxes stop the job, or are skipped; a later call fills the rest."""
    copy = random_cache(tmp_path / "cache")
    lcs = lightconers
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


def test_other_lightconer(cache, lightconers, tmp_path):
    """A file made with another lightconer is not resumed, unless clobbered."""
    lightconer = lightconers["healpix"]
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


def test_lightconer_metadata(cache, lightconers):
    """The options that change the slices, with large arrays hashed."""
    lcs = lightconers
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
    faster = attrs.evolve(lcs["healpix"], tile_first=True)
    assert same_metadata(lightconer_metadata(faster), hp)
    other = FastAngularLightconer.healpix(
        nside=2,
        subsample_level=1,
        order="nested",
        min_redshift=6.2,
        max_redshift=9.8,
        resolution=cache.inputs.simulation_options.cell_size,
    )
    assert lightconer_metadata(other)["latitude"] != hp["latitude"]


def test_errors(cache, lightconers, tmp_path):
    """Quantities that are not fields, and lightcones beyond the nodes."""
    lcs = lightconers
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


# --- The dv/dr correction and the 21cmFAST lightcone file -----------------------------


@pytest.fixture(scope="module")
def on_disk(tmp_path_factory):
    """Make lightcones on disk of what the dv/dr correction needs, from random boxes.

    They are made once for each value of USE_TS_FLUCT.
    """
    made = {}

    def make(ts: bool, lightconers: dict):
        if ts not in made:
            tmp = tmp_path_factory.mktemp("dvdr")
            cache = random_cache(tmp / "cache", ts=ts)
            quantities = ["brightness_temp", "los_velocity"]
            if ts:
                quantities.append("tau_21")
            make_lightcones_on_disk(lightconers, quantities, cache, tmp / "out")
            made[ts] = cache, tmp / "out"
        return made[ts]

    return make


@pytest.mark.parametrize("ts", [False, True], ids=["no_ts", "ts"])
@pytest.mark.parametrize("name", ["rect", "angular", "healpix"])
def test_include_dvdr_on_disk(on_disk, lightconers, ts, name):
    """The correction of run_lightcone, in blocks of lines of sight."""
    cache, out = on_disk(ts, lightconers)
    reference = construct_lightcone_from_cache(
        cache,
        attrs.evolve(lightconers[name], quantities=("brightness_temp",)),
        include_dvdr_in_tau21=True,
    )
    # a few lines of sight at a time
    path = include_dvdr_on_disk(out / name, cache.inputs, max_memory=1, clobber=True)
    assert path == out / name / f"{DVDR}.h5"
    assert not list((out / name).glob(".*.tmp"))
    with LightconeStore.open(path) as store:
        assert store.complete
        assert store.quantity == DVDR
        np.testing.assert_allclose(
            store.lightcone[()], reference.lightcones["brightness_temp"], rtol=1e-6
        )
        with LightconeStore.open(out / name / "brightness_temp.h5") as plain:
            assert not np.allclose(store.lightcone[()], plain.lightcone[()])

    # a complete file is kept, unless clobbered
    mtime = path.stat().st_mtime_ns
    assert include_dvdr_on_disk(out / name, cache.inputs) == path
    assert path.stat().st_mtime_ns == mtime


def test_include_dvdr_on_disk_in_whole_chunks(on_disk, lightconers):
    """Blocks of whole chunks give the same result as a few lines of sight at a time."""
    cache, out = on_disk(False, lightconers)
    path = include_dvdr_on_disk(out / "rect", cache.inputs, max_memory=1, clobber=True)
    with LightconeStore.open(path) as store:
        expected = store.lightcone[()]
    include_dvdr_on_disk(out / "rect", cache.inputs, clobber=True)
    with LightconeStore.open(path) as store:
        np.testing.assert_array_equal(store.lightcone[()], expected)


def test_include_dvdr_on_disk_errors(cache, lightconers, tmp_path):
    """The stores must be complete, and the slices one cell apart."""
    lightconer = lightconers["rect"]
    make_lightcones_on_disk({"rect": lightconer}, "brightness_temp", cache, tmp_path)
    with pytest.raises(ValueError, match=r"los_velocity\.h5"):
        include_dvdr_on_disk(tmp_path / "rect", cache.inputs)

    coarse = RectilinearLightconer.between_redshifts(
        min_redshift=6.2,
        max_redshift=9.8,
        resolution=2 * cache.inputs.simulation_options.cell_size,
    )
    make_lightcones_on_disk(
        {"coarse": coarse}, ["brightness_temp", "los_velocity"], cache, tmp_path
    )
    with pytest.raises(ValueError, match="one cell apart"):
        include_dvdr_on_disk(tmp_path / "coarse", cache.inputs)


def test_link_lightcone_file_stores(on_disk, lightconers, tmp_path):
    """The stores that the file links to, and stores that do not fit together."""
    cache, out = on_disk(False, lightconers)
    stores = tmp_path / "stores"
    stores.mkdir()
    for quantity in ("brightness_temp", "los_velocity"):
        shutil.copy(out / "rect" / f"{quantity}.h5", stores)

    # without the dv/dr correction, brightness_temp is the lightcone without it
    lightcone = LightCone.from_file(
        link_lightcone_file(stores, cache.inputs, tmp_path / "plain.h5")
    )
    assert set(lightcone.lightcones) == {"brightness_temp", "los_velocity"}
    with LightconeStore.open(stores / "brightness_temp.h5") as store:
        np.testing.assert_array_equal(
            lightcone.lightcones["brightness_temp"], store.lightcone[()]
        )

    # a file that is not a store is left out
    include_dvdr_on_disk(stores, cache.inputs)
    (stores / "brightness_temp.h5").write_bytes(b"not a lightcone store")
    lightcone = LightCone.from_file(
        link_lightcone_file(stores, cache.inputs, tmp_path / "dvdr.h5")
    )
    assert set(lightcone.lightcones) == {"brightness_temp", "los_velocity"}
    with LightconeStore.open(stores / f"{DVDR}.h5") as store:
        np.testing.assert_array_equal(
            lightcone.lightcones["brightness_temp"], store.lightcone[()]
        )

    # a store with other distances
    with LightconeStore.open(stores / "los_velocity.h5") as velocity:
        shape, distances = velocity.shape, velocity.distances
        redshifts, node_redshifts = velocity.redshifts, velocity.node_redshifts
    with LightconeStore.open_or_create(
        stores / "density.h5",
        quantity="density",
        shape=shape,
        distances=distances * 1.01,
        redshifts=redshifts,
        node_redshifts=node_redshifts,
    ) as other:
        other.mark_done(range(other.done.size))
    with pytest.raises(ValueError, match="other distances"):
        link_lightcone_file(stores, cache.inputs, tmp_path / "mixed.h5")


@pytest.mark.parametrize("ts", [False, True], ids=["no_ts", "ts"])
@pytest.mark.parametrize("name", ["rect", "healpix"])
def test_link_lightcone_file(on_disk, lightconers, tmp_path, monkeypatch, ts, name):
    """21cmFAST reads the lightcones on disk through the links of one file."""
    cache, out = on_disk(ts, lightconers)
    include_dvdr_on_disk(out / name, cache.inputs)
    path = link_lightcone_file(out / name, cache.inputs)
    assert path == out / f"{name}.h5"

    # the links are relative to the file
    monkeypatch.chdir(tmp_path)
    kind = LightCone if name == "rect" else AngularLightcone
    lightcone = kind.from_file(path)
    expected = {"brightness_temp", "brightness_temp_no_dvdr", "los_velocity"}
    assert set(lightcone.lightcones) == expected | ({"tau_21"} if ts else set())
    stored = {
        "brightness_temp": DVDR,
        "brightness_temp_no_dvdr": "brightness_temp",
        "los_velocity": "los_velocity",
    }
    for key, quantity in stored.items():
        with LightconeStore.open(out / name / f"{quantity}.h5") as store:
            np.testing.assert_array_equal(
                lightcone.lightcones[key], store.lightcone[()]
            )
    np.testing.assert_allclose(
        lightcone.lightcone_distances, lightconers[name].lc_distances, rtol=1e-12
    )
    assert lightcone.inputs == cache.inputs
    reference = construct_lightcone_from_cache(
        cache,
        attrs.evolve(lightconers[name], quantities=("brightness_temp",)),
        global_quantities=("brightness_temp",),
    )
    assert set(lightcone.global_quantities) == {"brightness_temp"} | (
        {"tau_21"} if ts else set()
    )
    np.testing.assert_allclose(
        lightcone.global_quantities["brightness_temp"],
        reference.global_quantities["brightness_temp"],
        rtol=1e-6,
    )

    with pytest.raises(ValueError, match="no complete lightcone"):
        link_lightcone_file(tmp_path, cache.inputs)


# --- RSDs ---------------------------------------------------------------------------


def classy_output(inputs):
    """Run CLASS for the cosmology of a run, as 21cmFAST does for the RSDs."""
    cosmo = inputs.cosmo_params
    return run_classy(
        h=cosmo.hlittle,
        Omega_cdm=cosmo.OMm - cosmo.OMb,
        Omega_b=cosmo.OMb,
        n_s=cosmo.POWER_INDEX,
        sigma8=cosmo.SIGMA_8,
        output="vTk",
    )


def test_rsd_lightconer(cache, lightconers):
    """The distances of run_lightcone with RSDs, with the same quantities."""
    lightconer = attrs.evolve(lightconers["rect"], quantities=("density",))
    extended = rsd_lightconer(lightconer, cache.inputs)
    reference = lightconer.validate_options(
        cache.inputs, include_dvdr_in_tau21=False, apply_rsds=True
    )
    np.testing.assert_array_equal(extended.lc_distances, reference.lc_distances)
    assert extended.quantities == ("density",)
    assert extended.lc_distances.min() < lightconer.lc_distances.min()
    assert extended.lc_distances.max() > lightconer.lc_distances.max()
    np.testing.assert_allclose(
        np.diff(extended.lc_distances), cache.inputs.simulation_options.cell_size
    )


@pytest.fixture(scope="module")
def rsd_on_disk(tmp_path_factory):
    """Make lightcones on disk with dv/dr and RSDs, from random boxes.

    They are made once for each value of USE_TS_FLUCT, a few lines of sight at a
    time.
    """
    made = {}

    def make(ts: bool, lightconers: dict):
        if ts not in made:
            tmp = tmp_path_factory.mktemp("rsds")
            cache = random_cache(tmp / "cache", ts=ts)
            classy = classy_output(cache.inputs)
            extended = {
                name: rsd_lightconer(lc, cache.inputs, classy_output=classy)
                for name, lc in lightconers.items()
            }
            quantities = ["brightness_temp", "density", "los_velocity"]
            if ts:
                quantities.append("tau_21")
            make_lightcones_on_disk(extended, quantities, cache, tmp / "out")
            rsds = [DVDR, *quantities[1:]]
            for name in lightconers:
                include_dvdr_on_disk(tmp / "out" / name, cache.inputs)
                apply_rsds_on_disk(tmp / "out" / name, cache.inputs, rsds, max_memory=1)
            made[ts] = cache, extended, tmp / "out"
        return made[ts]

    return make


@pytest.mark.parametrize("ts", [False, True], ids=["no_ts", "ts"])
@pytest.mark.parametrize("name", ["rect", "angular", "healpix"])
def test_rsds_on_disk(rsd_on_disk, lightconers, tmp_path, ts, name):
    """The lightcones of run_lightcone with dv/dr and RSDs, read from the links."""
    cache, extended, out = rsd_on_disk(ts, lightconers)
    reference = construct_lightcone_from_cache(
        cache,
        attrs.evolve(lightconers[name], quantities=("brightness_temp", "density")),
        global_quantities=("brightness_temp", "density"),
        include_dvdr_in_tau21=True,
        apply_rsds=True,
        n_rsd_subcells=4,
    )
    path = link_lightcone_file(
        out / name, cache.inputs, lc_distances=lightconers[name].lc_distances
    )
    kind = LightCone if name == "rect" else AngularLightcone
    lightcone = kind.from_file(path)

    # the slices outside the requested distances are left out, as run_lightcone does
    np.testing.assert_allclose(
        lightcone.lightcone_distances, reference.lightcone_distances, rtol=1e-12
    )
    expected = set(reference.lightcones) | {"brightness_temp_no_dvdr"}
    assert set(lightcone.lightcones) == expected
    assert f"{DVDR}_with_rsds.h5" in {p.name for p in (out / name).iterdir()}
    for key, values in reference.lightcones.items():
        # the dv/dr-corrected field is stored in float32, before its RSDs
        scale = np.abs(values).max()
        np.testing.assert_allclose(
            lightcone.lightcones[key], values, rtol=1e-5, atol=1e-6 * scale
        )
    with_rsds = lightcone.lightcones["density_with_rsds"]
    assert not np.allclose(with_rsds, lightcone.lightcones["density"])
    for key in ("brightness_temp", "density"):
        np.testing.assert_allclose(
            lightcone.global_quantities[key],
            reference.global_quantities[key],
            rtol=1e-6,
        )

    # without the requested distances, all the slices are read
    full = kind.from_file(
        link_lightcone_file(out / name, cache.inputs, tmp_path / "a.h5")
    )
    np.testing.assert_allclose(
        full.lightcone_distances, extended[name].lc_distances, rtol=1e-12
    )


def test_apply_rsds_on_disk_errors(cache, lightconers, tmp_path):
    """The stores must be complete and one cell apart; n_rsd_subcells is checked."""
    lightconer = lightconers["rect"]
    make_lightcones_on_disk({"rect": lightconer}, "density", cache, tmp_path)
    with pytest.raises(ValueError, match=r"los_velocity\.h5"):
        apply_rsds_on_disk(tmp_path / "rect", cache.inputs, "density")

    make_lightcones_on_disk({"rect": lightconer}, "los_velocity", cache, tmp_path)
    paths = apply_rsds_on_disk(tmp_path / "rect", cache.inputs, "density")
    path = paths["density"]
    assert path == tmp_path / "rect" / "density_with_rsds.h5"
    with LightconeStore.open(path) as store:
        assert store.complete
        assert store.quantity == "density_with_rsds"
        assert store.metadata["n_rsd_subcells"] == 4
    mtime = path.stat().st_mtime_ns
    apply_rsds_on_disk(tmp_path / "rect", cache.inputs, ["density"])
    assert path.stat().st_mtime_ns == mtime
    with pytest.raises(ValueError, match="n_rsd_subcells=4"):
        apply_rsds_on_disk(tmp_path / "rect", cache.inputs, "density", n_rsd_subcells=2)
    apply_rsds_on_disk(
        tmp_path / "rect", cache.inputs, "density", n_rsd_subcells=2, clobber=True
    )
    with LightconeStore.open(path) as store:
        assert store.metadata["n_rsd_subcells"] == 2

    coarse = RectilinearLightconer.between_redshifts(
        min_redshift=6.2,
        max_redshift=9.8,
        resolution=2 * cache.inputs.simulation_options.cell_size,
    )
    make_lightcones_on_disk(
        {"coarse": coarse}, ["density", "los_velocity"], cache, tmp_path
    )
    with pytest.raises(ValueError, match="one cell apart"):
        apply_rsds_on_disk(tmp_path / "coarse", cache.inputs, "density")
