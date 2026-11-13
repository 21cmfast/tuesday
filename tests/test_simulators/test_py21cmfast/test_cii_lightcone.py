"""Tests of building a [CII] lightcone from coeval boxes."""

import gc
import shutil
import weakref
from pathlib import Path
from types import SimpleNamespace

import attrs
import numpy as np
import py21cmfast as p21c
import pytest
from py21cmfast.drivers.lightcone import LightCone
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.io.h5 import read_output_struct

from tuesday.simulators.py21cmfast.lightcones import construct_lightcone_from_coevals
from tuesday.simulators.py21cmfast.postprocessing import cii_lightcone
from tuesday.simulators.py21cmfast.postprocessing.cii_calculation import calculate_cii
from tuesday.simulators.py21cmfast.postprocessing.cii_lightcone import (
    gen_coeval,
    gen_coeval_list,
    gen_lc,
)

NODE_REDSHIFTS = [7.0, 6.5, 6.0]


def make_run() -> tuple[p21c.InputParameters, p21c.RectilinearLightconer, LightCone]:
    """Create small inputs, a lightconer and a lightcone with two plain fields."""
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=NODE_REDSHIFTS
    ).evolve_input_structs(BOX_LEN=50, HII_DIM=10, DIM=20, PERTURB_ALGORITHM="LINEAR")
    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=6.1,
        max_redshift=6.9,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "density"),
    )
    shape = (10, 10, len(lightconer.lc_distances))
    lightcone = LightCone(
        lightcone_distances=lightconer.lc_distances,
        inputs=inputs,
        lightcones={
            "brightness_temp": np.full(shape, 5.0),
            "density": np.full(shape, 0.1),
        },
    )
    return inputs, lightconer, lightcone


class FakeCoeval(SimpleNamespace):
    """A stand-in for a [CII] coeval box that can be weakly referenced."""

    __slots__ = ("__weakref__",)
    __hash__ = object.__hash__


@pytest.fixture
def run(monkeypatch) -> SimpleNamespace:
    """A small run whose cache reading is replaced by uniform [CII] coeval boxes.

    Each box has an intensity equal to its redshift (times ``scale``), so the
    interpolated lightcone lies between the node values. The seeds passed to
    ``gen_coeval`` are recorded in ``calls["seeds"]``, and the number of boxes still
    in memory when each box is built in ``calls["alive"]``.
    """
    run_inputs, lightconer, lightcone = make_run()
    calls = {"seeds": [], "inputs": [], "scale": 1.0, "alive": []}
    boxes = weakref.WeakSet()

    def fake_gen_coeval(z, model, directory_path, seed=None, inputs=None):
        gc.collect()
        calls["alive"].append(len(boxes))
        calls["seeds"].append(seed)
        calls["inputs"].append(inputs)
        coeval = FakeCoeval(
            redshift=z,
            cosmo_params=run_inputs.cosmo_params,
            simulation_options=run_inputs.simulation_options,
            cii_intensity=np.full((10, 10, 10), calls["scale"] * z),
            initial_conditions=object(),
            halobox=object(),
        )
        boxes.add(coeval)
        return coeval

    monkeypatch.setattr(cii_lightcone, "gen_coeval", fake_gen_coeval)
    return SimpleNamespace(lightconer=lightconer, lightcone=lightcone, calls=calls)


def test_gen_lc_output(run):
    cii_lc = gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused", seed=3)

    assert isinstance(cii_lc, LightCone)
    intensity = cii_lc.lightcones["cii_intensity"]
    assert intensity.shape == run.lightcone.lightcones["brightness_temp"].shape
    # Interpolated between the node values, and no slice left empty.
    assert np.all(intensity >= min(NODE_REDSHIFTS))
    assert np.all(intensity <= max(NODE_REDSHIFTS))
    assert np.all(cii_lc.lightcones["cii_brightness"] > 0)
    # The boxes are read for the run of the lightcone.
    assert all(inputs is run.lightcone.inputs for inputs in run.calls["inputs"])


def test_gen_lc_shares_one_generator(run):
    # One generator, seeded once, is shared by all redshifts as in gen_coeval_list.
    gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused", seed=3)

    seeds = run.calls["seeds"]
    assert len(seeds) == len(NODE_REDSHIFTS)
    assert isinstance(seeds[0], np.random.Generator)
    assert all(seed is seeds[0] for seed in seeds)


def test_gen_lc_keeps_two_coevals_in_memory(run):
    # Each box is interpolated with the previous one as soon as it is built, so at
    # most the previous box is still alive when the next one is built.
    gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")

    assert run.calls["alive"] == [0, 1, 1]


def test_gen_lc_keeps_other_fields(run):
    cii_lc = gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")

    np.testing.assert_array_equal(cii_lc.lightcones["brightness_temp"], 5.0)
    np.testing.assert_array_equal(cii_lc.lightcones["density"], 0.1)


def test_gen_lc_does_not_mutate_inputs(run):
    original_tb = run.lightcone.lightcones["brightness_temp"]
    gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")

    assert set(run.lightcone.lightcones) == {"brightness_temp", "density"}
    assert run.lightcone.lightcones["brightness_temp"] is original_tb
    np.testing.assert_array_equal(original_tb, 5.0)
    assert run.lightconer.quantities == ("brightness_temp", "density")


def test_gen_lc_calls_are_independent(run):
    first = gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")
    first_intensity = first.lightcones["cii_intensity"].copy()

    run.calls["scale"] = 2.0
    second = gen_lc(run.lightcone, run.lightconer, "deLooze14", "unused")

    np.testing.assert_array_equal(first.lightcones["cii_intensity"], first_intensity)
    np.testing.assert_allclose(
        second.lightcones["cii_intensity"], 2 * first_intensity, rtol=1e-12
    )


def test_cii_lightcone_class_removed():
    assert not hasattr(cii_lightcone, "CIILightcone")


# --- A real (tiny) 21cmFAST run -----------------------------------------------------


@pytest.fixture(scope="module")
def real_run(tmp_path_factory) -> SimpleNamespace:
    """Run a tiny 21cmFAST lightcone and keep its output cache.

    The 21-cm velocity-gradient correction is switched off, so that the
    brightness-temperature lightcone is a plain interpolation of the cached coeval
    boxes.
    """
    cache_dir = tmp_path_factory.mktemp("cache")
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"],
        random_seed=1,
        node_redshifts=NODE_REDSHIFTS,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    ).evolve_input_structs(
        BOX_LEN=30,
        HII_DIM=10,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        PERTURB_ALGORITHM="LINEAR",
    )
    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=6.1,
        max_redshift=6.9,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp",),
    )
    lightcone = p21c.run_lightcone(
        lightconer=lightconer,
        inputs=inputs,
        cache=OutputCache(cache_dir),
        write=True,
        progressbar=False,
        include_dvdr_in_tau21=False,
    )
    return SimpleNamespace(
        cache_dir=cache_dir, lightconer=lightconer, lightcone=lightcone
    )


def read_cached(cache_dir: Path, z: float, name: str):
    """Read one cached 21cmFAST output at redshift ``z``."""
    path = next(
        f for f in OutputCache(cache_dir).list_datasets(redshift=z) if f.stem == name
    )
    return read_output_struct(path)


def copy_cache_without(cache_dir: Path, destination: Path, *names: str) -> Path:
    """Copy a 21cmFAST output cache, leaving out the outputs called ``names``."""
    shutil.copytree(cache_dir, destination)
    for name in names:
        for path in destination.rglob(f"{name}.h5"):
            path.unlink()
    return destination


def test_interpolation_matches_21cmfast_lightcone(real_run):
    # Interpolating the cached brightness-temperature boxes the way gen_lc
    # interpolates the [CII] boxes must reproduce the 21cmFAST lightcone.
    lc = real_run.lightcone
    coevals = [
        SimpleNamespace(
            redshift=z,
            simulation_options=lc.inputs.simulation_options,
            cosmo_params=lc.inputs.cosmo_params,
            brightness_temp=read_cached(real_run.cache_dir, z, "BrightnessTemp").get(
                "brightness_temp"
            ),
        )
        for z in NODE_REDSHIFTS
    ]
    brightness_temp = construct_lightcone_from_coevals(
        coevals, real_run.lightconer, lc.inputs
    ).lightcones["brightness_temp"]

    # 21cmFAST stores lightcones in float32 (~1e-7 relative precision).
    np.testing.assert_allclose(
        brightness_temp, lc.lightcones["brightness_temp"], rtol=1e-5, atol=1e-5
    )


def test_gen_coeval_conserves_flux(real_run, monkeypatch):
    z = NODE_REDSHIFTS[1]
    returned = {}

    def keep_intensity(halo_catalog, model, seed=None):
        luminosity, intensity = calculate_cii(halo_catalog, model, seed=seed)
        returned["intensity"] = intensity
        return luminosity, intensity

    monkeypatch.setattr(cii_lightcone, "calculate_cii", keep_intensity)
    coeval = gen_coeval(z, "Lagache18", real_run.cache_dir, seed=5)

    assert coeval.cii_intensity.shape == (10, 10, 10)
    assert returned["intensity"].sum() > 0
    # Every halo lies inside the box, so gridding conserves the total flux.
    np.testing.assert_allclose(
        coeval.cii_intensity.sum(), returned["intensity"].sum(), rtol=1e-10
    )


def halo_part(cache_dir: Path, z: float, field: str) -> np.ndarray:
    """The part of a cached EmissivityFields field that comes from the sampled halos.

    21cmFAST adds to its EmissivityFields the emission of halos below
    SAMPLER_MIN_MASS, from the conditional mass function. Recomputing the box with
    every halo mass set to zero (21cmFAST skips such halos) gives that part alone.
    """
    inputs = read_cached(cache_dir, z, "HaloCatalog").inputs
    run = RunCache.from_inputs(inputs, OutputCache(cache_dir))
    nodes = sorted(inputs.node_redshifts, reverse=True)
    i = nodes.index(z)
    no_halos = read_output_struct(run.HaloCatalog[z])
    no_halos.load_all()
    no_halos.set("halo_masses", np.zeros_like(no_halos.get("halo_masses")))
    # The boxes of the previous node redshift, as 21cmFAST's redshift loop passes them.
    previous = {}
    if i > 0:
        previous["previous_ionize_box"] = read_output_struct(
            run.IonizedBox[nodes[i - 1]]
        )
        if run.TsBox:
            previous["previous_spin_temp"] = read_output_struct(run.TsBox[nodes[i - 1]])
    below_sampler = p21c.compute_emissivity_fields(
        redshift=z,
        inputs=inputs,
        initial_conditions=read_output_struct(run.InitialConditions),
        perturbed_field=read_output_struct(run.PerturbedField[z]),
        halo_catalog=no_halos,
        write=False,
        regenerate=True,
        **previous,
    )
    full = read_output_struct(run.EmissivityFields[z]).get(field)
    return full.astype(np.float64) - below_sampler.get(field).astype(np.float64)


def grid_halo_property(monkeypatch, cache_dir: Path, z: float, prop: str):
    """Grid one halo property of the PerturbedHaloCatalog with gen_coeval, per Mpc^3."""
    monkeypatch.setattr(
        cii_lightcone,
        "calculate_cii",
        lambda halo_catalog, model, seed=None: (
            None,
            halo_catalog.get(prop)[: halo_catalog.n_halos].astype(np.float64),
        ),
    )
    coeval = gen_coeval(z, "Lagache18", cache_dir)
    options = coeval.simulation_options
    return coeval.cii_intensity / (options.BOX_LEN / options.HII_DIM) ** 3


def test_gen_coeval_grids_like_21cmfast(real_run, monkeypatch):
    # Grid each halo's ionising emissivity instead of its [CII] flux: gen_coeval must
    # place the halos where 21cmFAST does, so the box must equal the halo part of the
    # cached EmissivityFields n_ion, which 21cmFAST builds from the Eulerian halo
    # positions with cloud-in-cell. (21cmFAST does not store the gridded SFR of a run
    # without USE_TS_FLUCT.)
    z = NODE_REDSHIFTS[1]
    gridded = grid_halo_property(monkeypatch, real_run.cache_dir, z, "n_ion")
    expected = halo_part(real_run.cache_dir, z, "n_ion")

    # The EmissivityFields are stored in float32.
    np.testing.assert_allclose(gridded, expected, rtol=1e-5, atol=1e-5 * expected.max())


def test_gen_coeval_does_not_rely_on_calculate_cii_loading(real_run, monkeypatch):
    # 21cmFAST's perturb_halo_catalog crashes (segfault) if the per-halo random
    # numbers of the catalogue have not been read from disk, and only reads the
    # coordinates and masses itself. gen_coeval must load the catalogue, rather
    # than rely on calculate_cii having read it.
    monkeypatch.setattr(
        cii_lightcone,
        "calculate_cii",
        lambda halo_catalog, model, seed=None: (None, np.ones(halo_catalog.n_halos)),
    )
    coeval = gen_coeval(NODE_REDSHIFTS[0], "Lagache18", real_run.cache_dir)

    assert np.isfinite(coeval.cii_intensity).all()


def test_gen_lc_on_real_cache(real_run):
    cii_lc = gen_lc(
        real_run.lightcone, real_run.lightconer, "Lagache18", real_run.cache_dir, seed=1
    )
    brightness = cii_lc.lightcones["cii_brightness"]

    assert brightness.shape == real_run.lightcone.lightcones["brightness_temp"].shape
    assert np.all(np.isfinite(brightness))
    # Every slice receives [CII] emission from the halos.
    assert np.all(brightness.sum(axis=(0, 1)) > 0)


def test_gen_lc_matches_coeval_list(real_run):
    # Interpolating the boxes one by one gives the same lightcone as building all
    # of them first with gen_coeval_list.
    cii_lc = gen_lc(
        real_run.lightcone, real_run.lightconer, "Lagache18", real_run.cache_dir, seed=5
    )
    coevals = gen_coeval_list(
        real_run.lightcone, "Lagache18", real_run.cache_dir, seed=5
    )
    expected = construct_lightcone_from_coevals(
        coevals,
        attrs.evolve(real_run.lightconer, quantities=("cii_intensity",)),
        real_run.lightcone.inputs,
    ).lightcones["cii_intensity"]

    np.testing.assert_array_equal(cii_lc.lightcones["cii_intensity"], expected)


def test_gen_lc_without_21cm_outputs(real_run, tmp_path):
    # Only the halo catalogue and the initial conditions are needed for [CII]: the
    # 21-cm outputs of the run are not read.
    cache = copy_cache_without(
        real_run.cache_dir,
        tmp_path / "cache",
        "PerturbedField",
        "IonizedBox",
        "BrightnessTemp",
        "EmissivityFields",
    )
    args = (real_run.lightcone, real_run.lightconer, "Lagache18")
    reduced = gen_lc(*args, cache, seed=2)
    full = gen_lc(*args, real_run.cache_dir, seed=2)

    for key in ("cii_intensity", "cii_brightness"):
        np.testing.assert_array_equal(reduced.lightcones[key], full.lightcones[key])


@pytest.mark.parametrize("missing", ["HaloCatalog", "InitialConditions"])
def test_gen_coeval_needs_halos_and_initial_conditions(real_run, tmp_path, missing):
    cache = copy_cache_without(real_run.cache_dir, tmp_path / "cache", missing)

    with pytest.raises(ValueError, match=missing):
        gen_coeval(NODE_REDSHIFTS[0], "Lagache18", cache, seed=0)


# --- A (tiny) run with minihalos and reionization feedback --------------------------


@pytest.fixture(scope="module")
def feedback_run(tmp_path_factory) -> SimpleNamespace:
    """Run a tiny coeval with minihalos and reionization feedback (~1 min).

    The halo properties then depend on the TsBox and IonizedBox of the previous node
    redshift.
    """
    cache_dir = tmp_path_factory.mktemp("feedback_cache")
    inputs = p21c.InputParameters.from_template(
        ["minihalos-discrete", "size-tiny"], random_seed=3
    ).evolve_input_structs(
        BOX_LEN=20,
        HII_DIM=8,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        PERTURB_ALGORITHM="LINEAR",
        ZPRIME_STEP_FACTOR=1.2,
        R_BUBBLE_MAX=10.0,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=True,
    )
    assert inputs.astro_options.USE_MCGS
    nodes = sorted(inputs.node_redshifts, reverse=True)
    p21c.run_coeval(
        out_redshifts=[nodes[2]],
        inputs=inputs,
        cache=OutputCache(cache_dir),
        write=True,
        progressbar=False,
    )
    return SimpleNamespace(cache_dir=cache_dir, inputs=inputs, nodes=nodes)


@pytest.mark.parametrize("node", [1, 2])
def test_gen_coeval_has_the_feedback_of_the_run(feedback_run, monkeypatch, node):
    # Below Z_HEAT_MAX the halo SFRs of the run depend on the previous node's
    # Lyman-Werner background and reionization: gen_coeval must give the halos the
    # SFRs of the run, i.e. reproduce the halo part of the cached sfrd_acg.
    z = feedback_run.nodes[node]
    assert z < feedback_run.inputs.simulation_options.Z_HEAT_MAX
    gridded = grid_halo_property(monkeypatch, feedback_run.cache_dir, z, "sfr_acg")
    expected = halo_part(feedback_run.cache_dir, z, "sfrd_acg")

    assert expected.max() > 0
    np.testing.assert_allclose(gridded, expected, rtol=1e-5, atol=1e-5 * expected.max())


@pytest.mark.parametrize("missing", ["TsBox", "IonizedBox"])
def test_gen_coeval_needs_the_previous_boxes(feedback_run, tmp_path, missing):
    cache = copy_cache_without(feedback_run.cache_dir, tmp_path / "cache", missing)

    with pytest.raises(ValueError, match=missing):
        gen_coeval(feedback_run.nodes[2], "Lagache18", cache, seed=0)


# --- A (tiny) run with reionization feedback, without minihalos ---------------------


@pytest.fixture(scope="module")
def photoheating_run(tmp_path_factory) -> SimpleNamespace:
    """Run a tiny coeval with reionization feedback but no minihalos or Ts (~10 s).

    The cache then has no TsBox, and the halo properties depend on the IonizedBox of
    the previous node redshift. Z_HEAT_MAX is the first node redshift: 21cmFAST
    4.3.1 crashes (segfault) with reionization feedback at a first node below it.
    """
    cache_dir = tmp_path_factory.mktemp("photoheating_cache")
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"],
        random_seed=4,
        node_redshifts=NODE_REDSHIFTS,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=True,
    ).evolve_input_structs(
        BOX_LEN=30,
        HII_DIM=10,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        PERTURB_ALGORITHM="LINEAR",
        Z_HEAT_MAX=NODE_REDSHIFTS[0],
    )
    assert not inputs.astro_options.USE_MCGS
    p21c.run_coeval(
        out_redshifts=[NODE_REDSHIFTS[-1]],
        inputs=inputs,
        cache=OutputCache(cache_dir),
        write=True,
        progressbar=False,
    )
    return SimpleNamespace(cache_dir=cache_dir, inputs=inputs)


@pytest.mark.parametrize("node", [1, 2])
def test_gen_coeval_with_reionization_feedback_only(
    photoheating_run, monkeypatch, node
):
    # Reionization feedback without minihalos needs only the previous IonizedBox:
    # gen_coeval must not ask for a TsBox, which such a run does not have, and must
    # give the halos the properties of the run.
    run = RunCache.from_inputs(
        photoheating_run.inputs, OutputCache(photoheating_run.cache_dir)
    )
    assert not run.TsBox
    z = NODE_REDSHIFTS[node]
    gridded = grid_halo_property(monkeypatch, photoheating_run.cache_dir, z, "n_ion")
    expected = halo_part(photoheating_run.cache_dir, z, "n_ion")

    assert expected.max() > 0
    np.testing.assert_allclose(gridded, expected, rtol=1e-5, atol=1e-5 * expected.max())


def test_gen_coeval_with_reionization_feedback_needs_the_ionized_box(
    photoheating_run, tmp_path
):
    cache = copy_cache_without(
        photoheating_run.cache_dir, tmp_path / "cache", "IonizedBox"
    )

    with pytest.raises(ValueError, match="IonizedBox"):
        gen_coeval(NODE_REDSHIFTS[2], "Lagache18", cache, seed=0)
