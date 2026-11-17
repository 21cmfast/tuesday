"""Tests of the UV luminosity functions of 21cmFAST halo catalogues."""

from types import SimpleNamespace

import numpy as np
import py21cmfast as p21c
import pytest
from astropy import units as un
from py21cmfast.io.caching import OutputCache
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.wrapper.cfuncs import convert_halo_properties
from py21cmfast.wrapper.outputs import HaloCatalog, PerturbedHaloCatalog

from tuesday.core.summaries.luminosity_functions import (
    KUV_SUN16,
    uv_luminosity_function,
)
from tuesday.simulators.py21cmfast.summaries.luminosity_functions import (
    calculate_uvlf,
    get_halo_properties,
)

REDSHIFT = 8.0
SMALL_BOX = {
    "BOX_LEN": 20,
    "HII_DIM": 8,
    "DIM": None,
    "HIRES_TO_LOWRES_FACTOR": 2,
    "PERTURB_ALGORITHM": "LINEAR",
}
SECONDS_PER_YEAR = un.yr.to(un.s)


@pytest.fixture(scope="module")
def inputs() -> p21c.InputParameters:
    """Small inputs without minihalos or reionization feedback."""
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"], random_seed=1
    ).evolve_input_structs(**SMALL_BOX)
    assert not inputs.astro_options.USE_MCGS
    assert not inputs.astro_options.USE_REIONIZATION_PHOTOHEATING_FEEDBACK
    return inputs


@pytest.fixture(scope="module")
def mcg_inputs() -> p21c.InputParameters:
    """Small inputs with minihalos and relative-velocity fluctuations."""
    inputs = p21c.InputParameters.from_template(
        ["minihalos-discrete", "size-tiny"], random_seed=3
    ).evolve_input_structs(**SMALL_BOX)
    assert inputs.astro_options.USE_MCGS
    assert inputs.matter_options.V_CB_MODEL == "FLUCTS"
    return inputs


@pytest.fixture(scope="module")
def run(inputs) -> SimpleNamespace:
    """Initial conditions and a halo catalogue of a tiny run (a few seconds)."""
    initial_conditions = p21c.compute_initial_conditions(inputs=inputs)
    halo_catalog = p21c.determine_halo_catalog(
        redshift=REDSHIFT, initial_conditions=initial_conditions, inputs=inputs
    )
    return SimpleNamespace(
        inputs=inputs, initial_conditions=initial_conditions, halo_catalog=halo_catalog
    )


def make_perturbed(
    inputs: p21c.InputParameters, n_padding: int = 3, **properties: np.ndarray
) -> PerturbedHaloCatalog:
    """Create a PerturbedHaloCatalog with the given halo properties.

    The padding beyond the real halos is NaN, so any use of it shows.
    """
    n_halos = len(next(iter(properties.values())))
    catalog = PerturbedHaloCatalog.new(
        redshift=REDSHIFT, buffer_size=n_halos + n_padding, inputs=inputs
    )
    for name, value in properties.items():
        padded = np.full(n_halos + n_padding, np.nan, dtype=np.float32)
        padded[:n_halos] = value
        catalog.set(name, padded)
    catalog.n_halos = n_halos
    return catalog


def test_perturbed_catalogue(inputs):
    sfr = np.array([0.0, 0.1, 1.0, 1.0, 10.0])
    catalog = make_perturbed(
        inputs,
        # 21cmFAST gives the SFR in Msun/s.
        sfr_acg=sfr / SECONDS_PER_YEAR,
        stellar_masses_acg=np.full(5, 1e8),
    )
    result = calculate_uvlf(catalog)

    expected = uv_luminosity_function(
        sfr.astype(np.float32) / SECONDS_PER_YEAR * un.Msun / un.s, (20 * un.Mpc) ** 3
    )
    np.testing.assert_array_equal(result.bin_edges, expected.bin_edges)
    np.testing.assert_array_equal(result.counts, expected.counts)
    assert result.counts.sum() == 4
    assert result.number_density.unit == un.Mpc**-3
    np.testing.assert_allclose(
        result.number_density.value, expected.number_density.value, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("components", "sfr"), [("both", 3.0), ("acg", 1.0), ("mcg", 2.0)]
)
def test_components(mcg_inputs, components, sfr):
    calls = []

    def kuv(sfr, stellar_mass, redshift):
        calls.append((sfr, stellar_mass, redshift))
        return np.full_like(sfr, 1.15e-28)

    catalog = make_perturbed(
        mcg_inputs,
        sfr_acg=np.array([1.0]) / SECONDS_PER_YEAR,
        sfr_mcg=np.array([2.0]) / SECONDS_PER_YEAR,
        stellar_masses_acg=np.array([1e8]),
        stellar_masses_mcg=np.array([2e8]),
    )
    edges = np.linspace(-24.0, -12.0, 1201)
    result = calculate_uvlf(catalog, bins=edges, kuv=kuv, components=components)

    magnitude = 51.63 - 2.5 * np.log10(sfr / 1.15e-28)
    assert result.counts.sum() == 1
    assert (
        edges[np.argmax(result.counts)]
        <= magnitude
        < edges[np.argmax(result.counts) + 1]
    )
    # K_UV gets the SFR and stellar mass of the same components.
    ((got_sfr, got_mass, got_z),) = calls
    np.testing.assert_allclose(got_sfr, [sfr], rtol=1e-6)
    np.testing.assert_allclose(got_mass, [1e8 * sfr], rtol=1e-6)
    assert got_z == REDSHIFT


def test_runs_without_minihalos_count_atomic_cooling_galaxies(inputs):
    catalog = make_perturbed(
        inputs,
        sfr_acg=np.array([1.0, 10.0]) / SECONDS_PER_YEAR,
        stellar_masses_acg=np.array([1e8, 1e9]),
    )
    np.testing.assert_array_equal(
        calculate_uvlf(catalog, components="both").counts,
        calculate_uvlf(catalog, components="acg").counts,
    )


def test_minihalo_component_needs_minihalos(inputs):
    catalog = make_perturbed(
        inputs, sfr_acg=np.array([1.0]), stellar_masses_acg=np.array([1e8])
    )
    with pytest.raises(ValueError, match="USE_MCGS"):
        calculate_uvlf(catalog, components="mcg")


def test_unknown_components(inputs):
    catalog = make_perturbed(
        inputs, sfr_acg=np.array([1.0]), stellar_masses_acg=np.array([1e8])
    )
    with pytest.raises(ValueError, match="components must be"):
        calculate_uvlf(catalog, components="all")


def test_other_objects_are_refused(inputs):
    with pytest.raises(TypeError, match="HaloCatalog or a PerturbedHaloCatalog"):
        calculate_uvlf(SimpleNamespace(inputs=inputs))


def test_halo_catalogue_without_feedback(run):
    # The properties of a HaloCatalog are those that convert_halo_properties gives
    # without feedback grids, which are its defaults in runs without minihalos or
    # reionization feedback.
    catalog = run.halo_catalog
    props = convert_halo_properties(
        redshift=REDSHIFT,
        inputs=run.inputs,
        halo_masses=catalog.get("halo_masses"),
        star_rng=catalog.get("star_rng"),
        sfr_rng=catalog.get("sfr_rng"),
        xray_rng=catalog.get("xray_rng"),
    )
    got = get_halo_properties(catalog)
    assert set(got) == set(props)
    for name, value in props.items():
        np.testing.assert_array_equal(got[name], value)

    result = calculate_uvlf(catalog)
    sfr = props["sfr_acg"].astype(np.float64) * SECONDS_PER_YEAR
    with np.errstate(divide="ignore"):
        magnitude = 51.63 - 2.5 * np.log10(sfr / 1.15e-28)
    counts, _ = np.histogram(magnitude, np.linspace(-24, -12, 25))
    assert counts.sum() > 100
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_allclose(
        result.number_density.to_value(un.Mpc**-3), counts / 0.5 / 20.0**3
    )


def test_perturbed_catalogue_matches_halo_catalogue(run):
    # Without minihalos or reionization feedback, the PerturbedHaloCatalog of the run
    # has the properties of the galaxy model alone.
    perturbed = p21c.perturb_halo_catalog(
        initial_conditions=run.initial_conditions,
        inputs=run.inputs,
        halo_catalog=run.halo_catalog,
    )
    np.testing.assert_array_equal(
        calculate_uvlf(perturbed).counts, calculate_uvlf(run.halo_catalog).counts
    )


def test_cached_halo_catalogue(run, tmp_path):
    cache = OutputCache(tmp_path)
    cache.write(run.halo_catalog)
    (path,) = cache.list_datasets(kind="HaloCatalog")
    # The halo arrays are read from the file when they are used.
    cached = read_output_struct(path)
    np.testing.assert_array_equal(
        calculate_uvlf(cached).counts, calculate_uvlf(run.halo_catalog).counts
    )


def make_halo_catalog(inputs: p21c.InputParameters) -> HaloCatalog:
    """Create a HaloCatalog of 200 halos with random masses and positions."""
    rng = np.random.default_rng(0)
    n_halos = 200
    catalog = HaloCatalog.new(inputs=inputs, redshift=REDSHIFT, buffer_size=n_halos)
    catalog.set("halo_masses", (10 ** rng.uniform(6, 11, n_halos)).astype("f4"))
    for name in ("star_rng", "sfr_rng", "xray_rng"):
        catalog.set(name, rng.normal(size=n_halos).astype("f4"))
    catalog.set(
        "halo_coords",
        rng.uniform(0, inputs.simulation_options.BOX_LEN, (n_halos, 3)).astype("f4"),
    )
    catalog.n_halos = n_halos
    return catalog


def test_minihalos_need_initial_conditions(mcg_inputs):
    catalog = make_halo_catalog(mcg_inputs)
    with pytest.raises(ValueError, match="initial_conditions"):
        calculate_uvlf(catalog)


def test_minihalo_properties_use_relative_velocities(mcg_inputs):
    catalog = make_halo_catalog(mcg_inputs)
    shape = (mcg_inputs.simulation_options.HII_DIM,) * 3
    slow = SimpleNamespace(get=lambda name: np.zeros(shape, dtype="f4"))
    fast = SimpleNamespace(get=lambda name: np.full(shape, 60.0, dtype="f4"))

    props_slow = get_halo_properties(catalog, slow)
    props_fast = get_halo_properties(catalog, fast)
    assert props_slow["sfr_mcg"].sum() > 0
    # Faster streaming raises the minihalo turnover mass, so less star formation.
    assert props_fast["sfr_mcg"].sum() < props_slow["sfr_mcg"].sum()

    result = calculate_uvlf(catalog, initial_conditions=slow, bins=10)
    sfr = (props_slow["sfr_acg"] + props_slow["sfr_mcg"]).astype(np.float64)
    expected = uv_luminosity_function(
        sfr * un.Msun / un.s, (20 * un.Mpc) ** 3, bins=10, kuv=KUV_SUN16
    )
    np.testing.assert_array_equal(result.counts, expected.counts)
