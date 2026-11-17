"""Tests of the [CII] emission of halos below the 21cmFAST sampler mass."""

from types import SimpleNamespace

import numpy as np
import py21cmfast as p21c
import pytest
from astropy import units as un
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.io.h5 import read_output_struct

from tuesday.core.postprocessing.line_intensities import cii_flux
from tuesday.simulators.py21cmfast.postprocessing import cii_lightcone, cii_unresolved
from tuesday.simulators.py21cmfast.postprocessing.cii_lightcone import gen_coeval
from tuesday.simulators.py21cmfast.postprocessing.cii_unresolved import (
    grid_unresolved_halos,
    integral_stellar_ratio,
    mean_halo_properties,
    unresolved_cell_totals,
    unresolved_mass_range,
)

REDSHIFT = 7.0

# The tiny test box uses 2LPT at low resolution, for which 21cmFAST warns whenever
# its inputs are built or read. The accuracy of the density field does not matter
# here: the tests compare with the EmissivityFields of the same run. The size-tiny
# template also sets a small R_BUBBLE_MAX, and its box cannot hold halos up to the
# largest mass of the integrals, which 21cmFAST warns about too.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Resolution is likely too low:UserWarning"),
    pytest.mark.filterwarnings("ignore:You are setting R_BUBBLE_MAX:UserWarning"),
    pytest.mark.filterwarnings("ignore:The maximum halo mass:UserWarning"),
]


@pytest.fixture(scope="module")
def subsampler_run(tmp_path_factory) -> SimpleNamespace:
    """Run a tiny coeval box with halos below SAMPLER_MIN_MASS = 1e9 (~20 s).

    The spin temperature is computed (USE_TS_FLUCT), so that 21cmFAST grids the SFR
    in its EmissivityFields and models halo metallicities.
    """
    cache_dir = tmp_path_factory.mktemp("subsampler_cache")
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"], random_seed=2
    ).evolve_input_structs(
        BOX_LEN=40,
        HII_DIM=10,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        SAMPLER_MIN_MASS=1e9,
        ZPRIME_STEP_FACTOR=1.2,
        RECOMB_MODEL="none",
    )
    # End the node redshifts at REDSHIFT, so that every node is computed.
    nodes = [z for z in inputs.node_redshifts if z > REDSHIFT]
    inputs = inputs.clone(node_redshifts=sorted([*nodes, REDSHIFT], reverse=True))
    assert inputs.astro_options.USE_TS_FLUCT
    assert inputs.astro_options.USE_METALLICITY
    p21c.run_coeval(
        out_redshifts=[REDSHIFT],
        inputs=inputs,
        cache=OutputCache(cache_dir),
        write=True,
        progressbar=False,
    )
    run = RunCache.from_inputs(inputs, OutputCache(cache_dir))
    return SimpleNamespace(
        cache_dir=cache_dir,
        inputs=inputs,
        run=run,
        initial_conditions=read_output_struct(run.InitialConditions),
    )


def subsampler_sfr_grid(subsampler_run: SimpleNamespace) -> tuple[np.ndarray, float]:
    """The SFR (Msun/s) that 21cmFAST adds to its EmissivityFields below the sampler.

    The EmissivityFields recomputed with every halo mass set to zero (21cmFAST skips
    such halos) hold the halos below the sampler alone. Also returns the total SFR
    of the cached EmissivityFields.
    """
    inputs, run = subsampler_run.inputs, subsampler_run.run
    nodes = sorted(inputs.node_redshifts, reverse=True)
    previous = nodes[nodes.index(REDSHIFT) - 1]
    no_halos = read_output_struct(run.HaloCatalog[REDSHIFT])
    no_halos.load_all()
    no_halos.set("halo_masses", np.zeros_like(no_halos.get("halo_masses")))
    below_sampler = p21c.compute_emissivity_fields(
        redshift=REDSHIFT,
        inputs=inputs,
        initial_conditions=subsampler_run.initial_conditions,
        perturbed_field=read_output_struct(run.PerturbedField[REDSHIFT]),
        halo_catalog=no_halos,
        previous_spin_temp=read_output_struct(run.TsBox[previous]),
        previous_ionize_box=read_output_struct(run.IonizedBox[previous]),
        write=False,
        regenerate=True,
    )
    so = inputs.simulation_options
    cell_volume = (so.BOX_LEN / so.HII_DIM) ** 3
    full = read_output_struct(run.EmissivityFields[REDSHIFT]).get("sfrd_acg")
    return (
        below_sampler.get("sfrd_acg").astype(np.float64) * cell_volume,
        full.astype(np.float64).sum() * cell_volume,
    )


def test_unresolved_sfr_matches_emissivity_fields(subsampler_run):
    # The SFR of the halos below SAMPLER_MIN_MASS, gridded with the same CHMF,
    # Lagrangian-to-Eulerian move and cloud-in-cell as the [CII], must equal what
    # 21cmFAST adds to the EmissivityFields.
    expected, total = subsampler_sfr_grid(subsampler_run)
    grids = grid_unresolved_halos(
        subsampler_run.initial_conditions,
        REDSHIFT,
        models=[],
        inputs=subsampler_run.inputs,
    )

    # The halos below the sampler contribute a sizeable part of the SFR here.
    assert expected.sum() > 0.01 * total
    np.testing.assert_allclose(grids["sfr"].sum(), expected.sum(), rtol=5e-4)
    np.testing.assert_allclose(
        grids["sfr"], expected, rtol=2e-3, atol=1e-4 * expected.max()
    )


def test_unresolved_halo_mass_is_conserved(subsampler_run):
    # The CHMF integral of the halo mass, moved and gridded with cloud-in-cell,
    # keeps the total mass of the Lagrangian cells.
    ic, inputs = subsampler_run.initial_conditions, subsampler_run.inputs
    grids = grid_unresolved_halos(ic, REDSHIFT, models=[], inputs=inputs)
    per_cell = unresolved_cell_totals(ic, REDSHIFT, models=[], inputs=inputs)[
        "halo_mass"
    ]

    assert per_cell.shape == grids["halo_mass"].shape
    np.testing.assert_allclose(grids["halo_mass"].sum(), per_cell.sum(), rtol=1e-10)
    assert np.all(grids["halo_mass"] >= 0)


def test_gen_coeval_include_unresolved(subsampler_run):
    cache_dir = subsampler_run.cache_dir
    sampled = gen_coeval(REDSHIFT, "deLooze14", cache_dir, seed=4)
    total = gen_coeval(
        REDSHIFT, "deLooze14", cache_dir, seed=4, include_unresolved=True
    )
    unresolved = grid_unresolved_halos(
        subsampler_run.initial_conditions,
        REDSHIFT,
        models=["deLooze14"],
        inputs=subsampler_run.inputs,
    )["deLooze14"]

    assert unresolved.sum() > 0
    np.testing.assert_allclose(
        total.cii_intensity - sampled.cii_intensity,
        unresolved,
        rtol=1e-10,
        atol=1e-10 * unresolved.max(),
    )


@pytest.mark.parametrize("model", ["Vallini15", "Horlaville24"])
def test_metallicity_models_on_a_run_with_metallicities(subsampler_run, model: str):
    grids = grid_unresolved_halos(
        subsampler_run.initial_conditions,
        REDSHIFT,
        models=[model],
        inputs=subsampler_run.inputs,
    )
    assert np.all(np.isfinite(grids[model]))
    assert grids[model].sum() > 0


def test_gen_coeval_list_passes_include_unresolved(monkeypatch):
    calls = []

    def fake_gen_coeval(
        z, model, directory_path, seed=None, inputs=None, include_unresolved=False
    ):
        calls.append(include_unresolved)
        raise RuntimeError("stop")

    monkeypatch.setattr(cii_lightcone, "gen_coeval", fake_gen_coeval)
    lc = SimpleNamespace(inputs=SimpleNamespace(node_redshifts=[7.0]))
    with pytest.raises(RuntimeError, match="stop"):
        cii_lightcone.gen_coeval_list(
            lc, "deLooze14", "unused", include_unresolved=True
        )
    assert calls == [True]


def test_include_unresolved_keeps_sampled_scatter(subsampler_run):
    # The averages over the unresolved halos draw no random numbers, so the scatter
    # of the sampled halos, drawn from one generator for all node redshifts, is the
    # same with and without include_unresolved: the difference at every node is the
    # unresolved [CII] alone.
    lc = SimpleNamespace(inputs=subsampler_run.inputs)
    without = cii_lightcone.gen_coeval_list(
        lc, "deLooze14", subsampler_run.cache_dir, seed=4
    )
    with_unresolved = cii_lightcone.gen_coeval_list(
        lc, "deLooze14", subsampler_run.cache_dir, seed=4, include_unresolved=True
    )

    assert len(without) > 1
    for a, b in zip(without, with_unresolved, strict=True):
        unresolved = grid_unresolved_halos(
            subsampler_run.initial_conditions,
            a.redshift,
            models=["deLooze14"],
            inputs=subsampler_run.inputs,
        )["deLooze14"]
        np.testing.assert_allclose(
            b.cii_intensity - a.cii_intensity,
            unresolved,
            rtol=1e-10,
            atol=1e-10 * max(unresolved.max(), 1e-300),
        )


@pytest.mark.parametrize("model", ["Vallini15", "Horlaville24"])
def test_metallicity_models_need_metallicities(subsampler_run, model: str):
    # 21cmFAST models no halo metallicities without USE_METALLICITY.
    inputs = subsampler_run.inputs.evolve_input_structs(USE_METALLICITY=False)
    ic = subsampler_run.initial_conditions
    with pytest.raises(ValueError, match="USE_METALLICITY"):
        grid_unresolved_halos(ic, REDSHIFT, models=[model], inputs=inputs)
    with pytest.raises(ValueError, match="USE_METALLICITY"):
        mean_halo_properties(inputs, REDSHIFT, np.array([1e8]), models=[model])


def test_unknown_model_raises_without_unresolved_halos(subsampler_run, monkeypatch):
    # With no mass range to integrate, the model name must still be checked.
    monkeypatch.setattr(
        cii_unresolved, "unresolved_mass_range", lambda inputs: (1e9, 1e8)
    )
    ic, inputs = subsampler_run.initial_conditions, subsampler_run.inputs
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        unresolved_cell_totals(ic, REDSHIFT, models=["Silva15"], inputs=inputs)
    totals = unresolved_cell_totals(ic, REDSHIFT, models=["deLooze14"], inputs=inputs)
    assert not np.any(totals["deLooze14"])


def test_mean_cii_without_scatter_matches_power_law():
    # With no stellar-mass or SFR scatter, the mean deLooze14 luminosity is the
    # power law of the mean SFR times the mean of its 0.4 dex lognormal scatter.
    inputs = p21c.InputParameters.from_template(
        "latest-discrete", random_seed=1
    ).evolve_input_structs(SIGMA_STAR=0.0, SIGMA_SFR_LIM=0.0, SIGMA_SFR_INDEX=0.0)
    masses = np.array([3e8, 1e9])
    means = mean_halo_properties(inputs, REDSHIFT, masses, models=["deLooze14"])

    sfr_yr = (means["sfr"] * un.Msun / un.s).to_value(un.Msun / un.yr)
    expected = 10**7.10 * sfr_yr**1.05 * np.exp((0.4 * np.log(10)) ** 2 / 2)
    np.testing.assert_allclose(means["deLooze14"], expected, rtol=1e-2)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("model", ["Vallini15", "deLooze14"])
def test_halos_without_star_formation_do_not_emit(model: str):
    # Far below the turnover mass, 21cmFAST's float32 SFR is zero; the log of the
    # SFR in the [CII] relations must not turn the mean luminosity into NaN.
    inputs = p21c.InputParameters.from_template("latest-discrete", random_seed=1)
    means = mean_halo_properties(inputs, REDSHIFT, np.array([1e6, 1e9]), models=[model])
    assert means["sfr"][0] == 0
    assert means[model][0] == 0
    assert means[model][1] > 0


def test_mean_sfr_is_the_integrals_relation():
    # The mean SFR uses the single power law of the 21cmFAST CHMF integrals. At
    # z = 7 the turnover mass is M_TURN_STELLAR_FEEDBACK, above the atomic-cooling
    # threshold.
    inputs = p21c.InputParameters.from_template("latest-discrete", random_seed=1)
    ap = inputs.astro_params.cdict
    masses = np.array([1e8, 1e9])
    means = mean_halo_properties(inputs, REDSHIFT, masses, models=[])

    fstar = ap["F_STAR10_ACG"] * (masses / 1e10) ** ap["ALPHA_STAR_ACG"]
    fstar *= np.exp(-ap["M_TURN_STELLAR_FEEDBACK"] / masses)
    cosmo = inputs.cosmo_params
    hubble_time = (1 / cosmo.cosmo.H(REDSHIFT)).to_value(un.s)
    expected = fstar * cosmo.OMb / cosmo.OMm * masses / (ap["t_STAR"] * hubble_time)
    np.testing.assert_allclose(means["sfr"], expected, rtol=5e-3)


def test_integral_stellar_ratio():
    inputs = p21c.InputParameters.from_template("latest-discrete", random_seed=1)
    ap = inputs.astro_params.cdict
    low_mass = np.array([1e7])
    expected = 1 / (
        1
        + (ap["UPPER_STELLAR_TURNOVER_MASS"] / 1e10)
        ** (ap["UPPER_STELLAR_TURNOVER_INDEX"] - ap["ALPHA_STAR_ACG"])
    )
    np.testing.assert_allclose(
        integral_stellar_ratio(inputs, low_mass), expected, rtol=1e-3
    )

    single = inputs.evolve_input_structs(USE_UPPER_STELLAR_TURNOVER=False)
    np.testing.assert_array_equal(
        integral_stellar_ratio(single, np.array([1e8, 1e11])), 1.0
    )


def test_unresolved_mass_range():
    inputs = p21c.InputParameters.from_template(
        "latest-discrete", random_seed=1
    ).evolve_input_structs(SAMPLER_MIN_MASS=1e9)
    m_min, m_max = unresolved_mass_range(inputs)

    assert m_min == pytest.approx(10**inputs.astro_params.M_TURN_STELLAR_FEEDBACK / 50)
    assert m_max == 1e9


@pytest.mark.parametrize(
    ("template", "changes", "match"),
    [
        ("minihalos-discrete", {}, "USE_MCGS"),
        ("latest", {}, "CHMF-SAMPLER"),
        (
            "latest-discrete",
            {"USE_REIONIZATION_PHOTOHEATING_FEEDBACK": True},
            "USE_REIONIZATION_PHOTOHEATING_FEEDBACK",
        ),
    ],
)
def test_unsupported_runs_raise(template, changes, match):
    inputs = p21c.InputParameters.from_template(
        template, random_seed=1
    ).evolve_input_structs(**changes)
    with pytest.raises(NotImplementedError, match=match):
        unresolved_mass_range(inputs)


def test_cii_flux_of_unresolved_grid(subsampler_run):
    # The [CII] grid is the flux of the gridded luminosity: compare the box total
    # with the CHMF-weighted luminosity integrated over the Lagrangian cells.
    ic, inputs = subsampler_run.initial_conditions, subsampler_run.inputs
    grids = grid_unresolved_halos(ic, REDSHIFT, models=["Lagache18"], inputs=inputs)
    per_cell = unresolved_cell_totals(ic, REDSHIFT, models=["Lagache18"], inputs=inputs)
    luminosity = per_cell["Lagache18"].sum() * un.Lsun

    np.testing.assert_allclose(
        grids["Lagache18"].sum(),
        cii_flux(luminosity, REDSHIFT, inputs.cosmo_params.cosmo).value,
        rtol=1e-10,
    )
