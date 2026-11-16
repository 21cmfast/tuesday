"""Tests of the [CII] emission of halos below the 21cmFAST sampler mass."""

from types import SimpleNamespace

import attrs
import numpy as np
import py21cmfast as p21c
import pytest
from astropy import units as un
from py21cmfast.io.caching import OutputCache
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.wrapper.cfuncs import convert_halo_properties

from tuesday.core.postprocessing.line_intensities import cii_flux
from tuesday.simulators.py21cmfast.postprocessing import cii_lightcone, cii_unresolved
from tuesday.simulators.py21cmfast.postprocessing.cii_lightcone import (
    _cic_grid,
    gen_coeval,
)
from tuesday.simulators.py21cmfast.postprocessing.cii_unresolved import (
    fix_cached_inputs,
    grid_unresolved_halos,
    integral_stellar_ratio,
    mean_halo_properties,
    unresolved_cell_totals,
    unresolved_mass_range,
)

REDSHIFT = 7.0

# The tiny test box uses 2LPT at low resolution, for which 21cmFAST warns whenever
# its inputs are built or read. The accuracy of the density field does not matter
# here: the tests compare with the HaloBox of the same run.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Resolution is likely too low:UserWarning"
)


@pytest.fixture(scope="module")
def subsampler_run(tmp_path_factory) -> SimpleNamespace:
    """Run a tiny coeval box whose HaloBox has halos below SAMPLER_MIN_MASS = 1e9."""
    cache_dir = tmp_path_factory.mktemp("subsampler_cache")
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"],
        random_seed=2,
        node_redshifts=[7.5, REDSHIFT],
        USE_TS_FLUCT=False,
        INHOMO_RECO=False,
    ).evolve_input_structs(
        BOX_LEN=40,
        HII_DIM=10,
        DIM=None,
        HIRES_TO_LOWRES_FACTOR=2,
        SAMPLER_MIN_MASS=1e9,
    )
    p21c.run_coeval(
        out_redshifts=[REDSHIFT],
        inputs=inputs,
        cache=OutputCache(cache_dir),
        write=True,
        progressbar=False,
    )
    cache = {
        f.stem: read_output_struct(f)
        for f in OutputCache(cache_dir).list_datasets(redshift=REDSHIFT)
    }
    return SimpleNamespace(cache_dir=cache_dir, cache=cache)


def sampled_sfr_grid(cache: dict) -> np.ndarray:
    """Grid the SFR (Msun/s) of the sampled halos at their Eulerian positions."""
    halo_catalog = cache["HaloCatalog"]
    halo_catalog.load_all()
    n = halo_catalog.n_halos
    props = convert_halo_properties(
        redshift=REDSHIFT,
        inputs=halo_catalog.inputs,
        **{
            key: halo_catalog.get(key)[:n]
            for key in ("halo_masses", "halo_coords", "star_rng", "sfr_rng", "xray_rng")
        },
    )
    perturbed = p21c.perturb_halo_catalog(
        initial_conditions=cache["InitialConditions"],
        inputs=halo_catalog.inputs,
        halo_catalog=halo_catalog,
    )
    so = halo_catalog.simulation_options
    positions = perturbed.get("halo_coords")[: perturbed.n_halos] / (
        so.BOX_LEN / so.HII_DIM
    )
    return _cic_grid(positions, props["halo_sfr"], so.HII_DIM)


def test_unresolved_sfr_matches_halobox(subsampler_run):
    # The SFR of the halos below SAMPLER_MIN_MASS, gridded with the same CHMF,
    # Lagrangian-to-Eulerian move and cloud-in-cell as the [CII], must equal what
    # 21cmFAST adds to the HaloBox: its total SFR minus the sampled halos.
    cache = subsampler_run.cache
    so = cache["HaloCatalog"].simulation_options
    cell_volume = (so.BOX_LEN / so.HII_DIM) ** 3
    expected = cache["HaloBox"].get("halo_sfr").astype(float) * cell_volume
    expected -= sampled_sfr_grid(cache)
    grids = grid_unresolved_halos(
        cache["InitialConditions"], REDSHIFT, models=[], seed=0
    )

    # The halos below the sampler contribute a sizeable part of the SFR here.
    assert expected.sum() > 0.01 * cache["HaloBox"].get("halo_sfr").sum() * cell_volume
    np.testing.assert_allclose(grids["sfr"].sum(), expected.sum(), rtol=2e-3)
    np.testing.assert_allclose(
        grids["sfr"], expected, rtol=1e-2, atol=1e-3 * expected.max()
    )


def test_unresolved_halo_mass_is_conserved(subsampler_run):
    # The CHMF integral of the halo mass, moved and gridded with cloud-in-cell,
    # keeps the total mass of the Lagrangian cells.
    ic = subsampler_run.cache["InitialConditions"]
    grids = grid_unresolved_halos(ic, REDSHIFT, models=[], seed=0)
    per_cell = unresolved_cell_totals(ic, REDSHIFT, models=[], seed=0)["halo_mass"]

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
        subsampler_run.cache["InitialConditions"],
        REDSHIFT,
        models=["deLooze14"],
        seed=np.random.default_rng(4),
    )["deLooze14"]

    # The sampled part is unchanged by the option; the seed of the model scatter
    # is used first by the sampled halos.
    assert np.all(total.cii_intensity >= sampled.cii_intensity)
    assert total.cii_intensity.sum() > sampled.cii_intensity.sum()
    np.testing.assert_allclose(
        total.cii_intensity - sampled.cii_intensity,
        unresolved,
        rtol=2e-2,
        atol=1e-3 * unresolved.max(),
    )


def test_gen_coeval_list_passes_include_unresolved(monkeypatch):
    calls = []

    def fake_gen_coeval(z, model, directory_path, seed=None, include_unresolved=False):
        calls.append(include_unresolved)
        raise RuntimeError("stop")

    monkeypatch.setattr(cii_lightcone, "gen_coeval", fake_gen_coeval)
    lc = SimpleNamespace(inputs=SimpleNamespace(node_redshifts=[7.0]))
    with pytest.raises(RuntimeError, match="stop"):
        cii_lightcone.gen_coeval_list(
            lc, "deLooze14", "unused", include_unresolved=True
        )
    assert calls == [True]


def test_include_unresolved_keeps_sampled_scatter(subsampler_run, monkeypatch):
    # The averages over the unresolved halos draw their own random numbers: they
    # must not advance the generator of the sampled-halo scatter, or the sampled
    # [CII] of the later node redshifts would change with include_unresolved.
    def consuming_grid(initial_conditions, z, models, seed=None, **kwargs):
        np.random.default_rng(seed).standard_normal(1000)
        n = initial_conditions.inputs.simulation_options.HII_DIM
        return {model: np.zeros((n, n, n)) for model in models}

    monkeypatch.setattr(cii_lightcone, "grid_unresolved_halos", consuming_grid)
    lc = SimpleNamespace(inputs=SimpleNamespace(node_redshifts=[7.5, REDSHIFT]))
    without = cii_lightcone.gen_coeval_list(
        lc, "deLooze14", subsampler_run.cache_dir, seed=4
    )
    with_unresolved = cii_lightcone.gen_coeval_list(
        lc, "deLooze14", subsampler_run.cache_dir, seed=4, include_unresolved=True
    )

    for a, b in zip(without, with_unresolved, strict=True):
        np.testing.assert_array_equal(a.cii_intensity, b.cii_intensity)


@pytest.mark.parametrize("model", ["Vallini15", "Horlaville24"])
def test_metallicity_models_need_ts_fluct(subsampler_run, model: str):
    # The test run has USE_TS_FLUCT=False, so 21cmFAST gives no metallicities.
    ic = subsampler_run.cache["InitialConditions"]
    with pytest.raises(ValueError, match="USE_TS_FLUCT"):
        grid_unresolved_halos(ic, REDSHIFT, models=[model], seed=0)
    with pytest.raises(ValueError, match="USE_TS_FLUCT"):
        mean_halo_properties(
            fix_cached_inputs(ic.inputs), REDSHIFT, np.array([1e8]), models=[model]
        )


def test_unknown_model_raises_without_unresolved_halos(subsampler_run, monkeypatch):
    # With no mass range to integrate, the model name must still be checked.
    monkeypatch.setattr(
        cii_unresolved, "unresolved_mass_range", lambda inputs: (1e9, 1e8)
    )
    ic = subsampler_run.cache["InitialConditions"]
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        unresolved_cell_totals(ic, REDSHIFT, models=["Silva15"], seed=0)
    totals = unresolved_cell_totals(ic, REDSHIFT, models=["deLooze14"], seed=0)
    assert not np.any(totals["deLooze14"])


def test_mean_cii_without_scatter_matches_power_law():
    # With no stellar-mass or SFR scatter, the mean deLooze14 luminosity is the
    # power law of the mean SFR times the mean of its 0.4 dex lognormal scatter.
    inputs = p21c.InputParameters.from_template(
        "latest-discrete", random_seed=1
    ).evolve_input_structs(SIGMA_STAR=0.0, SIGMA_SFR_LIM=0.0, SIGMA_SFR_INDEX=0.0)
    masses = np.array([3e8, 1e9])
    means = mean_halo_properties(
        inputs, REDSHIFT, masses, models=["deLooze14"], n_draws=200_000, seed=1
    )

    sfr_yr = (means["sfr"] * un.Msun / un.s).to_value(un.Msun / un.yr)
    expected = 10**7.10 * sfr_yr**1.05 * np.exp((0.4 * np.log(10)) ** 2 / 2)
    np.testing.assert_allclose(means["deLooze14"], expected, rtol=1e-2)


def test_mean_sfr_is_the_integrals_relation():
    # The mean SFR uses the single power law of the 21cmFAST CHMF integrals.
    inputs = p21c.InputParameters.from_template("latest-discrete", random_seed=1)
    ap = inputs.astro_params
    masses = np.array([1e8, 1e9])
    means = mean_halo_properties(inputs, REDSHIFT, masses, models=[], seed=1)

    fstar = 10**ap.F_STAR10 * (masses / 1e10) ** ap.ALPHA_STAR
    fstar *= np.exp(-(10**ap.M_TURN) / masses)
    cosmo = inputs.cosmo_params
    hubble_time = (1 / cosmo.cosmo.H(REDSHIFT)).to_value(un.s)
    expected = fstar * cosmo.OMb / cosmo.OMm * masses / (ap.t_STAR * hubble_time)
    np.testing.assert_allclose(means["sfr"], expected, rtol=5e-3)


def test_integral_stellar_ratio():
    inputs = p21c.InputParameters.from_template("latest-discrete", random_seed=1)
    ap = inputs.astro_params
    low_mass = np.array([1e7])
    expected = 1 / (
        1
        + (10**ap.UPPER_STELLAR_TURNOVER_MASS / 1e10)
        ** (ap.UPPER_STELLAR_TURNOVER_INDEX - ap.ALPHA_STAR)
    )
    np.testing.assert_allclose(
        integral_stellar_ratio(inputs, low_mass), expected, rtol=1e-3
    )

    single = inputs.evolve_input_structs(USE_UPPER_STELLAR_TURNOVER=False)
    np.testing.assert_array_equal(
        integral_stellar_ratio(single, np.array([1e8, 1e11])), 1.0
    )


def test_fix_cached_inputs():
    # Inputs read back from HDF5 hold numpy scalars; a np.bool_ USE_SIGMA_8 is
    # skipped when 21cmFAST fills its C struct, which then reads False.
    inputs = p21c.InputParameters.from_template("latest-discrete", random_seed=1)
    tables = inputs.cosmo_tables
    cached = attrs.evolve(
        inputs,
        cosmo_tables=attrs.evolve(
            tables,
            ps_norm=np.float64(tables.ps_norm),
            USE_SIGMA_8=np.bool_(tables.USE_SIGMA_8),
        ),
    )

    fixed = fix_cached_inputs(cached)
    assert type(fixed.cosmo_tables.USE_SIGMA_8) is bool
    assert fixed.cosmo_tables.cstruct.USE_SIGMA_8
    assert fixed.cosmo_tables.ps_norm == tables.ps_norm


def test_unresolved_mass_range():
    inputs = p21c.InputParameters.from_template(
        "latest-discrete", random_seed=1
    ).evolve_input_structs(SAMPLER_MIN_MASS=1e9)
    m_min, m_max = unresolved_mass_range(inputs)

    assert m_min == pytest.approx(10**inputs.astro_params.M_TURN / 50)
    assert m_max == 1e9


@pytest.mark.parametrize(
    ("template", "match"),
    [("minihalos-discrete", "USE_MINI_HALOS"), ("latest", "CHMF-SAMPLER")],
)
def test_unsupported_runs_raise(template, match):
    inputs = p21c.InputParameters.from_template(template, random_seed=1)
    with pytest.raises(NotImplementedError, match=match):
        unresolved_mass_range(inputs)


def test_cii_flux_of_unresolved_grid(subsampler_run):
    # The [CII] grid is the flux of the gridded luminosity: compare the box total
    # with the CHMF-weighted luminosity integrated over the Lagrangian cells.
    ic = subsampler_run.cache["InitialConditions"]
    grids = grid_unresolved_halos(ic, REDSHIFT, models=["Lagache18"], seed=0)
    per_cell = unresolved_cell_totals(ic, REDSHIFT, models=["Lagache18"], seed=0)
    luminosity = per_cell["Lagache18"].sum() * un.Lsun
    cosmo = fix_cached_inputs(ic.inputs).cosmo_params.cosmo

    np.testing.assert_allclose(
        grids["Lagache18"].sum(),
        cii_flux(luminosity, REDSHIFT, cosmo).value,
        rtol=1e-10,
    )
