"""Tests of the UV luminosity functions of 21cmFAST halo catalogues."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un

from tuesday.core import CoevalCatalog
from tuesday.core.summaries.luminosity_functions import uv_luminosity_function
from tuesday.simulators.py21cmfast.catalogs import catalog_from_halo_catalog
from tuesday.simulators.py21cmfast.summaries.luminosity_functions import (
    calculate_uvlf,
)

SECONDS_PER_YEAR = un.yr.to(un.s)


def test_perturbed_catalogue(halo_inputs, mock_perturbed_halo_catalog):
    sfr = np.array([0.0, 0.1, 1.0, 1.0, 10.0])
    halos = mock_perturbed_halo_catalog(
        halo_inputs,
        # 21cmFAST gives the SFR in Msun/s.
        sfr_acg=sfr / SECONDS_PER_YEAR,
        stellar_masses_acg=np.full(5, 1e8),
    )
    result = calculate_uvlf(halos)

    expected = uv_luminosity_function(
        CoevalCatalog(
            positions=np.zeros((5, 3)) * un.Mpc,
            features={
                "sfr": (sfr / SECONDS_PER_YEAR).astype(np.float32) * un.Msun / un.s
            },
            volume=(20 * un.Mpc) ** 3,
        )
    )
    np.testing.assert_array_equal(result.bin_edges, expected.bin_edges)
    np.testing.assert_array_equal(result.counts, expected.counts)
    assert result.counts.sum() == 4
    assert result.number_density.unit == 1 / (un.mag * un.Mpc**3)
    np.testing.assert_allclose(
        result.number_density.value, expected.number_density.value, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("components", "sfr"), [("both", 3.0), ("acg", 1.0), ("mcg", 2.0)]
)
def test_components(minihalo_inputs, mock_perturbed_halo_catalog, components, sfr):
    calls = []

    def kuv(sfr, stellar_mass, redshift):
        calls.append((sfr, stellar_mass, redshift))
        return np.full_like(sfr, 1.15e-28)

    halos = mock_perturbed_halo_catalog(
        minihalo_inputs,
        sfr_acg=np.array([1.0]) / SECONDS_PER_YEAR,
        sfr_mcg=np.array([2.0]) / SECONDS_PER_YEAR,
        stellar_masses_acg=np.array([1e8]),
        stellar_masses_mcg=np.array([2e8]),
    )
    edges = np.linspace(-24.0, -12.0, 1201) * un.mag
    result = calculate_uvlf(halos, bins=edges, kuv=kuv, components=components)

    magnitude = (51.63 - 2.5 * np.log10(sfr / 1.15e-28)) * un.mag
    assert result.counts.sum() == 1
    i = np.argmax(result.counts)
    assert edges[i] <= magnitude < edges[i + 1]
    # K_UV gets the SFR and stellar mass of the same components.
    ((got_sfr, got_mass, got_z),) = calls
    np.testing.assert_allclose(got_sfr, [sfr], rtol=1e-6)
    np.testing.assert_allclose(got_mass, [1e8 * sfr], rtol=1e-6)
    assert got_z == halos.redshift


def test_runs_without_minihalos_count_atomic_cooling_galaxies(
    halo_inputs, mock_perturbed_halo_catalog
):
    halos = mock_perturbed_halo_catalog(
        halo_inputs,
        sfr_acg=np.array([1.0, 10.0]) / SECONDS_PER_YEAR,
        stellar_masses_acg=np.array([1e8, 1e9]),
    )
    np.testing.assert_array_equal(
        calculate_uvlf(halos, components="both").counts,
        calculate_uvlf(halos, components="acg").counts,
    )


def test_minihalo_component_needs_minihalos(halo_inputs, mock_perturbed_halo_catalog):
    halos = mock_perturbed_halo_catalog(
        halo_inputs, sfr_acg=np.array([1.0]), stellar_masses_acg=np.array([1e8])
    )
    with pytest.raises(ValueError, match="USE_MCGS"):
        calculate_uvlf(halos, components="mcg")


def test_unknown_components(halo_inputs, mock_perturbed_halo_catalog):
    halos = mock_perturbed_halo_catalog(
        halo_inputs, sfr_acg=np.array([1.0]), stellar_masses_acg=np.array([1e8])
    )
    with pytest.raises(ValueError, match="components must be"):
        calculate_uvlf(halos, components="all")


@pytest.mark.parametrize("with_inputs", [False, True])
def test_other_objects_are_refused(halo_inputs, with_inputs):
    other = SimpleNamespace(inputs=halo_inputs) if with_inputs else object()
    with pytest.raises(TypeError, match="HaloCatalog or a PerturbedHaloCatalog"):
        calculate_uvlf(other)


def test_halo_catalogue(halo_run):
    result = calculate_uvlf(halo_run.halo_catalog)
    catalog = catalog_from_halo_catalog(halo_run.halo_catalog)
    sfr = catalog["sfr_acg"].to_value(un.Msun / un.yr).astype(np.float64)
    with np.errstate(divide="ignore"):
        magnitude = 51.63 - 2.5 * np.log10(sfr / 1.15e-28)
    counts, _ = np.histogram(magnitude, np.linspace(-24, -12, 25))
    assert counts.sum() > 100
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_allclose(
        result.number_density.to_value(1 / (un.mag * un.Mpc**3)),
        counts / 0.5 / 20.0**3,
    )


def test_perturbed_catalogue_of_a_run(halo_run):
    np.testing.assert_array_equal(
        calculate_uvlf(halo_run.perturbed).counts,
        calculate_uvlf(halo_run.halo_catalog).counts,
    )


def test_minihalos(minihalo_inputs, mock_halo_catalog):
    halos = mock_halo_catalog(minihalo_inputs)
    shape = (minihalo_inputs.simulation_options.HII_DIM,) * 3
    ics = SimpleNamespace(get=lambda name: np.zeros(shape, dtype="f4"))
    with pytest.raises(ValueError, match="initial_conditions"):
        calculate_uvlf(halos)

    result = calculate_uvlf(halos, initial_conditions=ics, bins=10)
    catalog = catalog_from_halo_catalog(halos, initial_conditions=ics)
    sfr = catalog["sfr_acg"].astype(np.float64) + catalog["sfr_mcg"].astype(np.float64)
    expected = uv_luminosity_function(catalog.with_features(sfr=sfr), bins=10)
    np.testing.assert_array_equal(result.counts, expected.counts)
    assert result.counts.sum() > 0
