"""Tests of tuesday catalogues made from 21cmFAST halo catalogues."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from py21cmfast.io.caching import OutputCache
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.wrapper.cfuncs import convert_halo_properties

from tuesday.core import CoevalCatalog
from tuesday.simulators.py21cmfast.catalogs import (
    HALO_PROPERTY_UNITS,
    catalog_from_halo_catalog,
    get_halo_properties,
)

ACG_FEATURES = {"halo_mass", "stellar_mass_acg", "sfr_acg"}
MCG_FEATURES = {"stellar_mass_mcg", "sfr_mcg"}


def test_perturbed_catalogue(halo_inputs, mock_perturbed_halo_catalog):
    coords = np.arange(15.0).reshape(5, 3)
    halos = mock_perturbed_halo_catalog(
        halo_inputs,
        halo_coords=coords,
        halo_masses=np.full(5, 1e10),
        sfr_acg=np.arange(5.0),
        stellar_masses_acg=np.full(5, 1e8),
    )
    catalog = catalog_from_halo_catalog(halos)

    assert isinstance(catalog, CoevalCatalog)
    assert len(catalog) == 5
    assert catalog.redshift == halos.redshift
    assert catalog.volume == (20 * un.Mpc) ** 3
    np.testing.assert_array_equal(catalog.positions.to_value(un.Mpc), coords)
    assert set(catalog.features) == ACG_FEATURES
    for name in ACG_FEATURES:
        assert catalog[name].unit == HALO_PROPERTY_UNITS[name]
    np.testing.assert_array_equal(
        catalog["sfr_acg"].to_value(un.Msun / un.s), np.arange(5.0)
    )
    np.testing.assert_array_equal(catalog["stellar_mass_acg"].to_value(un.Msun), 1e8)
    # The arrays of the PerturbedHaloCatalog are not copied.
    assert np.shares_memory(catalog["sfr_acg"], halos.get("sfr_acg"))


def test_perturbed_catalogue_with_minihalos(
    minihalo_inputs, mock_perturbed_halo_catalog
):
    halos = mock_perturbed_halo_catalog(
        minihalo_inputs,
        sfr_acg=[1.0, 2.0],
        sfr_mcg=[3.0, 4.0],
        stellar_masses_acg=[1e8, 2e8],
        stellar_masses_mcg=[3e8, 4e8],
    )
    catalog = catalog_from_halo_catalog(halos)
    assert set(catalog.features) == ACG_FEATURES | MCG_FEATURES
    np.testing.assert_array_equal(catalog["sfr_mcg"].to_value(un.Msun / un.s), [3, 4])
    np.testing.assert_array_equal(
        catalog["stellar_mass_mcg"].to_value(un.Msun), [3e8, 4e8]
    )


@pytest.mark.parametrize("with_inputs", [False, True])
def test_other_objects_are_refused(halo_inputs, with_inputs):
    other = SimpleNamespace(inputs=halo_inputs) if with_inputs else object()
    with pytest.raises(TypeError, match="HaloCatalog or a PerturbedHaloCatalog"):
        catalog_from_halo_catalog(other)


def test_halo_catalogue_without_feedback(halo_run):
    # The properties of a HaloCatalog are those that convert_halo_properties gives
    # without feedback grids, which are its defaults in runs without minihalos or
    # reionization feedback.
    halos = halo_run.halo_catalog
    props = convert_halo_properties(
        redshift=halos.redshift,
        inputs=halo_run.inputs,
        halo_masses=halos.get("halo_masses"),
        star_rng=halos.get("star_rng"),
        sfr_rng=halos.get("sfr_rng"),
        xray_rng=halos.get("xray_rng"),
    )
    got = get_halo_properties(halos)
    assert set(got) == set(props)
    for name, value in props.items():
        np.testing.assert_array_equal(got[name], value)

    catalog = catalog_from_halo_catalog(halos)
    assert len(catalog) == halos.n_halos > 1000
    assert set(catalog.features) == ACG_FEATURES
    for name in ACG_FEATURES:
        np.testing.assert_array_equal(catalog[name].value, props[name])
    # Lagrangian positions.
    np.testing.assert_array_equal(catalog.positions.value, halos.get("halo_coords"))


def test_perturbed_catalogue_of_a_run(halo_run):
    # Without minihalos or reionization feedback, the PerturbedHaloCatalog of the run
    # has the properties of the galaxy model alone, at Eulerian positions.
    lagrangian = catalog_from_halo_catalog(halo_run.halo_catalog)
    eulerian = catalog_from_halo_catalog(halo_run.perturbed)
    assert len(eulerian) == len(lagrangian)
    for name in ACG_FEATURES:
        np.testing.assert_array_equal(eulerian[name], lagrangian[name])
    assert not np.array_equal(eulerian.positions, lagrangian.positions)


def test_cached_halo_catalogue(halo_run, tmp_path):
    cache = OutputCache(tmp_path)
    cache.write(halo_run.halo_catalog)
    (path,) = cache.list_datasets(kind="HaloCatalog")
    # The halo arrays are read from the file when they are used.
    cached = catalog_from_halo_catalog(read_output_struct(path))
    direct = catalog_from_halo_catalog(halo_run.halo_catalog)
    np.testing.assert_array_equal(cached.positions, direct.positions)
    for name in ACG_FEATURES:
        np.testing.assert_array_equal(cached[name], direct[name])


def test_minihalos_need_initial_conditions(minihalo_inputs, mock_halo_catalog):
    with pytest.raises(ValueError, match="initial_conditions"):
        catalog_from_halo_catalog(mock_halo_catalog(minihalo_inputs))


def test_minihalo_properties_use_relative_velocities(
    minihalo_inputs, mock_halo_catalog
):
    halos = mock_halo_catalog(minihalo_inputs)
    shape = (minihalo_inputs.simulation_options.HII_DIM,) * 3
    slow = SimpleNamespace(get=lambda name: np.zeros(shape, dtype="f4"))
    fast = SimpleNamespace(get=lambda name: np.full(shape, 60.0, dtype="f4"))

    props_slow = get_halo_properties(halos, slow)
    props_fast = get_halo_properties(halos, fast)
    assert props_slow["sfr_mcg"].sum() > 0
    # Faster streaming raises the minihalo turnover mass, so less star formation.
    assert props_fast["sfr_mcg"].sum() < props_slow["sfr_mcg"].sum()

    catalog = catalog_from_halo_catalog(halos, initial_conditions=slow)
    assert set(catalog.features) == ACG_FEATURES | MCG_FEATURES
    np.testing.assert_array_equal(catalog["sfr_mcg"].value, props_slow["sfr_mcg"])
