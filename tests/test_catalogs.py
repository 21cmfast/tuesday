"""Tests of the Catalog class."""

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology.units import littleh
from astropy.table import QTable, Table

from tuesday.core import Catalog

N = 5


@pytest.fixture
def catalog() -> Catalog:
    rng = np.random.default_rng(0)
    return Catalog(
        positions=rng.uniform(0, 10, (N, 3)) * un.Mpc,
        features={
            "mass": np.arange(1.0, N + 1) * 1e10 * un.Msun,
            "spin": rng.uniform(0, 0.1, N) * un.dimensionless_unscaled,
        },
        volume=(10 * un.Mpc) ** 3,
        redshift=7.0,
    )


def test_catalog(catalog):
    assert len(catalog) == N
    assert catalog.redshift == 7.0
    assert catalog.volume == 1000 * un.Mpc**3
    np.testing.assert_allclose(
        catalog["mass"].to_value(un.Msun), [1e10, 2e10, 3e10, 4e10, 5e10]
    )
    assert set(catalog.features) == {"mass", "spin"}


def test_positions_only():
    catalog = Catalog(positions=np.zeros((3, 3)) * un.Mpc / littleh)
    assert len(catalog) == 3
    assert dict(catalog.features) == {}
    assert catalog.volume is None
    assert catalog.redshift is None


def test_missing_feature(catalog):
    with pytest.raises(KeyError, match=r"no feature 'sfr'.*'mass', 'spin'"):
        catalog["sfr"]


def test_features_are_read_only(catalog):
    with pytest.raises(TypeError):
        catalog.features["sfr"] = np.ones(N) * un.Msun / un.yr
    with pytest.raises(AttributeError):
        catalog.volume = 1 * un.Mpc**3
    with pytest.raises(AttributeError):
        catalog.other = 1


def test_arrays_are_not_copied():
    positions = np.zeros((N, 3), dtype=np.float32) * un.Mpc
    mass = np.ones(N, dtype=np.float32) * un.Msun
    catalog = Catalog(positions=positions, features={"mass": mass})
    assert np.shares_memory(catalog.positions, positions)
    assert np.shares_memory(catalog["mass"], mass)
    assert catalog["mass"].dtype == np.float32


def test_positions_are_not_a_feature(catalog):
    assert "position" not in catalog.features
    with pytest.raises(KeyError, match="no feature 'position'"):
        catalog["position"]
    with pytest.raises(ValueError, match="No feature may be called 'position'"):
        Catalog(
            positions=np.zeros((N, 3)) * un.Mpc,
            features={"position": np.zeros((N, 3)) * un.Mpc},
        )


def test_repr(catalog):
    assert repr(catalog) == (
        "<Catalog of 5 objects, features ['mass', 'spin'], volume 1000.0 Mpc3, "
        "redshift 7.0>"
    )


def test_given_features_dict_is_not_used(catalog):
    features = {"mass": np.ones(N) * un.Msun}
    catalog = Catalog(positions=catalog.positions, features=features)
    features["sfr"] = np.ones(N) * un.Msun / un.yr
    assert "sfr" not in catalog.features


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((N, 3)),
        np.zeros((N, 3)) * un.Msun,
        np.zeros((N, 2)) * un.Mpc,
        np.zeros(3) * un.Mpc,
    ],
)
def test_bad_positions(positions):
    with pytest.raises(ValueError, match="positions"):
        Catalog(positions=positions)


def test_features_must_be_quantities():
    with pytest.raises(TypeError, match="'mass' must be an astropy Quantity"):
        Catalog(positions=np.zeros((N, 3)) * un.Mpc, features={"mass": np.ones(N)})


@pytest.mark.parametrize("shape", [(N + 1,), (), (2, N)])
def test_features_need_one_value_per_object(shape):
    with pytest.raises(ValueError, match="'mass' has shape"):
        Catalog(
            positions=np.zeros((N, 3)) * un.Mpc,
            features={"mass": np.ones(shape) * un.Msun},
        )


def test_features_may_have_more_axes():
    catalog = Catalog(
        positions=np.zeros((N, 3)) * un.Mpc,
        features={"velocity": np.zeros((N, 3)) * un.km / un.s},
    )
    assert catalog["velocity"].shape == (N, 3)


def test_volume_must_be_a_volume():
    with pytest.raises(ValueError, match="volume must be a volume"):
        Catalog(positions=np.zeros((N, 3)) * un.Mpc, volume=10 * un.Mpc**2)


@pytest.mark.parametrize(
    "volume",
    [
        0.0 * un.Mpc**3,
        -1.0 * un.Mpc**3,
        np.inf * un.Mpc**3,
        np.nan * un.Mpc**3,
        np.array([1.0, 2.0]) * un.Mpc**3,
    ],
    ids=["zero", "negative", "infinite", "nan", "array"],
)
def test_volume_must_be_finite_and_positive(volume):
    with pytest.raises(ValueError, match="single finite, positive value"):
        Catalog(positions=np.zeros((N, 3)) * un.Mpc, volume=volume)


def test_volume_must_be_a_quantity():
    with pytest.raises(ValueError, match="Quantity with a unit of volume"):
        Catalog(positions=np.zeros((N, 3)) * un.Mpc, volume=1000.0)


def test_select_checks_the_volume(catalog):
    with pytest.raises(ValueError, match="single finite, positive value"):
        catalog.select(np.array([0, 1]), volume=0.0 * un.Mpc**3)


def test_volume_with_littleh():
    catalog = Catalog(
        positions=np.zeros((N, 3)) * un.Mpc, volume=(10 * un.Mpc / littleh) ** 3
    )
    assert catalog.volume.unit == (un.Mpc / littleh) ** 3


def test_with_features(catalog):
    sfr = np.ones(N) * un.Msun / un.yr
    new = catalog.with_features(sfr=sfr, mass=2 * catalog["mass"])
    assert set(new.features) == {"mass", "spin", "sfr"}
    np.testing.assert_allclose(new["mass"], 2 * catalog["mass"])
    # The arrays are not copied.
    assert np.shares_memory(new["sfr"], sfr)
    assert np.shares_memory(new.positions, catalog.positions)
    assert new.volume == catalog.volume
    assert new.redshift == catalog.redshift
    # The original catalogue is unchanged.
    assert set(catalog.features) == {"mass", "spin"}


def test_with_features_validates(catalog):
    with pytest.raises(ValueError, match="'sfr' has shape"):
        catalog.with_features(sfr=np.ones(N + 1) * un.Msun / un.yr)


def test_select_by_feature(catalog):
    heavy = catalog.select(catalog["mass"] > 2.5e10 * un.Msun)
    assert len(heavy) == 3
    np.testing.assert_allclose(heavy["mass"].to_value(un.Msun), [3e10, 4e10, 5e10])
    np.testing.assert_array_equal(heavy.positions, catalog.positions[2:])
    np.testing.assert_array_equal(heavy["spin"], catalog["spin"][2:])
    assert heavy.volume is catalog.volume
    assert heavy.redshift == catalog.redshift


def test_select_region(catalog):
    slab = catalog.select(np.array([0, 1]), volume=(5 * un.Mpc) ** 3)
    assert len(slab) == 2
    assert slab.volume == 125 * un.Mpc**3


def test_to_table(catalog):
    table = catalog.to_table()
    assert isinstance(table, QTable)
    assert table.colnames == ["position", "mass", "spin"]
    assert table.meta == {"volume": catalog.volume, "redshift": 7.0}
    assert np.shares_memory(table["mass"], catalog["mass"])
    # Changing the table does not change the catalogue.
    table["sfr"] = np.ones(N) * un.Msun / un.yr
    table.meta["redshift"] = 8.0
    assert "sfr" not in catalog.features
    assert catalog.redshift == 7.0


def test_from_table(catalog):
    new = Catalog.from_table(catalog.to_table())
    np.testing.assert_array_equal(new.positions, catalog.positions)
    assert set(new.features) == {"mass", "spin"}
    np.testing.assert_array_equal(new["mass"], catalog["mass"])
    assert new.volume == catalog.volume
    assert new.redshift == catalog.redshift
    assert np.shares_memory(new["mass"], catalog["mass"])


def test_from_plain_table():
    table = Table(
        {
            "position": np.zeros((N, 3)) * un.Mpc,
            "mass": np.ones(N) * un.Msun,
        }
    )
    catalog = Catalog.from_table(table)
    assert catalog["mass"].unit == un.Msun
    assert catalog.volume is None
    assert catalog.redshift is None


def test_from_table_needs_positions():
    with pytest.raises(ValueError, match="no column 'position'"):
        Catalog.from_table(QTable({"mass": np.ones(N) * un.Msun}))


def test_from_table_needs_units():
    table = QTable({"position": np.zeros((N, 3)) * un.Mpc, "id": np.arange(N)})
    with pytest.raises(TypeError, match="'id' must be an astropy Quantity"):
        Catalog.from_table(table)
