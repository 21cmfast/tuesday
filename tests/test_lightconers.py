"""Tests of the lightconers, without any simulator."""

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck15, Planck18

from tuesday.core.lightcones import (
    AngularLightconer,
    BracketingLightconer,
    Chunk,
    LightconeContext,
    RectilinearLightconer,
    Snapshot,
    build_lightcone,
    interpolate_in_distance,
)

N = 8
CELL = 2.0 * un.Mpc
REDSHIFTS = (8.0, 7.6, 7.2, 6.8)


@pytest.fixture(scope="module")
def context() -> LightconeContext:
    return LightconeContext(
        cosmo=Planck18,
        box_shape=(N, N, N),
        cell_size=CELL,
        node_redshifts=REDSHIFTS,
    )


def _distance(z: float) -> float:
    return Planck18.comoving_distance(z).to_value(un.Mpc)


def _snapshots(rng=None, fields=("density",)) -> list[Snapshot]:
    """Snapshots of fields that are a fixed box times the comoving distance."""
    rng = rng or np.random.default_rng(0)
    patterns = {f: rng.uniform(1, 2, size=(N, N, N)) for f in fields}
    return [
        Snapshot(z, {f: p * _distance(z) for f, p in patterns.items()})
        for z in REDSHIFTS
    ], patterns


def _between(cls, **kw):
    kw.setdefault("quantities", ("density",))
    return cls.between_redshifts(
        min_redshift=6.9, max_redshift=7.9, resolution=CELL, **kw
    )


# ------------------------------------------------------------------------------
# The small objects
# ------------------------------------------------------------------------------
def test_snapshot_is_lazy():
    calls = []

    def read():
        calls.append(1)
        return np.ones((2, 2, 2))

    snapshot = Snapshot(7, {"density": read, "xH": np.zeros((2, 2, 2))})
    assert snapshot.redshift == 7.0
    assert "density" in snapshot
    assert "velocity" not in snapshot
    assert not calls
    assert snapshot["density"] is snapshot["density"]
    assert len(calls) == 1
    assert "density" in repr(snapshot)
    with pytest.raises(KeyError):
        snapshot["velocity"]


def test_interpolate_in_distance():
    a, b = np.array([1.0, -1.0, 2.0]), np.array([3.0, 5.0, -2.0])
    np.testing.assert_allclose(interpolate_in_distance(2, a, b, 1, 3), [2, 2, 0])
    np.testing.assert_allclose(interpolate_in_distance(1, a, b, 1, 3), a)
    # Where the signs differ, "mean_max" takes the larger value.
    np.testing.assert_allclose(
        interpolate_in_distance(2, a, b, 1, 3, kind="mean_max"), [2, 5, 2]
    )
    with pytest.raises(ValueError, match="kind must be"):
        interpolate_in_distance(2, a, b, 1, 3, kind="nearest")
    with pytest.raises(ValueError, match="same shape"):
        interpolate_in_distance(2, a, b[:2], 1, 3)


def test_spec():
    with pytest.raises(ValueError, match="Either lc_distances or lc_redshifts"):
        RectilinearLightconer()
    with pytest.raises(ValueError, match="lc_distances must be non-negative"):
        RectilinearLightconer(lc_distances=np.linspace(-1, 1, 5) * un.Mpc)
    with pytest.raises(ValueError, match="lc_redshifts must be non-negative"):
        RectilinearLightconer(lc_redshifts=np.linspace(-1, 1, 5))

    from_z = RectilinearLightconer(lc_redshifts=np.array([6.0, 7.0]))
    assert from_z.cosmo is Planck18
    np.testing.assert_allclose(
        from_z.lc_distances.to_value(un.Mpc), [_distance(6), _distance(7)]
    )

    lightconer = _between(RectilinearLightconer)
    np.testing.assert_allclose(np.diff(lightconer.lc_distances.to_value(un.Mpc)), 2.0)
    np.testing.assert_allclose(lightconer.lc_redshifts[0], 6.9, rtol=1e-4)
    assert lightconer == _between(RectilinearLightconer)
    assert lightconer != _between(RectilinearLightconer, quantities=("xH",))
    np.testing.assert_allclose(
        lightconer.get_lc_distances_in_pixels(CELL).value,
        lightconer.lc_distances.to_value(un.Mpc) / 2,
    )

    other = _between(RectilinearLightconer, cosmo=Planck15)
    assert other.cosmo is Planck15


def test_evolve():
    lightconer = _between(RectilinearLightconer)
    more = lightconer.evolve(quantities=("density", "los_velocity"))
    assert more.quantities == ("density", "los_velocity")
    assert more.required_fields() == ("density", "velocity_z")
    assert lightconer.quantities == ("density",)

    shorter = lightconer.evolve(lc_distances=lightconer.lc_distances[:5])
    assert len(shorter.lc_distances) == len(shorter.lc_redshifts) == 5
    assert shorter.index_offset == lightconer.index_offset


def test_wrong_cosmology(context):
    lightconer = _between(RectilinearLightconer, cosmo=Planck15)
    with pytest.raises(ValueError, match="The cosmology of the lightconer"):
        lightconer.start(context)


# ------------------------------------------------------------------------------
# Rectilinear lightcones
# ------------------------------------------------------------------------------
def test_rectilinear(context):
    snapshots, patterns = _snapshots()
    lightconer = _between(RectilinearLightconer)
    session = lightconer.start(context)
    n = len(lightconer.lc_distances)
    assert session.shape == (N, N, n)
    assert session.resume_depth == 1

    lightcone = build_lightcone(lightconer, snapshots, context, dtype=float)["density"]
    # The field is linear in distance, so the interpolation is exact. By default
    # (index_offset = n) slice i of the lightcone is slice i of the box, which wraps.
    index = np.arange(n) % N
    expected = patterns["density"][:, :, index] * lightconer.lc_distances.value
    np.testing.assert_allclose(lightcone, expected, rtol=1e-12)

    shifted = _between(RectilinearLightconer, index_offset=3)
    lightcone = build_lightcone(shifted, snapshots, context, dtype=float)["density"]
    expected = patterns["density"][:, :, (index + 3 - n) % N]
    np.testing.assert_allclose(
        lightcone, expected * lightconer.lc_distances.value, rtol=1e-12
    )


def test_chunks_cover_the_lightcone_once(context):
    snapshots, _ = _snapshots(fields=("density", "velocity_z"))
    lightconer = _between(RectilinearLightconer, quantities=("density", "los_velocity"))
    session = lightconer.start(context)
    counts = {q: np.zeros(session.shape[-1], dtype=int) for q in lightconer.quantities}

    assert list(session.ingest(snapshots[0])) == []
    for snapshot in snapshots[1:]:
        for chunk in session.ingest(snapshot):
            assert isinstance(chunk, Chunk)
            assert chunk.index[0] is Ellipsis
            assert chunk.values.shape == (N, N)
            counts[chunk.quantity][chunk.index[-1]] += 1
    assert session.finish() == ()
    for count in counts.values():
        assert np.all(count == 1)


def test_snapshots_are_only_read_when_used(context):
    """The boxes of an interval without slices are not asked for."""
    rng = np.random.default_rng(1)
    read = []

    def box(z):
        def get():
            read.append(z)
            return rng.uniform(size=(N, N, N))

        return get

    snapshots = [Snapshot(z, {"density": box(z)}) for z in REDSHIFTS]
    # Slices between the last two snapshots only.
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=6.9, max_redshift=7.1, resolution=CELL, quantities=("density",)
    )
    lightcone = build_lightcone(lightconer, snapshots, context)["density"]
    assert np.all(lightcone != 0)
    assert sorted(read) == [6.8, 7.2]


# ------------------------------------------------------------------------------
# Writing a lightconer
# ------------------------------------------------------------------------------
class Recording(RectilinearLightconer):
    """A lightconer that records what it is asked, and defines a quantity."""

    def fields_for(self, quantity):
        if quantity == "overdensity":
            return ("density",)
        return super().fields_for(quantity)

    def prepare_slice(self, slice_, session):
        calls = session.cache.run.setdefault("prepared", [])
        calls.append(slice_.index)
        # What depends on the two snapshots is made once for all their slices.
        if "mean" not in session.cache.interval:
            session.cache.run["means"] = session.cache.run.get("means", 0) + 1
            session.cache.interval["mean"] = np.mean(slice_.far["density"])
        session.cache.slice["quantities"] = []
        return super().prepare_slice(slice_, session)

    def make_slice(self, quantity, slice_, session):
        session.cache.slice["quantities"].append(quantity)
        session.cache.run["last"] = list(session.cache.slice["quantities"])
        if quantity == "overdensity":
            density = self.make_field_slice(
                slice_.near["density"], slice_.far["density"], slice_, session
            )
            return density / session.cache.interval["mean"] - 1
        return super().make_slice(quantity, slice_, session)


def test_hooks_and_cache_lifetimes(context):
    snapshots, _ = _snapshots()
    lightconer = _between(Recording, quantities=("density", "overdensity"))
    assert lightconer.required_fields() == ("density",)

    session = lightconer.start(context)
    n = session.shape[-1]
    out = {q: np.zeros(session.shape) for q in lightconer.quantities}
    for snapshot in snapshots:
        for quantity, index, values in session.ingest(snapshot):
            out[quantity][index] = values
        # Between snapshots, only what is kept for the run remains.
        assert session.cache.slice == {}
        assert session.cache.interval == {}

    # Each slice was prepared once, for both quantities.
    assert sorted(session.cache.run["prepared"]) == list(range(n))
    assert session.cache.run["last"] == ["density", "overdensity"]
    # There are slices in all three intervals between the four snapshots.
    assert session.cache.run["means"] == 3
    assert np.all(out["density"] > 0)
    assert np.all(np.abs(out["overdensity"]) < 1)

    session.finish()
    assert session.cache.run == {}


def test_base_class_is_abstract(context):
    lightconer = _between(BracketingLightconer)
    with pytest.raises(NotImplementedError):
        lightconer.start(context)


# ------------------------------------------------------------------------------
# Angular lightcones
# ------------------------------------------------------------------------------
def _ring(n: int, latitude: float = 0.2) -> dict:
    return {
        "latitude": np.full(n, latitude),
        "longitude": np.arange(n) * 2 * np.pi / n,
    }


def test_angular_validation():
    distances = np.array([100.0]) * un.Mpc
    lon = np.linspace(0, 1, 6)
    with pytest.raises(ValueError, match="longitude must be 1-dimensional"):
        AngularLightconer(longitude=lon[None], latitude=lon, lc_distances=distances)
    with pytest.raises(ValueError, match="longitude must be in the range"):
        AngularLightconer(longitude=lon + 7, latitude=lon, lc_distances=distances)
    with pytest.raises(ValueError, match="must have the same shape"):
        AngularLightconer(longitude=lon, latitude=lon[:3], lc_distances=distances)
    with pytest.raises(ValueError, match="cannot be split"):
        AngularLightconer(
            longitude=lon, latitude=lon, n_subsamples=4, lc_distances=distances
        )


def test_angular_uniform_field(context):
    """A field that is uniform in each snapshot gives slices that are uniform."""
    snapshots = [
        Snapshot(z, {"density": np.full((N, N, N), _distance(z))}) for z in REDSHIFTS
    ]
    for kw in ({}, {"tile_first": True}, {"tile_first": False}, {"n_subsamples": 4}):
        lightconer = _between(AngularLightconer, **_ring(16), **kw)
        session = lightconer.start(context)
        assert session.shape == (
            16 // kw.get("n_subsamples", 1),
            len(lightconer.lc_distances),
        )
        lightcone = build_lightcone(lightconer, snapshots, context, dtype=float)
        np.testing.assert_allclose(
            lightcone["density"],
            np.broadcast_to(lightconer.lc_distances.value, session.shape),
            rtol=1e-10,
        )


@pytest.mark.parametrize("order", [0, 1, 3])
def test_angular_both_ways_agree(context, order):
    """Sampling the boxes first, or interpolating them first, gives the same."""
    rng = np.random.default_rng(4)
    fields = ("density", "z_reion", "velocity_x", "velocity_y", "velocity_z")
    snapshots = []
    for z in REDSHIFTS:
        boxes = {f: rng.normal(size=(N, N, N)) for f in fields}
        # "Not yet reionized" cells, more of them at higher redshift.
        boxes["z_reion"] = np.where(
            rng.uniform(size=(N, N, N)) < z / 10, -1.0, rng.uniform(6, 12, (N, N, N))
        )
        snapshots.append(Snapshot(z, boxes))

    kw = {
        **_ring(24),
        "interpolation_order": order,
        "origin": np.full(3, N / 2) * un.pixel,
        "quantities": ("density", "z_reion", "los_velocity"),
    }
    first = build_lightcone(
        _between(AngularLightconer, tile_first=True, **kw), snapshots, context, float
    )
    second = build_lightcone(
        _between(AngularLightconer, tile_first=False, **kw), snapshots, context, float
    )
    for quantity in kw["quantities"]:
        assert np.any(first[quantity] != 0)
        np.testing.assert_allclose(
            first[quantity], second[quantity], rtol=0, atol=1e-10, err_msg=quantity
        )


def test_angular_subsamples_average(context):
    """A pixel with sub-samples is the mean of the field over its directions."""
    snapshots, _ = _snapshots(
        fields=("density", "velocity_x", "velocity_y", "velocity_z")
    )
    rng = np.random.default_rng(2)
    directions = {
        "latitude": rng.uniform(-1, 1, 4 * 10),
        "longitude": rng.uniform(0, 2 * np.pi, 4 * 10),
    }
    kw = {"quantities": ("density", "los_velocity"), **directions}
    plain = build_lightcone(
        _between(AngularLightconer, **kw), snapshots, context, float
    )
    averaged = build_lightcone(
        _between(AngularLightconer, n_subsamples=4, **kw), snapshots, context, float
    )
    for quantity, values in plain.items():
        assert averaged[quantity].shape == (10, values.shape[-1])
        np.testing.assert_allclose(
            averaged[quantity],
            values.reshape(4, 10, -1).mean(axis=0),
            rtol=1e-10,
            err_msg=quantity,
        )


def test_like_rectilinear(context):
    """A patch of sky that spans the box at one redshift, with a pixel per cell."""
    angular = AngularLightconer.like_rectilinear(
        n_cells=N,
        cell_size=CELL,
        match_at_z=6.9,
        max_redshift=7.9,
        quantities=("density",),
    )
    assert angular.n_pixels == N * N
    assert angular.lc_redshifts[0] == pytest.approx(6.9, rel=1e-4)
    np.testing.assert_allclose(np.diff(angular.lc_distances.to_value(un.Mpc)), 2.0)
    # The patch spans the box at the distance of the first slice.
    width = np.ptp(angular.longitude) * angular.lc_distances[0]
    assert width.to_value(un.Mpc) == pytest.approx(N * 2.0)
    # The observer is that far behind the box.
    assert angular.origin[2].value == pytest.approx(-_distance(6.9) / 2)

    snapshots, _ = _snapshots()
    lightcone = build_lightcone(angular, snapshots, context, float)["density"]
    assert lightcone.shape == (N * N, len(angular.lc_distances))
    assert np.all(lightcone > 0)


def test_healpix():
    lightconer = AngularLightconer.healpix(
        2, subsample_level=1, lc_distances=np.array([100.0, 101.0]) * un.Mpc
    )
    assert lightconer.longitude.size == 4 * 48
    assert lightconer.n_pixels == 48
    assert lightconer.n_subsamples == 4
    between = AngularLightconer.healpix(
        2, min_redshift=7.0, max_redshift=7.1, resolution=CELL
    )
    assert between.n_pixels == 48
    assert between.lc_redshifts[0] == pytest.approx(7.0, rel=1e-4)


def test_follows_the_protocol_of_21cmfast(context):
    """The lightconers can be passed to py21cmfast.run_lightcone."""
    protocol = pytest.importorskip("py21cmfast.lightconers")
    for cls, kw in ((RectilinearLightconer, {}), (AngularLightconer, _ring(4))):
        lightconer = _between(cls, **kw)
        assert isinstance(lightconer, protocol.LightconerProtocol)
        assert isinstance(lightconer.start(context), protocol.LightconeSession)
    assert isinstance(Snapshot(7.0, {}), protocol.Snapshot)
