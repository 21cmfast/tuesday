"""Tests of the lightcone stores on disk."""

import astropy.units as un
import h5py
import numpy as np
import pytest

from tuesday.core.lightcones import LightconeStore, default_chunks
from tuesday.core.lightcones.store import CHUNK_VALUES, LINE_OF_SIGHT_VALUES


def test_default_chunks():
    """Chunks of about CHUNK_VALUES, whose lines of sight hold LINE_OF_SIGHT_VALUES."""
    for shape in [(1500, 1500, 2500), (12 * 2048**2, 2500), (512, 512, 3000)]:
        chunks = default_chunks(shape)
        assert chunks[1:-1] == shape[1:-1]
        values = np.prod(chunks)
        assert CHUNK_VALUES / 2 < values <= CHUNK_VALUES
        lines = np.prod(chunks[:-1]) * shape[-1]
        assert LINE_OF_SIGHT_VALUES / 2 < lines <= LINE_OF_SIGHT_VALUES
    assert default_chunks((1500, 1500, 2500)) == (4, 1500, 43)
    # small lightcones are one chunk
    assert default_chunks((100, 5)) == (100, 5)
    assert default_chunks((16, 16, 30)) == (16, 16, 30)
    # one row per chunk, if a row is large
    assert default_chunks((3, 2**21, 4)) == (1, 2**21, 1)
    with pytest.raises(ValueError, match="at least one"):
        default_chunks((10,))
    with pytest.raises(ValueError, match="at least one"):
        default_chunks((10, 0))


def _layout(**kw):
    layout = {
        "quantity": "density",
        "shape": (6, 4),
        "distances": np.linspace(100, 130, 4) * un.Mpc,
        "redshifts": np.linspace(6.0, 6.3, 4),
        "node_redshifts": [7.0, 6.5, 6.0],
        "metadata": {"lightconer": "Example", "nside": 1, "kinds": '{"a": "b"}'},
    }
    return layout | kw


def test_create_fill_and_read(tmp_path):
    """A new store is NaN and not done; it is complete once every interval is."""
    path = tmp_path / "sub" / "density.h5"
    with LightconeStore.open_or_create(path, **_layout()) as store:
        assert store.path == path
        assert store.quantity == "density"
        assert store.shape == (6, 4)
        assert np.all(np.isnan(store.lightcone[()]))
        assert store.lightcone.dtype == np.float32
        assert store.lightcone.chunks == (6, 4)
        np.testing.assert_array_equal(store.done, [False, False])
        assert not store.complete
        assert np.all(np.isnan(store.global_mean))
        assert store.metadata["nside"] == 1

        values = np.arange(12.0).reshape(6, 2)
        store.write(slice(0, 2), values)
        store.set_global_mean(0, 1.5)
        store.mark_done(0)
        assert not store.complete
        store.write([3], values[:, :1])
        store.mark_done([1])
        assert store.complete

    with LightconeStore.open(path) as store:
        np.testing.assert_array_equal(store.lightcone[:, :2], values)
        assert np.all(np.isnan(store.lightcone[:, 2]))
        np.testing.assert_array_equal(store.lightcone[:, 3], values[:, 0])
        np.testing.assert_array_equal(store.done, [True, True])
        assert store.complete
        assert store.global_mean[0] == 1.5
        np.testing.assert_allclose(store.distances, _layout()["distances"])
        np.testing.assert_array_equal(store.redshifts, _layout()["redshifts"])
        np.testing.assert_array_equal(store.node_redshifts, [7.0, 6.5, 6.0])
        with h5py.File(path) as f:
            assert f["lightcone_distances"].attrs["unit"] == "Mpc"
        with pytest.raises((OSError, ValueError)):
            store.write(slice(0, 1), values[:, :1])


def test_resume(tmp_path):
    """The same layout resumes the store; another raises, unless clobbered."""
    path = tmp_path / "density.h5"
    with LightconeStore.open_or_create(path, **_layout()) as store:
        store.write(slice(0, 1), np.ones((6, 1)))
        store.mark_done(0)

    with LightconeStore.open_or_create(path, **_layout()) as store:
        np.testing.assert_array_equal(store.done, [True, False])
        np.testing.assert_array_equal(store.lightcone[:, 0], 1)

    other = {
        "quantity": "z_reion",
        "shape": (5, 4),
        "distances": np.linspace(100, 131, 4) * un.Mpc,
        "redshifts": np.linspace(6.0, 6.4, 4),
        "node_redshifts": [7.0, 6.6, 6.0],
        "metadata": {"lightconer": "Example", "nside": 2, "kinds": '{"a": "b"}'},
    }
    for key, value in other.items():
        with pytest.raises(ValueError, match="was made with another"):
            LightconeStore.open_or_create(path, **_layout(**{key: value}))
    with pytest.raises(ValueError, match="metadata 'extra'"):
        LightconeStore.open_or_create(
            path, **_layout(metadata=_layout()["metadata"] | {"extra": 1})
        )

    with LightconeStore.open_or_create(path, **_layout(**other), clobber=True) as store:
        assert store.quantity == "z_reion"
        assert not store.done.any()
        assert np.all(np.isnan(store.lightcone[()]))


def test_errors(tmp_path):
    """Layouts that do not fit, and files that are not stores."""
    with pytest.raises(ValueError, match="one distance and one redshift"):
        LightconeStore.open_or_create(
            tmp_path / "a.h5", **_layout(redshifts=np.zeros(3))
        )
    with pytest.raises(ValueError, match="at least two node redshifts"):
        LightconeStore.open_or_create(tmp_path / "a.h5", **_layout(node_redshifts=[7]))

    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f["x"] = 1
    with pytest.raises(ValueError, match="not a lightcone store"):
        LightconeStore.open(path)
    with pytest.raises(ValueError, match="mode"):
        LightconeStore.open(path, "w")
    path.write_bytes(b"not an HDF5 file")
    with pytest.raises(OSError, match="create it again"):
        LightconeStore.open(path)
    with pytest.raises(FileNotFoundError):
        LightconeStore.open(tmp_path / "missing.h5")
