"""Tests of the two ways of making a slice of an angular lightcone, and the choice."""

import numpy as np
import pytest

from tuesday.core import lightcones
from tuesday.core.lightcones import (
    CELLS_PER_SAMPLE,
    interpolate_then_sample,
    sample_then_interpolate,
    samples_first,
    shell_slice,
)


@pytest.fixture(scope="module")
def boxes() -> tuple[np.ndarray, np.ndarray]:
    """Two random boxes of 6 x 7 x 8 cells, with values of both signs."""
    rng = np.random.default_rng(1)
    return tuple(rng.normal(size=(2, 6, 7, 8)))


@pytest.fixture(scope="module")
def directions() -> np.ndarray:
    """Random unit vectors."""
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(3, 300))
    return vectors / np.linalg.norm(vectors, axis=0)


def linear(a, b):
    """A linear interpolation, at 30 % of the way from a to b."""
    return 0.7 * a + 0.3 * b


def mean_max(a, b):
    """21cmFAST's "mean_max": the mean, or the larger value where the signs differ."""
    out = linear(a, b)
    flag = a * b < 0
    out[flag] = np.maximum(a, b)[flag]
    return out


KW = {"origin": (2.0, 3.5, 4.0), "threads": 2, "chunk_size": 64}


@pytest.mark.parametrize("order", [0, 1])
def test_sample_then_interpolate_equals_interpolate_then_sample(
    boxes, directions, order
):
    """Both ways give the same slice, for a linear interpolation and for mean_max."""
    for interpolate, is_linear in ((linear, True), (mean_max, False)):
        expected = interpolate_then_sample(
            interpolate, *boxes, directions, 11.3, order=order, **KW
        )
        result = sample_then_interpolate(
            interpolate, *boxes, directions, 11.3, order=order, linear=is_linear, **KW
        )
        assert result.shape == (directions.shape[1],)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_nonlinear_needs_linear_false(boxes, directions):
    """Treating mean_max as linear when sampling first gives a different slice."""
    expected = interpolate_then_sample(mean_max, *boxes, directions, 11.3, **KW)
    wrong = sample_then_interpolate(
        mean_max, *boxes, directions, 11.3, linear=True, **KW
    )
    assert not np.allclose(wrong, expected)


def test_linear_has_no_default(boxes, directions):
    """Callers must say whether the interpolation is linear."""
    with pytest.raises(TypeError, match="linear"):
        sample_then_interpolate(linear, *boxes, directions, 11.3)
    with pytest.raises(TypeError, match="linear"):
        shell_slice(linear, *boxes, directions, 11.3)


def test_samples_first():
    """Sample first above CELLS_PER_SAMPLE cells per direction, for orders 0 and 1."""
    assert samples_first(100 * CELLS_PER_SAMPLE + 1, 100)
    assert not samples_first(100 * CELLS_PER_SAMPLE, 100)
    assert samples_first(10**9, 1, order=0)
    assert not samples_first(10**9, 1, order=3)
    # the examples of the docstring
    assert samples_first(1500**3, 2**15)
    assert not samples_first(228**3, 12 * 1024**2)


@pytest.mark.parametrize(
    ("sample_first", "n_directions", "expected"),
    [
        (None, 10, "sample"),
        (None, 30, "interpolate"),
        (True, 30, "sample"),
        (False, 10, "interpolate"),
    ],
)
def test_shell_slice_chooses(
    monkeypatch, boxes, directions, sample_first, n_directions, expected
):
    """shell_slice takes the cheaper way (6*7*8 = 336 cells: 21 per direction at 16)."""
    called = []
    for name in ("sample_then_interpolate", "interpolate_then_sample"):
        original = getattr(lightcones.slices, name)

        def spy(*args, _name=name, _original=original, **kw):
            called.append(_name.split("_")[0])
            return _original(*args, **kw)

        monkeypatch.setattr(lightcones.slices, name, spy)

    result = shell_slice(
        mean_max,
        *boxes,
        directions[:, :n_directions],
        11.3,
        linear=False,
        sample_first=sample_first,
        **KW,
    )
    assert called == [expected]
    np.testing.assert_allclose(
        result,
        interpolate_then_sample(
            mean_max, *boxes, directions[:, :n_directions], 11.3, **KW
        ),
        rtol=1e-12,
        atol=1e-12,
    )
