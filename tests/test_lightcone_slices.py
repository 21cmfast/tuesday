"""Tests of the two ways of making a slice of an angular lightcone, and the choice."""

import numpy as np
import pytest
from cosmotile import make_lightcone_slice_interpolator
from scipy.ndimage import map_coordinates

from tuesday.core import lightcones
from tuesday.core.lightcones import (
    CELLS_PER_SAMPLE,
    interpolate_then_sample,
    n_samples,
    sample_cellwise,
    sample_then_interpolate,
    samples_first,
    shell_slice,
)


@pytest.fixture(scope="module")
def boxes() -> tuple[np.ndarray, np.ndarray]:
    """Two random boxes of 6 x 7 x 8 cells, with values of both signs."""
    rng = np.random.default_rng(1)
    return tuple(rng.normal(size=(2, 6, 7, 8)))


def interpolator(n: int, order: int = 1, n_subsamples: int = 1):
    """A cosmotile interpolator onto n random directions (n_subsamples each)."""
    rng = np.random.default_rng(3)
    shape = (n_subsamples, n)
    return make_lightcone_slice_interpolator(
        latitude=rng.uniform(-1.5, 1.5, shape),
        longitude=rng.uniform(0, 2 * np.pi, shape),
        distance_to_shell=11.3,
        interpolation_order=order,
        origin=np.array([2.0, 3.5, 4.0]),
    )


def linear(a, b):
    """A linear interpolation, at 30 % of the way from a to b."""
    return 0.7 * a + 0.3 * b


def mean_max(a, b):
    """21cmFAST's "mean_max": the mean, or the larger value where the signs differ."""
    out = linear(a, b)
    flag = a * b < 0
    out[flag] = np.maximum(a, b)[flag]
    return out


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("n_subsamples", [1, 4])
def test_sample_then_interpolate_equals_interpolate_then_sample(
    boxes, order, n_subsamples
):
    """Both ways give the same slice, for a linear interpolation and for mean_max."""
    shell = interpolator(300, order=order, n_subsamples=n_subsamples)
    for interpolate, is_linear in ((linear, True), (mean_max, False)):
        expected = interpolate_then_sample(interpolate, *boxes, shell)
        result = sample_then_interpolate(interpolate, *boxes, shell, linear=is_linear)
        assert result.shape == (300,)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_nonlinear_needs_linear_false(boxes):
    """Treating mean_max as linear when sampling first gives a different slice."""
    shell = interpolator(300)
    expected = interpolate_then_sample(mean_max, *boxes, shell)
    wrong = sample_then_interpolate(mean_max, *boxes, shell, linear=True)
    assert not np.allclose(wrong, expected)


def test_linear_has_no_default(boxes):
    """Callers must say whether the interpolation is linear."""
    with pytest.raises(TypeError, match="linear"):
        sample_then_interpolate(linear, *boxes, interpolator(10))
    with pytest.raises(TypeError, match="linear"):
        shell_slice(linear, *boxes, interpolator(10))


def test_higher_orders_interpolate_first(boxes):
    """Order 3 cannot sample first; shell_slice interpolates first for it."""
    shell = interpolator(5, order=3)
    with pytest.raises(ValueError, match="orders 0 and 1"):
        sample_then_interpolate(linear, *boxes, shell, linear=True)
    np.testing.assert_array_equal(
        shell_slice(mean_max, *boxes, shell, linear=False, sample_first=True),
        interpolate_then_sample(mean_max, *boxes, shell),
    )


def test_sample_cellwise(boxes):
    """The rule at the eight cells around each point, weighted trilinearly."""
    rng = np.random.default_rng(7)
    coordinates = rng.uniform(-20, 20, size=(3, 50))
    expected = map_coordinates(
        mean_max(*boxes), coordinates, order=1, mode="grid-wrap", prefilter=False
    )
    np.testing.assert_allclose(
        sample_cellwise(mean_max, boxes, coordinates), expected, rtol=1e-12
    )
    weights = np.array([0.5, 0.25, 0.25])
    np.testing.assert_allclose(
        sample_cellwise(mean_max, boxes, coordinates[:, :48], weights=weights),
        (weights[:, None] * expected[:48].reshape(3, 16)).sum(axis=0),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="same shape"):
        sample_cellwise(mean_max, (boxes[0], boxes[1][:-1]), coordinates)


def test_samples_first():
    """Sample first above CELLS_PER_SAMPLE cells per point, for orders 0 and 1."""
    assert samples_first(100 * CELLS_PER_SAMPLE + 1, 100)
    assert not samples_first(100 * CELLS_PER_SAMPLE, 100)
    assert samples_first(10**9, 1, order=0)
    assert not samples_first(10**9, 1, order=3)
    # the examples of the docstring
    assert samples_first(1500**3, 2**15)
    assert not samples_first(228**3, 12 * 1024**2)
    assert n_samples(interpolator(10, n_subsamples=4)) == 40


@pytest.mark.parametrize(
    ("sample_first", "n_directions", "expected"),
    [
        (None, 10, "sample"),
        (None, 30, "interpolate"),
        (True, 30, "sample"),
        (False, 10, "interpolate"),
    ],
)
def test_shell_slice_chooses(monkeypatch, boxes, sample_first, n_directions, expected):
    """shell_slice takes the cheaper way (6*7*8 = 336 cells: 21 per point at 16)."""
    called = []
    for name in ("sample_then_interpolate", "interpolate_then_sample"):
        original = getattr(lightcones.slices, name)

        def spy(*args, _name=name, _original=original, **kw):
            called.append(_name.split("_")[0])
            return _original(*args, **kw)

        monkeypatch.setattr(lightcones.slices, name, spy)

    shell = interpolator(n_directions)
    result = shell_slice(
        mean_max, *boxes, shell, linear=False, sample_first=sample_first
    )
    assert called == [expected]
    np.testing.assert_allclose(
        result,
        interpolate_then_sample(mean_max, *boxes, shell),
        rtol=1e-12,
        atol=1e-12,
    )
