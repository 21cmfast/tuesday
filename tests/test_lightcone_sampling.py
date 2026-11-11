"""Tests of the sampling of periodic boxes on spherical shells."""

import numpy as np
import pytest

from tuesday.core.lightcones import (
    average_subsamples,
    line_of_sight_component,
    sample_cellwise,
    sample_periodic,
    sample_shell,
    sample_shell_cellwise,
    shell_coordinates,
)


@pytest.fixture(scope="module")
def boxes() -> tuple[np.ndarray, np.ndarray]:
    """Two random boxes of 6 x 7 x 8 cells, with values of both signs."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(6, 7, 8)), rng.normal(size=(6, 7, 8))


@pytest.fixture(scope="module")
def coordinates() -> np.ndarray:
    """Random points inside and outside the box, also at negative coordinates."""
    rng = np.random.default_rng(2)
    return rng.uniform(-20, 30, size=(3, 500))


@pytest.fixture(scope="module")
def directions() -> np.ndarray:
    """Random unit vectors."""
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(3, 300))
    return vectors / np.linalg.norm(vectors, axis=0)


def test_sample_periodic_at_cells(boxes):
    """At the centre of a cell (integer coordinates), any order gives the cell value."""
    box = boxes[0]
    index = np.array([[0, 5, -1, 13], [0, 6, 7, -8], [0, 7, 3, 25]])
    expected = box[tuple(index % np.array(box.shape)[:, None])]
    for order in (0, 1):
        np.testing.assert_allclose(
            sample_periodic(box, index.astype(float), order=order),
            expected,
            rtol=0,
            atol=1e-14,
        )


def test_sample_periodic_linear_between_cells(boxes):
    """Order 1 halfway between two cells along an axis gives their mean, wrapping."""
    box = boxes[0]
    coordinates = np.array([[2.5, 5.5, 2.0], [3.0, 1.0, 6.5], [4.0, 7.0, 7.0]])
    expected = [
        (box[2, 3, 4] + box[3, 3, 4]) / 2,
        (box[5, 1, 7] + box[0, 1, 7]) / 2,
        (box[2, 6, 7] + box[2, 0, 7]) / 2,
    ]
    np.testing.assert_allclose(
        sample_periodic(box, coordinates, order=1), expected, rtol=1e-14
    )


def test_sample_periodic_is_periodic(boxes, coordinates):
    """Shifting the points by whole box lengths does not change the samples."""
    box = boxes[0]
    shift = np.array(box.shape)[:, None] * np.array([[2], [-1], [3]])
    for order in (0, 1):
        np.testing.assert_allclose(
            sample_periodic(box, coordinates + shift, order=order),
            sample_periodic(box, coordinates, order=order),
            rtol=1e-12,
            atol=1e-12,
        )


def test_sample_periodic_order_0_takes_nearest_cell(boxes, coordinates):
    """Order 0 takes the value of the nearest cell."""
    box = boxes[0]
    nearest = np.round(coordinates).astype(int) % np.array(box.shape)[:, None]
    np.testing.assert_array_equal(
        sample_periodic(box, coordinates, order=0), box[tuple(nearest)]
    )


def test_sample_periodic_errors(boxes, coordinates):
    """Only orders 0 and 1, and only 3D boxes."""
    with pytest.raises(ValueError, match="order must be 0 or 1"):
        sample_periodic(boxes[0], coordinates, order=3)
    with pytest.raises(ValueError, match="box must be 3D"):
        sample_periodic(boxes[0][0], coordinates[:2])


def test_sample_cellwise_linear_rule(boxes, coordinates):
    """For a linear rule, combining at the cells equals combining the samples."""
    box_a, box_b = boxes
    result = sample_cellwise(lambda a, b: 0.3 * a + 0.7 * b, boxes, coordinates)
    expected = 0.3 * sample_periodic(box_a, coordinates) + 0.7 * sample_periodic(
        box_b, coordinates
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sample_cellwise_nonlinear_rule(boxes, coordinates):
    """For any rule, the result is the trilinear sample of the combined box."""

    def rule(a, b):
        out = (a + b) / 2
        flag = a * b < 0
        out[flag] = np.maximum(a, b)[flag]
        return out

    expected = sample_periodic(rule(*boxes), coordinates, order=1)
    result = sample_cellwise(rule, boxes, coordinates)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    # and it is not the same as combining the samples
    assert not np.allclose(
        result, rule(*(sample_periodic(b, coordinates) for b in boxes))
    )


def test_sample_cellwise_errors(boxes, coordinates):
    """The boxes must be 3D and of the same shape."""
    with pytest.raises(ValueError, match="same shape"):
        sample_cellwise(np.add, (boxes[0], boxes[1][:-1]), coordinates)


def test_shell_coordinates(directions):
    """Points at the radius from the origin along each direction."""
    coordinates = shell_coordinates(directions, 7.5, origin=(1.0, 2.0, 3.0))
    offsets = coordinates - np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=0), 7.5, rtol=1e-12)
    np.testing.assert_allclose(offsets / 7.5, directions, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(shell_coordinates(directions, 2.0), 2.0 * directions)


def test_line_of_sight_component(directions):
    """Positive towards the observer, zero across the line of sight."""
    speeds = np.linspace(-3, 3, directions.shape[1])
    np.testing.assert_allclose(
        line_of_sight_component(speeds * directions, directions), -speeds, atol=1e-14
    )
    np.testing.assert_allclose(
        line_of_sight_component(speeds * directions, 5 * directions),
        -speeds,
        atol=1e-14,
    )
    across = np.cross(directions.T, [0.0, 0.0, 1.0]).T
    np.testing.assert_allclose(
        line_of_sight_component(across, directions), 0, atol=1e-14
    )


def test_average_subsamples():
    """The mean over the sub-samples, which are ordered sub-sample by sub-sample."""
    values = np.arange(12.0)
    np.testing.assert_array_equal(average_subsamples(values, 3), [4, 5, 6, 7])
    np.testing.assert_array_equal(average_subsamples(values, 1), values)
    np.testing.assert_array_equal(
        average_subsamples(values.reshape(6, 2), 2),
        values.reshape(2, 3, 2).mean(axis=0),
    )
    with pytest.raises(ValueError, match="cannot be split"):
        average_subsamples(values, 5)


@pytest.mark.parametrize("order", [0, 1])
def test_sample_shell(boxes, directions, order):
    """sample_shell samples each box on the shell, in any chunks and threads."""
    origin = (2.0, 3.5, 4.0)
    coordinates = shell_coordinates(directions, 11.3, origin)
    expected = [sample_periodic(box, coordinates, order=order) for box in boxes]
    np.testing.assert_array_equal(
        sample_shell(boxes, directions, 11.3, origin=origin, order=order), expected
    )
    np.testing.assert_array_equal(
        sample_shell(
            boxes,
            directions,
            11.3,
            origin=origin,
            order=order,
            threads=3,
            chunk_size=37,
        ),
        expected,
    )


def test_sample_shell_cellwise(boxes, directions):
    """sample_shell_cellwise is sample_cellwise at the shell's points, in any chunks."""
    origin = (2.0, 3.5, 4.0)
    expected = sample_cellwise(
        np.maximum, boxes, shell_coordinates(directions, 11.3, origin)
    )
    np.testing.assert_array_equal(
        sample_shell_cellwise(
            np.maximum, boxes, directions, 11.3, origin=origin, threads=2, chunk_size=50
        ),
        expected,
    )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sample_shell_cellwise(np.maximum, boxes, directions, 11.3, chunk_size=0)
