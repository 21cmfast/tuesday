"""Tests for lightcone noise generation."""

import astropy.units as un
import numpy as np
import pytest
from py21cmsense import Observation, Observatory

from tuesday.core import (
    grid_baselines_uv,
    sample_from_rms_noise,
    sample_lc_noise,
    thermal_noise_per_voxel,
)


@pytest.fixture
def observation():
    """Fixture to create an observatory instance."""
    return Observation(
        observatory=Observatory.from_ska("AA4"),
        time_per_day=1.0 * un.hour,
        lst_bin_size=1.0 * un.hour,
        integration_time=120.0 * un.second,
        bandwidth=50 * un.kHz,
        n_days=1000,
    )


def test_grid_baselines(observation):
    """Test the grid_baselines function."""

    observatory = observation.observatory
    hours_tracking = observation.time_per_day
    integration_time = observation.integration_time
    freqs = np.array([150.0]) * un.MHz
    time_offsets = observatory.time_offsets_from_obs_int_time(
        integration_time, hours_tracking
    )

    baseline_groups = observatory.get_redundant_baselines()
    baselines = observatory.baseline_coords_from_groups(baseline_groups)

    weights = observatory.baseline_weights_from_groups(baseline_groups)

    # Call the function
    proj_bls = observatory.projected_baselines(
        baselines=baselines, time_offset=time_offsets
    )
    lc_shape = np.array([20, 20, 1945])
    boxlength = 30.0 * un.Mpc
    uv_coverage = np.zeros((lc_shape[0], lc_shape[0], len(freqs)))

    for i, freq in enumerate(freqs):
        # uv coverage integrated over one field
        uv_coverage[..., i] += grid_baselines_uv(
            proj_bls[::2] * freq / freqs[0], freq, boxlength, lc_shape, weights[::2]
        )


def test_thermal_noise_per_voxel(observation):
    """Test the thermal_noise_per_voxel function."""
    boxlength = 300.0 * un.Mpc
    lc_shape = (20, 20, 1945)
    thermal_noise_per_voxel(
        observation,
        150 * un.MHz,
        boxlength,
        lc_shape,
        antenna_effective_area=[517.7] * un.m**2,
    )
    thermal_noise_per_voxel(
        observation, np.array([150.0, 120.0]) * un.MHz, boxlength, lc_shape
    )
    with pytest.raises(
        ValueError,
        match="You cannot provide both beam_area "
        "and antenna_effective_area."
        " Proceding with beam_area.",
    ):
        thermal_noise_per_voxel(
            observation,
            np.array([150.0, 120.0]) * un.MHz,
            boxlength,
            lc_shape,
            antenna_effective_area=517.7 * un.m**2,
            beam_area=1.0 * un.arcmin**2,
        )
    with pytest.raises(
        ValueError,
        match="Antenna effective area must either be a float or have the"
        " same shape as freqs.",
    ):
        thermal_noise_per_voxel(
            observation,
            np.array([150.0, 120.0, 100.0]) * un.MHz,
            boxlength,
            lc_shape,
            antenna_effective_area=[517.7, 200.0] * un.m**2,
        )
    with pytest.raises(
        ValueError, match="Beam area must be a float or have the same shape as freqs."
    ):
        thermal_noise_per_voxel(
            observation,
            np.array([150.0, 120.0, 100.0]) * un.MHz,
            boxlength,
            lc_shape,
            beam_area=[517.7, 200.0] * un.rad**2,
        )


def test_sample_from_rms_noise():
    """Test the sample_from_rms_noise function."""
    sample_from_rms_noise(
        np.random.default_rng(0).normal(5.0, 1.0, (10, 10, 2)) * un.mK,
        seed=4,
        nsamples=10,
    )


def test_sample_lc_noise(observation):
    """Test the sample_lc_noise function."""
    sample_lc_noise(
        observation,
        np.array([150.0, 120.0]) * un.MHz,
        300.0 * un.Mpc,
        (20, 20, 15),
    )
