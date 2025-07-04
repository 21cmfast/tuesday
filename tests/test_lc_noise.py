"""Tests for lightcone noise generation."""

import astropy.units as un
import numpy as np
from py21cmsense import Observation, Observatory

from tuesday.core import grid_baselines, sample_lc_noise, thermal_noise


def test_lc_noise_sampling():
    """Test the grid_baselines function."""

    observatory = Observatory.from_ska("AA4")
    hours_tracking = 1.0 * un.hour
    integration_time = 120.0 * un.second
    freqs = np.array([150.0]) * un.MHz
    time_offsets = observatory.time_offsets_from_obs_int_time(
        integration_time, hours_tracking
    )

    baseline_groups = observatory.get_redundant_baselines()
    baselines = observatory.baseline_coords_from_groups(baseline_groups)
    print("We have", baselines.shape[0], "baseline groups.")
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
        uv_coverage[..., i] += grid_baselines(
            proj_bls[::2] * freq / freqs[0], freq, boxlength, lc_shape, weights[::2]
        )

    obs = Observation(
        observatory=observatory,
        time_per_day=hours_tracking,
        lst_bin_size=hours_tracking,
        integration_time=integration_time,
        bandwidth=50 * un.kHz,
        n_days=int(np.ceil(1000 / hours_tracking.value)),
    )
    sigma_rms = thermal_noise(obs, freqs, boxlength, lc_shape, a_eff=[517.7] * un.m**2)
    sigma_rms = thermal_noise(obs, freqs, boxlength, lc_shape)
    sigma = sigma_rms / np.sqrt(uv_coverage * obs.n_days)
    sigma[uv_coverage == 0.0] = 0.0

    sample_lc_noise(sigma, seed=4, nsamples=10)
