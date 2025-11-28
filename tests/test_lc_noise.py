"""Tests for lightcone noise generation."""

import astropy.units as un
import numpy as np
import pytest
from py21cmsense import Observation, Observatory

from tuesday.core import (
    compute_thermal_rms_per_snapshot_vis,
    compute_thermal_rms_uvgrid,
    observe_lightcone,
    sample_from_rms_uvgrid,
)


@pytest.fixture
def observation():
    """Fixture to create an observatory instance."""
    return Observation(
        observatory=Observatory.from_ska("LOW_FULL_AA4"),
        lst_bin_size=1.0 * un.hour,
        integration_time=120.0 * un.second,
        bandwidth=50 * un.kHz,
        n_days=1000,
    )


def test_rms_per_snapshot_vis(observation):
    """Test the thermal_noise_per_voxel function."""
    boxlength = 300.0 * un.Mpc
    boxnside = 20
    compute_thermal_rms_per_snapshot_vis(
        observation,
        150 * un.MHz,
        boxlength,
        boxnside,
        antenna_effective_area=[517.7] * un.m**2,
    )
    compute_thermal_rms_per_snapshot_vis(
        observation, np.array([150.0, 120.0]) * un.MHz, boxlength, boxnside
    )
    with pytest.raises(
        ValueError, match="You cannot provide both beam_area and antenna_effective_area"
    ):
        compute_thermal_rms_per_snapshot_vis(
            observation,
            np.array([150.0, 120.0]) * un.MHz,
            boxlength,
            boxnside,
            antenna_effective_area=517.7 * un.m**2,
            beam_area=1.0 * un.arcmin**2,
        )
    with pytest.raises(
        ValueError,
        match="Antenna effective area must either be a float or have the"
        " same shape as freqs",
    ):
        compute_thermal_rms_per_snapshot_vis(
            observation,
            np.array([150.0, 120.0, 100.0]) * un.MHz,
            boxlength,
            boxnside,
            antenna_effective_area=[517.7, 200.0] * un.m**2,
        )
    with pytest.raises(
        ValueError, match="Beam area must be a float or have the same shape as freqs"
    ):
        compute_thermal_rms_per_snapshot_vis(
            observation,
            np.array([150.0, 120.0, 100.0]) * un.MHz,
            boxlength,
            boxnside,
            beam_area=[517.7, 200.0] * un.rad**2,
        )
    _, sigma = compute_thermal_rms_uvgrid(
        observation,
        np.array([150.0, 120.0, 100.0]) * un.MHz,
        boxlength,
        boxnside,
        min_nbls_per_uv_cell=15,
    )

    samples = sample_from_rms_uvgrid(
        sigma,
        seed=4,
        nsamples=10,
    )
    assert samples.shape == (10, *sigma.shape)


class TestSampleFromRmsNoise:
    @pytest.mark.parametrize("nsamples", [1, 2])
    @pytest.mark.parametrize("ncells", [10, 11])
    def test_image_noise_reality(self, nsamples, ncells):
        """Test that the UV noise is Hermitian."""
        img_noise = sample_from_rms_uvgrid(
            np.ones((ncells, ncells)) * un.mK,
            nsamples=nsamples,
            seed=4,
            return_in_uv=False,
        )
        np.testing.assert_allclose(img_noise.imag, 0.0)


class TestObserveLightcone:
    def setup_class(self):
        self.lc_freqs = np.linspace(100.0, 105.0, 50) * un.MHz
        self.ncells = 20
        obs = Observation(
            observatory=Observatory.from_ska("LOW_INNER_R350M_AA4"),
            lst_bin_size=0.5 * un.hour,
            integration_time=120.0 * un.second,
            bandwidth=50 * un.kHz,
            n_days=1000,
        )

        _, sigma = compute_thermal_rms_uvgrid(
            obs,
            freqs=self.lc_freqs,
            box_length=300.0 * un.Mpc,
            box_ncells=self.ncells,
            min_nbls_per_uv_cell=15,
        )
        self.sigma = sigma

    @pytest.mark.parametrize("wedge_slope", [0.0, 1.0])
    @pytest.mark.parametrize("wedge_buffer", [0.0, 300 * un.ns])
    @pytest.mark.parametrize("wedge_mode", ["rolling", "chunk"])
    def test_sample_lc_noise(self, observation, wedge_slope, wedge_buffer, wedge_mode):
        """Test the sample_lc_noise function."""
        lc = np.zeros((self.ncells, self.ncells, self.lc_freqs.size)) * un.mK

        out, _ = observe_lightcone(
            lightcone=lc,
            thermal_rms_uv=self.sigma,
            box_length=300.0 * un.Mpc,
            lightcone_freqs=self.lc_freqs,
            remove_wedge=True,
            nsamples=1,
            wedge_slope=wedge_slope,
            wedge_buffer=wedge_buffer,
            wedge_mode=wedge_mode,
            wedge_chunk_size=self.lc_freqs.size,
            seed=4,
        )
        assert np.sum(np.abs(out)) > 0
        assert out.shape == (1, *lc.shape)
