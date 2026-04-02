"""Tests for lightcone noise generation."""

import astropy.units as un
import numpy as np
import pytest
from py21cmsense import Observation, Observatory

from tuesday.core import (
    compute_thermal_rms_per_snapshot_vis,
    compute_thermal_rms_uvgrid,
    observe_coeval,
    observe_lightcone,
    sample_from_rms_uvgrid,
)
from tuesday.core.instrument_models.noise import compute_uv_sampling


@pytest.fixture
def observation():
    """Fixture to create an observatory instance."""
    return Observation(
        observatory=Observatory.from_ska("LOW_FULL_AA4"),
        lst_bin_size=1.0 * un.hour,
        time_per_day=1.0 * un.hour,
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
        freqs=150 * un.MHz,
        box_res=boxlength / boxnside,
        antenna_effective_area=[517.7] * un.m**2,
    )
    compute_thermal_rms_per_snapshot_vis(
        observation,
        freqs=np.array([150.0, 120.0]) * un.MHz,
        box_res=boxlength / boxnside,
        antenna_effective_area=[517.7] * un.m**2,
    )
    with pytest.raises(
        ValueError, match="You cannot provide both beam_area and antenna_effective_area"
    ):
        compute_thermal_rms_per_snapshot_vis(
            observation,
            freqs=np.array([150.0, 120.0]) * un.MHz,
            box_res=boxlength / boxnside,
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
            freqs=np.array([150.0, 120.0, 100.0]) * un.MHz,
            box_res=boxlength / boxnside,
            antenna_effective_area=[517.7, 200.0] * un.m**2,
        )
    with pytest.raises(
        ValueError, match="Beam area must have length one or the same shape as freqs"
    ):
        compute_thermal_rms_per_snapshot_vis(
            observation,
            freqs=np.array([150.0, 120.0, 100.0]) * un.MHz,
            box_res=boxlength / boxnside,
            beam_area=[517.7, 200.0] * un.rad**2,
        )

    _, _, uvcov = compute_uv_sampling(
        observation,
        freqs=np.array([150.0, 120.0, 100.0]) * un.MHz,
        box_length=boxlength,
        box_ncells=boxnside,
    )

    sigma = compute_thermal_rms_uvgrid(
        observation,
        uv_coverage=uvcov,
        box_length=boxlength,
        freqs=np.array([150.0, 120.0, 100.0]) * un.MHz,
        min_nbls_per_uv_cell=15,
    )

    samples = sample_from_rms_uvgrid(
        sigma,
        seed=4,
        nrealizations=10,
    )
    assert samples.shape == (10, boxnside, boxnside, 3)


class TestSampleFromRmsNoise:
    @pytest.mark.parametrize("nsamples", [1, 2])
    @pytest.mark.parametrize("ncells", [10, 11])
    def test_image_noise_reality(self, nsamples, ncells):
        """Test that the UV noise is Hermitian."""
        img_noise = sample_from_rms_uvgrid(
            np.ones((ncells, ncells // 2 + 1)) * un.mK,
            nrealizations=nsamples,
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
            time_per_day=0.5 * un.hour,
            integration_time=120.0 * un.second,
            bandwidth=50 * un.kHz,
            n_days=1000,
        )

        _, _, uvcov = compute_uv_sampling(
            obs, freqs=self.lc_freqs, box_length=300.0 * un.Mpc, box_ncells=self.ncells
        )
        sigma = compute_thermal_rms_uvgrid(
            obs,
            uv_coverage=uvcov,
            freqs=self.lc_freqs,
            box_length=300.0 * un.Mpc,
            min_nbls_per_uv_cell=15,
        )
        self.sigma = sigma

    @pytest.mark.parametrize("wedge_slope", [0.0, 1.0])
    @pytest.mark.parametrize("wedge_buffer", [0.0 * un.ns, 300 * un.ns])
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
            nrealizations=1,
            wedge_slope=wedge_slope,
            wedge_buffer=wedge_buffer,
            wedge_mode=wedge_mode,
            wedge_chunk_size=self.lc_freqs.size,
            seed=4,
        )
        assert np.sum(np.abs(out)) > 0
        assert out.shape == (1, *lc.shape)


class TestObserveCoeval:
    def setup_class(self):
        self.ncells = 20
        self.obs = Observation(
            observatory=Observatory.from_ska("LOW_INNER_R350M_AA4"),
            lst_bin_size=0.5 * un.hour,
            time_per_day=0.5 * un.hour,
            integration_time=120.0 * un.second,
            bandwidth=50 * un.kHz,
            n_days=1000,
        )

    @pytest.mark.parametrize("spatial_taper", [None, "hann"])
    @pytest.mark.parametrize("remove_wedge", [False, True])
    @pytest.mark.parametrize("remove_mean", [False, True])
    def test_it_runs_through(self, spatial_taper, remove_wedge, remove_mean):
        """Test that observe_coeval runs through without error."""
        box = np.zeros((self.ncells, self.ncells, self.ncells)) * un.mK

        out = observe_coeval(
            box=box,
            box_length=35.0 * un.Mpc,
            observation=self.obs,
            redshift=7.0,
            seed=1,
            nrealizations=1,
            remove_wedge=remove_wedge,
            wedge_slope=1.0,
            wedge_buffer=100 * un.ns,
            spatial_taper=spatial_taper,
            remove_mean=remove_mean,
        )

        assert out.unit == un.mK
        assert np.sum(np.abs(out)) > 0
        assert out.shape == (1, *box.shape)

    def test_slope_zero_equals_no_wedge_removal(self):
        """Test that setting wedge slope to zero is same as not removing the wedge."""
        box = np.zeros((self.ncells, self.ncells, self.ncells)) * un.mK

        out_no_wedge_removal = observe_coeval(
            box=box,
            box_length=35.0 * un.Mpc,
            observation=self.obs,
            redshift=7.0,
            seed=1,
            nrealizations=1,
            remove_wedge=False,
        )

        out_zero_slope = observe_coeval(
            box=box,
            box_length=35.0 * un.Mpc,
            observation=self.obs,
            redshift=7.0,
            seed=1,
            nrealizations=1,
            remove_wedge=True,
            wedge_slope=0.0,
        )

        np.testing.assert_allclose(out_no_wedge_removal, out_zero_slope)

    def test_no_freq_or_redshift(self):
        """Test that observe_coeval raises an error if neither z nor f is provided."""
        box = np.zeros((self.ncells, self.ncells, self.ncells)) * un.mK

        with pytest.raises(
            ValueError, match="You must provide either frequency or redshift"
        ):
            observe_coeval(
                box=box,
                box_length=300.0 * un.Mpc,
                observation=self.obs,
                seed=1,
                nrealizations=1,
            )
