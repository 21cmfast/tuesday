"""Tests for lightcone noise generation."""

import astropy.units as un
import numpy as np
import pytest
from astropy.cosmology import Planck18
from astropy.cosmology import units as cu
from astropy.cosmology.units import littleh
from py21cmsense import Observation, Observatory
from py21cmsense.conversions import f2z, z2f

from tuesday.core import (
    compute_thermal_rms_per_snapshot_vis,
    compute_thermal_rms_uvgrid,
    observe_coeval,
    observe_lightcone,
    sample_from_rms_uvgrid,
)
from tuesday.core.instrument_models.noise import (
    apply_beam,
    apply_wedge_filter,
    apply_wedge_filter_coeval,
    compute_uv_sampling,
)


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

    @pytest.mark.parametrize("nsamples", [10, 20])
    @pytest.mark.parametrize("ncells", [100, 101])
    def test_half_plane_unity_noise(self, nsamples, ncells):
        """Test that the UV noise is Hermitian."""
        uv_noise = sample_from_rms_uvgrid(
            np.ones((ncells, ncells // 2 + 1)) * un.mK,
            nrealizations=nsamples,
            seed=4,
            return_in_uv=True,
        )

        assert np.isclose(np.std(uv_noise.real), 1.0 * un.mK, rtol=0.01)
        assert np.isclose(np.std(uv_noise.imag), 1.0 * un.mK, rtol=0.01)


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

        out = observe_lightcone(
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


# Shared small setup for the tests below.
NCELLS = 20
BOX_LENGTH = 300.0 * un.Mpc
LC_FREQS = np.linspace(100.0, 105.0, 30) * un.MHz


@pytest.fixture(scope="module")
def small_obs():
    return Observation(
        observatory=Observatory.from_ska("LOW_INNER_R350M_AA4"),
        lst_bin_size=0.5 * un.hour,
        time_per_day=0.5 * un.hour,
        integration_time=120.0 * un.second,
        bandwidth=50 * un.kHz,
        n_days=1000,
    )


@pytest.fixture(scope="module", params=[False, True], ids=["half", "full"])
def lc_sigma(request, small_obs):  # noqa: PLR0206
    """Thermal RMS on the UV grid of a lightcone, on the half or full UV plane."""
    *_, uvcov = compute_uv_sampling(
        small_obs,
        freqs=LC_FREQS,
        box_length=BOX_LENGTH,
        box_ncells=NCELLS,
        full_plane=request.param,
    )
    return compute_thermal_rms_uvgrid(
        small_obs, uv_coverage=uvcov, freqs=LC_FREQS, box_length=BOX_LENGTH
    )


class TestThermalRmsPerSnapshotVis:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"box_slice_depth": 20 * un.Mpc},
            {"box_slice_depth": 100 * un.kHz},
            {"beam_area": 0.01 * un.rad**2},
            {"beam_area": [0.01, 0.02] * un.rad**2},
        ],
    )
    def test_options(self, small_obs, kwargs):
        rms = compute_thermal_rms_per_snapshot_vis(
            small_obs,
            freqs=[150.0, 120.0] * un.MHz,
            box_res=BOX_LENGTH / NCELLS,
            **kwargs,
        )
        assert rms.shape == (2,)
        assert np.all(rms > 0)

    @pytest.mark.parametrize("box_slice_depth", [None, 15 * un.Mpc])
    def test_littleh_consistent(self, small_obs, box_slice_depth):
        """Giving lengths in Mpc/h should give the same answer as in Mpc."""
        h = small_obs.cosmo.h
        kw = {"observation": small_obs, "freqs": [150.0, 120.0] * un.MHz}

        rms = compute_thermal_rms_per_snapshot_vis(
            box_res=15 * un.Mpc, box_slice_depth=box_slice_depth, **kw
        )
        with un.add_enabled_equivalencies(cu.with_H0(small_obs.cosmo.H0)):
            rms_h = compute_thermal_rms_per_snapshot_vis(
                box_res=15 * h * un.Mpc / littleh,
                box_slice_depth=(
                    None if box_slice_depth is None else box_slice_depth * h / littleh
                ),
                **kw,
            )
        np.testing.assert_allclose(rms, rms_h)

    def test_default_beam_area(self, small_obs):
        """Passing the observatory's own beam area is the same as passing nothing."""
        kw = {
            "observation": small_obs,
            "freqs": 150.0 * un.MHz,
            "box_res": BOX_LENGTH / NCELLS,
        }
        np.testing.assert_allclose(
            compute_thermal_rms_per_snapshot_vis(**kw),
            compute_thermal_rms_per_snapshot_vis(
                beam_area=small_obs.observatory.beam.area(150.0 * un.MHz), **kw
            ),
        )


def test_rms_uvgrid_bad_lst_bin_size(small_obs):
    obs = small_obs.clone(lst_bin_size=0.25 * un.hour)
    with pytest.raises(NotImplementedError, match="LST-bin size"):
        compute_thermal_rms_uvgrid(
            obs,
            uv_coverage=np.ones((NCELLS, NCELLS // 2 + 1, 1)),
            freqs=[150.0] * un.MHz,
            box_length=BOX_LENGTH,
        )


@pytest.mark.parametrize("ncells", [10, 11])
@pytest.mark.parametrize("full_plane", [False, True])
@pytest.mark.parametrize("freq_dependent_uv_grid", [False, True])
def test_uv_sampling_shapes(small_obs, ncells, full_plane, freq_dependent_uv_grid):
    freqs = [150.0, 151.0] * un.MHz
    ugrid, vgrid, uvcov = compute_uv_sampling(
        small_obs,
        freqs=freqs,
        box_length=BOX_LENGTH,
        box_ncells=ncells,
        full_plane=full_plane,
        freq_dependent_uv_grid=freq_dependent_uv_grid,
    )
    nv = ncells if full_plane else ncells // 2 + 1
    assert ugrid.shape == (ncells + 1, len(freqs))
    assert vgrid.shape == (nv + 1, len(freqs))
    assert uvcov.shape == (ncells, nv, len(freqs))
    if not full_plane:
        # The v=0 row must be symmetric in u (excluding the unpaired Nyquist mode).
        v0 = uvcov[ncells % 2 == 0 :, 0]
        np.testing.assert_allclose(v0, v0[::-1])


class TestApplyBeam:
    @pytest.mark.parametrize("ncells", [10, 11])
    @pytest.mark.parametrize("shape", [(), (3,), (2, 3)], ids=["2d", "3d", "4d"])
    def test_shapes(self, small_obs, ncells, shape):
        """The beam is applied to 2, 3 and 4D inputs, and preserves their shape."""
        nfreq = 3 if shape else 1
        freqs = np.linspace(150, 151, nfreq) * un.MHz
        if len(shape) == 2:
            lc_shape = (shape[0], ncells, ncells, shape[1])
        else:
            lc_shape = (ncells, ncells, *shape)
        lc = np.ones(lc_shape) * un.mK

        out = apply_beam(small_obs, lc, freqs=freqs, box_length=BOX_LENGTH)

        assert out.shape == lc.shape
        assert np.all(out <= lc)
        assert np.all(lc == 1 * un.mK)  # not modified in place

    def test_in_place(self, small_obs):
        lc = np.ones((NCELLS, NCELLS, 2)) * un.mK
        apply_beam(
            small_obs,
            lc,
            freqs=[150, 151] * un.MHz,
            box_length=BOX_LENGTH,
            in_place=True,
        )
        assert np.all(lc <= 1 * un.mK)
        assert np.any(lc < 1 * un.mK)

    @pytest.mark.parametrize(
        ("lc_shape", "match"),
        [
            ((1, 1, NCELLS, NCELLS, 2), "must be either 2, 3 or 4D"),
            ((NCELLS, NCELLS + 1, 2), "same number of pixels in x and y"),
            ((NCELLS, NCELLS, 3), "must match the length of freqs"),
        ],
    )
    def test_bad_shapes(self, small_obs, lc_shape, match):
        with pytest.raises(ValueError, match=match):
            apply_beam(
                small_obs,
                np.ones(lc_shape) * un.mK,
                freqs=[150, 151] * un.MHz,
                box_length=BOX_LENGTH,
            )


class TestSampleFromRmsUVGridOptions:
    @pytest.mark.parametrize("ncells", [10, 11])
    @pytest.mark.parametrize("full_plane", [False, True])
    @pytest.mark.parametrize("return_in_uv", [False, True])
    @pytest.mark.parametrize("spatial_taper", [None, "hann"])
    def test_shapes(self, ncells, full_plane, return_in_uv, spatial_taper):
        nv = ncells if full_plane else ncells // 2 + 1
        out = sample_from_rms_uvgrid(
            np.ones((ncells, nv, 2)) * un.mK,
            nrealizations=3,
            return_in_uv=return_in_uv,
            spatial_taper=spatial_taper,
        )
        assert out.shape == (3, ncells, nv if return_in_uv else ncells, 2)
        assert out.unit == un.mK
        if not return_in_uv:
            assert not np.iscomplexobj(out)

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape of rms_noise is not correct"):
            sample_from_rms_uvgrid(np.ones((10, 3, 2)) * un.mK)

    def test_unseeded_is_random(self):
        rms = np.ones((10, 6)) * un.mK
        assert not np.allclose(sample_from_rms_uvgrid(rms), sample_from_rms_uvgrid(rms))


class TestApplyWedgeFilter:
    kperp_x = np.fft.fftfreq(NCELLS, d=15.0) / un.Mpc
    kperp_y = np.fft.rfftfreq(NCELLS, d=15.0) / un.Mpc

    def _uv(self, ny=NCELLS // 2 + 1):
        rng = np.random.default_rng(1)
        shape = (1, NCELLS, ny, len(LC_FREQS))
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * un.mK

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"uv_lightcones_ny": 3}, "shape of uv_lightcones is not correct"),
            ({"mode": "foo"}, "mode must be either"),
            ({"mode": "rolling"}, "must provide a window_size"),
        ],
    )
    def test_errors(self, kwargs, match):
        uv = self._uv(kwargs.pop("uv_lightcones_ny", NCELLS // 2 + 1))
        with (
            un.add_enabled_equivalencies(cu.with_H0(Planck18.H0)),
            pytest.raises(ValueError, match=match),
        ):
            apply_wedge_filter(
                uv,
                kperp_x=self.kperp_x,
                kperp_y=self.kperp_y,
                lightcone_freqs=LC_FREQS,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("mode", "window_size"),
        [("chunk", None), ("chunk", 10), ("chunk", 12), ("rolling", 10)],
    )
    def test_littleh_consistent(self, mode, window_size):
        """kperp in h/Mpc should give the same answer as kperp in 1/Mpc."""
        h = Planck18.h
        kw = {
            "uv_lightcones": self._uv(),
            "lightcone_freqs": LC_FREQS,
            "mode": mode,
            "window_size": window_size,
        }
        with un.add_enabled_equivalencies(cu.with_H0(Planck18.H0)):
            out = apply_wedge_filter(kperp_x=self.kperp_x, kperp_y=self.kperp_y, **kw)
            out_h = apply_wedge_filter(
                kperp_x=self.kperp_x / h * littleh,
                kperp_y=self.kperp_y / h * littleh,
                **kw,
            )
        np.testing.assert_allclose(out, out_h)
        # Something should have been filtered, but not everything.
        assert 0 < np.sum(np.abs(out)) < np.sum(np.abs(kw["uv_lightcones"]))


class TestObserveLightconeOptions:
    @pytest.mark.parametrize("remove_mean", [False, True])
    @pytest.mark.parametrize("spatial_taper", [None, "hann"])
    @pytest.mark.parametrize("remove_wedge", [False, True])
    def test_options(self, lc_sigma, remove_mean, spatial_taper, remove_wedge):
        lc = np.ones((NCELLS, NCELLS, len(LC_FREQS))) * un.mK
        out = observe_lightcone(
            lightcone=lc,
            thermal_rms_uv=lc_sigma,
            box_length=BOX_LENGTH,
            lightcone_redshifts=f2z(LC_FREQS),
            nrealizations=2,
            seed=1,
            remove_mean=remove_mean,
            spatial_taper=spatial_taper,
            remove_wedge=remove_wedge,
        )
        assert out.shape == (2, *lc.shape)
        assert not np.iscomplexobj(out)

    @pytest.mark.parametrize(
        ("lc_shape", "kwargs", "match"),
        [
            (
                (NCELLS, NCELLS + 1, len(LC_FREQS)),
                {"lightcone_freqs": LC_FREQS},
                "same number of pixels",
            ),
            ((NCELLS, NCELLS, len(LC_FREQS)), {}, "either lightcone_freqs or"),
            (
                (NCELLS, NCELLS, len(LC_FREQS)),
                {"lightcone_freqs": LC_FREQS[1:]},
                "length of freqs must be the same",
            ),
        ],
    )
    def test_errors(self, lc_sigma, lc_shape, kwargs, match):
        with pytest.raises(ValueError, match=match):
            observe_lightcone(
                lightcone=np.zeros(lc_shape) * un.mK,
                thermal_rms_uv=lc_sigma,
                box_length=BOX_LENGTH,
                **kwargs,
            )


class TestObserveCoevalOptions:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frequency": z2f(7.0)},
            {"redshift": 7.0, "frequency": z2f(7.0)},
            {"redshift": 7.0, "multiply_by_beam": False},
        ],
    )
    def test_options(self, small_obs, kwargs):
        box = np.zeros((NCELLS, NCELLS, NCELLS)) * un.mK
        out = observe_coeval(
            box=box, box_length=35.0 * un.Mpc, observation=small_obs, seed=1, **kwargs
        )
        assert out.shape == (1, *box.shape)

    def test_frequency_same_as_redshift(self, small_obs):
        kw = {
            "box": np.zeros((NCELLS, NCELLS, NCELLS)) * un.mK,
            "box_length": 35.0 * un.Mpc,
            "observation": small_obs,
            "seed": 1,
        }
        np.testing.assert_allclose(
            observe_coeval(redshift=7.0, **kw), observe_coeval(frequency=z2f(7.0), **kw)
        )

    @pytest.mark.parametrize(
        "wedge_buffer",
        [100 * un.ns, 0.05 / un.Mpc, 0.05 / Planck18.h * littleh / un.Mpc],
        ids=["delay", "kpar", "kpar-h"],
    )
    def test_wedge_buffer_units(self, small_obs, wedge_buffer):
        """A buffer in kpar is converted to delay; with or without littleh."""
        box = np.zeros((NCELLS, NCELLS, NCELLS)) * un.mK
        kw = {
            "box": box,
            "box_length": 35.0 * un.Mpc,
            "observation": small_obs,
            "redshift": 7.0,
            "seed": 1,
            "remove_wedge": True,
        }
        out = observe_coeval(wedge_buffer=wedge_buffer, **kw)
        out_nobuffer = observe_coeval(**kw)
        assert out.shape == (1, *box.shape)
        # A bigger buffer removes more power.
        assert np.sum(out**2) < np.sum(out_nobuffer**2)


def test_coeval_wedge_littleh_consistent():
    """The coeval wedge filter gives the same answer with or without littleh."""
    h = Planck18.h
    rng = np.random.default_rng(1)
    box_uv = rng.normal(size=(1, NCELLS, NCELLS // 2 + 1, NCELLS)) * un.mK
    kperp_x = np.fft.fftfreq(NCELLS, d=2.0) / un.Mpc
    kperp_y = np.fft.rfftfreq(NCELLS, d=2.0) / un.Mpc
    kw = {"box_uv_nu": box_uv, "redshift": 7.0, "wedge_buffer": 0.05 / un.Mpc}

    with un.add_enabled_equivalencies(cu.with_H0(Planck18.H0)):
        out = apply_wedge_filter_coeval(
            kperp_x=kperp_x, kperp_y=kperp_y, box_res=2.0 * un.Mpc, **kw
        )
        out_h = apply_wedge_filter_coeval(
            kperp_x=kperp_x / h * littleh,
            kperp_y=kperp_y / h * littleh,
            box_res=2.0 * h * un.Mpc / littleh,
            **kw,
        )
    np.testing.assert_allclose(out, out_h)
