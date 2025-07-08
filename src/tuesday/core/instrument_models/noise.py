"""A module to add thermal noise to lightcones."""

import logging

import astropy.units as un
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18
from astropy.cosmology.units import littleh
from py21cmsense import Observation
from py21cmsense.conversions import dk_du, f2z
from scipy.signal import windows

logger = logging.getLogger(__name__)


def grid_baselines_uv(
    uvws: np.ndarray,
    freq: un.Quantity,
    boxlength: un.Quantity,
    lc_shape: tuple[int, int, int],
    weights: np.ndarray,
    include_mirrored_bls: bool = True,
    avg_mirrored_bls: bool = True,
):
    r"""Grid positive baselines in uv space.

    Parameters
    ----------
    uvws : np.ndarray
        Baselines in uv space with shape (N bls, N time offsets, 3).
    freq : un.Quantity
        Frequency at which the baselines are projected.
    boxlength : un.Quantity
        Transverse length of the simulation box.
    lc_shape : tuple
        Shape of the lightcone (Nx, Ny, Nz).
        We assume that Nx = Ny to be sky-plane dimensions,
        and Nz to be to line-of-sight (frequency) dimension.
    weights : np.ndarray
        Weights for each baseline group with shape (N bls).
    include_mirrored_bls : bool, optional
        If True, include the inverse aka mirrored baselines in the histogram.
        Mirrored baselines are baselines with u,v -> -u,-v.
    avg_mirrored_bls : bool, optional
        If True, average the mirrored baselines by two since they do
        not carry any additional information to the positive baselines.
        You may not want to divide by two if your plan is to only use
        half of the uv plane in a later step to estimate sensitivity.

    Returns
    -------
    uvsum : np.ndarray
        2D histogram of uv counts for one day
        of observation with shape (Nu=Nx, Nv=Nx).

    """
    if "littleh" in boxlength.unit.to_string():
        boxlength = boxlength.to(un.Mpc / littleh)
    else:
        boxlength = boxlength.to(un.Mpc) * Planck18.h / littleh
    dx = float(boxlength.value) / float(lc_shape[0])
    ugrid_edges = (
        np.fft.fftshift(np.fft.fftfreq(lc_shape[0], d=dx)) * 2 * np.pi * boxlength.unit
    )

    du = ugrid_edges[1] - ugrid_edges[0]
    ugrid_edges = np.append(ugrid_edges - du / 2.0, ugrid_edges[-1] + du / 2.0)

    ugrid_edges /= dk_du(f2z(freq))

    weights = np.repeat(weights, uvws.shape[1])
    uvws = uvws.reshape((uvws.shape[0] * uvws.shape[1], -1))
    uvsum = np.histogram2d(
        uvws[:, 0], uvws[:, 1], bins=ugrid_edges.value, weights=weights
    )[0]

    if include_mirrored_bls:
        uvsum += np.flip(uvsum)
        if avg_mirrored_bls:
            uvsum /= 2.0

    return uvsum


def thermal_noise_per_voxel(
    observation: Observation,
    freqs: np.ndarray,
    boxlen: float,
    lc_shape: tuple[int, int, int],
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
):
    r"""
    Calculate thermal noise RMS per baseline per integration snapshot.

    Eqn 3 from Prelogovic+22 2107.00018 without the last sqrt term
    That eqn comes from converting Eqn 9 in Ghara+16 1511.07448
    that's a flux density [Jy] to temperature [mK],
    but without the assumption of a circular symmetry of antenna distribution.

    Parameters
    ----------
    observation : py21cmsense.Observation
        Instance of `Observation`.
    freqs : astropy.units.Quantity
        Frequencies at which the noise is calculated.
    boxlen : astropy.units.Quantity
        Transverse length of the simulation box.
    lc_shape : tuple
        Shape of the lightcone (Nx, Ny, Nz).
        We assume that Nx = Ny to be sky-plane dimensions,
        and Nz to be to line-of-sight (frequency) dimension.
    antenna_effective_area : astropy.units.Quantity, optional
        Effective area of the antenna with shape (Nfreqs,).
    beam_area : astropy.units.Quantity, optional
        Beam area of the antenna with shape (Nfreqs,).
    """
    try:
        len(freqs)
    except TypeError:
        freqs = np.array([freqs.value]) * freqs.unit

    if beam_area is not None:
        try:
            len(beam_area)
        except TypeError:
            beam_area = np.array([beam_area.value] * len(freqs)) * beam_area.unit
        if antenna_effective_area is not None:
            raise ValueError(
                "You cannot provide both beam_area and antenna_effective_area."
                " Proceding with beam_area."
            )
        omega_beam = beam_area.to(un.rad**2)
        if len(omega_beam) > 1 and len(omega_beam) != len(freqs):
            raise ValueError(
                "Beam area must be a float or have the same shape as freqs."
            )
    elif antenna_effective_area is not None:
        try:
            len(antenna_effective_area)
        except TypeError:
            antenna_effective_area = (
                np.array([antenna_effective_area.value] * len(freqs))
                * antenna_effective_area.unit
            )
        if len(antenna_effective_area) > 1 and len(antenna_effective_area) != len(
            freqs
        ):
            raise ValueError(
                "Antenna effective area must either be a float or "
                "have the same shape as freqs."
            )
        a_eff = antenna_effective_area.to(un.m**2)
        omega_beam = (c / freqs.to("Hz")) ** 2 / a_eff * un.rad**2
    else:
        omega_beam = None

    sig_uv = np.zeros(len(freqs))
    for i, nu in enumerate(freqs):
        obs = observation.clone(
            observatory=observation.observatory.clone(
                beam=observation.observatory.beam.clone(frequency=nu)
            )
        )

        tsys = obs.Tsys.to(un.mK)

        d = Planck18.comoving_distance(f2z(nu)).to(un.Mpc)  # Mpc
        theta_box = (boxlen.to(un.Mpc) / d) * un.rad
        omega_pix = theta_box**2 / np.prod(lc_shape[:2])

        sqrt = np.sqrt(2.0 * observation.bandwidth.to("Hz") * obs.integration_time).to(
            un.dimensionless_unscaled
        )
        # I need this 1e6 to get the same numbers as tools...
        sig_uv[i] = (
            tsys.value
            / omega_pix
            / sqrt
            / 1e6
            * (
                observation.observatory.beam.area
                if omega_beam is None
                else omega_beam[i]
            )
        )
    return sig_uv * tsys.unit


def taper2d(n: int, taper: str = "blackmanharris"):
    r"""2D window function.

    Parameters
    ----------
    n : int
        Size of the window function, assumed to be square.

    Returns
    -------
    wf : np.ndarray
        2D Blackman-Harris window function with shape (n, n)

    """
    wf = getattr(windows, taper)(n)
    return np.sqrt(np.outer(wf, wf))


def sample_from_rms_noise(
    rms_noise: un.Quantity,
    seed: int | None = None,
    nsamples: int = 1,
    window_fnc: str = "blackmanharris",
):
    """Sample noise for a lightcone slice given the corresponding rms noise in uv space.

    Parameters
    ----------
    rms_noise : astropy.units.Quantity
        RMS noise in uv space, shape (Nx, Ny, Nfreqs).
    nsamples : int, optional
        Number of noise realisations to sample, by default 1.
    window_fnc : str, optional
        Name of window function to be applied to the noise sampled in uv space,
        by default windows.blackmanharris.

    Returns
    -------
    lc_noise : un.Quantity
        Noise sampled in real space, shape (nsamples, Nx, Ny, Nfreqs

    """
    if len(rms_noise.shape) == 2:
        rms_noise = rms_noise[..., None]
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31 - 1)
        logger.info(f"Setting random seed to {seed}", stacklevel=2)
    rng = np.random.default_rng(seed)

    window_fnc = taper2d(rms_noise.shape[0], window_fnc)

    noise = (
        rng.normal(size=(nsamples, *rms_noise.shape))
        + 1j * rng.normal(size=(nsamples, *rms_noise.shape))
    ) * rms_noise.value[None, ...]

    noise *= window_fnc[None, ..., None]
    noise = (noise + np.conj(noise)) / 2.0
    noise = np.fft.ifft2(np.fft.ifftshift(noise, axes=(1, 2)), axes=(1, 2))

    return noise.real * rms_noise.unit


def sample_lc_noise(
    observation: Observation,
    freqs: un.Quantity,
    boxlength: un.Quantity,
    lc_shape: tuple[int, int, int],
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
    seed: int | None = None,
    nsamples: int = 1,
    window_fnc: str = "blackmanharris",
):
    """Test the grid_baselines function."""
    observatory = observation.observatory
    time_offsets = observatory.time_offsets_from_obs_int_time(
        observation.integration_time, observation.time_per_day
    )

    baseline_groups = observatory.get_redundant_baselines()
    baselines = observatory.baseline_coords_from_groups(baseline_groups)
    weights = observatory.baseline_weights_from_groups(baseline_groups)

    proj_bls = observatory.projected_baselines(
        baselines=baselines, time_offset=time_offsets
    )

    uv_coverage = np.zeros((lc_shape[0], lc_shape[0], len(freqs)))

    for i, freq in enumerate(freqs):
        uv_coverage[..., i] += grid_baselines_uv(
            proj_bls[::2] * freq / freqs[0], freq, boxlength, lc_shape, weights[::2]
        )

    sigma_rms = thermal_noise_per_voxel(
        observation,
        freqs,
        boxlength,
        lc_shape,
        antenna_effective_area=antenna_effective_area,
        beam_area=beam_area,
    )
    sigma = sigma_rms / np.sqrt(uv_coverage * observation.n_days)
    sigma[uv_coverage == 0.0] = 0.0

    return sample_from_rms_noise(
        sigma, seed=seed, nsamples=nsamples, window_fnc=window_fnc
    )
