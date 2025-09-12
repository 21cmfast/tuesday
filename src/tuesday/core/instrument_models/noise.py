"""A module to add thermal noise to lightcones."""

import logging
from collections.abc import Callable

import astropy.units as un
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18
from astropy.cosmology.units import littleh
from py21cmsense import Observation
from py21cmsense.conversions import dk_du, f2z, z2f
from scipy.signal import windows

logger = logging.getLogger(__name__)


def grid_baselines_uv(
    uvws: np.ndarray,
    freq: un.Quantity,
    box_length: un.Quantity,
    box_ncells: int,
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
    box_length : un.Quantity
        Transverse length of the simulation box.
    box_ncells : int
        Number of voxels Nx = Ny of a lightcone or coeval box.
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
    if "littleh" in box_length.unit.to_string():
        box_length = box_length.to(un.Mpc / littleh)
    else:
        box_length = box_length.to(un.Mpc) * Planck18.h / littleh
    dx = float(box_length.value) / float(box_ncells)
    ugrid_edges = (
        np.fft.fftshift(np.fft.fftfreq(box_ncells, d=dx)) * 2 * np.pi * box_length.unit
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
    box_length: float,
    box_ncells: int,
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
    box_length : astropy.units.Quantity
        Transverse length of the simulation box.
    box_ncells : int
        Number of voxels Nx = Ny of a lightcone or coeval box.
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
        theta_box = (box_length.to(un.Mpc) / d) * un.rad
        omega_pix = theta_box**2 / box_ncells**2

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
    return_in_uv: bool = False,
):
    """Sample noise for a lightcone slice given the corresponding rms noise in uv space.

    Parameters
    ----------
    rms_noise : astropy.units.Quantity
        RMS noise in uv space, shape (Nx, Ny, Nfreqs).
    seed : int, optional
        Random seed for reproducibility, by default None.
    nsamples : int, optional
        Number of noise realisations to sample, by default 1.
    window_fnc : str, optional
        Name of window function to be applied to the noise sampled in uv space,
        by default windows.blackmanharris.
    return_in_uv : bool, optional
        If True, return the noise sampled in uv space instead of real space,
        by default False.

    Returns
    -------
    noise : un.Quantity
        Noise sampled in real or uv space, shape
        (nsamples, Nx or Nu, Ny or Nv, Nfreqs)

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
    noise = np.fft.ifftshift(noise, axes=(1, 2))
    if not return_in_uv:
        noise = np.fft.ifft2(noise, axes=(1, 2)).real * rms_noise.unit
    else:
        noise = noise * rms_noise.unit

    return noise


def thermal_noise_uv(
    observation: Observation,
    freqs: un.Quantity,
    box_length: un.Quantity,
    box_ncells: int,
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
    min_nbls_per_uv_cell: int = 1,
):
    """Thermal noise RMS per voxel in uv space.

    Parameters
    ----------
    observation : py21cmsense.Observation
        Instance of `Observation`.
    freqs : astropy.units.Quantity
        Frequencies at which the noise is calculated.
    box_length : astropy.units.Quantity
        Length of the box in which the noise is calculated.
    box_ncells : int
        Number of voxels Nx = Ny of a lightcone or coeval box.
    antenna_effective_area : astropy.units.Quantity, optional
        Effective area of the antenna with shape (Nfreqs,).
    beam_area : astropy.units.Quantity, optional
        Beam area of the antenna with shape (Nfreqs,).
        Must only provide one of antenna_effective_area or beam_area.
    min_nbls_per_uv_cell : int, optional
        Minimum number of baselines per uv cell to consider
        the cell to be measured, by default 1.
        sigma is set to zero for uv cells with less than
        this number of baselines.

    Returns
    -------
    sigma : astropy.units.Quantity
        Thermal noise RMS per voxel in uv space
        with shape (Nx, Ny, Nfreqs).

    """
    observatory = observation.observatory.clone(
        beam=observation.observatory.beam.clone(frequency=freqs[0])
    )
    time_offsets = observatory.time_offsets_from_obs_int_time(
        observation.integration_time, observation.time_per_day
    )

    baseline_groups = observatory.get_redundant_baselines()
    baselines = observatory.baseline_coords_from_groups(baseline_groups)
    weights = observatory.baseline_weights_from_groups(baseline_groups)

    proj_bls = observatory.projected_baselines(
        baselines=baselines, time_offset=time_offsets
    )

    uv_coverage = np.zeros((box_ncells, box_ncells, len(freqs)))

    for i, freq in enumerate(freqs):
        uv_coverage[..., i] += grid_baselines_uv(
            proj_bls[::2] * freq / freqs[0], freq, box_length, box_ncells, weights[::2]
        )

    sigma_rms = thermal_noise_per_voxel(
        observation,
        freqs,
        box_length,
        box_ncells,
        antenna_effective_area=antenna_effective_area,
        beam_area=beam_area,
    )
    sigma = sigma_rms / np.sqrt(uv_coverage * observation.n_days)
    sigma[uv_coverage < min_nbls_per_uv_cell] = 0.0
    return sigma


def alpha(z, cosmo=Planck18):
    nu_21 = z2f(z).to(un.MHz)
    speed_of_light = c.to(un.m / un.s)
    return (speed_of_light * (1 + z) / (nu_21 * cosmo.H(z))).to(un.Mpc / un.MHz)


def delay2kpar(tau, freq=None, z=None):
    if z is None:
        z = f2z(freq)
    kpar = 2 * np.pi * np.abs(tau) / alpha(z)
    return kpar.to(1 / un.Mpc)


def kperp2baseline(kperp, cosmo=Planck18, freq=None, z=None):
    if z is None:
        z = f2z(freq)
    if freq is None:
        freq = z2f(z).to(un.MHz)
    baseline = np.abs(kperp * cosmo.comoving_distance(z) * c / (2 * np.pi * freq))
    return baseline.to(un.m)


def horizon_limit(buffer=0.0 * un.ns, cosmo=Planck18):
    def horizon_lim(kperp, freq=None, z=None):
        if z is None:
            z = f2z(freq)
        if freq is None:
            freq = z2f(z).to(un.MHz)
        baseline = kperp2baseline(kperp, cosmo=cosmo, freq=freq, z=z)
        tau_wedge = baseline / c  # Eqn 8 in HERA+23
        return delay2kpar(tau_wedge + buffer.to(un.ns), freq, z).to(1 / un.Mpc)

    return horizon_lim


def sample_lc_noise(
    lightcone: un.Quantity,
    box_length: un.Quantity,
    lightcone_redshifts: np.ndarray,
    *,
    observation: Observation | None = None,
    thermal_noise_uv_sigma: un.Quantity | None = None,
    freqs: un.Quantity | None = None,
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
    seed: int | None = None,
    nsamples: int = 1,
    window_fnc: str = "blackmanharris",
    min_nbls_per_uv_cell: int = 1,
    remove_wedge: bool = False,
    wedge_kpar: Callable | None = None,
    wedge_chunk_size: np.ndarray | int = 20,
    wedge_chunk_skip: np.ndarray | int = None,
    cosmo=Planck18,
):
    """Sample thermal noise and add it to a lightcone in Fourier space.

    Parameters
    ----------
    lightcone : astropy.units.Quantity
        Lightcone slice with shape (Nx, Ny, Nz).
    box_length : astropy.units.Quantity
        Length of the lightcone box side.
    lightcone_redshifts : np.ndarray
        Redshifts corresponding to the last axis of the lightcone.
    observation : py21cmsense.Observation, optional
        Instance of `Observation`. Needed if thermal_noise_uv_sigma is not provided.
    thermal_noise_uv_sigma : astropy.units.Quantity, optional
        Precomputed thermal noise RMS in uv space with shape (Nx, Ny, Nz).
        Can be computed with `thermal_noise_uv`.
    freqs : astropy.units.Quantity, optional
        Frequencies at which the thermal noise is calculated.
        Must have the same length as the lightcone frequency axis.
        If not provided, freqs are calculated from lightcone_redshifts.
    antenna_effective_area : astropy.units.Quantity, optional
        Effective area of the antenna with shape (Nfreqs,).
    beam_area : astropy.units.Quantity, optional
        Beam area of the antenna with shape (Nfreqs,).
        Must only provide one of antenna_effective_area or beam_area.
    nsamples : int, optional
        Number of noise realisations to sample, by default 1.
    seed : int, optional
        Random seed for reproducibility, by default None.
    window_fnc : str, optional
        Name of window function to be applied to the noise sampled in uv space,
        by default windows.blackmanharris.
    min_nbls_per_uv_cell : int, optional
        Minimum number of baselines per uv cell to consider
        the cell to be measured, by default 1.
        Thermal noise in uv space is set to zero for
        uv cells with less than this number of baselines.
    remove_wedge : bool, optional
        If True, remove the wedge from the noisy lightcone,
        using wedge_kpar to determine the wedge boundary,
        by default False.
    wedge_kpar : callable, optional
        Function that takes kperp and returns the corresponding kpar
        below which modes are considered to be contaminated by foregrounds.
        By default, the horizon limit with no buffer is used.
    cosmo : astropy.cosmology, optional
        Cosmology to use, by default Planck18.
    wedge_chunk_size : int or np.ndarray, optional
        Number of slices per chunk used to perform wedge removal.
        See Prelogovic+23 page 4 step (i).
        If an int is provided, all chunks will have the same size.
        If an array is provided, it must have the same length as the number of chunks.
        Must be provided if remove_wedge is True.
    wedge_chunk_skip : int or np.ndarray, optional
        Number of redshift slices to skip between chunks.
        If not provided, independent cubic chunks are assumed
        with wedge_chunk_skip = wedge_chunk_size.
        If an int is provided, all chunks will be
        separated by the same number of slices.

    Returns
    -------
    lightcone samples with noise
    """
    if freqs is None:
        if lightcone_redshifts is None:
            raise ValueError("You must provide either freqs or lightcone_redshifts.")
        freqs = (1420.0 / (1 + lightcone_redshifts)) * un.MHz

    if len(freqs) != lightcone.shape[2]:
        raise ValueError(
            "The length of freqs must be the same as the "
            "length of the lightcone frequency axis."
        )
    if thermal_noise_uv_sigma is None:
        thermal_noise_uv_sigma = thermal_noise_uv(
            observation,
            freqs,
            box_length,
            lightcone.shape[0],
            antenna_effective_area=antenna_effective_area,
            beam_area=beam_area,
            min_nbls_per_uv_cell=min_nbls_per_uv_cell,
        )

    noise_realisation_uv = sample_from_rms_noise(
        thermal_noise_uv_sigma,
        seed=seed,
        nsamples=nsamples,
        window_fnc=window_fnc,
        return_in_uv=True,
    )

    lightcone -= lightcone.mean(axis=(0, 1), keepdims=True)

    lc_ft = np.fft.fft2(lightcone.value, axes=(0, 1)) * lightcone.unit
    lc_ft[thermal_noise_uv_sigma == 0] = 0.0
    noisy_lc_ft = lc_ft + noise_realisation_uv
    noisy_lc_ft[..., thermal_noise_uv_sigma == 0] = 0.0

    noisy_lc_real = np.fft.ifft2(noisy_lc_ft, axes=(1, 2)).real.to(lightcone.unit)
    if remove_wedge:
        if wedge_chunk_size is None:
            raise ValueError(
                "You must provide wedge_chunk_size if remove_wedge is True."
            )
        if isinstance(wedge_chunk_size, int):
            nchunks = int(np.floor(lightcone.shape[2] / wedge_chunk_size))
            wedge_chunk_size = np.array([wedge_chunk_size] * nchunks)
        else:
            wedge_chunk_size = np.asarray(wedge_chunk_size)
            nchunks = len(wedge_chunk_size)
        if wedge_chunk_skip is None:
            wedge_chunk_skip = wedge_chunk_size
        elif isinstance(wedge_chunk_skip, int):
            wedge_chunk_skip = np.array([wedge_chunk_skip] * nchunks)
        else:
            wedge_chunk_skip = np.asarray(wedge_chunk_skip)
            if len(wedge_chunk_skip) != nchunks:
                raise ValueError(
                    "wedge_chunk_skip must have the same length as wedge_chunk_size."
                )
        if wedge_kpar is None:
            wedge_kpar = horizon_limit(cosmo=cosmo)

        kperp = (
            np.fft.fftshift(
                np.fft.fftfreq(
                    lightcone.shape[0],
                    d=(box_length / lightcone.shape[0]).to(un.Mpc).value,
                )
            )
            * 2
            * np.pi
        )

        final_lc_real = np.zeros((nsamples, *lightcone.shape[:-1], nchunks))
        final_lc_redshifts = np.zeros(nchunks)
        for i in range(nchunks):
            chunk_start = np.sum(wedge_chunk_size[:i]) if i > 0 else 0
            chunk_end = np.min([chunk_start + wedge_chunk_skip[i], lightcone.shape[2]])
            chunk_z = lightcone_redshifts[(chunk_start + chunk_end) // 2]
            lightcone_chunk = noisy_lc_real[..., chunk_start:chunk_end]
            lightcone_chunk *= windows.blackmanharris(lightcone_chunk.shape[-1])[
                None, None, :
            ]
            chunk_cdist = cosmo.comoving_distance(
                lightcone_redshifts[chunk_end - 1]
            ).to(un.Mpc) - cosmo.comoving_distance(lightcone_redshifts[chunk_start]).to(
                un.Mpc
            )
            kpar = (
                np.fft.fftshift(
                    np.fft.fftfreq(
                        lightcone_chunk.shape[-1],
                        d=(chunk_cdist / lightcone_chunk.shape[-1]).to(un.Mpc).value,
                    )
                )
                * 2
                * np.pi
            )

            kperp_mesh, kpar_mesh = np.meshgrid(kperp, kpar, indexing="ij")

            lightcone_chunk_ft = np.fft.fft2(lightcone_chunk, axes=(1, 2, 3))
            wedge_kpar_min = (
                wedge_kpar(kperp=np.abs(kperp_mesh) / un.Mpc, z=chunk_z)
                .to(1 / un.Mpc)
                .value
            )
            wedge_mask = np.abs(kpar_mesh) < wedge_kpar_min

            lightcone_chunk_ft[..., wedge_mask] = 0.0
            final_lc_real[..., i] = np.fft.ifft2(lightcone_chunk_ft, axes=(1, 2, 3))[
                ..., lightcone_chunk.shape[-1] // 2
            ].real
            final_lc_redshifts[i] = chunk_z
        final_lc_real = final_lc_real * lightcone.unit
    else:
        final_lc_real = noisy_lc_real
        final_lc_redshifts = lightcone_redshifts
    return final_lc_real, final_lc_redshifts
