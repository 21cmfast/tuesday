"""A module to add thermal noise to lightcones."""

import logging
from typing import Literal

import astropy.units as un
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18
from py21cmsense import Observation
from py21cmsense import units as tp
from py21cmsense._utils import grid_baselines
from py21cmsense.conversions import dk_du, f2z, z2f
from scipy.signal import windows

logger = logging.getLogger(__name__)


def compute_thermal_rms_per_snapshot_vis(
    observation: Observation,
    freqs: np.ndarray,
    box_length: tp.Length,
    box_ncells: int,
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
) -> tp.Temperature:
    r"""
    Calculate thermal noise RMS per baseline per integration snapshot.

    Eqn 3 from Prelogovic+22 2107.00018 without the last sqrt term
    That eqn comes from converting Eqn 9 in Ghara+16 1511.07448
    that's a flux density [Jy] to temperature [mK],
    but without the assumption of a circular symmetry of antenna distribution.

    The result here is in temperature units, which requires knowing the
    pixel solid angle. This is calculated assuming the box has a transverse
    comoving length of `box_length` at the redshifts corresponding to `freqs`.

    The usual formula to convert from flux density (Jy)  to temperature involves only
    the beam area, but this assumes that the flux comes from a source that fills the
    beam. Here we scale this by the final pixel solid angle desired, which is frequency
    dependent, and assumes that the input is the same comoving size at all frequencies.

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
        Effective area of the antenna with shape (Nfreqs,). Either this or beam_area
        can be provided. If neither is provided, the observation.beam.area is used.
    beam_area : astropy.units.Quantity, optional
        Beam area of the antenna with shape (Nfreqs,). Either this or
        antenna_effective_area can be provided. If neither is provided, the
        observation.beam.area is used.
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
        beam_area = (
            observation.observatory.beam.area if omega_beam is None else omega_beam[i]
        )
        sig_uv[i] = tsys.value * beam_area / omega_pix / sqrt

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


def compute_thermal_rms_uvgrid(
    observation: Observation,
    freqs: un.Quantity,
    box_length: un.Quantity,
    box_ncells: int,
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
    min_nbls_per_uv_cell: int = 1,
) -> tuple[np.ndarray, un.Quantity]:
    """Thermal noise RMS per voxel in uv space.

    This function integrates over the `lst_bin_size` of the observation at a resolution
    of `observation.integration_time` to compute the uv coverage of the observation.

    While this will compute the RMS of the thermal noise on a UV grid at any
    given array of frequencies, it does not account for the frequency dependence
    of the UV size of a box that has a fixed comoving transverse size.
    Therefore, if the frequency range is very large, it is recommended to
    split the lightcone into smaller chunks in frequency and compute the
    thermal noise RMS for each chunk separately.

    This function *does* however account for the frequency dependence of both
    the baseline UV coverage, and the evolution of the system temperature, and
    the changing cell size and its impact on the RMS of a single baseline.

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
    ugrid_edges
        The edges of the uv grid cells (along one dimension) in which the noise
        level is calculated. Each redshift has a different uv grid. Shape (Nx+1, Nz).
    sigma : astropy.units.Quantity
        Thermal noise RMS per voxel in uv space
        with shape (Nx, Ny, Nfreqs).
    """
    with_h = "hlittle" in box_length.unit.to_string()

    observatory = observation.observatory
    time_offsets = observatory.time_offsets_from_obs_int_time(
        observation.integration_time, observation.lst_bin_size
    )

    # Combine redundant baselines together to reduce memory/computation time.
    # Weights is the number of baselines in each redundant group.
    # Note that we add conjugates here because it makes prettier symmetric plots.
    # We have to divide by two later to get the right noise level.
    baseline_groups = observatory.get_redundant_baselines(add_conjugates=True)
    baselines = observatory.baseline_coords_from_groups(baseline_groups)
    weights = observatory.baseline_weights_from_groups(baseline_groups)

    kperp = np.fft.fftshift(
        np.fft.fftfreq(box_ncells, d=(box_length / box_ncells).value)
    ) * (2 * np.pi / box_length.unit)

    kperp_to_u = 1 / dk_du(f2z(freqs), with_h=with_h)

    # ugrid is frequency dependent, so this is (Nu, Nz)
    ugrid_edges = np.outer(kperp, kperp_to_u).to(un.dimensionless_unscaled).value
    du = ugrid_edges[1] - ugrid_edges[0]
    ugrid_edges -= du
    ugrid_edges = np.vstack((ugrid_edges, ugrid_edges[-1] + du))

    # Divide by two to account for the conjugate baselines added above.
    uv_coverage = (
        grid_baselines(
            coherent=True,
            baselines=baselines,
            weights=weights,
            time_offsets=time_offsets,
            frequencies=freqs,
            ugrid_edges=ugrid_edges.T,
            phase_center_dec=observation.phase_center_dec,
            telescope_latitude=observatory.latitude,
            world=observatory.world,
        )
        / 2
    )

    sigma_rms = compute_thermal_rms_per_snapshot_vis(
        observation,
        freqs,
        box_length,
        box_ncells,
        antenna_effective_area=antenna_effective_area,
        beam_area=beam_area,
    )
    sigma = sigma_rms / np.sqrt(uv_coverage.T * observation.n_days)
    sigma[min_nbls_per_uv_cell > uv_coverage.T] = 0.0
    return ugrid_edges, sigma, uv_coverage.T


def sample_from_rms_uvgrid(
    rms_noise: un.Quantity,
    seed: int | None = None,
    nsamples: int = 1,
    window_fnc: str = "blackmanharris",
    return_in_uv: bool = False,
    apply_inverse_variance_weighting: bool = False,
):
    """Sample noise for a lightcone slice given the corresponding rms noise in uv space.

    Note that this function assumes that the rms_noise is on a 3D grid (2 uv dimensions,
    one frequency dimension), whose central UV pixel corresponds to the zero baseline.
    This is the format output for example by `compute_thermal_rms_uvgrid`.

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
    apply_inverse_variance_weighting : bool, optional
        If True, apply inverse variance weighting to the noise samples in uv space.
        This ensures that uv cells with lower noise contribute more to the final
        real-space noise. By default False.

    Returns
    -------
    noise : un.Quantity
        Noise sampled in real or uv space, shape
        (nsamples, Nx or Nu, Ny or Nv, Nfreqs). If in UV space, note that the
        ordering of the grid is switched to be in standard FFT format (i.e.
        zero-mode first, then negatives then positives).
    """
    if rms_noise.ndim == 2:
        rms_noise = rms_noise[..., None]
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31 - 1)
        logger.info(f"Setting random seed to {seed}", stacklevel=2)

    rng = np.random.default_rng(seed)

    noise = (
        rng.normal(size=(nsamples, *rms_noise.shape))
        + 1j * rng.normal(size=(nsamples, *rms_noise.shape))
    ) * rms_noise.value[None, ...]

    if not return_in_uv:
        window_fnc = taper2d(rms_noise.shape[0], window_fnc)
        noise *= window_fnc[None, ..., None]

    # FIXME: this seems to be suppressing noise a LOT. Is this correct?
    if apply_inverse_variance_weighting:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / (rms_noise**2).value
        w[np.isinf(w)] = 0.0
        wsum = w.sum(axis=(0, 1), keepdims=True)
        noise *= w.shape[0] * w.shape[1] * w[None, ...] / wsum[None, ...]

    noise = np.fft.ifftshift(noise, axes=(1, 2))

    # Make the noise Hermitian so that the real-space noise is real.
    noise[:, 0, 0] += noise[:, 0, 0].conj()
    noise[:, 0, 1:] += noise[:, 0, 1:][:, ::-1].conj()
    noise[:, 1:, 0] += noise[:, 1:, 0][:, ::-1].conj()
    noise[:, 1:, 1:] += noise[:, 1:, 1:][:, ::-1, ::-1].conj()

    if not return_in_uv:
        noise = np.fft.ifft2(noise, axes=(1, 2)) * rms_noise.unit
    else:
        noise = noise * rms_noise.unit

    return noise


def apply_wedge_filter(
    uv_lightcones: un.Quantity,
    kperp_grid: tp.Wavenumber,
    lightcone_freqs: un.Quantity,
    window_size: int | None = None,
    mode: Literal["rolling", "chunk"] = "chunk",
    wedge_slope: float = 1.0,
    buffer: un.Quantity = 0.0 * un.ns,
):
    """Apply a wedge filter to a lightcone in uv space.

    This function computes the Fourier transform along the frequency axis
    in chunks of size `chunk_size`, and zeros all modes that fall into the wedge
    before Fourier-transforming back to frequency space.

    Note that we don't apply a frequency taper here. Frequency tapers are important
    if there really are foregrounds in the data, but are less important if the data
    is simply the cosmological signal + thermal noise. Furthermore, applying a frequency
    taper causes channels towards the edges to be significantly downweighted in the
    final filtered lightcone, which is not desirable.

    Note that there is no uniquely "correct" way to apply this wedge filter to
    a rectilinear lightcone such as is assumed in this function. This is because the
    wedge is naturally defined in (b, tau) space, where b is the baseline length
    and tau is the delay, the fourier dual of frequency *along a single baseline*.
    In other words, the correct modes to remove depends on redshift, but we only have
    a single slice at each redshift, and we can't do the line-of-sight FT of a single
    slice.

    We allow two methods to get the "approximate" wedge cut:

    1. A 'rolling' window approach, where a window of a specified is used to compute
       the FT, and the wedge is defined at the central frequency of the window, and the
       window is rolled along the frequency axis, so only one slice is changed at a
       time, each with a "correct" wedge cut. The downsides here are:
         - The slices at the edges of the lightcone must be dealt with specially, as
           they can't be central slices. Here, we set these via the chunk method (below)
         - The computational cost is higher, as the FT must be computed many times.
         - The slices are not independent, as each slice gets mixed in with its
           neighbours within the chunk. How exactly this effects the output lightcone
           with respect to the "true" wedge cut (or with respect to the second method
           below) is not clear.
    2. A 'chunk' approach, where the lightcone is split into independent chunks
       and the FT is computed for each chunk. The wedge is defined at the
       central frequency of each chunk, and all modes below the wedge are removed.
       This method should in principle yield the same power spectrum, within the chunks,
       as would be normally computed.
       The downsides here are:
         - The wedge is not precisely correct within each chunk, especially for the
           edge channels of the chunk.

    Parameters
    ----------
    uv_lightcones : astropy.units.Quantity
        Lightcones in uv space, shape (Nsamples, Nx, Ny, Nz).
    kperp_grid : np.ndarray
        The kperp grid cell centers (along one dimension) in which the noise
        level is calculated. Shape (Nkperp,). The assumption is that the kperp grid
        is the same at each redshift (this is true for a standard lightcone with fixed
        comoving transverse size).
    lightcone_freqs : astropy.units.Quantity
        Frequencies corresponding to the last axis of the lightcone.
    chunk_size : int or np.ndarray, optional
        Number of slices per chunk used to perform wedge removal.
        See Prelogovic+23 page 4 step (i).
        If an int is provided, all chunks will have the same size.
        If an array is provided, it must have the same length as the number of chunks.
        If None, the entire lightcone is treated as a single chunk.
    wedge_slope : float, optional
        Slope of the wedge in (b, tau) space, by default 1.0 (horizon limit).
    buffer : astropy.units.Quantity, optional
        Additional buffer to add to the wedge in delay space, by default 0.0 ns.
    """
    if mode not in ["rolling", "chunk"]:
        raise ValueError("mode must be either 'rolling' or 'chunk'.")

    with_h = "hlittle" in kperp_grid.unit.to_string()

    nz = uv_lightcones.shape[-1]

    if window_size is None and mode == "chunk":
        window_size = nz

    if window_size is None:
        raise ValueError("You must provide a window_size if mode is 'rolling'.")

    def filter_chunk(uv_chunk, freqs_chunk):
        n = len(freqs_chunk)
        f0 = freqs_chunk[n // 2]

        uvtau = np.fft.fft(uv_chunk, axis=-1)
        this_dnu = np.mean(np.diff(freqs_chunk))
        tau = (
            np.fft.fftshift(
                np.fft.fftfreq(uv_chunk.shape[-1], d=this_dnu.to(un.Hz).value)
            )
            * un.s
        )

        kperp_mag = np.add.outer(kperp_grid**2, kperp_grid**2) ** 0.5
        umag = kperp_mag / dk_du(f2z(f0), with_h=with_h)

        # This wedge is exact for the central slice (except for the fact that the
        # frequencies are probably not exactly regular).
        wedge = wedge_slope * umag / f0

        mask = tau[None, None] < wedge[:, :, None] + buffer
        uvtau[:, mask] = 0.0
        return np.fft.ifft(uvtau, axis=-1)

    filtered = np.zeros_like(uv_lightcones)

    # Rolling mode needs some extra handling for the first and last chunks.
    if mode == "rolling":
        first_chunk = filter_chunk(
            uv_lightcones[..., :window_size], freqs_chunk=lightcone_freqs[:window_size]
        )
        last_chunk = filter_chunk(
            uv_lightcones[..., -window_size:],
            freqs_chunk=lightcone_freqs[-window_size:],
        )
        filtered[..., : window_size // 2] = first_chunk[..., : window_size // 2]
        filtered[..., -window_size // 2 :] = last_chunk[..., -window_size // 2 :]

    chunk_start = 0
    chunk_end = chunk_start + window_size
    while chunk_end <= nz:
        uv_chunk = uv_lightcones[..., chunk_start:chunk_end]
        freqs_chunk = lightcone_freqs[chunk_start:chunk_end]
        out = filter_chunk(uv_chunk, freqs_chunk=freqs_chunk)

        if mode == "rolling":
            if chunk_start == 0:
                filtered[..., : window_size // 2] = out[..., : window_size // 2]
            elif chunk_end == nz:
                filtered[..., -window_size // 2 :] = out[..., -window_size // 2 :]
            else:
                filtered[..., chunk_start + window_size // 2] = out[
                    ..., window_size // 2
                ]
        else:
            filtered[..., chunk_start:chunk_end] = out

        if mode == "chunk":
            chunk_start += window_size
            chunk_end += window_size
            # On the last chunk, get all the straggling frequencies.
            if chunk_end < nz < chunk_end + window_size:
                chunk_end = nz
        elif mode == "rolling":
            chunk_start += 1
            chunk_end += 1

    return filtered


def observe_lightcone(
    lightcone: un.Quantity,
    box_length: tp.Length,
    *,
    thermal_rms_uv: un.Quantity[un.K],
    lightcone_redshifts: np.ndarray | None = None,
    lightcone_freqs: un.Quantity | None = None,
    seed: int | None = None,
    nsamples: int = 1,
    spatial_taper: str = "blackmanharris",
    apply_spatial_taper: bool = True,
    remove_wedge: bool = False,
    wedge_chunk_size: int | None = None,
    wedge_slope: float = 1.0,
    wedge_buffer: un.Quantity[un.ns] = 0.0,
    wedge_mode: Literal["rolling", "chunk"] = "chunk",
    cosmo=Planck18,
    remove_mean: bool = True,
):
    """Mock observe a lightcone.

    This adds thermal noise consistent with a given telescope's UV coverage, accounting
    for rotation synthesis over the observation's `lst_bin_size` and `n_days`.

    Optionally, the foreground wedge can be removed from the noisy lightcone.

    Parameters
    ----------
    lightcone : astropy.units.Quantity
        Lightcone slice with shape (Nx, Ny, Nz).
    ugrid_edges : np.ndarray
        The edges of the uv grid cells (along one dimension) in which the noise
        level is calculated. Each redshift has a different uv grid. Shape (Nx+1, Nz).

    box_length : astropy.units.Quantity
        Length of the lightcone box side.
    lightcone_redshifts : np.ndarray
        Redshifts corresponding to the last axis of the lightcone.
    lightcone_freqs : astropy.units.Quantity, optional
        Frequencies at which the thermal noise is calculated.
        Must have the same length as the lightcone frequency axis.
        If not provided, freqs are calculated from lightcone_redshifts.
    observation : py21cmsense.Observation, optional
        Instance of `Observation`. Needed if thermal_noise_uv_sigma is not provided.
    thermal_noise_uv_sigma : astropy.units.Quantity, optional
        Precomputed thermal noise RMS in uv space with shape (Nx, Ny, Nz).
        Can be computed with `thermal_noise_uv`.
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
    if lightcone_freqs is None:
        if lightcone_redshifts is None:
            raise ValueError(
                "You must provide either lightcone_freqs or lightcone_redshifts."
            )
        lightcone_freqs = z2f(lightcone_redshifts)

    if len(lightcone_freqs) != lightcone.shape[2]:
        raise ValueError(
            "The length of freqs must be the same as the "
            "length of the lightcone frequency axis."
        )

    # Hera the noise realizations are a 4D array (nsamples, Nx, Ny, Nfreqs).
    # The ordering of the Nx, Ny axes is in standard format for FFT (i.e. zero-mode
    # first, then negatives then positives).
    noise_realisation_uv = sample_from_rms_uvgrid(
        thermal_rms_uv,
        seed=seed,
        nsamples=nsamples,
        return_in_uv=True,
    )

    if remove_mean:
        # Don't subtract in-place or the user could get a nasty surprise.
        lightcone = lightcone - lightcone.mean(axis=(0, 1), keepdims=True)

    lc_uv_nu = np.fft.fft2(lightcone.value, axes=(0, 1)) * lightcone.unit
    thermal_rms_uv = np.fft.fftshift(thermal_rms_uv, axes=(0, 1))
    kperp_grid = np.fft.fftfreq(
        lightcone.shape[0], d=(box_length / lightcone.shape[0]).value
    ) * (1 / box_length.unit)

    lc_uv_nu = lc_uv_nu + noise_realisation_uv
    lc_uv_nu[:, thermal_rms_uv == 0] = 0.0

    if remove_wedge:
        lc_uv_nu = apply_wedge_filter(
            lc_uv_nu,
            kperp_grid=kperp_grid,
            lightcone_freqs=lightcone_freqs,
            window_size=wedge_chunk_size,
            wedge_slope=wedge_slope,
            buffer=wedge_buffer,
            mode=wedge_mode,
        )

    if apply_spatial_taper:
        window_fnc = np.fft.fftshift(taper2d(lightcone.shape[0], spatial_taper))
        lc_uv_nu *= window_fnc[None, ..., None]

    noisy_lc_real = np.fft.ifft2(lc_uv_nu, axes=(1, 2)).real.to(lightcone.unit)

    return noisy_lc_real, lightcone_redshifts
