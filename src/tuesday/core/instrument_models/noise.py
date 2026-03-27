"""A module to add thermal noise to lightcones."""

import logging
from typing import Literal

import astropy.units as un
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18, z_at_value
from astropy.cosmology.units import with_H0
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

    The usual formula to convert from flux density (Jy) to temperature involves only
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

    Returns
    -------
    sig_uv : astropy.units.Quantity
        Thermal noise RMS per baseline per integration snapshot in temperature units.
        Shape (Nfreqs,).
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

    dx = box_length / box_ncells

    sig_uv = np.zeros(len(freqs))
    for i, nu in enumerate(freqs):
        obs = observation.clone(frequency=nu)

        tsys = obs.Tsys.to(un.mK)

        d = Planck18.comoving_distance(f2z(nu)).to(un.Mpc)  # Mpc
        theta_box = (box_length.to(un.Mpc) / d) * un.rad
        omega_pix = theta_box**2 / box_ncells**2

        df = np.abs(
            z2f(z_at_value(Planck18.comoving_distance, d + dx / 2))
            - z2f(z_at_value(Planck18.comoving_distance, d - dx / 2))
        ).to(un.Hz)

        npolarizations = 2  # assume dual polarization
        sqrt = np.sqrt(npolarizations * df * obs.integration_time).to(
            un.dimensionless_unscaled
        )
        beam_area = (
            observation.observatory.beam.area(observation.frequency)
            if omega_beam is None
            else omega_beam[i]
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


def compute_uv_sampling(
    observation: Observation,
    freqs: un.Quantity,
    box_length: tp.Length,
    box_ncells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the UV sampling of an observation.

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

    Returns
    -------
    ugrid_edges
        The edges of the uv grid cells (along one dimension) in which the noise
        level is calculated. Each redshift has a different uv grid. Shape (Nx+1, Nz).
    vgrid_edges
        The edges of the v grid cells (along one dimension) in which the noise
        level is calculated. Each redshift has a different v grid. Shape (Nv+1, Nz).
    uv_coverage : np.ndarray
        Number of baseline samples in each uv cell, with shape (Nx, Ny, Nfreqs).
        This includes the effect of rotation synthesis over the lst_bin_size and n_days
        of the observation.

    Notes
    -----
    The output of this function has vgrid_edges only be the positive half of the plane,
    so that it has shape (Nx//2 + 1, Nz). This is because the negative half of the plane
    is redundant, and we don't need to compute the coverage for both halves.

    The ordering of the uv_coverage output is such that the first axis corresponds to
    the full u grid in ascending order (i.e. from negative to positive u), and the
    second axis corresponds to the non-negative v modes in ascending order (i.e. from
    zero to the Nyquist frequency of the simulation box that has been specified via
    box_length and box_ncells).
    """
    observatory = observation.observatory
    time_offsets = observatory.time_offsets_from_obs_int_time(
        observation.integration_time, observation.lst_bin_size
    )

    # Combine redundant baselines together to reduce memory/computation time.
    # Weights is the number of baselines in each redundant group.
    # Note that only one of the conjugate-pairs are in each baseline (specifically,
    # the one with positive v), so we don't need to worry about double counting.
    baselines = observatory.redundant_baseline_vectors
    weights = observatory.redundant_baseline_weights

    kperp = np.fft.fftshift(
        np.fft.fftfreq(box_ncells, d=(box_length / box_ncells).value)
    ) * (2 * np.pi / box_length.unit)

    kperp_to_u = 1 / dk_du(f2z(freqs)).to(
        box_length.unit**-1, with_H0(observation.cosmo.H0)
    )

    # ugrid is frequency dependent, so this is (Nu, Nz)
    ugrid_edges = np.outer(kperp, kperp_to_u).to(un.dimensionless_unscaled).value

    # In the v direction we only need the non-negative half of the plane. We take all
    # the non-negative modes here, being careful to always include the highest-frequency
    # component (which, in the case of an even number of pixels, is the nyquist
    # frequency and must be included).
    vgrid_edges = np.abs(ugrid_edges[: box_ncells // 2 + 1])[::-1]
    du = ugrid_edges[1] - ugrid_edges[0]
    ugrid_edges -= du / 2
    vgrid_edges -= du / 2

    # So far we've been dealing with centres. Turn them into edges.
    ugrid_edges = np.vstack((ugrid_edges, ugrid_edges[-1] + du))
    vgrid_edges = np.vstack((vgrid_edges, vgrid_edges[-1] + du))

    uv_coverage = grid_baselines(
        coherent=True,
        baselines=baselines,
        weights=weights,
        time_offsets=time_offsets,
        frequencies=freqs,
        ugrid_edges=ugrid_edges.T,
        vgrid_edges=vgrid_edges.T,
        phase_center_dec=observation.phase_center_dec,
        telescope_latitude=observatory.latitude,
        world=observatory.world,
    ).transpose(1, 2, 0)  # (Nu, Nv, Nfreqs)

    return ugrid_edges, vgrid_edges, uv_coverage


def compute_thermal_rms_uvgrid(
    observation: Observation,
    uv_coverage: np.ndarray,
    freqs: un.Quantity,
    box_length: un.Quantity,
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
    min_nbls_per_uv_cell: int = 0.1,
) -> un.Quantity:
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
    vgrid_edges
        The edges of the v grid cells (along one dimension) in which the noise
        level is calculated. Each redshift has a different v grid. Shape (Nv+1, Nz).
    sigma : astropy.units.Quantity
        Thermal noise RMS per voxel in uv space
        with shape (Nx, Ny, Nfreqs).
    uv_coverage : np.ndarray
        Number of baseline samples in each uv cell, with shape (Nx, Ny, Nfreqs).
        This includes the effect of rotation synthesis over the lst_bin_size and n_days
        of the observation.
    """
    nx, ny, nfreqs = uv_coverage.shape
    assert ny == nx // 2 + 1, "uv_coverage should have shape (Nx, Nx//2 + 1, Nfreqs)"
    assert nfreqs == len(freqs), (
        "uv_coverage should have the same number of frequency channels as freqs"
    )

    if observation.lst_bin_size != observation.time_per_day:
        raise NotImplementedError(
            "Cannot deal with an LST-bin size (i.e. time over which a field is tracked)"
            " that differs from the total time observed in a particular day. This would"
            "imply that separate fields (in RA) are observed, but this function only"
            "computes one output box."
        )

    sigma_rms = compute_thermal_rms_per_snapshot_vis(
        observation=observation,
        freqs=freqs,
        box_length=box_length,
        box_ncells=nx,
        antenna_effective_area=antenna_effective_area,
        beam_area=beam_area,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = sigma_rms / np.sqrt(uv_coverage * observation.n_days)

    sigma[min_nbls_per_uv_cell > uv_coverage] = 0.0
    return sigma


def _prepare_2d_complex_noise_for_irfft2(x: np.ndarray):
    # Note that this function is not very generic.
    # we assume the input has a first dimension of nrealization which should not
    # be fourier transformed. The FT axes should be (1, 2). The array can
    # have as many other axes as we want. We only call this internally from the
    # sample_from_rms_uvgrid function, and we know that the shape of the noise array
    # there is
    assert x.ndim >= 3

    n = x.shape[1]

    x[:, 0, 0] = np.real(x[:, 0, 0])
    if n % 2 != 0:
        x[:, 1 : n // 2 + 1, 0] = np.conj(x[:, -1 : n // 2 : -1, 0])
    if n % 2 == 0:
        x[:, 1 : n // 2 + 1, 0] = np.conj(x[:, -1 : n // 2 - 1 : -1, 0])
        x[:, n // 2, 0] = np.real(x[:, n // 2, 0])
        x[:, 0, -1] = np.real(x[:, 0, -1])
        x[:, 1 : n // 2 + 1, -1] = np.conj(x[:, -1 : n // 2 - 1 : -1, -1])
        x[:, n // 2, -1] = np.real(x[:, n // 2, -1])


def sample_from_rms_uvgrid(
    rms_noise: un.Quantity,
    seed: int | None = None,
    nrealizations: int = 1,
    window_fnc: str | None = None,
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
        RMS noise in uv space, shape (Nx, Ny, Nfreqs). The shape and ordering
        of this array must be such that in the first dimension (u) the pixels go
        from negative to positive u in ascending order, such that the central pixel
        is zero. In the case that the number of pixels is even, the zero mode must
        still be indexed by Nx//2. This is the standard format for FFT frequency output
        from numpy, and also the format output by `compute_thermal_rms_uvgrid`.
        The second axis should only have the non-negative modes, in ascending order,
        starting with zero and including the Nyquist frequency (i.e. the same
        assumptions that ``np.fft.irrft`` uses). Note that this function
        cannot check this ordering, so it is up to the user to ensure that the input is
        in the correct format.
    seed : int, optional
        Random seed for reproducibility, by default None.
    nrealizations : int, optional
        Number of noise realisations to sample, by default 1.
    window_fnc : str, optional
        Name of window function to be applied (in 2D) to the noise sampled in uv space.
        By default don't apply a window function.
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
        (nrealizations, Nx or Nu, Ny or Nv, Nfreqs). If in UV space, note that the
        ordering of the grid is switched to be in standard FFT format (i.e.
        zero-mode first, then negatives then positives).
    """
    if rms_noise.ndim == 2:
        # Add a last dimension of frequency.
        rms_noise = rms_noise[..., None]

    # Check that the shape of rms_noise is correct.
    nx, ny, nfreqs = rms_noise.shape
    if nx // 2 + 1 != ny:
        raise ValueError(
            "The shape of rms_noise is not correct. The first dimension should be "
            "the full u grid, and the second dimension should be the non-negative v "
            "modes. If the first dimension has size Nx, the second "
            f"dimension should have size Ny = Nx//2 + 1. Got {rms_noise.shape}."
        )

    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31 - 1)
        logger.info(f"Setting random seed to {seed}", stacklevel=2)

    rng = np.random.default_rng(seed)

    # Get some complex-value noise. The shape of the noise
    # is (Nrealizations, Nu, Nv, Nfreqs), where Nu is the full u grid and Nv is the
    # non-negative v modes.
    noise = (
        rng.normal(size=(nrealizations, *rms_noise.shape))
        + 1j * rng.normal(size=(nrealizations, *rms_noise.shape))
    ) * rms_noise.value[None, ...]

    if not return_in_uv and window_fnc is not None:
        window_fnc = taper2d(rms_noise.shape[0], window_fnc)
        window_fnc = window_fnc[:, -ny:]  # restrict to the half-plane
        noise *= window_fnc[None, ..., None]

    if apply_inverse_variance_weighting:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / (rms_noise**2).value
        w[np.isinf(w)] = 0.0
        wsum = w.sum(axis=(0, 1), keepdims=True)
        # We multiply by the number of pixels so that the overall normalization of the
        # noise is not changed by this weighting (i.e. consider a uniform visibility
        # as a function of (u, v) and a uniform weighting... we should get the same
        # result as if we had NOT weighted, which requires multiplying by the number of
        # pixels).
        noise *= w.shape[0] * w.shape[1] * w[None, ...] / wsum[None, ...]

    # The second axis needs to be ifftshifted such that it is in the right format
    # for ifft.
    noise = np.fft.ifftshift(noise, axes=(1,))

    # In a 2D array, some of the entries in the noise are still redundant, and must
    # be set properly to be either real or the conjugate of another entry. Do that now.
    _prepare_2d_complex_noise_for_irfft2(noise)

    if not return_in_uv:
        # We need to normalise by n^2 here because of the normalization convention of
        # the FFT in numpy. This will mean that we regain the correct normalisation of
        # the noise power spectrum. This was set by looking at the normalization of the
        # inverse FT in powerbox (applied to the sampled k-modes from a power spectrum).
        noise = np.fft.irfft2(noise, s=(nx, nx), axes=(1, 2)) * rms_noise.unit
    else:
        noise = noise * rms_noise.unit

    return noise


def apply_wedge_filter(
    uv_lightcones: un.Quantity,
    kperp_x: tp.Wavenumber,
    kperp_y: tp.Wavenumber,
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
        Lightcones in uv space, shape (Nrealizations, Nx, Ny, Nz).
        Note that Ny should have a size Nx//2 + 1, i.e. only the non-negative modes.
        This can be gotten from doing ``rfft2`` of the real-space lightcone.
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
    nrealizations, nx, ny, nz = uv_lightcones.shape

    if ny != nx // 2 + 1:
        raise ValueError(
            "The shape of uv_lightcones is not correct. The second dimension should be "
            "the non-negative v modes. If the first dimension has size Nx, the second "
            f"dimension should have size Ny=ceil(Nx/2) + 1. Got {uv_lightcones.shape}."
        )

    if mode not in ["rolling", "chunk"]:
        raise ValueError("mode must be either 'rolling' or 'chunk'.")

    with_h = "hlittle" in kperp_x.unit.to_string()

    if window_size is None and mode == "chunk":
        window_size = nz

    if window_size is None:
        raise ValueError("You must provide a window_size if mode is 'rolling'.")

    def filter_chunk(uv_chunk, freqs_chunk):
        n = len(freqs_chunk)
        f0 = freqs_chunk[n // 2]

        uvtau = np.fft.fft(uv_chunk, axis=-1)
        this_dnu = np.mean(np.diff(freqs_chunk))
        tau = np.fft.fftfreq(uv_chunk.shape[-1], d=this_dnu.to(un.Hz).value) * un.s

        kperp_mag = np.add.outer(kperp_x**2, kperp_y**2) ** 0.5
        umag = kperp_mag / dk_du(f2z(f0), with_h=with_h)

        # This wedge is exact for the central slice (except for the fact that the
        # frequencies are probably not exactly regular).
        wedge = wedge_slope * umag / f0

        mask = np.abs(tau)[None, None] < wedge[:, :, None] + buffer
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
    nrealizations: int = 1,
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
    nrealizations : int, optional
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
    nx, ny, nz = lightcone.shape

    if lightcone_freqs is None:
        if lightcone_redshifts is None:
            raise ValueError(
                "You must provide either lightcone_freqs or lightcone_redshifts."
            )
        lightcone_freqs = z2f(lightcone_redshifts)

    if len(lightcone_freqs) != nz:
        raise ValueError(
            "The length of freqs must be the same as the "
            "length of the lightcone frequency axis."
        )

    # Hera the noise realizations are a 4D array (nrealizations, Nx, Ny, Nfreqs).
    # The ordering of the Nx, Ny axes is in standard format for FFT (i.e. zero-mode
    # first, then negatives then positives).
    noise_realisation_uv = sample_from_rms_uvgrid(
        thermal_rms_uv,
        seed=seed,
        nrealizations=nrealizations,
        return_in_uv=True,
    )

    if remove_mean:
        # Don't subtract in-place or the user could get a nasty surprise.
        lightcone = lightcone - lightcone.mean(axis=(0, 1), keepdims=True)

    # TODO: check all the orderings of axes here
    lc_uv_nu = np.fft.rfft2(lightcone.value, axes=(0, 1)) * lightcone.unit

    # Only shift the first axis here, because the second axis only has the non-negative
    # modes, so there is no negative half to shift.
    thermal_rms_uv = np.fft.fftshift(thermal_rms_uv, axes=(0,))

    kperp_x = np.fft.fftfreq(nx, d=(box_length / nx).value) * (1 / box_length.unit)
    kperp_y = np.fft.rfftfreq(ny, d=(box_length / ny).value) * (1 / box_length.unit)

    lc_uv_nu = lc_uv_nu + noise_realisation_uv
    lc_uv_nu[:, thermal_rms_uv == 0] = 0.0

    if remove_wedge:
        lc_uv_nu = apply_wedge_filter(
            lc_uv_nu,
            kperp_x=kperp_x,
            kperp_y=kperp_y,
            lightcone_freqs=lightcone_freqs,
            window_size=wedge_chunk_size,
            wedge_slope=wedge_slope,
            buffer=wedge_buffer,
            mode=wedge_mode,
        )

    if apply_spatial_taper:
        _, nx, ny = lc_uv_nu.shape
        window_fnc = taper2d(lightcone.shape[0], spatial_taper)[:, -ny:]
        window_fnc = np.fft.fftshift(
            window_fnc, axes=(0,)
        )  # shift the window to be in the right format for FFT
        lc_uv_nu *= window_fnc[None, ..., None]

    noisy_lc_real = np.fft.irfft2(lc_uv_nu, s=(nx, nx), axes=(1, 2)).to(lightcone.unit)

    return noisy_lc_real, lightcone_redshifts
