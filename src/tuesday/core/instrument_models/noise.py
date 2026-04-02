"""A module with functions to mock observe simulated data.

This module contains functions that, when combined together, can produce samples of
thermal noise on either a coeval or lightcone box, given the specifications of an
"observation" with a real telescope, and a simulation box size and resolution.
Furthermore, input simulations can be "observed" by adding such thermal noise samples
as well as applying observational effects such as the UV sampling and the wedge filter.

This code utilises the well-tested ``py21cmsense`` package to compute the UV sampling
and thermal noise RMS on a UV grid, using the ``Observatory`` and ``Observation``
classes to specify the telescope and observation parameters.

UV Sampling
-----------
The fundamental computation that needs to happen to produce noise samples is to
calculate a grid that specifies how many baselines sample each Fourier mode over the
course of an observation.

There are a few subtle points to consider when computing the number of samples on a UV
grid when that UV grid is defined by a *simulated lightcone*. Here we outline these
considerations, and how ``tuesday`` deals with them. The main point is that the UV grid
is defined by the simulation box, but the UV sampling is defined by the telescope and
observation parameters.

1. An input lightcone can be of any transverse size, whereas an observation with a real
   telescope only observes a transverse size defined by the primary beam.

   * In ``tuesday``, the low-level function :func:`sample_from_rms_uvgrid`, will produce
     samples (whether in UV or image space) that *do not* account for this size
     difference. However, the beam size can be applied to output from this function,
     if it is in image-space, using the ``apply_beam`` function. In principle, the
     UV-space samples with this correction (corresponding to a convolution) can be
     produced by going to image space, using ``apply_beam``, and then going back to
     UV space.
   * On the other hand, the high-level function :func:`observe_lightcone` does account
     for this size difference, by applying the beam before returning the "observed"
     lightcone.
2. The Fourier-space extent of the lightcone may differ from the extent of the array
   layout.

   * This is simply dealt with by assigning zero weight to UV cells that are outside the
     Fourier-space extent of the lightcone. When transforming to image space, this
     corresponds to a convolution with the "synthesized beam" of the observation,
     effectively limiting the resolution of the mock observation to that defined by
     the instrument, rather than the simulation box.
3. The frequency range and binning of the simulation may be different from the
   observation. In particular, instruments typically have a fixed frequency binning,
   whereas a lightcone usually has a fixed comoving distance binning, which corresponds
   to a frequency binning that changes with redshift.

   * In ``tuesday``, all frequency-based information must be directly specified, and is
     **not** taken from the ``py21cmsense.Observation``. This is primarily because
     ``py21cmsense`` is designed specifically for instrumental sensitivity to the
     power spectrum, and considers only a short frequency range (or "spectral window")
     at a time. This is too inflexible for our purposes. Thus, the user must directly
     specify the frequencies at which to compute the noise, and these frequencies must
     match the shape of the lightcone along the last axis (or else, for a coeval cube,
     the whole box is assumed to be at a single frequency).
4. There are subtle questions about how to deal with the evolution of scales across
   frequency for a lightcone. In particular, the following quantities all change with
   frequency (and therefore should be computed per lightcone slice):

   1. The UV coordinates of each baseline. These are proportional to frequency.
   2. The UV coordinates of the Fourier grid defined by the lightcone. The Fourier modes
      of the lightcone at each slice are constant (in units of inverse comoving length),
      but their conversion to UV coordinates changes with redshift/frequency. This
      dependency is much weaker than the baseline proportionality (something like
      redshift to the power of 1/5 at high redshift).
   3. The frequency-dependence of the beam size, which is essentially linear.
   4. The pixel solid angle. This comes into the conversion from flux density to
      temperature.

    * In ``tuesday``, by default all of these quantities are computed per lightcone
      slice, so that the noise is as accurate as possible. However, the user can choose
      to ignore the frequency dependence of the UV grid of the lightcone Fourier modes,
      which is the weakest effect, to speed up the computation.
"""

import logging
from typing import Literal

import astropy.units as un
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18, z_at_value
from astropy.cosmology import units as cu
from py21cmsense import Observation
from py21cmsense import units as tp
from py21cmsense._utils import grid_baselines
from py21cmsense.conversions import dk_du, f2z, z2f
from scipy.signal import windows

logger = logging.getLogger(__name__)


@un.quantity_input
def compute_thermal_rms_per_snapshot_vis(
    observation: Observation,
    freqs: tp.Frequency,
    box_res: tp.Length,
    box_slice_depth: tp.Length | tp.Frequency | None = None,
    antenna_effective_area: un.Quantity[un.m**2] | None = None,
    beam_area: un.Quantity[un.rad**2] | None = None,
) -> un.Quantity[un.mK]:
    r"""
    Calculate thermal noise RMS per baseline per integration snapshot.

    Eqn 3 from Prelogovic+22 2107.00018 without the last sqrt term
    That eqn comes from converting Eqn 9 in Ghara+16 1511.07448
    that's a flux density [Jy] to temperature [mK],
    but without the assumption of a circular symmetry of antenna distribution.

    The result here is in temperature units, which requires knowing the
    pixel solid angle. This is calculated assuming the box has a transverse
    comoving resolution of `box_res` at the redshifts corresponding to `freqs`.

    The usual formula to convert from flux density (Jy) to temperature involves only
    the beam area, but this assumes that the flux comes from a source that fills the
    beam. Here we scale this by the final pixel solid angle desired, which is frequency
    dependent, and assumes that the input is the same comoving size at all frequencies.

    There is a slightly involved interplay between the parameters `freqs`, `box_res`
    and `box_slice_depth`. The `freqs` define the central frequencies of bins for which
    the resulting noise level is returned. If the result is intended to be applied
    directly to a standard lightcone in which each cell is a cube with the same comoving
    size in all dimensions, then `freqs` is *technically* all that is required, since
    these must correspond to the redshift slices of the lightcone, and thus must be
    spaced equally in comoving distance, thus defining the `box_res` and
    `box_slice_depth`. However, there are situations in which two disjoint redshift
    slices are considered, in which case at least `box_res` must be specified (and the
    `box_slice_depth` is considered equal to `box_res`). In yet other cases, it might be
    useful to assume that the slices do not have the same comoving depth as their
    transverse cell size, so `box_slice_depth` can be specified independently, either as
    a comoving size or a frequency delta.

    Note that, as a rule throughout the functions in this module, the ``bandwidth``
    and ``n_channels`` attributes of the ``py21cmsense.Observation`` are *never used*,
    since they refer to the frequency binning of the instrument, which is not
    necessarily the same as that of an input lightcone.

    Parameters
    ----------
    observation : py21cmsense.Observation
        Instance of `Observation`.
    freqs : astropy.units.Quantity
        Frequencies at which the noise is calculated.
    box_res : astropy.units.Quantity
        Transverse resolution of the simulation box.
    box_slice_depth : astropy.units.Quantity, optional
        Depth of each slice in the simulation box. If not provided, it is assumed to be
        equal to `box_res`.
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
    freqs = np.atleast_1d(freqs)

    if beam_area is not None:
        beam_area = np.atleast_1d(beam_area)

        if antenna_effective_area is not None:
            raise ValueError(
                "You cannot provide both beam_area and antenna_effective_area."
                " Proceding with beam_area."
            )
        omega_beam = beam_area
        if len(omega_beam) > 1 and len(omega_beam) != len(freqs):
            raise ValueError(
                "Beam area must have length one or the same shape as freqs."
            )

    elif antenna_effective_area is not None:
        antenna_effective_area = np.atleast_1d(antenna_effective_area)
        if len(antenna_effective_area) > 1 and len(antenna_effective_area) != len(
            freqs
        ):
            raise ValueError(
                "Antenna effective area must either be a float or "
                "have the same shape as freqs."
            )
        a_eff = antenna_effective_area
        omega_beam = un.rad**2 * (c / freqs) ** 2 / a_eff
    else:
        omega_beam = None

    # By default assume that the slice depth is the same as the transverse resolution,
    # but allow it to be specified
    if box_slice_depth is None:
        box_slice_depth = box_res

    sig_uv = np.zeros(len(freqs)) * un.mK
    for i, nu in enumerate(freqs):
        obs = observation.clone(frequency=nu)

        tsys = obs.Tsys

        # transverse comoving distance per radian
        d = Planck18.comoving_distance(f2z(nu)).to(
            box_res.unit,
        )

        with un.set_enabled_equivalencies(
            cu.with_H0(observation.cosmo.H0) + cu.dimensionless_redshift()
        ):
            omega_pix = (box_res / (d / un.rad)) ** 2

            if box_slice_depth.unit.is_equivalent(un.Hz):
                df = box_slice_depth
            else:
                df = np.abs(
                    z2f(z_at_value(Planck18.comoving_distance, d + box_slice_depth / 2))
                    - z2f(
                        z_at_value(Planck18.comoving_distance, d - box_slice_depth / 2)
                    )
                )

        npolarizations = 2  # assume dual polarization
        sqrt = np.sqrt(npolarizations * df * obs.integration_time).to(
            un.dimensionless_unscaled
        )
        beam_area = (
            observation.observatory.beam.area(nu)
            if omega_beam is None
            else omega_beam[i]
        )
        sig_uv[i] = tsys * beam_area / omega_pix / sqrt

    return sig_uv


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
    return np.outer(wf, wf)


def compute_uv_sampling(
    observation: Observation,
    freqs: un.Quantity,
    box_length: tp.Length,
    box_ncells: int,
    freq_dependent_uv_grid: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the UV sampling of an observation.

    This function integrates over the `lst_bin_size` of the observation at a resolution
    of `observation.integration_time` to compute the uv coverage of the observation.

    Note that while the frequency-dependent UV coordinates of baselines is always
    accounted for in this function, the frequency-dependence of the UV coordinates of
    the Fourier grid defined by the lightcone is optional to account for, since it is a
    much weaker effect. By default, the UV grid is computed at each frequency, but if
    `freq_dependent_uv_grid` is set to False, the UV grid is computed at the central
    frequency and assumed to be the same at all frequencies. This can speed up the
    computation significantly, and is a good approximation if the frequency range is
    small.

    If you want no frequency dependence at all across a range of redshifts, simply
    pass only the central frequency to this function, to obtain a single UV grid, and
    then apply that to an entire box. This approximation is made in codes like
    21cmSense when computing power spectra in narrow spectral windows.

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
    freq_dependent_uv_grid : bool, optional
        Whether to compute a frequency-dependent uv grid, accounting for the change in
        the conversion from comoving transverse length to angle with redshift. By
        default True. If False, the uv grid is computed at the central frequency and
        assumed to be the same at all frequencies. This is a good approximation if the
        frequency range is small, and can speed up the computation significantly.

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
        box_length.unit**-1, cu.with_H0(observation.cosmo.H0)
    )

    if not freq_dependent_uv_grid:
        kperp_to_u = np.mean(kperp_to_u)

    # ugrid is potentially frequency dependent, so this is (Nu, Nz)
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


def convert_half_to_full_uv_plane(uv: np.ndarray, inverse_counts: bool = False):
    """Convert a UV grid that only has the non-negative v modes to a full UV grid.

    Parameters
    ----------
    uv : np.ndarray
        UV grid whose first two axes have shape (Nu, Nv//2 + 1) that only has the
        non-negative v modes. Arbitrary axes can follow the first two.
    inverse_counts : bool, optional
        If True, the inverse of the counts are used to weight the UV grid.
        This is useful if the input UV grid is a noise RMS grid, which should be
        weighted by the inverse of the number of samples in each cell, rather than the
        number of samples itself. By default False.
    """
    nu, nv = uv.shape[:2]
    if not (nv == nu // 2 + 1):
        raise ValueError(
            "Input UV grid must have shape (Nu, Nu//2 + 1, ...), but has shape "
            f"{uv.shape}."
        )

    if nu % 2 == 0:
        # even case, we need to mirror all but the last point (the Nyquist frequency)
        # note that in this case we don't have any data for the u=-nyquist, v<0 modes,
        # since they would correspond with u=+nyquist, v>0 modes, which we didn't bin
        # at all. We just set these to zero here.
        rev = np.roll(uv[::-1, ::-1, ...], shift=1, axis=0)
        rev[0] = 0.0
        uv = np.concatenate((rev, uv[:, 1:-1, ...]), axis=1)

        # At v=0, we've only counted one baseline instead of both:
        if inverse_counts:
            uv[1:, nu // 2] = 1 / (
                1 / uv[1:, nu // 2, ...] + 1 / uv[1:, nu // 2, ...][::-1]
            )
        else:
            uv[1:, nu // 2] += uv[1:, nu // 2, ...][::-1]
    else:
        # odd case, we can mirror all points
        uv = np.concatenate((uv[::-1, ::-1, ...], uv[:, 1:, ...]), axis=1)

        # At v=0, we've only counted one baseline instead of both:
        if inverse_counts:
            uv[:, nu // 2] = 1 / (1 / uv[:, nu // 2, ...] + 1 / uv[::-1, nu // 2, ...])
        else:
            uv[:, nu // 2] += uv[::-1, nu // 2, ...]

    return uv


@un.quantity_input
def compute_thermal_rms_uvgrid(
    observation: Observation,
    uv_coverage: np.ndarray,
    freqs: tp.Frequency,
    box_length: tp.Length,
    antenna_effective_area: un.Quantity | None = None,
    beam_area: un.Quantity | None = None,
    box_slice_depth: tp.Frequency | tp.Length | None = None,
    min_nbls_per_uv_cell: int = 1,
) -> un.Quantity[un.mK]:
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
        box_res=box_length / nx,
        antenna_effective_area=antenna_effective_area,
        beam_area=beam_area,
        box_slice_depth=box_slice_depth,
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


@un.quantity_input
def compute_beam(
    observation: Observation,
    freqs: tp.Frequency,
    box_ncells: int,
    box_length: tp.Length,
    in_place: bool = False,
) -> np.ndarray:
    """Compute the primary beam effect for a given observation and frequency grid.

    This computes a Gaussian beam in the transverse direction, with a FWHM given by the
    primary beam of the telescope at each frequency. The beam is computed in real space.

    Parameters
    ----------
    observation : Observation
        The observation defining the beam to apply.
    freqs : astropy.units.Quantity
        The frequencies of the lightcone slices, at which to apply the beam.
        Shape (Nz,).
    box_ncells : int
        The number of cells in the transverse direction of the lightcone.
    box_length : astropy.units.Quantity
        The transverse size of the lightcone.
    in_place : bool, optional
        Whether to apply the beam in place (modifying the input lightcone) or to return
        a new array with the beam applied. By default False (i.e. return a new array).

    Returns
    -------
    lightcone_with_beam : astropy.units.Quantity
        The lightcone with the beam applied, with the same shape as the input lightcone.
    """
    boxres = box_length / box_ncells

    obs = observation.observatory

    out = np.zeros((box_ncells, box_ncells, len(freqs)))
    for i, fq in enumerate(freqs):
        z = f2z(fq)
        dz = observation.cosmo.comoving_distance(z).to(
            box_length.unit, cu.with_H0(observation.cosmo.H0)
        )

        theta_grid = np.arange(-box_length / dz / 2, box_length / dz / 2, boxres / dz)[
            :box_ncells
        ]
        if box_ncells % 2 == 1:
            theta_grid += boxres / dz / 2
        zenith_angle = np.sqrt(np.add.outer(theta_grid**2, theta_grid**2)) * un.rad
        bmsig = obs.beam.fwhm(fq) / (2 * np.sqrt(2 * np.log(2)))
        out[..., i] = np.exp(-((zenith_angle) ** 2) / (2 * bmsig**2))

    return out


@un.quantity_input
def apply_beam(
    observation: Observation,
    lightcone: tp.Temperature,
    freqs: tp.Frequency,
    box_length: tp.Length,
    in_place: bool = False,
) -> un.mK:
    """Apply the effect of the primary beam to a lightcone.

    This applies a Gaussian beam in the transverse direction, with a FWHM given by the
    primary beam of the telescope at each frequency. The beam is applied in real space.

    Parameters
    ----------
    observation : Observation
        The observation defining the beam to apply.
    lightcone : astropy.units.Quantity
        The lightcone to which to apply the beam. The shape of the lightcone can be
        either (Nx, Ny), (Nx, Ny, Nz) or (nrealizations, Nx, Ny, Nz).
    freqs : astropy.units.Quantity
        The frequencies of the lightcone slices, at which to apply the beam.
        Shape (Nz,).
    box_length : astropy.units.Quantity
        The transverse size of the lightcone.
    in_place : bool, optional
        Whether to apply the beam in place (modifying the input lightcone) or to return
        a new array with the beam applied. By default False (i.e. return a new array).

    Returns
    -------
    lightcone_with_beam : astropy.units.Quantity
        The lightcone with the beam applied, with the same shape as the input lightcone.
    """
    gauss = compute_beam(
        observation=observation,
        freqs=freqs,
        box_ncells=lightcone.shape[1],
        box_length=box_length,
    )

    # We allow the lightcone to be 2,3 or 4D. If 2D, we assume it's (x, y). If 3D, we
    # assume it's (x, y, z). If 4D, we assume it's (nrealizations, x, y, z).
    ndim = lightcone.ndim
    if lightcone.ndim == 2:
        lightcone = lightcone[None, :, :, None]
    elif lightcone.ndim == 3:
        lightcone = lightcone[None, :, :, :]
    elif lightcone.ndim != 4:
        raise ValueError("lightcone must be either 2, 3 or 4D.")

    nrealizations, nx, ny, nz = lightcone.shape
    if nx != ny:
        raise ValueError("lightcone must have the same number of pixels in x and y.")
    if nz != len(freqs):
        raise ValueError(
            "The number of frequency channels in the lightcone must match the length "
            "of freqs."
        )

    lc = lightcone if in_place else lightcone.copy()
    lc *= gauss

    if ndim == 2:
        return lc[0, :, :, 0]
    if ndim == 3:
        return lc[0, :, :, :]
    return lc


def sample_from_rms_uvgrid(
    rms_noise: un.Quantity,
    seed: int | None = None,
    nrealizations: int = 1,
    return_in_uv: bool = False,
    spatial_taper: str | None = None,
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
    return_in_uv : bool, optional
        If True, return the noise sampled in uv space instead of real space,
        by default False.
    spatial_taper : str or None, optional
        If not None, the name of a window function to apply in uv space before taking
        the inverse FT. This can be used to mitigate ringing effects in real space that
        arise from sharp edges in uv space. The window function is applied in 2D,
        and the same window function is applied in both dimensions.

    Returns
    -------
    noise : un.Quantity
        Noise sampled in real or uv space, shape
        (nrealizations, Nx or Nu, Ny or Nv, Nfreqs). If in UV space, note that the
        ordering of the grid is switched to be in standard FFT format (i.e.
        zero-mode first, then negatives then positives).
    """
    # TODO: add the ability to weight the samples in UV space by an arbitrary weighting
    #       before taking the inverse FT, to e.g. have natural vs uniform weighting.
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

    if not return_in_uv and spatial_taper is not None:
        spatial_taper = taper2d(rms_noise.shape[0], spatial_taper)
        spatial_taper = spatial_taper[:, -ny:]  # restrict to the half-plane
        noise *= spatial_taper[None, ..., None]

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


@un.quantity_input
def apply_wedge_filter(
    uv_lightcones: tp.Temperature,
    kperp_x: tp.Wavenumber,
    kperp_y: tp.Wavenumber,
    lightcone_freqs: tp.Frequency,
    window_size: int | None = None,
    mode: Literal["rolling", "chunk"] = "chunk",
    wedge_slope: float = 1.0,
    buffer: tp.Time = 0.0 * un.ns,
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


@un.quantity_input
def observe_lightcone(
    lightcone: tp.Temperature,
    box_length: tp.Length,
    *,
    thermal_rms_uv: tp.Temperature,
    lightcone_redshifts: np.ndarray | None = None,
    lightcone_freqs: tp.Frequency | None = None,
    seed: int | None = None,
    nrealizations: int = 1,
    spatial_taper: str | None = None,
    remove_wedge: bool = False,
    wedge_chunk_size: int | None = None,
    wedge_slope: float = 1.0,
    wedge_buffer: tp.Time = 0.0,
    wedge_mode: Literal["rolling", "chunk"] = "chunk",
    cosmo=Planck18,
    remove_mean: bool = True,
    image_weighting : str | None = None,
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
   image_weighting : str, optional
        Filters the last image according to the chosen filter
        If not provided, the last image is unfiltered

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

    # Here the noise realizations are a 4D array (nrealizations, Nu, Nv, Nfreqs).
    # The ordering of the Nx, Ny axes is in standard format for FFT (i.e. zero-mode
    # first, then negatives then positives). Also, Nv=Nu//2 + 1, i.e. only the
    # non-negative modes in the second axis.
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

    lc_uv_nu = lc_uv_nu + noise_realisation_uv
    lc_uv_nu[:, thermal_rms_uv == 0] = 0.0

    with un.set_enabled_equivalencies(
        cu.with_H0(cosmo.H0) + cu.dimensionless_redshift()
    ):
        d = box_length / nx
        kperp_x = np.fft.fftfreq(nx, d=d)
        kperp_y = np.fft.rfftfreq(ny, d=d)

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

    if spatial_taper is not None:
        _, nx, ny = lc_uv_nu.shape
        window_fnc = taper2d(lightcone.shape[0], spatial_taper)[:, -ny:]
        window_fnc = np.fft.fftshift(
            window_fnc, axes=(0,)
        )  # shift the window to be in the right format for FFT
        lc_uv_nu *= window_fnc[None, ..., None]

    if image_weighting != None:
        _apply_weighting_for_imaging(weight_type=image_weighting, 
                                     thermal_rms_uv_nu = thermal_rms_uv , 
                                     noisy_lightcone_uv_nu = lc_uv_nu, 
                                     intrinsic_lightcone_uv_nu = lightcone)

    noisy_lc_real = np.fft.irfft2(lc_uv_nu, s=(nx, nx), axes=(1, 2)).to(lightcone.unit)

    return noisy_lc_real, lightcone_redshifts




def _apply_weighting_for_imaging(weight_type: str, 
                                 thermal_rms_uv_nu: tp.Temperature, 
                                 noisy_lightcone_uv_nu: tp.Temperature, 
                                 intrinsic_lightcone_uv_nu: tp.Temperature):
    """Applies weighting in the Fourier transform to real space for imaging. 
    
    The choice of filtering is not obvious and is application-specific, 
    WARNING: the simple wiener filtering takes into account the pure intrinsic signal (i.e., not realistic).
    
    Parameters
    ----------
    weight_type : str
        weighting scheme to use between ['none', 'inverse_variance', 'wiener', 'realistic_wiener'] or aliases ['n','iv','w', 'rw']
    thermal_rms_uv_nu : astropy.units.Quantity
        The uv coverage in uv_nu space
    noisy_lightcone_uv_nu : astropy.units.Quantity
        The previous unfiltered observed lightcone in uv_nu space
    intrinsic_lightcone_uv_nu : astropy.units.Quantity
        The intrinsic signal of the observed lightcone in uv_nu space (be careful with the simple 'wiener' filter as it is unrealistic)
    
    Returns
    -------
    lc_uv_nu : astropy.units.Quantity
        lightcone samples with filtered noise in uv_nu space 
    """
    lc_uv_nu  = noisy_lightcone_uv_nu
    lightcone = intrinsic_lightcone_uv_nu
    thermal_rms_uv = thermal_rms_uv_nu
    
    if weight_type == 'none' or weight_type == None:
        return lc_uv_nu
    
    elif weight_type in ('inverse_variance', 'iv'):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / (2*(thermal_rms_uv**2).value)   # inverse variance weights

    elif weight_type in ('wiener', 'w'): # essentially matching the power spectrum of the observed with the signal
        # Estimate signal power spectrum from the measured signal and the known noise
        # Wiener filter is defined usually for the power spectrum as w = (p_signal)/(p_signal + p_noise)
        # the sigma in our case is applied in the real and imaginary parts separately. 
        # So the w computed should be further processed to w = np.sqrt(w)/2 (like in generating ICs for cosmological simulations)
        p_signal = np.abs(np.fft.rfft2(lightcone.value, axes=(0,1)))**2  # shape (nx, ny_rfft)

        # Now p_signal is the pure signal, just fourier transformed. Let's take the mean along the frequency axis and repeat it to be used as a prior
        p_signal = np.mean(p_signal, axis=2)
        
        p_signal = p_signal[..., np.newaxis]
        print(p_signal.shape)
        p_signal = np.repeat(p_signal, 200, axis=2)
        # Perhaps smooth it (maybe good but also sort of prior dependent)
        # p_signal = gaussian_filter(p_signal, sigma=2)  # from scipy.ndimage
        p_noise = 2*(thermal_rms_uv.value)**2  # our existing noise variance (real and imaginary parts so the variance is 2 sigma^2, hence the "2")
        with np.errstate(divide="ignore", invalid="ignore"):
            w = p_signal / (p_signal + p_noise)   # Wiener filter, values in [0,1]
            w = np.sqrt(w/2) # applied on the 2D UV space on real and imag parts. 
       
    elif weight_type in ('realistic_wiener', 'rw'):
        # Same as "Wiener" but estimate p_signal from observation
        
        p_measured = 2*np.abs((lc_uv_nu[0].value))**2
        p_noise = 2*(thermal_rms_uv.value)**2 
        
        p_signal = p_measured - p_noise
        with np.errstate(divide="ignore", invalid="ignore"):
            w = p_signal / (p_signal + p_noise)   
            w = np.sqrt(w/2) 
            
    else:
        raise ValueError(
            "Invalid_filtering "
            "Choose between ['none', 'inverse_variance', 'wiener', 'realistic_wiener']"
        )
    
    # Make sure unphysical values are skipped
    w[np.isinf(w)] = 0.0                # set unsampled cells to 0, w is in uv_nu space
    w[np.isnan(w)] = 0.0
    w[thermal_rms_uv.value == 0] = 0.0  # zero where unsampled, probably covered by the previous line
    wsum = w.sum(axis=(0,1), keepdims=True)
    lc_uv_nu *= w.shape[0] * w.shape[1] * w[None, ...] / wsum[None, ...] # ad-hoc normalization
    # 2nd and 3rd dimensions of w and wsum will not match but numpy is broadcasting
        
    return lc_uv_nu
