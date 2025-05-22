"""Code to calculate the 1D and 2D power spectrum of a lightcone."""

import warnings
from collections.abc import Callable

import astropy.units as un
import numpy as np
from powerbox.tools import (
    _magnitude_grid,
    above_mu_min_angular_generator,
    angular_average,
    get_power,
    ignore_zero_ki,
    power2delta,
    regular_angular_generator,
)
from scipy.interpolate import RegularGridInterpolator

from tuesday.core import CylindricalPS, SphericalPS
from tuesday.core.units import validate


def get_chunk_indices(
    lc_redshifts: np.ndarray,
    chunk_size: int | np.ndarray,
    ps_redshifts: np.ndarray | None = None,
    chunk_skip: np.ndarray | None = None,
):
    """Get the start and end indices for each lightcone chunk."""
    n_slices = lc_redshifts.shape[0]

    if ps_redshifts is None:
        if chunk_skip is None:
            chunk_skip = chunk_size
        if isinstance(chunk_size, int):
            chunk_starts = list(range(0, n_slices - chunk_size, chunk_skip))
            chunk_ends = np.array(chunk_starts) + chunk_size
        if isinstance(chunk_size, np.ndarray):
            raise ValueError(
                "chunk_size should be an int or ps_redshifts should be provided."
            )
    else:
        if not np.iterable(ps_redshifts):
            ps_redshifts = np.array([ps_redshifts])

        if np.min(np.round(ps_redshifts, 5)) < np.min(
            np.round(lc_redshifts, 5)
        ) or np.max(np.round(ps_redshifts, 5)) > np.max(np.round(lc_redshifts, 5)):
            raise ValueError("ps_redshifts should be within the range of lc_redshifts")
        if isinstance(chunk_size, int):
            chunk_size = np.array([chunk_size] * len(ps_redshifts))
        chunk_starts = np.array(
            [
                np.max([np.argmin(abs(lc_redshifts - z)) - s // 2, 0])
                for z, s in zip(ps_redshifts, chunk_size, strict=False)
            ],
            dtype=np.int32,
        )
        chunk_ends = np.min(
            [
                np.array(chunk_starts) + chunk_size,
                np.zeros_like(ps_redshifts) + n_slices,
            ],
            axis=0,
        )
    chunk_starts = np.array(chunk_starts, dtype=np.int32)
    chunk_ends = np.array(chunk_ends, dtype=np.int32)
    chunk_indices = [(s, e) for s, e in zip(chunk_starts, chunk_ends, strict=False)]
    return chunk_indices


def calculate_ps(
    chunk: un.Quantity,
    box_length: un.Quantity,
    *,
    chunk_redshift: float | None = None,
    calc_2d: bool | None = True,
    kperp_bins: int | np.ndarray | None = None,
    k_weights_2d: Callable | None = ignore_zero_ki,
    log_bins: bool | None = True,
    calc_1d: bool | None = False,
    k_bins: int | None = None,
    k_weights_1d: Callable | None = ignore_zero_ki,
    bin_ave: bool | None = True,
    interp: bool | None = None,
    prefactor_fnc: Callable | None = power2delta,
    interp_points_generator: Callable | None = None,
    get_variance: bool | None = False,
) -> dict:
    r"""Calculate power spectra from a lightcone or coeval box.

    Parameters
    ----------
    chunk : un.Quantity
        The 3D chunk whose power spectrum we want to calculate.
        This can be either a coeval box or a lightcone chunk.
    box_length : un.Quantity
        The side length of the box.
        Accepted units are: Mpc and Mpc/h.
    chunk_redshift : float, optional
        The central redshift of the lightcone chunk or coeval box.
    calc_2d : bool, optional
        If True, calculate the 2D power spectrum.
    kperp_bins : int, optional
        The number of bins to use for the kperp axis of the 2D PS.
    k_weights : callable, optional
        A function that takes a frequency tuple and returns
        a boolean mask for the k values to ignore.
        See powerbox.tools.ignore_zero_ki for an example
        and powerbox.tools.get_power documentation for more details.
        Default is powerbox.tools.ignore_zero_ki, which excludes
        the power any k_i = 0 mode.
        Typically, only the central zero mode |k| = 0 is excluded,
        in which case use powerbox.tools.ignore_zero_absk.
    calc_1d : bool, optional
        If True, calculate the 1D power spectrum.
    k_bins : int, optional
        The number of bins on which to calculate 1D PS.
    bin_ave : bool, optional
        If True, return the center value of each kperp and kpar bin
        i.e. len(kperp) = ps_2d.shape[0].
        If False, return the left edge of each bin
        i.e. len(kperp) = ps_2d.shape[0] + 1.
    interp : str, optional
        If True, use linear interpolation to calculate the PS
        at the points specified by interp_points_generator.
        Note that this significantly slows down the calculation.
    prefactor_fnc : callable, optional
        A function that takes a frequency tuple and returns the prefactor
        to multiply the PS with.
        Default is powerbox.tools.power2delta, which converts the power
        P [mK^2 Mpc^{-3}] to the dimensionless power :math:`\\delta^2` [mK^2].
    interp_points_generator : callable, optional
        A function that generates the points at which to interpolate the PS.
        See powerbox.tools.get_power documentation for more details.
    """
    if not calc_1d and not calc_2d:
        raise ValueError("At least one of calc_1d or calc_2d must be True.")

    if not interp:
        interp = None
    if not isinstance(chunk, un.Quantity):
        raise TypeError("chunk should be a Quantity.")

    if not isinstance(box_length, un.Quantity):
        raise TypeError("box_length should be a Quantity.")
    # Split the lightcone into chunks for each redshift bin
    # Infer HII_DIM from lc side shape
    box_side_shape = chunk.shape[0]
    if get_variance and interp is not None:
        raise NotImplementedError("Cannot get variance while interpolating.")

    out = {}
    if calc_1d:
        out["ps_1d"] = {}
    if calc_2d:
        out["ps_2d"] = {}

    if interp:
        interp = "linear"

    if prefactor_fnc is None:
        ps_unit = chunk.unit**2 * box_length.unit**3
    elif prefactor_fnc == power2delta:
        ps_unit = chunk.unit**2
    else:
        warnings.warn(
            "The prefactor function is not the default. PS unit may not be correct.",
            stacklevel=2,
        )
        ps_unit = chunk.unit**2

    if calc_2d:
        results = get_power(
            chunk,
            (
                box_length.value,
                box_length.value,
                box_length.value * chunk.shape[-1] / box_side_shape,
            ),
            res_ndim=2,
            bin_ave=bin_ave,
            bins=kperp_bins,
            log_bins=log_bins,
            nthreads=1,
            k_weights=k_weights_2d,
            prefactor_fnc=prefactor_fnc,
            interpolation_method=interp,
            return_sumweights=True,
            get_variance=get_variance,
        )
        if get_variance:
            ps_2d, kperp, var, nmodes, kpar = results
            lc_var_2d = var
        else:
            ps_2d, kperp, nmodes, kpar = results

        kpar = np.array(kpar).squeeze()
        lc_ps_2d = ps_2d[..., kpar > 0]
        kpar = kpar[kpar > 0]
        out["ps_2d"] = CylindricalPS(
            ps=lc_ps_2d * ps_unit,
            kperp=kperp.squeeze() / box_length.unit,
            kpar=kpar / box_length.unit,
            redshift=chunk_redshift,
            Nmodes=nmodes,
            var=lc_var_2d * ps_unit**2 if get_variance else None,
            delta=True if prefactor_fnc is not None else False,
        )

    if calc_1d:
        results = get_power(
            chunk,
            (
                box_length.value,
                box_length.value,
                box_length.value * chunk.shape[-1] / box_side_shape,
            ),
            bin_ave=bin_ave,
            bins=k_bins,
            log_bins=log_bins,
            k_weights=k_weights_1d,
            prefactor_fnc=prefactor_fnc,
            interpolation_method=interp,
            interp_points_generator=interp_points_generator,
            return_sumweights=True,
            get_variance=get_variance,
        )
        if get_variance:
            ps_1d, k, var_1d, nmodes_1d = results
            lc_var_1d = var_1d
        else:
            ps_1d, k, nmodes_1d = results
        lc_ps_1d = ps_1d

        out["ps_1d"] = SphericalPS(
            ps=lc_ps_1d * ps_unit,
            k=k.squeeze() / box_length.unit,
            redshift=chunk_redshift,
            Nmodes=nmodes_1d.squeeze(),
            var=lc_var_1d * ps_unit**2 if get_variance else None,
            delta=True if prefactor_fnc is not None else False,
        )

    return out


def calculate_ps_lc(
    lc: un.Quantity,
    box_length: un.Quantity,
    lc_redshifts: np.ndarray,
    *,
    ps_redshifts: float | np.ndarray | None = None,
    chunk_indices: list | None = None,
    chunk_size: int | None = None,
    chunk_skip: int | None = None,
    calc_2d: bool | None = True,
    kperp_bins: int | None = None,
    k_weights_2d: Callable | None = ignore_zero_ki,
    k_weights_1d: Callable | None = ignore_zero_ki,
    postprocess: bool | None = True,
    kpar_bins: int | np.ndarray | None = None,
    log_bins: bool | None = True,
    crop: list | np.ndarray | None = None,
    calc_1d: bool | None = True,
    k_bins: int | None = None,
    mu_min: float | None = None,
    bin_ave: bool | None = True,
    interp: bool | None = None,
    delta: bool | None = True,
    interp_points_generator: Callable | None = None,
    get_variance: bool | None = False,
) -> dict:
    """
    Calculate the PS by chunking a lightcone.

    Parameters
    ----------
    lc : un.Quantity
        The lightcone with units of temperature or dimensionless.
    box_length : un.Quantity
        The side length of the box, accepted units are length.
    lc_redshifts : np.ndarray
        The redshift of each lightcone slice.
        Array has same length as lc.shape[-1].
    chunk_redshift : float, optional
        The central redshift of the lightcone chunk or coeval box.
    calc_2d : bool, optional
        If True, calculate the 2D power spectrum.
    kperp_bins : int, optional
        The number of bins to use for the kperp axis of the 2D PS.
    k_weights : callable, optional
        A function that takes a frequency tuple and returns
        a boolean mask for the k values to ignore.
        See powerbox.tools.ignore_zero_ki for an example
        and powerbox.tools.get_power documentation for more details.
        Default is powerbox.tools.ignore_zero_ki, which excludes
        the power any k_i = 0 mode.
        Typically, only the central zero mode |k| = 0 is excluded,
        in which case use powerbox.tools.ignore_zero_absk.
    calc_1d : bool, optional
        If True, calculate the 1D power spectrum.
    k_bins : int, optional
        The number of bins on which to calculate 1D PS.
    mu_min : float, optional
        The minimum value of
        :math:`\\cos(\theta), \theta = \arctan (k_\\perp/k_\\parallel)`
        for all calculated PS.
        If None, all modes are included.
    bin_ave : bool, optional
        If True, return the center value of each kperp and kpar bin
        i.e. len(kperp) = ps_2d.shape[0].
        If False, return the left edge of each bin
        i.e. len(kperp) = ps_2d.shape[0] + 1.
    interp : str, optional
        If True, use linear interpolation to calculate the PS
        at the points specified by interp_points_generator.
        Note that this significantly slows down the calculation.
    delta : bool, optional
        Whether to convert the power P [mK^2 Mpc^{-3}] to the dimensionless
        power :math:`\\delta^2` [mK^2].
        Default is True.
    interp_points_generator : callable, optional
        A function that generates the points at which to interpolate the PS.
        See powerbox.tools.get_power documentation for more details.
    """
    validate(lc, "temperature")
    validate(box_length, "length")
    if chunk_indices is None:
        chunk_indices = get_chunk_indices(
            lc_redshifts,
            lc.shape[0] if chunk_size is None else chunk_size,
            ps_redshifts=ps_redshifts,
            chunk_skip=chunk_skip,
        )
    if mu_min is not None:
        if interp is None:
            k_weights_1d_input = k_weights_1d

            def mask_fnc(freq, absk):
                kz_mesh = np.zeros((len(freq[0]), len(freq[1]), len(freq[2])))
                kz = freq[2]
                for i in range(len(kz)):
                    kz_mesh[:, :, i] = kz[i]
                phi = np.arccos(kz_mesh / absk)
                mu_mesh = abs(np.cos(phi))
                kmag = _magnitude_grid([c for i, c in enumerate(freq) if i < 2])
                return np.logical_and(mu_mesh > mu_min, k_weights_1d_input(freq, kmag))

            k_weights_1d = mask_fnc

        if interp is not None:
            k_weights_1d = ignore_zero_ki

            interp_points_generator = above_mu_min_angular_generator(mu=mu_min)
    else:
        k_weights_1d = ignore_zero_ki
        if interp is not None:
            interp_points_generator = regular_angular_generator()
    if delta:
        prefactor_fnc = power2delta
    else:
        prefactor_fnc = None
    out = {}
    if calc_1d:
        out["ps_1d"] = {}
    if calc_2d:
        out["ps_2d"] = {}
    for chunk in chunk_indices:
        start = chunk[0]
        end = chunk[1]

        chunk = lc[..., start:end]
        if lc_redshifts is not None:
            chunk_z = lc_redshifts[(start + end) // 2]
        ps_chunk = calculate_ps(
            chunk=chunk,
            box_length=box_length,
            chunk_redshift=chunk_z,
            calc_2d=calc_2d,
            kperp_bins=kperp_bins,
            k_weights_2d=k_weights_2d,
            k_weights_1d=k_weights_1d,
            log_bins=log_bins,
            calc_1d=calc_1d,
            k_bins=k_bins,
            bin_ave=bin_ave,
            interp=interp,
            prefactor_fnc=prefactor_fnc,
            interp_points_generator=interp_points_generator,
            get_variance=get_variance,
        )
        if calc_1d:
            out["ps_1d"]["z = " + str(np.round(chunk_z, 2))] = ps_chunk["ps_1d"]
        if calc_2d:
            out["ps_2d"]["z = " + str(np.round(chunk_z, 2))] = ps_chunk["ps_2d"]
    return out


def calculate_ps_coeval(
    box: un.Quantity,
    box_length: un.Quantity,
    *,
    box_redshift: float | None = None,
    calc_2d: bool | None = True,
    kperp_bins: int | None = None,
    k_weights_2d: Callable | None = ignore_zero_ki,
    k_weights_1d: Callable | None = ignore_zero_ki,
    postprocess: bool | None = True,
    kpar_bins: int | np.ndarray | None = None,
    log_bins: bool | None = True,
    crop: list | np.ndarray | None = None,
    calc_1d: bool | None = True,
    k_bins: int | None = None,
    mu_min: float | None = None,
    bin_ave: bool | None = True,
    interp: bool | None = None,
    delta: bool | None = True,
    interp_points_generator: Callable | None = None,
    get_variance: bool | None = False,
) -> dict:
    """
    Calculate the PS by chunking a lightcone.

    Parameters
    ----------
    box : un.Quantity
        The coeval box with units of temperature or dimensionless.
    box_length : un.Quantity
        The side length of the box, accepted units are length.
    box_redshift : float, optional
        The redshift value of the coeval box.
    chunk_redshift : float, optional
        The central redshift of the lightcone chunk or coeval box.
    calc_2d : bool, optional
        If True, calculate the 2D power spectrum.
    kperp_bins : int, optional
        The number of bins to use for the kperp axis of the 2D PS.
    k_weights : callable, optional
        A function that takes a frequency tuple and returns
        a boolean mask for the k values to ignore.
        See powerbox.tools.ignore_zero_ki for an example
        and powerbox.tools.get_power documentation for more details.
        Default is powerbox.tools.ignore_zero_ki, which excludes
        the power any k_i = 0 mode.
        Typically, only the central zero mode |k| = 0 is excluded,
        in which case use powerbox.tools.ignore_zero_absk.
    calc_1d : bool, optional
        If True, calculate the 1D power spectrum.
    k_bins : int, optional
        The number of bins on which to calculate 1D PS.
    mu_min : float, optional
        The minimum value of
        :math:`\\cos(\theta), \theta = \arctan (k_\\perp/k_\\parallel)`
        for all calculated PS.
        If None, all modes are included.
    bin_ave : bool, optional
        If True, return the center value of each kperp and kpar bin
        i.e. len(kperp) = ps_2d.shape[0].
        If False, return the left edge of each bin
        i.e. len(kperp) = ps_2d.shape[0] + 1.
    interp : str, optional
        If True, use linear interpolation to calculate the PS
        at the points specified by interp_points_generator.
        Note that this significantly slows down the calculation.
    delta : bool, optional
        Whether to convert the power P [mK^2 Mpc^{-3}] to the dimensionless
        power :math:`\\delta^2` [mK^2].
        Default is True.
    interp_points_generator : callable, optional
        A function that generates the points at which to interpolate the PS.
        See powerbox.tools.get_power documentation for more details.
    """
    validate(box, "temperature")
    validate(box_length, "length")
    if mu_min is not None:
        if interp is None:
            k_weights_1d_input = k_weights_1d

            def mask_fnc(freq, absk):
                kz_mesh = np.zeros((len(freq[0]), len(freq[1]), len(freq[2])))
                kz = freq[2]
                for i in range(len(kz)):
                    kz_mesh[:, :, i] = kz[i]
                phi = np.arccos(kz_mesh / absk)
                mu_mesh = abs(np.cos(phi))
                kmag = _magnitude_grid([c for i, c in enumerate(freq) if i < 2])
                return np.logical_and(mu_mesh > mu_min, k_weights_1d_input(freq, kmag))

            k_weights_1d = mask_fnc

        if interp is not None:
            k_weights_1d = ignore_zero_ki

            interp_points_generator = above_mu_min_angular_generator(mu=mu_min)
    else:
        k_weights_1d = ignore_zero_ki
        if interp is not None:
            interp_points_generator = regular_angular_generator()
    if delta:
        prefactor_fnc = power2delta
    else:
        prefactor_fnc = None
    return calculate_ps(
        chunk=box,
        box_length=box_length,
        chunk_redshift=box_redshift,
        calc_2d=calc_2d,
        kperp_bins=kperp_bins,
        k_weights_2d=k_weights_2d,
        k_weights_1d=k_weights_1d,
        log_bins=log_bins,
        calc_1d=calc_1d,
        k_bins=k_bins,
        bin_ave=bin_ave,
        interp=interp,
        prefactor_fnc=prefactor_fnc,
        interp_points_generator=interp_points_generator,
        get_variance=get_variance,
    )


def bin_kpar(ps, kperp, kpar, bins=None, interp=None, log=False, redshifts=None):
    r"""
    Bin a 2D PS along the kpar axis and crop out empty bins in both axes.

    Parameters
    ----------
    ps : np.ndarray
        The 2D power spectrum of shape [len(redshifts), len(kperp), len(kpar)].
    kperp : np.ndarray
        Values of kperp.
    kpar : np.ndarray
        Values of kpar.
    bins : np.ndarray or int, optional
        The number of bins or the bin edges to use for binning the kpar axis.
        If None, produces 16 bins logarithmically spaced between
        the minimum and maximum `kpar` supplied.
    interp : str, optional
        If 'linear', use linear interpolation to calculate the PS at the specified
        kpar bins.
    log : bool, optional
        If 'False', kpar is binned linearly. If 'True', it is binned logarithmically.
    redshifts : np.ndarray, optional
        The redshifts at which the PS was calculated.
    """
    ps = ps if len(ps.shape) == 3 else ps[np.newaxis, ...]
    if bins is None:
        if log:
            bins = np.logspace(
                np.log10(kpar[0]), np.log10(kpar[-1]), len(kpar) // 2 + 1
            )
        else:
            bins = np.linspace(kpar[0], kpar[-1], len(kpar) // 2 + 1)
    elif isinstance(bins, int):
        if log:
            bins = np.logspace(np.log10(kpar[0]), np.log10(kpar[-1]), bins + 1)
        else:
            bins = np.linspace(kpar[0], kpar[-1], bins + 1)
    elif isinstance(bins, np.ndarray | list):
        bins = np.array(bins)
    else:
        raise ValueError("Bins should be np.ndarray or int")
    if log:
        bin_centers = np.exp((np.log(bins[1:]) + np.log(bins[:-1])) / 2)
    else:
        bin_centers = (bins[1:] + bins[:-1]) / 2
    if interp == "linear":
        new_ps = np.zeros((ps.shape[0], len(kperp), len(bins)))
        modes = np.zeros(len(bins))
        interp_fnc = RegularGridInterpolator(
            (redshifts, kperp, kpar) if redshifts is not None else (kperp, kpar),
            ps.squeeze(),
            bounds_error=False,
            fill_value=np.nan,
        )

        if redshifts is None:
            kperp_grid, kpar_grid = np.meshgrid(
                kperp, bin_centers, indexing="ij", sparse=True
            )
            new_ps = interp_fnc((kperp_grid, kpar_grid))
        else:
            redshifts_grid, kperp_grid, kpar_grid = np.meshgrid(
                redshifts, kperp, bin_centers, indexing="ij", sparse=True
            )
            new_ps = interp_fnc((redshifts_grid, kperp_grid, kpar_grid))

        idxs = np.digitize(kpar, bins) - 1
        for i in range(len(bins) - 1):
            modes[i] = np.sum(idxs == i)
    else:
        new_ps = np.zeros((ps.shape[0], len(kperp), len(bins) - 1))
        modes = np.zeros(len(bins) - 1)
        idxs = np.digitize(kpar, bins) - 1
        for i in range(len(bins) - 1):
            m = idxs == i
            new_ps[..., i] = np.nanmean(ps[..., m], axis=-1)
            modes[i] = np.sum(m)

    return new_ps.squeeze(), kperp, bin_centers, modes


def postprocess_ps(
    ps,
    kperp,
    kpar,
    kpar_bins=None,
    log_bins=True,
    crop=None,
    kperp_modes=None,
    return_modes=False,
    interp=None,
):
    r"""
    Postprocess a 2D PS by cropping out empty bins and log binning the kpar axis.

    Parameters
    ----------
    ps : np.ndarray
        The 2D power spectrum of shape [len(kperp), len(kpar)].
    kperp : np.ndarray
        Values of kperp.
    kpar : np.ndarray
        Values of kpar.
    kpar_bins : np.ndarray or int, optional
        The number of bins or the bin edges to use for binning the kpar axis.
        If None, produces 16 bins log spaced between the min and max `kpar` supplied.
    log_bins : bool, optional
        If True, log bin the kpar axis.
    crop : list, optional
        The crop range for the log-binned PS. If None, crops out all empty bins.
    kperp_modes : np.ndarray, optional
        The number of modes in each kperp bin.
    return_modes : bool, optional
        If True, return a grid with the number of modes in each bin.
        Requires kperp_modes to be supplied.
    """
    kpar = kpar[0]
    m = kpar > 1e-10
    if ps.shape[0] < len(kperp):
        if log_bins:
            kperp = np.exp((np.log(kperp[1:]) + np.log(kperp[:-1])) / 2.0)
        else:
            kperp = (kperp[1:] + kperp[:-1]) / 2
    kpar = kpar[m]
    ps = ps[:, m]
    mkperp = ~np.isnan(kperp)
    if kperp_modes is not None:
        kperp_modes = kperp_modes[mkperp]
    kperp = kperp[mkperp]
    ps = ps[mkperp, :]

    # maybe rebin kpar in log
    rebinned_ps, kperp, log_kpar, kpar_weights = bin_kpar(
        ps, kperp, kpar, bins=kpar_bins, interp=interp, log=log_bins
    )
    if crop is None:
        crop = [0, rebinned_ps.shape[-2] + 1, 0, rebinned_ps.shape[-1] + 1]
    # Find last bin that is NaN and cut out all bins before
    try:
        lastnan_perp = np.where(np.isnan(np.nanmean(rebinned_ps, axis=1)))[0][-1] + 1
        crop[0] = crop[0] + lastnan_perp
    except IndexError:
        pass
    try:
        lastnan_par = np.where(np.isnan(np.nanmean(rebinned_ps, axis=0)))[0][-1] + 1
        crop[2] = crop[2] + lastnan_par
    except IndexError:
        pass
    if kperp_modes is not None:
        kperp_modes = kperp_modes[crop[0] : crop[1]]
        kpar_grid, kperp_grid = np.meshgrid(
            kpar_weights[crop[2] : crop[3]], kperp_modes
        )

        nmodes = np.sqrt(kperp_grid**2 + kpar_grid**2)
        if return_modes:
            return (
                rebinned_ps[..., crop[0] : crop[1], :][..., crop[2] : crop[3]],
                kperp[crop[0] : crop[1]],
                log_kpar[crop[2] : crop[3]],
                nmodes,
            )
        return (
            rebinned_ps[None, ..., crop[0] : crop[1], :][..., crop[2] : crop[3]],
            kperp[crop[0] : crop[1]],
            log_kpar[crop[2] : crop[3]],
        )
    return (
        rebinned_ps[None, ..., crop[0] : crop[1], :][..., crop[2] : crop[3]],
        kperp[crop[0] : crop[1]],
        log_kpar[crop[2] : crop[3]],
    )


def cylindrical_to_spherical(
    ps,
    kperp,
    kpar,
    nbins=16,
    weights=1,
    interp=False,
    mu_min=None,
    generator=None,
    bin_ave=True,
):
    r"""
    Angularly average 2D PS to 1D PS.

    Parameters
    ----------
    ps : np.ndarray
        The 2D power spectrum of shape [len(kperp), len(kpar)].
    kperp : np.ndarray
        Values of kperp.
    kpar : np.ndarray
        Values of kpar.
    nbins : int, optional
        The number of bins on which to calculate 1D PS. Default is 16
    weights : np.ndarray, optional
        Weights to apply to the PS before averaging.
        Note that to obtain a 1D PS from the 2D PS that is consistent with
        the 1D PS obtained directly from the 3D PS, the weights should be
        the number of modes in each bin of the 2D PS (`Nmodes`).
    interp : bool, optional
        If True, use linear interpolation to calculate the 1D PS.
    mu_min : float, optional
        The minimum value of
        :math:`\\cos(\theta), \theta = \arctan (k_\\perp/k_\\parallel)`
        for all calculated PS.
        If None, all modes are included.
    generator : callable, optional
        A function that generates the points at which to interpolate the PS.
        See powerbox.tools.get_power documentation for more details.
    bin_ave : bool, optional
        If True, return the center value of each k bin
        i.e. len(k) = ps_1d.shape[0].
        If False, return the left edge of each bin
        i.e. len(k) = ps_1d.shape[0] + 1.
    """
    if mu_min is not None and interp and generator is None:
        generator = above_mu_min_angular_generator(mu=mu_min)

    if mu_min is not None and not interp:
        kpar_mesh, kperp_mesh = np.meshgrid(kpar, kperp)
        theta = np.arctan(kperp_mesh / kpar_mesh)
        mu_mesh = np.cos(theta)
        weights = mu_mesh >= mu_min

    ps_1d, k, sws = angular_average(
        ps,
        coords=[kperp, kpar],
        bins=nbins,
        weights=weights,
        bin_ave=bin_ave,
        log_bins=True,
        return_sumweights=True,
        interpolation_method="linear" if interp else None,
        interp_points_generator=generator,
    )
    return ps_1d, k, sws
