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

from ..units import validate
from .psclasses import CylindricalPS, SphericalPS


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
    return list(zip(chunk_starts, chunk_ends, strict=False))


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
) -> tuple[SphericalPS | None, CylindricalPS | None]:
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
    get_variance : bool, optional
        If True, compute the variance of the PS over the modes within each bin.
        Default is False.

    Returns
    -------
    ps1d : SphericalPS or None
        The 1D power spectrum.
        None if calc_1d is False.
    ps2d : CylindricalPS or None
        The 2D power spectrum.
        None if calc_2d is False.
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

    ps2d = None
    ps1d = None
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
            bins_upto_boxlen=True,
        )
        if get_variance:
            ps_2d, kperp, variance, nmodes, kpar = results
            lc_var_2d = variance
        else:
            ps_2d, kperp, nmodes, kpar = results

        kpar = np.array(kpar).squeeze()
        lc_ps_2d = ps_2d[..., kpar > 0]
        if get_variance:
            lc_var_2d = lc_var_2d[..., kpar > 0]
        kpar = kpar[kpar > 0]
        ps2d = CylindricalPS(
            ps=lc_ps_2d * ps_unit,
            kperp=kperp.squeeze() / box_length.unit,
            kpar=kpar / box_length.unit,
            redshift=chunk_redshift,
            n_modes=nmodes,
            variance=lc_var_2d * ps_unit**2 if get_variance else None,
            is_deltasq=prefactor_fnc is not None,
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
            bins_upto_boxlen=True,
        )
        if get_variance:
            ps_1d, k, var_1d, nmodes_1d = results
            lc_var_1d = var_1d
        else:
            ps_1d, k, nmodes_1d = results
        lc_ps_1d = ps_1d

        ps1d = SphericalPS(
            ps=lc_ps_1d * ps_unit,
            k=k.squeeze() / box_length.unit,
            redshift=chunk_redshift,
            n_modes=nmodes_1d.squeeze(),
            variance=lc_var_1d * ps_unit**2 if get_variance else None,
            is_deltasq=prefactor_fnc is not None,
        )

    return ps1d, ps2d


def calculate_ps_lc(
    lc: un.Quantity,
    box_length: un.Quantity,
    lc_redshifts: np.ndarray,
    *,
    ps_redshifts: float | np.ndarray | None = None,
    chunk_indices: list | None = None,
    chunk_size: int | None = None,
    chunk_skip: int | None = None,
    calc_2d: bool = True,
    kperp_bins: int | None = None,
    k_weights_2d: Callable | None = ignore_zero_ki,
    k_weights_1d: Callable | None = ignore_zero_ki,
    log_bins: bool = True,
    calc_1d: bool = True,
    k_bins: int | None = None,
    mu_min: float | None = None,
    bin_ave: bool = True,
    interp: bool | None = None,
    deltasq: bool = True,
    interp_points_generator: Callable | None = None,
    get_variance: bool = False,
    transform_ps1d: Callable | None = None,
    transform_ps2d: Callable | None = None,
) -> tuple[list[SphericalPS] | None, list[CylindricalPS] | None]:
    r"""
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
    transform_ps2d : Callable, optional
        A function that takes in a CylindricalPS object and returns
        a new CylindricalPS object.
    transform_ps1d : Callable, optional
        A function that takes in a SphericalPS object and returns
        a new SphericalPS object.
    get_variance : bool, optional
        Whether to calculate the variance of the PS.
        Default is False.
    chunk_indices : list, optional
        A list of tuples specifying the start and end indices of the lightcone
        chunks for which power spectra are calculated.

    Returns
    -------
    ps1d : list of SphericalPS or None
        The 1D power spectrum for each chunk.
        None if calc_1d is False.
    ps2d : list of CylindricalPS or None
        The 2D power spectrum for each chunk.
        None if calc_2d is False.
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

    prefactor_fnc = power2delta if deltasq else None

    ps2ds = []
    ps1ds = []

    for chunk in chunk_indices:
        start = chunk[0]
        end = chunk[1]

        chunk = lc[..., start:end]
        if lc_redshifts is not None:
            chunk_z = lc_redshifts[(start + end) // 2]
        ps1d, ps2d = calculate_ps(
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
        if ps1d is not None and transform_ps1d is not None:
            ps1d = transform_ps1d(ps1d)
        if ps2d is not None and transform_ps2d is not None:
            ps2d = transform_ps2d(ps2d)

        ps1ds.append(ps1d)
        ps2ds.append(ps2d)

    return ps1ds if calc_1d else None, ps2ds if calc_2d else None


def calculate_ps_coeval(
    box: un.Quantity,
    box_length: un.Quantity,
    *,
    box_redshift: float | None = None,
    calc_2d: bool | None = True,
    kperp_bins: int | None = None,
    k_weights_2d: Callable | None = ignore_zero_ki,
    k_weights_1d: Callable | None = ignore_zero_ki,
    log_bins: bool | None = True,
    calc_1d: bool | None = True,
    k_bins: int | None = None,
    mu_min: float | None = None,
    bin_ave: bool | None = True,
    interp: bool | None = None,
    deltasq: bool | None = True,
    interp_points_generator: Callable | None = None,
    get_variance: bool | None = False,
    transform_ps1d: Callable | None = None,
    transform_ps2d: Callable | None = None,
) -> tuple[SphericalPS | None, CylindricalPS | None]:
    r"""
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
    get_variance : bool, optional
        Whether to calculate the variance of the PS.
        Default is False.
    transform_ps2d : Callable, optional
        A function that takes in a CylindricalPS object and returns
        a new CylindricalPS object.
    transform_ps1d : Callable, optional
        A function that takes in a SphericalPS object and returns
        a new SphericalPS object.
    get_variance : bool, optional
        Whether to calculate the variance of the PS.
        Default is False.
    interp : bool, optional
        If True, use linear interpolation to calculate the PS
        at the points specified by interp_points_generator.
        Note that this significantly slows down the calculation.
    deltasq : bool, optional
        Whether to convert the power P [mK^2 Mpc^{-3}] to the dimensionless
        power :math:`\\delta^2` [mK^2].
        Default is True.
    transform_ps1d : Callable, optional
        A function that takes in a SphericalPS object and returns
        a new SphericalPS object.
    transform_ps2d : Callable, optional
        A function that takes in a CylindricalPS object and returns
        a new CylindricalPS object.

    Returns
    -------
    ps1d : SphericalPS or None
        The 1D power spectrum.
        None if calc_1d is False.
    ps2d : CylindricalPS or None
        The 2D power spectrum.
        None if calc_2d is False.
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
    prefactor_fnc = power2delta if deltasq else None

    ps1d, ps2d = calculate_ps(
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
    if calc_1d and transform_ps1d is not None:
        ps1d = transform_ps1d(ps1d)

    if calc_2d and transform_ps2d is not None:
        ps2d = transform_ps2d(ps2d)

    return ps1d, ps2d


def bin_kpar(
    bins_kpar: int | un.Quantity | None = None,
    log_kpar: bool | None = False,
    interp_kpar: bool | None = False,
    crop_kperp: tuple[int, int] | None = None,
    crop_kpar: tuple[int, int] | None = None,
):
    """
    Bins the kpar axis of a CylindricalPS object.

    Parameters
    ----------
    bins_kpar : int, astropy.units.Quantity, or None, optional
        Number of bins (if int), array of bin edges (if array-like),
        or None to use default binning (half the number of original kpar bins).
        Default is None.
    log_kpar : bool or None, optional
        If True, use logarithmic binning for kpar.
        If False or None, use linear binning. Default is False.
    interp_kpar : bool or None, optional
        If True, interpolate the power spectrum onto the new kpar bins.
        If False or None, aggregate using bin means. Default is False.
    crop_kperp : tuple of int or None, optional
        Tuple specifying the (start, end) indices to crop the kperp axis after binning.
        If None, no cropping is applied. Default is None.
    crop_kpar : tuple of int or None, optional
        Tuple specifying the (start, end) indices to crop the kpar axis after binning.
        If None, no cropping is applied. Default is None.

    Returns
    -------
    transform_ps : callable
        A function that takes a CylindricalPS object and returns a new CylindricalPS
        object with the binned kpar axis.

    Raises
    ------
    ValueError
        If `bins_kpar` is not an int or a valid array of bin edges/centres.

    Notes
    -----
    - If `interp_kpar` is True, the power spectrum and its variance (if present) are
      interpolated onto the new kpar bins.
    - If `interp_kpar` is False, the power spectrum and its variance are aggregated
      using the mean within each bin.
    - Cropping is applied after binning/interpolation.
    """

    def transform_ps(ps: CylindricalPS):
        if bins_kpar is None:
            if log_kpar:
                final_bins_kpar = (
                    np.logspace(
                        np.log10(ps.kpar.value[0]),
                        np.log10(ps.kpar.value[-1]),
                        len(ps.kpar) // 2 + 1,
                    )
                    * ps.kpar.unit
                )
            else:
                final_bins_kpar = np.linspace(
                    ps.kpar[0], ps.kpar[-1], len(ps.kpar) // 2 + 1
                )
        elif isinstance(bins_kpar, int):
            if log_kpar:
                final_bins_kpar = (
                    np.logspace(
                        np.log10(ps.kpar.value[0]),
                        np.log10(ps.kpar.value[-1]),
                        bins_kpar,
                    )
                    * ps.kpar.unit
                )
            else:
                final_bins_kpar = np.linspace(ps.kpar[0], ps.kpar[-1], bins_kpar)
        else:
            if not isinstance(bins_kpar, np.ndarray):
                raise ValueError("bins_kpar must be an array of bin edges or centres.")
            final_bins_kpar = bins_kpar
        if interp_kpar:
            mask = np.isnan(np.nanmean(ps.ps, axis=-1))
            interp_fnc = RegularGridInterpolator(
                (ps.kperp.value[~mask], ps.kpar.value),
                ps.ps[~mask].squeeze(),
                bounds_error=False,
                fill_value=np.nan,
            )
            kperp_grid, kpar_grid = np.meshgrid(
                ps.kperp, final_bins_kpar, indexing="ij", sparse=True
            )
            final_ps = interp_fnc((kperp_grid, kpar_grid)) * ps.ps.unit
            if ps.variance is not None:
                interp_fnc = RegularGridInterpolator(
                    (ps.kperp[~mask].value, ps.kpar.value),
                    ps.variance[~mask].squeeze(),
                    bounds_error=False,
                    fill_value=np.nan,
                )
                kperp_grid, kpar_grid = np.meshgrid(
                    ps.kperp, final_bins_kpar, indexing="ij", sparse=True
                )
                final_var = interp_fnc((kperp_grid, kpar_grid)) * ps.variance.unit

            idxs = np.digitize(ps.kpar.value, final_bins_kpar.value) - 1
            final_nmodes = np.zeros(len(final_bins_kpar))
            for i in range(len(final_bins_kpar)):
                final_nmodes[i] = np.sum(idxs == i)

        else:
            final_ps = np.zeros((len(ps.kperp), len(final_bins_kpar) - 1))
            final_nmodes = np.zeros(len(final_bins_kpar) - 1)
            idxs = np.digitize(ps.kpar.value, final_bins_kpar.value) - 1
            if ps.variance is not None:
                final_var = np.zeros((len(ps.kperp), len(final_bins_kpar)))
            for i in range(len(final_bins_kpar) - 1):
                m = idxs == i
                final_ps[..., i] = np.nanmean(ps.ps.value[..., m], axis=-1)
                final_nmodes[i] = np.sum(m)
                if ps.variance is not None:
                    final_var[..., i] = np.nanmean(ps.variance.value[..., m], axis=-1)
            if log_kpar:
                final_bins_kpar = (
                    np.exp(
                        (
                            np.log(final_bins_kpar.value[1:])
                            + np.log(final_bins_kpar.value[:-1])
                        )
                        / 2
                    )
                    * ps.kpar.unit
                )
            else:
                final_bins_kpar = (final_bins_kpar[1:] + final_bins_kpar[:-1]) / 2
            final_ps = final_ps * ps.ps.unit
            if ps.variance is not None:
                final_var = final_var * ps.variance.unit
        if crop_kperp is not None:
            final_ps = final_ps[crop_kperp[0] : crop_kperp[1]]
            if ps.variance is not None:
                final_var = final_var[crop_kperp[0] : crop_kperp[1]]
        if crop_kpar is not None:
            final_ps = final_ps[:, crop_kpar[0] : crop_kpar[1]]
            if ps.variance is not None:
                final_var = final_var[:, crop_kpar[0] : crop_kpar[1]]
        if ps.n_modes.ndim != 1 and np.all(ps.n_modes[:, :1] == ps.n_modes):
            raise ValueError("Must provide only kperp n_modes in a 1D array.")
        final_kperp_modes = (
            ps.n_modes[crop_kperp[0] : crop_kperp[1]]
            if crop_kperp is not None
            else ps.n_modes
        )
        final_kpar_modes = (
            final_nmodes[crop_kpar[0] : crop_kpar[1]]
            if crop_kpar is not None
            else final_nmodes
        )
        kpar_grid, kperp_grid = np.meshgrid(
            final_kperp_modes, final_kpar_modes, indexing="ij"
        )

        final_nmodes = np.sqrt(kperp_grid**2 + kpar_grid**2)

        return CylindricalPS(
            ps=final_ps,
            kperp=ps.kperp
            if crop_kperp is None
            else ps.kperp[crop_kperp[0] : crop_kperp[1]],
            kpar=final_bins_kpar
            if crop_kpar is None
            else final_bins_kpar[crop_kpar[0] : crop_kpar[1]],
            redshift=ps.redshift,
            n_modes=final_nmodes,
            variance=final_var if ps.variance is not None else None,
            is_deltasq=ps.is_deltasq,
        )

    return transform_ps


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
