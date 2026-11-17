"""Number densities of halos in bins of a property, and UV luminosity functions.

:func:`bin_number_density` bins any property of a set of halos or galaxies (for
example the halo mass or the UV magnitude) and divides the counts by the bin width
and the volume, which gives e.g. the halo mass function or the luminosity function.
:func:`sfr_to_uv_magnitude` converts star formation rates to absolute UV magnitudes,
and :func:`uv_luminosity_function` combines the two.

These functions work on plain arrays and do not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.summaries.luminosity_functions` for a wrapper that
takes 21cmFAST halo catalogues.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from astropy import units as un

from ..units import without_littleh

#: Unit of the conversion factor ``K_UV = SFR / L_UV``.
KUV_UNIT = (un.Msun / un.yr) / (un.erg / un.s / un.Hz)

#: Conversion factor ``K_UV = SFR / L_UV`` used by 21cmFAST for its UV luminosity
#: functions, from Sun & Furlanetto (2016), MNRAS, 460, 417.
KUV_SUN16 = 1.15e-28 * KUV_UNIT

#: Default bin edges of :func:`uv_luminosity_function`: 0.5 mag wide, from
#: ``M_UV = -24`` to ``-12``.
UV_MAGNITUDE_EDGES = np.linspace(-24.0, -12.0, 25)
UV_MAGNITUDE_EDGES.flags.writeable = False


@dataclass(frozen=True)
class BinnedNumberDensity:
    """Number density of objects in bins of one of their properties.

    Parameters
    ----------
    bin_edges : np.ndarray or un.Quantity
        Edges of the bins, in the unit of the binned property.
    counts : np.ndarray
        Number of objects in each bin.
    number_density : un.Quantity
        Number of objects per unit volume and per unit of the binned property in each
        bin, ``counts / (bin width * volume)``. For a property without a unit (e.g.
        a magnitude or the logarithm of a mass) its unit is the inverse of the volume
        unit, and it is per unit of the property (e.g. per magnitude).
    poisson_error : un.Quantity
        Poisson error of ``number_density``, ``sqrt(counts) / (bin width * volume)``.
    """

    bin_edges: np.ndarray | un.Quantity
    counts: np.ndarray
    number_density: un.Quantity
    poisson_error: un.Quantity

    @property
    def bin_centres(self) -> np.ndarray | un.Quantity:
        """Centres of the bins."""
        return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    @property
    def bin_widths(self) -> np.ndarray | un.Quantity:
        """Widths of the bins."""
        return np.diff(self.bin_edges)


def bin_number_density(
    values: np.ndarray | un.Quantity,
    bins: int | np.ndarray | un.Quantity,
    volume: un.Quantity,
) -> BinnedNumberDensity:
    """Bin a property of a set of objects into a number density.

    The number density in bin ``i`` is ``N_i / (width_i * volume)``, where ``N_i`` is
    the number of objects whose property falls in the bin, and its Poisson error is
    ``sqrt(N_i) / (width_i * volume)``. Binning the absolute UV magnitudes of the
    galaxies in a volume gives the UV luminosity function (per magnitude), and
    binning the natural logarithm of the halo masses gives the halo mass function
    ``dn/dlnM``.

    Parameters
    ----------
    values : np.ndarray or un.Quantity
        The property of each object. Non-finite values (such as the infinite UV
        magnitude of a galaxy without star formation) are not counted.
    bins : int or np.ndarray or un.Quantity
        Edges of the bins, increasing (in the unit of ``values`` if it has one; bin
        widths may differ), or the number of bins of equal width between the smallest
        and largest finite value. The last bin includes its upper edge, as in
        :func:`numpy.histogram`.
    volume : un.Quantity
        The volume that contains the objects, e.g. that of the simulation box.

    Returns
    -------
    BinnedNumberDensity
        The counts, number density and Poisson error in each bin.

    Raises
    ------
    ValueError
        If ``volume`` is not a volume.
    """
    if without_littleh(volume.unit).physical_type != "volume":
        raise ValueError(f"Expected a volume, but got {volume.unit.physical_type}.")
    if not isinstance(values, un.Quantity):
        values = np.asarray(values)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    norm = np.diff(edges) * volume
    return BinnedNumberDensity(
        bin_edges=edges,
        counts=counts,
        number_density=counts / norm,
        poisson_error=np.sqrt(counts) / norm,
    )


def uv_luminosity_to_magnitude(
    luminosity: un.Quantity[un.erg / un.s / un.Hz],
) -> np.ndarray:
    """Convert a UV luminosity per unit frequency to an absolute AB magnitude.

    Uses ``M_UV = 51.63 - 2.5 log10(L_UV / [erg s^-1 Hz^-1])``, as 21cmFAST does.

    Parameters
    ----------
    luminosity : un.Quantity[un.erg / un.s / un.Hz]
        UV luminosity per unit frequency.

    Returns
    -------
    np.ndarray
        Absolute AB magnitude. It is infinite where the luminosity is zero.
    """
    with np.errstate(divide="ignore"):
        return 51.63 - 2.5 * np.log10(luminosity.to_value(un.erg / un.s / un.Hz))


def sfr_to_uv_magnitude(
    sfr: un.Quantity[un.Msun / un.yr],
    kuv: un.Quantity | Callable = KUV_SUN16,
    stellar_mass: un.Quantity[un.Msun] | None = None,
    redshift: float | None = None,
) -> np.ndarray:
    """Calculate the absolute UV magnitude of galaxies from their star formation rate.

    The UV luminosity per unit frequency is ``L_UV = SFR / K_UV``, which is converted
    to a magnitude with :func:`uv_luminosity_to_magnitude`.

    Parameters
    ----------
    sfr : un.Quantity[un.Msun / un.yr]
        Star formation rate of each galaxy.
    kuv : un.Quantity or callable, optional
        The conversion factor ``K_UV = SFR / L_UV``, in units of
        ``Msun yr^-1 / (erg s^-1 Hz^-1)``. Either one value for all galaxies (by
        default :data:`KUV_SUN16`, the value 21cmFAST uses), or a function
        ``kuv(sfr, stellar_mass, redshift)`` that returns the value of each galaxy,
        e.g. an interpolation of a table computed with a stellar population model. It
        is called with ``sfr`` in Msun/yr and ``stellar_mass`` in Msun as plain
        arrays (``stellar_mass`` is None if not given) and with ``redshift``, and
        returns a Quantity or a plain array in ``Msun yr^-1 / (erg s^-1 Hz^-1)``.
    stellar_mass : un.Quantity[un.Msun], optional
        Stellar mass of each galaxy. Only passed on to ``kuv`` if it is a function.
    redshift : float, optional
        Redshift of the galaxies. Only passed on to ``kuv`` if it is a function.

    Returns
    -------
    np.ndarray
        Absolute UV (AB) magnitude of each galaxy. It is infinite for galaxies without
        star formation.
    """
    sfr = np.asarray(sfr.to_value(un.Msun / un.yr), dtype=np.float64)
    if callable(kuv):
        if stellar_mass is not None:
            stellar_mass = np.asarray(stellar_mass.to_value(un.Msun), dtype=np.float64)
        kuv = kuv(sfr, stellar_mass, redshift)
    if isinstance(kuv, un.Quantity):
        kuv = kuv.to_value(KUV_UNIT)
    luminosity = sfr / np.asarray(kuv, dtype=np.float64)
    return uv_luminosity_to_magnitude(luminosity * un.erg / un.s / un.Hz)


def uv_luminosity_function(
    sfr: un.Quantity[un.Msun / un.yr],
    volume: un.Quantity,
    bins: int | np.ndarray | None = None,
    kuv: un.Quantity | Callable = KUV_SUN16,
    stellar_mass: un.Quantity[un.Msun] | None = None,
    redshift: float | None = None,
) -> BinnedNumberDensity:
    """Calculate the UV luminosity function of the galaxies in a volume.

    The absolute UV magnitudes are computed with :func:`sfr_to_uv_magnitude` and
    binned with :func:`bin_number_density`. Galaxies without star formation are not
    counted.

    Parameters
    ----------
    sfr : un.Quantity[un.Msun / un.yr]
        Star formation rate of each galaxy.
    volume : un.Quantity
        The volume that contains the galaxies, e.g. that of the simulation box.
    bins : int or np.ndarray, optional
        Edges of the magnitude bins, or their number (see
        :func:`bin_number_density`). By default :data:`UV_MAGNITUDE_EDGES`, 0.5 mag
        wide from ``M_UV = -24`` to ``-12``.
    kuv : un.Quantity or callable, optional
        The conversion factor ``K_UV = SFR / L_UV``; see :func:`sfr_to_uv_magnitude`.
    stellar_mass : un.Quantity[un.Msun], optional
        Stellar mass of each galaxy, passed on to ``kuv`` if it is a function.
    redshift : float, optional
        Redshift of the galaxies, passed on to ``kuv`` if it is a function.

    Returns
    -------
    BinnedNumberDensity
        The luminosity function: ``bin_centres`` are the magnitudes and
        ``number_density`` is the number of galaxies per unit volume and per
        magnitude.
    """
    magnitudes = sfr_to_uv_magnitude(
        sfr, kuv=kuv, stellar_mass=stellar_mass, redshift=redshift
    )
    return bin_number_density(
        magnitudes, UV_MAGNITUDE_EDGES if bins is None else bins, volume
    )
