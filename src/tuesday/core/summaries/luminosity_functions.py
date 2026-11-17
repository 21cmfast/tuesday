"""Number densities of objects in bins of a feature, and UV luminosity functions.

:func:`bin_number_density` bins one feature of a :class:`~tuesday.core.catalogs.Catalog`
(for example the halo mass or the UV magnitude) and divides the counts by the bin
width and the volume of the catalogue, which gives e.g. the halo mass function or the
luminosity function. :func:`sfr_to_uv_magnitude` converts star formation rates to
absolute UV magnitudes, and :func:`uv_luminosity_function` bins the UV magnitudes of
the objects of a catalogue.

These functions do not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.summaries.luminosity_functions` for a wrapper that
takes 21cmFAST halo catalogues.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from astropy import units as un

from ..catalogs import Catalog

#: Unit of the conversion factor ``K_UV = SFR / L_UV``.
KUV_UNIT = (un.Msun / un.yr) / (un.erg / un.s / un.Hz)

#: Conversion factor ``K_UV = SFR / L_UV`` used by 21cmFAST for its UV luminosity
#: functions, from Sun & Furlanetto (2016), MNRAS, 460, 417.
KUV_SUN16 = 1.15e-28 * KUV_UNIT

#: Default bin edges of :func:`uv_luminosity_function`: 0.5 mag wide, from
#: ``M_UV = -24`` to ``-12``.
UV_MAGNITUDE_EDGES = np.linspace(-24.0, -12.0, 25) * un.mag
UV_MAGNITUDE_EDGES.flags.writeable = False


@dataclass(frozen=True)
class BinnedNumberDensity:
    """Number density of objects in bins of one of their features.

    Attributes
    ----------
    feature : str
        Name of the binned feature.
    bin_edges : un.Quantity
        Edges of the bins, in the unit of the feature.
    counts : np.ndarray
        Number of objects in each bin.
    number_density : un.Quantity
        Number of objects per unit volume and per unit of the feature in each bin,
        ``counts / (bin width * volume)``, e.g. in ``1 / (mag Mpc3)`` for magnitudes.
    poisson_error : un.Quantity
        Poisson error of ``number_density``, ``sqrt(counts) / (bin width * volume)``.
    """

    feature: str
    bin_edges: un.Quantity
    counts: np.ndarray
    number_density: un.Quantity
    poisson_error: un.Quantity

    @property
    def bin_centres(self) -> un.Quantity:
        """Centres of the bins."""
        return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    @property
    def bin_widths(self) -> un.Quantity:
        """Widths of the bins."""
        return np.diff(self.bin_edges)


def bin_number_density(
    catalog: Catalog,
    feature: str,
    bins: int | un.Quantity,
) -> BinnedNumberDensity:
    """Bin one feature of the objects of a catalogue into a number density.

    The number density in bin ``i`` is ``N_i / (width_i * volume)``, where ``N_i`` is
    the number of objects whose feature falls in the bin and ``volume`` is the volume
    of the catalogue, and its Poisson error is ``sqrt(N_i) / (width_i * volume)``.
    Binning the absolute UV magnitudes of the galaxies in a volume gives the UV
    luminosity function, and binning the natural logarithm of the halo masses gives
    the halo mass function ``dn/dlnM``. A feature that is derived from others (e.g.
    the UV magnitude from the star formation rate) is added to the catalogue with
    :meth:`~tuesday.core.catalogs.Catalog.with_features` first.

    Parameters
    ----------
    catalog : Catalog
        The catalogue. Its ``volume`` must be set.
    feature : str
        Name of the feature to bin, which must have one value per object. Non-finite
        values (such as the infinite UV magnitude of a galaxy without star formation)
        are not counted.
    bins : int or un.Quantity
        Edges of the bins, increasing, in a unit compatible with that of the feature
        (bin widths may differ), or the number of bins of equal width between the
        smallest and largest finite value (between 0 and 1 if there is none, with
        zero counts). The last bin includes its upper edge, as in
        :func:`numpy.histogram`.

    Returns
    -------
    BinnedNumberDensity
        The counts, number density and Poisson error in each bin.

    Raises
    ------
    ValueError
        If the catalogue has no volume, or the feature has more than one value per
        object.
    KeyError
        If the catalogue has no feature called ``feature``.
    """
    if catalog.volume is None:
        raise ValueError(
            "Number densities need the volume of the catalogue, which is not set."
        )
    values = catalog[feature]
    if values.ndim != 1:
        raise ValueError(
            f"Feature '{feature}' has shape {values.shape}; only features with one "
            "value per object can be binned."
        )
    counts, edges = np.histogram(values[np.isfinite(values)], bins=bins)
    norm = np.diff(edges) * catalog.volume
    return BinnedNumberDensity(
        feature=feature,
        bin_edges=edges,
        counts=counts,
        number_density=counts / norm,
        poisson_error=np.sqrt(counts) / norm,
    )


def uv_luminosity_to_magnitude(
    luminosity: un.Quantity[un.erg / un.s / un.Hz],
) -> un.Quantity[un.mag]:
    """Convert a UV luminosity per unit frequency to an absolute AB magnitude.

    Uses ``M_UV = 51.63 - 2.5 log10(L_UV / [erg s^-1 Hz^-1])``, as 21cmFAST does.

    Parameters
    ----------
    luminosity : un.Quantity[un.erg / un.s / un.Hz]
        UV luminosity per unit frequency.

    Returns
    -------
    un.Quantity[un.mag]
        Absolute AB magnitude. It is infinite where the luminosity is zero.
    """
    with np.errstate(divide="ignore"):
        magnitude = 51.63 - 2.5 * np.log10(luminosity.to_value(un.erg / un.s / un.Hz))
    return magnitude * un.mag


def sfr_to_uv_magnitude(
    sfr: un.Quantity[un.Msun / un.yr],
    kuv: un.Quantity | Callable = KUV_SUN16,
    stellar_mass: un.Quantity[un.Msun] | None = None,
    redshift: float | None = None,
) -> un.Quantity[un.mag]:
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
        float64 arrays (``stellar_mass`` is None if not given) and with ``redshift``,
        and returns a Quantity or a plain array in ``Msun yr^-1 / (erg s^-1 Hz^-1)``.
    stellar_mass : un.Quantity[un.Msun], optional
        Stellar mass of each galaxy. Only passed on to ``kuv`` if it is a function.
    redshift : float, optional
        Redshift of the galaxies. Only passed on to ``kuv`` if it is a function.

    Returns
    -------
    un.Quantity[un.mag]
        Absolute UV (AB) magnitude of each galaxy. It is infinite for galaxies without
        star formation.
    """
    sfr = np.asarray(sfr.astype(np.float64).to_value(un.Msun / un.yr))
    if callable(kuv):
        if stellar_mass is not None:
            stellar_mass = np.asarray(stellar_mass.astype(np.float64).to_value(un.Msun))
        kuv = kuv(sfr, stellar_mass, redshift)
    if isinstance(kuv, un.Quantity):
        kuv = kuv.to_value(KUV_UNIT)
    luminosity = sfr / np.asarray(kuv, dtype=np.float64)
    return uv_luminosity_to_magnitude(luminosity * un.erg / un.s / un.Hz)


def uv_luminosity_function(
    catalog: Catalog,
    bins: int | un.Quantity | None = None,
    kuv: un.Quantity | Callable = KUV_SUN16,
    sfr: str = "sfr",
    stellar_mass: str | None = "stellar_mass",
) -> BinnedNumberDensity:
    """Calculate the UV luminosity function of the galaxies of a catalogue.

    The absolute UV magnitude of each galaxy is computed from its star formation rate
    with :func:`sfr_to_uv_magnitude`, added to the catalogue as the feature
    ``"uv_magnitude"``, and binned with :func:`bin_number_density`. Galaxies without
    star formation are not counted.

    Parameters
    ----------
    catalog : Catalog
        The galaxies. Its ``volume`` must be set, and its ``redshift`` is passed on to
        ``kuv`` if it is a function.
    bins : int or un.Quantity, optional
        Edges of the magnitude bins (in ``un.mag``), or their number (see
        :func:`bin_number_density`). By default :data:`UV_MAGNITUDE_EDGES`, 0.5 mag
        wide from ``M_UV = -24`` to ``-12``.
    kuv : un.Quantity or callable, optional
        The conversion factor ``K_UV = SFR / L_UV``; see :func:`sfr_to_uv_magnitude`.
    sfr : str, optional
        Name of the feature with the star formation rate of each galaxy.
    stellar_mass : str or None, optional
        Name of the feature with the stellar mass of each galaxy, passed on to
        ``kuv`` if it is a function. None passes no stellar mass. It is not used if
        ``kuv`` is a Quantity.

    Returns
    -------
    BinnedNumberDensity
        The luminosity function: ``bin_centres`` are the magnitudes and
        ``number_density`` is the number of galaxies per unit volume and per
        magnitude.
    """
    magnitudes = sfr_to_uv_magnitude(
        catalog[sfr],
        kuv=kuv,
        stellar_mass=(
            catalog[stellar_mass] if callable(kuv) and stellar_mass else None
        ),
        redshift=catalog.redshift,
    )
    return bin_number_density(
        catalog.with_features(uv_magnitude=magnitudes),
        "uv_magnitude",
        UV_MAGNITUDE_EDGES if bins is None else bins,
    )
