"""CO interloper foregrounds of [CII] line-intensity maps.

A [CII] map at observed frequency ``nu_obs`` also contains the CO rotational lines
emitted at the lower redshifts ``z_co = nu_co / nu_obs - 1``. The CO emitters are
drawn from the CO luminosity functions of Popping et al. (2016), MNRAS, 461, 93
(arXiv:1602.02761), and trace the matter density with the effective bias of
arXiv:2510.07596 (Table A.1).

In each [CII] slice and for each CO line, map pixel ``(j, k)`` holds the emitters
of density cell ``(j, k)``, with no projection of the cells onto the sky:

- The emitters fill ``cell_size**2`` times the comoving depth of the channel at the
  CO redshift, ``depth = c dz_co / H(z_co)``, with ``dz_co`` the CO redshift range
  of the slice. For CO J=3-2 to 6-5 in front of [CII] at z = 6-7, this is 1.6-1.8
  cells.
- The density of a pixel is the average of the density slices whose centres lie
  within ``chi_co -/+ depth / 2`` of the CO distance.
- The luminosity is spread over the solid angle ``(cell_size / chi_co)**2`` and the
  observed frequency width ``dnu`` of the slice, so that the mean brightness is
  ``eps / (4 pi H(z_co) (1 + z_co))`` for a luminosity density ``eps``.

The foreground can include the clustering of the emitters and their shot noise: the
weights ``clip(1 + b delta, 0)`` are renormalised to unit mean in each slice, and
the number of emitters in each pixel is drawn from a Poisson distribution. It can
also include the clustering only: each pixel then holds its expected brightness,
with the weights ``1 + b delta``, neither clipped nor renormalised.

The density lightcone must be an evolved (Eulerian) field, such as the 2LPT density
of 21cmFAST: the effective bias relates the emitters at their final positions to the
matter at the same time. With 2LPT cells of 1.5 Mpc at z = 0-1.9, ``1 + b delta`` is
negative in 10-40% of the cells, mostly voids.

The angular scale is approximate: at the CO distance a pixel spans
``cell_size * chi_co / chi_cii`` across, 1.7-7 times less than the density cell
that sets its weight.

These functions work on plain arrays and do not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.postprocessing.co_foregrounds` for building the
density lightcone with 21cmFAST and adding the foreground to a [CII] lightcone.
"""

import csv
from functools import cache
from pathlib import Path

import attrs
import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import FLRW

from ..cosmology import comoving_distance_to_redshift
from .line_intensities import NU_CII_REST

DATA_FILE = Path(__file__).parent / "data" / "co_schechter_popping16.csv"

#: CO rotational transitions with a luminosity function in Popping et al. (2016).
CO_TRANSITIONS = tuple(f"CO J={j + 1}-{j}" for j in range(6))

#: Default grid of log10(L / L*) on which the luminosity functions are sampled.
LOG_L_RATIO = np.linspace(-3, 1, 100)

#: Unit of the CO luminosities of Popping et al. (2016).
CO_LUMINOSITY_UNIT = un.Jy * un.km / un.s * un.Mpc**2

# Effective bias of CO emitters, arXiv:2510.07596, Table A.1 (analytic b_eff).
_BIAS_REDSHIFTS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
_BIAS_VALUES = np.array([1.19, 1.49, 1.88, 2.13, 2.63, 2.81])


@cache
def _schechter_table() -> dict[str, dict[str, np.ndarray]]:
    """Read the Schechter parameters of each transition, sorted by redshift."""
    with DATA_FILE.open() as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    table = {}
    for transition in CO_TRANSITIONS:
        selected = sorted(
            (r for r in rows if r["transition"] == transition),
            key=lambda r: float(r["redshift"]),
        )
        table[transition] = {
            key: np.array([float(r[key]) for r in selected])
            for key in (
                "redshift",
                "observed_frequency",
                "alpha",
                "log_L_star",
                "log_phi_star",
            )
        }
    return table


def _params(transition: str) -> dict[str, np.ndarray]:
    """Return the Schechter parameters of a transition."""
    try:
        return _schechter_table()[transition]
    except KeyError:
        raise ValueError(
            f"Unknown CO transition '{transition}'. Available transitions are "
            f"{list(CO_TRANSITIONS)}."
        ) from None


def co_rest_frequency(transition: str) -> un.Quantity[un.GHz]:
    """Return the rest-frame frequency of a CO transition.

    Parameters
    ----------
    transition : str
        One of :data:`CO_TRANSITIONS`, e.g. ``"CO J=4-3"``.

    Returns
    -------
    un.Quantity[un.GHz]
        Rest-frame frequency of the line.

    Raises
    ------
    ValueError
        If ``transition`` is not one of :data:`CO_TRANSITIONS`.
    """
    return _params(transition)["observed_frequency"][0] * un.GHz


def co_luminosity_function(
    transition: str, redshift: float, log_l_ratio: np.ndarray
) -> tuple[np.ndarray, float]:
    """Evaluate the CO luminosity function of Popping et al. (2016).

    The Schechter function ``phi(L) = ln(10) phi* (L/L*)**(alpha + 1) exp(-L/L*)``
    per dex of luminosity is fitted at redshifts 0, 1, 2, 4 and 6. Between them,
    ``phi`` at fixed ``L/L*`` and ``log10 L*`` are interpolated linearly in redshift;
    above redshift 6, the fit at redshift 6 is used.

    Parameters
    ----------
    transition : str
        One of :data:`CO_TRANSITIONS`.
    redshift : float
        Redshift of the emitters.
    log_l_ratio : np.ndarray
        Values of ``log10(L / L*)`` at which to evaluate the luminosity function.

    Returns
    -------
    phi : np.ndarray
        Comoving number density per dex of luminosity, in Mpc^-3, at each
        ``log_l_ratio``.
    log_l_star : float
        ``log10 L*`` at this redshift, with ``L*`` in Jy km/s Mpc^2.

    Raises
    ------
    ValueError
        If ``transition`` is unknown or ``redshift`` is negative.
    """
    if redshift < 0:
        raise ValueError(f"The redshift must not be negative; got {redshift}.")
    p = _params(transition)
    z_nodes = p["redshift"]
    z = min(redshift, z_nodes[-1])
    # Bracketing nodes, with z on a node giving that node (f = 0, or f = 1 at the end).
    i = min(np.searchsorted(z_nodes, z, side="right") - 1, len(z_nodes) - 2)
    f = (z - z_nodes[i]) / (z_nodes[i + 1] - z_nodes[i])

    x = 10 ** np.asarray(log_l_ratio)

    def phi_at(j: int) -> np.ndarray:
        return (
            np.log(10)
            * 10 ** p["log_phi_star"][j]
            * x ** (p["alpha"][j] + 1)
            * np.exp(-x)
        )

    if f == 0:
        return phi_at(i), float(p["log_L_star"][i])
    if f == 1:
        return phi_at(i + 1), float(p["log_L_star"][i + 1])
    phi = phi_at(i) + f * (phi_at(i + 1) - phi_at(i))
    log_l_star = p["log_L_star"][i] + f * (p["log_L_star"][i + 1] - p["log_L_star"][i])
    return phi, float(log_l_star)


def co_bias(redshift: float) -> float:
    """Return the effective bias of CO emitters.

    Analytic ``b_eff`` of arXiv:2510.07596, Table A.1, tabulated at redshifts
    0.5-3, interpolated linearly and held constant outside this range.

    Parameters
    ----------
    redshift : float
        Redshift of the emitters.

    Returns
    -------
    float
        Effective bias.
    """
    return float(np.interp(redshift, _BIAS_REDSHIFTS, _BIAS_VALUES))


def bias_weights(density: np.ndarray, bias: float) -> np.ndarray:
    """Weight the mean number of emitters in each cell by the local density.

    The weight ``1 + bias * density`` is set to zero where it is negative, which
    happens where the density field has strong fluctuations. The weights are not
    renormalised here; :func:`co_foreground_brightness` renormalises them to unit
    mean in each slice.

    Parameters
    ----------
    density : np.ndarray
        Matter density contrast in each cell.
    bias : float
        Bias of the emitters.

    Returns
    -------
    np.ndarray
        Non-negative weights, with the shape of ``density``.
    """
    return np.clip(1 + bias * np.asarray(density), 0, None)


@attrs.define(frozen=True, kw_only=True)
class DensityLightcone:
    """A rectilinear lightcone of the matter density contrast.

    Parameters
    ----------
    density : np.ndarray
        Density contrast, with shape ``(n, n, n_slices)``; the last axis runs
        along the line of sight.
    distances : un.Quantity[un.Mpc]
        Comoving distance of each slice.
    cell_size : un.Quantity[un.Mpc]
        Comoving side length of a (cubic) cell.
    """

    density: np.ndarray = attrs.field(converter=np.asarray)
    distances: un.Quantity[un.Mpc] = attrs.field(converter=lambda d: d.to(un.Mpc))
    cell_size: un.Quantity[un.Mpc] = attrs.field(converter=lambda d: d.to(un.Mpc))

    @density.validator
    def _check_density(self, attribute: attrs.Attribute, value: np.ndarray) -> None:
        if value.ndim != 3 or value.shape[0] != value.shape[1]:
            raise ValueError(
                f"density must have shape (n, n, n_slices); got {value.shape}."
            )

    @distances.validator
    def _check_distances(self, attribute: attrs.Attribute, value: un.Quantity) -> None:
        if value.shape != (self.density.shape[-1],):
            raise ValueError(
                f"distances must have one entry per slice ({self.density.shape[-1]}); "
                f"got shape {value.shape}."
            )

    def nearest_slice(self, distance: un.Quantity[un.Mpc]) -> np.ndarray:
        """Return the density slice nearest to a comoving distance.

        Parameters
        ----------
        distance : un.Quantity[un.Mpc]
            Comoving distance.

        Returns
        -------
        np.ndarray
            The density of the nearest slice, with shape ``(n, n)``.

        Raises
        ------
        ValueError
            If the distance is beyond the lightcone.
        """
        half = self.cell_size / 2
        if not self.distances[0] - half <= distance <= self.distances[-1] + half:
            raise ValueError(
                f"The density lightcone covers {self.distances[0] - half:.1f} to "
                f"{self.distances[-1] + half:.1f}, but {distance:.1f} is needed."
            )
        return self.density[..., np.argmin(np.abs(self.distances - distance))]

    def depth_average(
        self, distance: un.Quantity[un.Mpc], depth: un.Quantity[un.Mpc]
    ) -> np.ndarray:
        """Average the density over a comoving depth along the line of sight.

        The slices whose centres lie in ``[distance - depth / 2, distance +
        depth / 2)`` are averaged; if there is none, the nearest slice is used.

        Parameters
        ----------
        distance : un.Quantity[un.Mpc]
            Comoving distance of the middle of the depth.
        depth : un.Quantity[un.Mpc]
            Comoving depth to average over.

        Returns
        -------
        np.ndarray
            The averaged density, with shape ``(n, n)``.

        Raises
        ------
        ValueError
            If the distance is beyond the lightcone.
        """
        nearest = self.nearest_slice(distance)
        inside = (self.distances >= distance - depth / 2) & (
            self.distances < distance + depth / 2
        )
        if not inside.any():
            return nearest
        return self.density[..., inside].mean(axis=-1)


def _co_slices(
    transition: str,
    distances: un.Quantity[un.Mpc],
    cosmo: FLRW,
    rest_frequency: un.Quantity[un.GHz],
) -> dict[str, np.ndarray | un.Quantity]:
    """Observed frequency, CO redshift, distances and channel width of each slice.

    The observed width of slice ``i`` is ``nu_co / (1 + z_co)**2 * dz_co``, and the
    comoving depth of its channel at the CO redshift is ``c dz_co / H(z_co)``, with
    ``dz_co`` the gradient of the CO redshift along the slices.
    """
    if len(distances) < 2:
        raise ValueError(
            "At least two slices are needed for the frequency width of a slice."
        )
    nu_obs = rest_frequency / (1 + comoving_distance_to_redshift(distances, cosmo))
    nu_co = co_rest_frequency(transition)
    z_co = (nu_co / nu_obs).to_value("") - 1
    dz_co = np.abs(np.gradient(z_co))
    delta_nu = nu_co / (1 + z_co) ** 2 * dz_co
    depth = (const.c / cosmo.H(np.clip(z_co, 0, None))).to(un.Mpc) * dz_co
    valid = z_co > 0
    z_co = np.where(valid, z_co, 0)
    return {
        "valid": valid,
        "nu_obs": nu_obs,
        "z_co": z_co,
        "chi_co": cosmo.comoving_distance(z_co),
        "d_lum": cosmo.luminosity_distance(z_co),
        "delta_nu": delta_nu,
        "depth": depth,
    }


def _to_brightness(
    slices: dict[str, np.ndarray | un.Quantity], i: int, cell_size: un.Quantity
) -> float:
    """Brightness in Jy/sr of a unit CO luminosity in a pixel of slice ``i``."""
    solid_angle = (cell_size / slices["chi_co"][i]) ** 2 * un.sr
    return (
        CO_LUMINOSITY_UNIT
        * slices["nu_obs"][i]
        / (
            4
            * np.pi
            * slices["d_lum"][i] ** 2
            * const.c
            * slices["delta_nu"][i]
            * solid_angle
        )
    ).to_value(un.Jy / un.sr)


def _check_log_l_ratio(log_l_ratio: np.ndarray) -> float:
    """Return the spacing of a uniform grid of log10(L / L*)."""
    log_l_ratio = np.asarray(log_l_ratio)
    step = np.diff(log_l_ratio)
    if log_l_ratio.ndim != 1 or step.size == 0 or not np.allclose(step, step[0]):
        raise ValueError("log_l_ratio must be a uniform grid of at least two values.")
    return float(step[0])


def mean_co_brightness(
    distances: un.Quantity[un.Mpc],
    cell_size: un.Quantity[un.Mpc],
    cosmo: FLRW,
    transitions: tuple[str, ...] = CO_TRANSITIONS,
    rest_frequency: un.Quantity[un.GHz] = NU_CII_REST,
    log_l_ratio: np.ndarray = LOG_L_RATIO,
) -> un.Quantity[un.Jy / un.sr]:
    """Calculate the mean CO surface brightness of each slice for a uniform density.

    For each CO line, the emitters of a pixel fill ``cell_size**2`` times the
    comoving depth of the channel at the CO redshift, ``depth = c dz_co / H(z_co)``,
    and have the luminosity ``eps(z_co) cell_size**2 depth``, with ``eps`` the CO
    luminosity density of the luminosity function. Their brightness is

    ``B = eps cell_size**2 depth nu_obs / (4 pi D_L**2 c dnu Omega)``,

    with ``Omega = (cell_size / chi_co)**2``, ``dnu`` the observed width of the
    slice and ``D_L`` the luminosity distance, so that the flux of a source of
    luminosity ``L`` is ``L / (4 pi D_L**2)`` in Jy km/s. In a flat cosmology this
    is ``eps / (4 pi H(z_co) (1 + z_co))``, whatever the cell size. This is the
    mean of :func:`co_foreground_brightness`; lines that would come from a
    negative redshift are skipped.

    Parameters
    ----------
    distances : un.Quantity[un.Mpc]
        Comoving distance of each slice of the [CII] lightcone (at least two).
    cell_size : un.Quantity[un.Mpc]
        Comoving side length of a cell of the [CII] lightcone.
    cosmo : FLRW
        Cosmology of the distance-redshift relation.
    transitions : tuple of str, optional
        The CO transitions to include; all of :data:`CO_TRANSITIONS` by default.
    rest_frequency : un.Quantity[un.GHz], optional
        Rest-frame frequency of the target line of the lightcone. Defaults to [CII].
    log_l_ratio : np.ndarray, optional
        Uniform grid of ``log10(L / L*)`` over which the luminosity function is
        integrated.

    Returns
    -------
    un.Quantity[un.Jy / un.sr]
        Mean CO surface brightness of each slice.

    Raises
    ------
    ValueError
        If a transition is unknown or there are fewer than two slices.
    """
    step = _check_log_l_ratio(log_l_ratio)
    area = cell_size.to_value(un.Mpc) ** 2
    total = np.zeros(len(distances))
    for transition in transitions:
        slices = _co_slices(transition, distances, cosmo, rest_frequency)
        for i in np.flatnonzero(slices["valid"]):
            phi, log_l_star = co_luminosity_function(
                transition, slices["z_co"][i], log_l_ratio
            )
            volume = area * slices["depth"][i].to_value(un.Mpc)
            luminosity = np.sum(phi * step * 10 ** (log_l_ratio + log_l_star)) * volume
            total[i] += luminosity * _to_brightness(slices, i, cell_size)
    return total * un.Jy / un.sr


def co_foreground_brightness(
    distances: un.Quantity[un.Mpc],
    n_pixels: int,
    cell_size: un.Quantity[un.Mpc],
    cosmo: FLRW,
    density_lightcone: DensityLightcone,
    transitions: tuple[str, ...] = CO_TRANSITIONS,
    rest_frequency: un.Quantity[un.GHz] = NU_CII_REST,
    log_l_ratio: np.ndarray = LOG_L_RATIO,
    shot_noise: bool = True,
    seed: int | np.random.Generator | None = None,
) -> un.Quantity[un.Jy / un.sr]:
    """Simulate the CO interloper foreground of a rectilinear [CII] lightcone.

    For each slice and CO line, the density is averaged over the comoving depth of
    the channel at the CO redshift (:meth:`DensityLightcone.depth_average`), and
    density cell ``(j, k)`` weights pixel ``(j, k)``. The mean number of emitters in
    each luminosity bin of a pixel is ``phi * dlog(L) * cell_size**2 * depth *
    weight``, and their luminosity is converted to a surface brightness as in
    :func:`mean_co_brightness`. Two models are available:

    - clustering and shot noise (``shot_noise=True``, the default): the weights
      ``clip(1 + b delta, 0)`` of :func:`bias_weights` are renormalised to unit mean
      over the slice, and the number of emitters is drawn from a Poisson
      distribution with this mean. The mean of the map is that of
      :func:`mean_co_brightness`.
    - clustering only (``shot_noise=False``): each pixel holds the expected
      brightness, with the linear weight ``1 + b delta``. It is neither clipped nor
      renormalised, so the brightness is negative where ``b delta < -1``, and its
      mean over a slice is :func:`mean_co_brightness` times ``1 + b <delta>``.

    See the module docstring for the approximations of this geometry.

    Parameters
    ----------
    distances : un.Quantity[un.Mpc]
        Comoving distance of each slice of the [CII] lightcone (at least two).
    n_pixels : int
        Number of pixels along each side of a slice.
    cell_size : un.Quantity[un.Mpc]
        Comoving side length of a cell of the [CII] lightcone.
    cosmo : FLRW
        Cosmology of the distance-redshift relation.
    density_lightcone : DensityLightcone
        Matter density contrast covering the comoving distances of the CO lines,
        with ``n_pixels`` cells of ``cell_size`` along each transverse side.
    transitions : tuple of str, optional
        The CO transitions to include; all of :data:`CO_TRANSITIONS` by default.
    rest_frequency : un.Quantity[un.GHz], optional
        Rest-frame frequency of the target line of the lightcone. Defaults to [CII].
    log_l_ratio : np.ndarray, optional
        Uniform grid of ``log10(L / L*)`` of the luminosity bins.
    shot_noise : bool, optional
        If True (the default), simulate clustering and shot noise: Poisson draws
        with the clipped and renormalised weights. If False, simulate the
        clustering only: the expected brightness with the unclipped weights.
    seed : int or np.random.Generator, optional
        Seed or generator for the Poisson draws, unused if ``shot_noise`` is False.
        If None, a random seed is used.

    Returns
    -------
    un.Quantity[un.Jy / un.sr]
        CO surface brightness, with shape ``(n_pixels, n_pixels, len(distances))``.

    Raises
    ------
    ValueError
        If a transition is unknown, there are fewer than two slices, the density
        lightcone does not have ``n_pixels`` cells of ``cell_size`` across, or it
        does not cover the comoving distances of the CO lines.
    """
    if density_lightcone.density.shape[0] != n_pixels:
        raise ValueError(
            f"The density lightcone has {density_lightcone.density.shape[0]} cells "
            f"across, but the map has {n_pixels} pixels; density cells are mapped "
            "to pixels one to one."
        )
    if not np.isclose(density_lightcone.cell_size, cell_size, rtol=1e-6):
        raise ValueError(
            f"The density lightcone has a cell size of {density_lightcone.cell_size}, "
            f"but the [CII] lightcone has {cell_size}; they must be equal."
        )
    step = _check_log_l_ratio(log_l_ratio)
    rng = np.random.default_rng(seed)
    area = cell_size.to_value(un.Mpc) ** 2

    brightness = np.zeros((n_pixels, n_pixels, len(distances)))
    for transition in transitions:
        slices = _co_slices(transition, distances, cosmo, rest_frequency)
        for i in np.flatnonzero(slices["valid"]):
            z_co = slices["z_co"][i]
            phi, log_l_star = co_luminosity_function(transition, z_co, log_l_ratio)
            depth = slices["depth"][i]
            mean_counts = phi * step * area * depth.to_value(un.Mpc)
            luminosities = 10 ** (log_l_ratio + log_l_star)
            delta = density_lightcone.depth_average(slices["chi_co"][i], depth)
            bias = co_bias(z_co)

            if shot_noise:
                weights = bias_weights(delta, bias)
                if weights.mean() > 0:
                    weights = weights / weights.mean()
                counts = rng.poisson(weights[..., None] * mean_counts)
                luminosity = counts @ luminosities
            else:
                weights = 1 + bias * np.asarray(delta, dtype=float)
                luminosity = weights * np.sum(mean_counts * luminosities)
            brightness[..., i] += luminosity * _to_brightness(slices, i, cell_size)
    return brightness * un.Jy / un.sr
