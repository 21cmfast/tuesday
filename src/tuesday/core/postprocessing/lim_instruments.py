"""Instrument and survey specifications for line-intensity-mapping (LIM) noise.

Each instrument is described by a :class:`LIMInstrument`. Ready-made instruments are
stored as YAML templates in the ``lim_instrument_templates`` directory, which give the
value, unit and literature reference of every parameter. Load one with
:meth:`LIMInstrument.from_template`, and override any parameter with keyword
arguments::

    ccatp = LIMInstrument.from_template("ccatp")
    deeper = LIMInstrument.from_template("ccatp", survey_time=4000 * un.hour)

A custom instrument can be loaded from a YAML file with the same layout using
:meth:`LIMInstrument.from_yaml`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import attrs
import numpy as np
import yaml
from astropy import units as un
from astropy.stats import gaussian_fwhm_to_sigma
from py21cmsense.beam import GaussianBeam

TEMPLATE_DIR = Path(__file__).parent / "lim_instrument_templates"

#: Units in which a sensitivity can be given: a noise-equivalent intensity (NEI), a
#: noise-equivalent flux density (NEFD) per beam, or an rms flux density per beam.
SENSITIVITY_UNITS = (
    un.Jy * un.s**0.5 / un.sr,
    un.Jy * un.s**0.5,
    un.Jy,
)


def _to_unit(unit: un.UnitBase) -> Callable[[un.Quantity | None], un.Quantity | None]:
    """Build an attrs converter that expresses a quantity in ``unit``."""

    def converter(value: un.Quantity | None) -> un.Quantity | None:
        return None if value is None else un.Quantity(value).to(unit)

    return converter


def _sensitivity_converter(value: un.Quantity | None) -> un.Quantity | None:
    """Check that a sensitivity is an NEI, an NEFD or an rms flux density."""
    if value is None:
        return None
    value = un.Quantity(value)
    if not any(value.unit.is_equivalent(unit) for unit in SENSITIVITY_UNITS):
        raise un.UnitConversionError(
            f"A sensitivity must have units equivalent to one of "
            f"{[str(unit) for unit in SENSITIVITY_UNITS]}, got '{value.unit}'."
        )
    return value


def beam_fwhm(
    nu_obs: un.Quantity[un.GHz], diameter: un.Quantity[un.m]
) -> un.Quantity[un.rad]:
    """Get the full width at half maximum of a diffraction-limited beam.

    Parameters
    ----------
    nu_obs : un.Quantity[un.GHz]
        Observed frequency.
    diameter : un.Quantity[un.m]
        Diameter of the primary dish.

    Returns
    -------
    un.Quantity[un.rad]
        Beam FWHM, taken as 1.22 lambda / D, from
        :meth:`py21cmsense.beam.GaussianBeam.first_null`.

    Notes
    -----
    For a uniformly illuminated circular aperture, 1.22 lambda / D is the first null
    of the Airy pattern, and the FWHM is about 1.03 lambda / D. The illumination of
    real dishes is tapered, which broadens the main beam, and [CII] intensity-mapping
    forecasts take 1.22 lambda / D as the FWHM (e.g. Chung et al. 2020,
    arXiv:1812.08135; CONCERTO Collaboration 2020, arXiv:2007.14246). This is close to
    the beam FWHMs quoted for CCAT-prime and measured for CONCERTO.
    """
    return GaussianBeam(dish_size=diameter).first_null(nu_obs).to(un.rad)


def beam_solid_angle(
    nu_obs: un.Quantity[un.GHz], diameter: un.Quantity[un.m]
) -> un.Quantity[un.sr]:
    """Get the solid angle of a Gaussian beam with FWHM from :func:`beam_fwhm`.

    Parameters
    ----------
    nu_obs : un.Quantity[un.GHz]
        Observed frequency.
    diameter : un.Quantity[un.m]
        Diameter of the primary dish.

    Returns
    -------
    un.Quantity[un.sr]
        Beam solid angle, 2 pi sigma^2.
    """
    sigma = beam_fwhm(nu_obs, diameter) * gaussian_fwhm_to_sigma
    return (2.0 * np.pi * sigma**2).to(un.sr)


@attrs.define(frozen=True, kw_only=True)
class LIMInstrument:
    """Specification of a line-intensity-mapping instrument and survey.

    Parameters
    ----------
    name : str
        Name of the instrument; used to label the noisy lightcone fields.
    spectral_resolution : float
        Spectral resolving power, R = nu / delta_nu.
    dish_diameter : un.Quantity[un.m]
        Diameter of the primary dish, which sets the diffraction-limited beam.
    min_frequency, max_frequency : un.Quantity[un.GHz]
        Observed frequency range of the instrument.
    n_feeds : int
        Number of independent spatial feeds (spectrometers) observing at once.
    survey_time : un.Quantity[un.hour]
        Total on-sky integration time of the survey.
    survey_area : un.Quantity[un.deg**2]
        Solid angle covered by the survey.
    sensitivity : un.Quantity | None
        Detector sensitivity: a noise-equivalent intensity (NEI, in
        ``Jy s^(1/2) / sr``), a noise-equivalent flux density per beam (NEFD, in
        ``Jy s^(1/2)``) or an rms flux density per beam (in ``Jy``). If None, it is
        computed with the AtLAST sensitivity calculator.
    low_frequency_sensitivity : un.Quantity | None
        Sensitivity used below ``break_frequency``, for instruments whose
        sensitivity differs between two sub-bands.
    break_frequency : un.Quantity[un.GHz] | None
        Frequency below which ``low_frequency_sensitivity`` is used.
    pixel_time : un.Quantity[un.hour] | None
        Integration time per beam-sized pixel. If None, it is computed from the
        survey parameters with :func:`~.add_noise_cii.calculate_tpix`.
    description : str
        Free-text description of the instrument and survey.
    references : dict[str, str]
        Literature reference for each parameter, keyed by parameter name.
    """

    name: str = attrs.field(converter=str)
    spectral_resolution: float = attrs.field(converter=float)
    dish_diameter: un.Quantity[un.m] = attrs.field(converter=_to_unit(un.m))
    min_frequency: un.Quantity[un.GHz] = attrs.field(converter=_to_unit(un.GHz))
    max_frequency: un.Quantity[un.GHz] = attrs.field(converter=_to_unit(un.GHz))
    n_feeds: int = attrs.field(converter=int)
    survey_time: un.Quantity[un.hour] = attrs.field(converter=_to_unit(un.hour))
    survey_area: un.Quantity[un.deg**2] = attrs.field(converter=_to_unit(un.deg**2))
    sensitivity: un.Quantity | None = attrs.field(
        default=None, converter=_sensitivity_converter
    )
    low_frequency_sensitivity: un.Quantity | None = attrs.field(
        default=None, converter=_sensitivity_converter
    )
    break_frequency: un.Quantity[un.GHz] | None = attrs.field(
        default=None, converter=_to_unit(un.GHz)
    )
    pixel_time: un.Quantity[un.hour] | None = attrs.field(
        default=None, converter=_to_unit(un.hour)
    )
    description: str = attrs.field(default="", converter=str)
    references: dict[str, str] = attrs.field(factory=dict)

    @max_frequency.validator
    def _check_frequency_range(
        self, attribute: attrs.Attribute, value: un.Quantity
    ) -> None:
        if value <= self.min_frequency:
            raise ValueError(
                f"min_frequency ({self.min_frequency}) must be below "
                f"max_frequency ({value})."
            )

    @classmethod
    def available_templates(cls) -> list[str]:
        """List the names of the built-in instrument templates.

        Returns
        -------
        list of str
            Template names accepted by :meth:`from_template`.
        """
        return sorted(path.stem for path in TEMPLATE_DIR.glob("*.yaml"))

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> Self:
        """Load an instrument from a YAML file.

        The file has a ``name``, an optional ``description`` and a ``parameters``
        mapping. Each parameter is a mapping with a ``value`` and optional ``unit``
        (any string understood by :class:`astropy.units.Unit`) and ``reference``.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file.
        **kwargs
            Parameters that override those in the file.

        Returns
        -------
        LIMInstrument
            The instrument.
        """
        with Path(path).open() as fl:
            data = yaml.safe_load(fl)

        params = {"name": data["name"], "description": data.get("description", "")}
        references = {}
        for key, spec in data["parameters"].items():
            value = spec["value"]
            if value is not None and "unit" in spec:
                value = value * un.Unit(spec["unit"])
            params[key] = value
            if spec.get("reference"):
                references[key] = spec["reference"]
        params["references"] = references

        return cls(**(params | kwargs))

    @classmethod
    def from_template(cls, name: str, **kwargs: Any) -> Self:
        """Load a built-in instrument template.

        Parameters
        ----------
        name : str
            Name of the template; see :meth:`available_templates`.
        **kwargs
            Parameters that override those of the template.

        Returns
        -------
        LIMInstrument
            The instrument.

        Raises
        ------
        ValueError
            If ``name`` is not a built-in template.
        """
        available = cls.available_templates()
        if name not in available:
            raise ValueError(
                f"Input a correct instrument template. Got '{name}', available "
                f"templates are {available}."
            )
        return cls.from_yaml(TEMPLATE_DIR / f"{name}.yaml", **kwargs)

    def sensitivity_at(self, nu_obs: un.Quantity[un.GHz]) -> un.Quantity | None:
        """Get the sensitivity at an observed frequency.

        Parameters
        ----------
        nu_obs : un.Quantity[un.GHz]
            Observed frequency.

        Returns
        -------
        un.Quantity or None
            The sensitivity, or None if it is not tabulated.
        """
        if self.break_frequency is not None and nu_obs < self.break_frequency:
            return self.low_frequency_sensitivity
        return self.sensitivity

    def beam_fwhm(self, nu_obs: un.Quantity[un.GHz]) -> un.Quantity[un.rad]:
        """Get the FWHM of the instrument beam; see :func:`beam_fwhm`.

        Parameters
        ----------
        nu_obs : un.Quantity[un.GHz]
            Observed frequency.

        Returns
        -------
        un.Quantity[un.rad]
            Beam FWHM.
        """
        return beam_fwhm(nu_obs, self.dish_diameter)

    def beam_solid_angle(self, nu_obs: un.Quantity[un.GHz]) -> un.Quantity[un.sr]:
        """Get the solid angle of the instrument beam; see :func:`beam_solid_angle`.

        Parameters
        ----------
        nu_obs : un.Quantity[un.GHz]
            Observed frequency.

        Returns
        -------
        un.Quantity[un.sr]
            Beam solid angle.
        """
        return beam_solid_angle(nu_obs, self.dish_diameter)
