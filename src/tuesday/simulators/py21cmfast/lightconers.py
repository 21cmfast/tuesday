"""Lightconers for 21cmFAST.

The lightconers themselves are in :mod:`tuesday.core.lightcones`: they do not depend
on 21cmFAST, and can be passed directly to :func:`py21cmfast.run_lightcone` (they
follow :class:`py21cmfast.lightconers.LightconerProtocol`) and to
:func:`tuesday.simulators.py21cmfast.lightcones.construct_lightcone_from_cache`.

This module only keeps :class:`FastAngularLightconer`, the name under which the
angular lightconer with sub-sampled pixels was introduced, for existing code.
"""

from __future__ import annotations

import warnings

import attrs
from py21cmfast.wrapper.inputs import Planck18

from tuesday.core.lightcones import AngularLightconer

__all__ = ["FastAngularLightconer"]


@attrs.define(kw_only=True, slots=False)
class FastAngularLightconer(AngularLightconer):
    """An angular lightconer whose pixels can average several directions.

    .. deprecated::
        This is now :class:`tuesday.core.lightcones.AngularLightconer`, which has the
        same parameters. The only difference is the default cosmology, which is
        that of 21cmFAST here (and Planck18 of astropy there): pass
        ``cosmo=inputs.cosmo_params.cosmo`` to either.
    """

    #: The cosmology that is used if none is given: the Planck18 of 21cmFAST.
    default_cosmo = Planck18

    def __attrs_post_init__(self) -> None:
        """Warn that the class was renamed."""
        warnings.warn(
            "FastAngularLightconer is deprecated: use "
            "tuesday.core.lightcones.AngularLightconer, which has the same parameters "
            "(pass it the cosmology of the simulation: "
            "cosmo=inputs.cosmo_params.cosmo).",
            DeprecationWarning,
            stacklevel=3,
        )
