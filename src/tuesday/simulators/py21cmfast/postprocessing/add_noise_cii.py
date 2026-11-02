"""Add instrumental noise to a 21cmFAST [CII] surface-brightness lightcone.

This is a thin wrapper around
:func:`tuesday.core.postprocessing.lim_noise.observe_lim_lightcone`, which reads the
geometry and cosmology from a 21cmFAST :class:`~py21cmfast.LightCone`.
"""

import copy

from astropy import units as un
from py21cmfast import LightCone

from tuesday.core.postprocessing.lim_instruments import LIMInstrument
from tuesday.core.postprocessing.lim_noise import observe_lim_lightcone


def add_cii_noise(
    lc_obj: LightCone,
    noise_template: LIMInstrument | str,
    seed: int | None = None,
) -> LightCone:
    """Mock-observe a [CII] lightcone with a line-intensity-mapping instrument.

    See :func:`~tuesday.core.postprocessing.lim_noise.observe_lim_lightcone` for the
    channel averaging, beam smoothing and noise model. Slices outside the frequency
    range of the instrument are set to NaN.

    Parameters
    ----------
    lc_obj : LightCone
        Rectilinear lightcone with a ``"CII_BrightnessDensity"`` field in Jy/sr, as
        produced by :func:`~.cii_lightcone.gen_lc`. It is not modified.
    noise_template : LIMInstrument or str
        The instrument, or the name of a built-in instrument template; see
        :func:`~tuesday.core.postprocessing.lim_noise.calculate_noise` for the
        built-in templates.
    seed : int, optional
        Seed for the noise realisation. If None, a random seed is used.

    Returns
    -------
    LightCone
        A copy of ``lc_obj`` with three new fields in Jy/sr, named after the
        instrument: ``"noise_<name>"`` (the noise), ``"Bcii_<name>"`` (the
        beam-smoothed signal) and ``"Bcii_<name>_noise"`` (their sum). The other
        fields are shared with ``lc_obj``.

    Raises
    ------
    ValueError
        If ``noise_template`` is not the name of a built-in template, or the
        lightcone is not rectilinear.

    Warns
    -----
    UserWarning
        If some slices are outside the frequency range of the instrument.
    """
    instrument = (
        noise_template
        if isinstance(noise_template, LIMInstrument)
        else LIMInstrument.from_template(noise_template)
    )
    cell_size = (
        lc_obj.simulation_options.BOX_LEN / lc_obj.simulation_options.HII_DIM * un.Mpc
    )
    signal, noise = observe_lim_lightcone(
        lc_obj.lightcones["CII_BrightnessDensity"] * un.Jy / un.sr,
        redshifts=lc_obj.lightcone_redshifts,
        cell_size=cell_size,
        cosmo=lc_obj.cosmo_params.cosmo,
        noise_template=instrument,
        seed=seed,
    )
    signal = signal.to_value(un.Jy / un.sr)
    noise = noise.to_value(un.Jy / un.sr)

    observed = copy.copy(lc_obj)
    observed.lightcones = {
        **lc_obj.lightcones,
        f"noise_{instrument.name}": noise,
        f"Bcii_{instrument.name}": signal,
        f"Bcii_{instrument.name}_noise": signal + noise,
    }
    return observed
