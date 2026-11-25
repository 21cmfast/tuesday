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
    brightness_field: str = "cii_brightness",
) -> LightCone:
    """Mock-observe a [CII] lightcone with a line-intensity-mapping instrument.

    See :func:`~tuesday.core.postprocessing.lim_noise.observe_lim_lightcone` for the
    channel averaging, beam smoothing and noise model. Slices outside the frequency
    range of the instrument are set to NaN.

    Parameters
    ----------
    lc_obj : LightCone
        Rectilinear lightcone with a ``brightness_field`` field in Jy/sr, e.g.
        ``"cii_brightness"`` from :func:`~.cii_lightcone.gen_lc`. It is not
        modified.
    noise_template : LIMInstrument or str
        The instrument, or the name of a built-in instrument template; see
        :func:`~tuesday.core.postprocessing.lim_noise.calculate_noise` for the
        built-in templates.
    seed : int, optional
        Seed for the noise realisation. If None, a random seed is used.
    brightness_field : str, optional
        Name of the field to observe, ending in ``"_brightness"``: by default the
        [CII] signal, or e.g. ``"cii_co_brightness"`` for [CII] plus the CO
        foreground of :func:`~.co_foregrounds.add_co_foregrounds`.

    Returns
    -------
    LightCone
        A copy of ``lc_obj`` with three new fields in Jy/sr, named after the
        observed field (``<field>_brightness``) and the instrument:
        ``"<field>_noise_<name>"`` (the noise), ``"<field>_signal_<name>"`` (the
        beam-smoothed signal) and ``"<field>_observed_<name>"`` (their sum), e.g.
        ``"cii_noise_ccatp"``. The other fields are shared with ``lc_obj``.

    Raises
    ------
    ValueError
        If ``noise_template`` is not the name of a built-in template,
        ``brightness_field`` does not end in ``"_brightness"``, or the lightcone is
        not rectilinear.

    Warns
    -----
    UserWarning
        If some slices are outside the frequency range of the instrument.
    """
    if not brightness_field.endswith("_brightness"):
        raise ValueError(
            f"brightness_field must end in '_brightness'; got '{brightness_field}'."
        )
    prefix = brightness_field.removesuffix("_brightness")
    instrument = (
        noise_template
        if isinstance(noise_template, LIMInstrument)
        else LIMInstrument.from_template(noise_template)
    )
    cell_size = (
        lc_obj.simulation_options.BOX_LEN / lc_obj.simulation_options.HII_DIM * un.Mpc
    )
    signal, noise = observe_lim_lightcone(
        lc_obj.lightcones[brightness_field] * un.Jy / un.sr,
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
        f"{prefix}_noise_{instrument.name}": noise,
        f"{prefix}_signal_{instrument.name}": signal,
        f"{prefix}_observed_{instrument.name}": signal + noise,
    }
    return observed
