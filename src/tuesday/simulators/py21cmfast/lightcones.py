"""Generate lightcones from 21cmFAST output cache."""

from collections.abc import Sequence

import numpy as np
from py21cmfast.drivers.lightcone import (
    AngularLightcone,
    LightCone,
    setup_lightcone_instance,
)
from py21cmfast.io.caching import RunCache
from py21cmfast.lightcones import Lightconer


def construct_lightcone_from_cache(
    cache: RunCache,
    lightconer: Lightconer,
    global_quantities: Sequence[str] = (),
) -> LightCone:
    """Construct a lightcone from a cached coeval simulation run.

    This function takes a RunCache object, a Lightconer object, and a list of global
    quantities, and constructs a lightcone by iterating through the redshifts in the
    cache. It retrieves coeval boxes from the cache, extracts global quantities, and
    uses the Lightconer to generate lightcone slices.

    Parameters
    ----------
    cache
        The cache containing the coeval simulation data.
    lightconer
        The object used to generate lightcone slices.
    global_quantities
        A list of global quantities to extract.

    Returns
    -------
    Lightcone
        The constructed (21cmFAST) lightcone object.

    Raises
    ------
    ValueError
        If the provided cache is not complete.
    """
    if not cache.is_complete():
        raise ValueError("The cache specified is not complete!")

    inputs = cache.inputs
    node_redshifts = sorted(cache.BrightnessTemp.keys(), reverse=True)

    lightconer.validate_options(cache.inputs.matter_options, cache.inputs.astro_options)

    # Create the LightCone instance, loading from file if needed
    lightcone = setup_lightcone_instance(
        lightconer=lightconer,
        inputs=inputs,
        scrollz=node_redshifts,
        global_quantities=global_quantities,
        photon_nonconservation_data={},
    )

    lightcone._last_completed_node = -1
    lightcone._last_completed_lcidx = (
        np.sum(
            lightcone.lightcone_redshifts
            >= node_redshifts[lightcone._last_completed_node]
        )
        - 1
    )

    prev_coeval = None
    for iz, z in enumerate(node_redshifts):
        # Here we read all the boxes that we might need, without actually reading
        # any data.
        coeval = cache.get_coeval_at_z(z=z)

        # Save mean/global quantities
        for quantity in global_quantities:
            if quantity == "log10_mturn_acg":
                lightcone.global_quantities[quantity][iz] = (
                    coeval.ionized_box.log10_Mturnover_ave
                )
            elif quantity == "log10_mturn_mcg":
                lightcone.global_quantities[quantity][iz] = (
                    coeval.ionized_box.log10_Mturnover_MINI_ave
                )
            else:
                lightcone.global_quantities[quantity][iz] = np.mean(
                    getattr(coeval, quantity)
                )

        # Get lightcone slices
        if prev_coeval is not None:
            for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                coeval, prev_coeval
            ):
                if this_lc is not None:
                    lightcone.lightcones[quantity][..., idx] = this_lc

        prev_coeval = coeval

        # last redshift things
        if iz == len(node_redshifts) - 1 and (
            isinstance(lightcone, AngularLightcone) and lightconer.get_los_velocity
        ):
            lightcone.lightcones["brightness_temp_with_rsds"] = lightcone.compute_rsds(
                n_subcells=inputs.astro_params.N_RSD_STEPS
            )

    return lightcone
