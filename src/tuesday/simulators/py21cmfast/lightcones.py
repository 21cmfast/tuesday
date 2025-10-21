"""Generate lightcones from 21cmFAST output cache."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from py21cmfast.drivers.lightcone import (
    LightCone,
    setup_lightcone_instance,
)
from py21cmfast.io import read_output_struct
from py21cmfast.io.caching import RunCache
from py21cmfast.lightconers import Lightconer


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

    lightconer.validate_options(
        cache.inputs, include_dvdr_in_tau21=False, apply_rsds=False
    )

    # Create the LightCone instance, loading from file if needed
    lightcone = setup_lightcone_instance(
        lightconer=lightconer,
        inputs=inputs,
        scrollz=node_redshifts,
        include_dvdr_in_tau21=False,
        apply_rsds=False,
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

    return lightcone


def construct_lightcone_from_filelist(
    filelist: Sequence[Path],
    lightconer: Lightconer,
) -> LightCone:
    """Construct a Lightcone from a list of OutputStruct files.

    This function assumes that the list of files are 21cmFAST output structs, each
    at a different redshift. These may be from a RunCache, but don't need to be.

    Parameters
    ----------
    filelist
        A list of paths to 21cmFAST output struct files.
    lightconer
        The object used to generate lightcone slices.
    global_quantities
        A list of global quantities to extract. These must exist in the files
        given.
    """
    boxes = [read_output_struct(fl) for fl in filelist]
    inputs = boxes[0].inputs
    boxes = sorted(boxes, key=lambda b: b.redshift, reverse=True)
    node_redshifts = [b.redshift for b in boxes]

    lightconer.validate_options(inputs, apply_rsds=False, include_dvdr_in_tau21=False)

    # Create the LightCone instance, loading from file if needed
    lightcone = setup_lightcone_instance(
        lightconer=lightconer,
        inputs=inputs,
        scrollz=node_redshifts,
        photon_nonconservation_data={},
        include_dvdr_in_tau21=False,
        apply_rsds=False,
    )

    lightcone._last_completed_node = -1
    lightcone._last_completed_lcidx = (
        np.sum(
            lightcone.lightcone_redshifts
            >= node_redshifts[lightcone._last_completed_node]
        )
        - 1
    )

    prev_box = None
    for box in boxes:
        # This hacks an OutputStruct to look like a Coeval, which has all the
        # boxes defined directly as attributes. Later versions
        # of 21cmFAST will not require this.
        _box = SimpleNamespace()
        _box.__dict__.update(box.__dict__)
        _box.simulation_options = box.inputs.simulation_options
        _box.cosmo_params = box.inputs.cosmo_params
        for q in lightconer.quantities:
            setattr(_box, q, box.get(q))

        # Get lightcone slices
        if prev_box is not None:
            for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                _box, prev_box
            ):
                if this_lc is not None:
                    lightcone.lightcones[quantity][..., idx] = this_lc

        prev_box = _box

    return lightcone
