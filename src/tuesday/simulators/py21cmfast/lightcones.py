"""Generate lightcones from 21cmFAST output cache."""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from py21cmfast.drivers.lightcone import (
    AngularLightcone,
    LightCone,
    do_rsds,
    setup_lightcone_instance,
)
from py21cmfast.io import read_output_struct
from py21cmfast.io.caching import RunCache
from py21cmfast.lightconers import Lightconer
from py21cmfast.wrapper.inputs import InputParameters


def construct_lightcone_from_cache(
    cache: RunCache,
    lightconer: Lightconer,
    global_quantities: Sequence[str] = (),
    apply_rsds: bool = False,
    include_dvdr_in_tau21: bool = False,
    n_rsd_subcells: int = 10,
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
        cache.inputs, apply_rsds=apply_rsds, include_dvdr_in_tau21=include_dvdr_in_tau21
    )

    # Create the LightCone instance, loading from file if needed
    lightcone = setup_lightcone_instance(
        lightconer=lightconer,
        inputs=inputs,
        scrollz=node_redshifts,
        include_dvdr_in_tau21=include_dvdr_in_tau21,
        apply_rsds=apply_rsds,
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
            isinstance(lightcone, AngularLightcone) and apply_rsds
        ):
            field_with_rsds = do_rsds(
                field=lightcone.lightcones["brightness_temp"],
                los_velocity=lightcone.lightcones["los_velocity"],
                redshifts=lightcone.lightcone_redshifts,
                inputs=inputs,
                periodic=False,
                n_rsd_subcells=n_rsd_subcells,
            )
            lightcone.lightcones["brightness_temp_with_rsds"] = field_with_rsds

    return lightcone


def construct_lightcone_from_coevals(
    coevals: Iterable[Any],
    lightconer: Lightconer,
    inputs: InputParameters,
    node_redshifts: Sequence[float] | None = None,
) -> LightCone:
    """Construct a Lightcone by interpolating a sequence of coeval boxes.

    Each pair of consecutive boxes is interpolated onto the lightcone with
    :meth:`~py21cmfast.lightconers.Lightconer.make_lightcone_slices`, as in
    :func:`py21cmfast.run_lightcone`. The boxes do not need to be 21cmFAST
    :class:`~py21cmfast.Coeval` objects, nor to be stored on disk.

    ``coevals`` is iterated once, and only the current and previous boxes are
    referenced, so if it is a generator that builds each box when it is requested,
    at most two boxes are in memory at a time.

    Parameters
    ----------
    coevals
        The coeval boxes, in order of decreasing redshift. Each must have the
        attributes ``redshift``, ``simulation_options`` and ``cosmo_params``, and
        one attribute per quantity of ``lightconer``.
    lightconer
        The object used to generate lightcone slices.
    inputs
        The input parameters of the simulation.
    node_redshifts
        The redshifts of the boxes. Defaults to ``inputs.node_redshifts``.

    Returns
    -------
    LightCone
        The constructed (21cmFAST) lightcone object.

    Raises
    ------
    ValueError
        If the lightcone extends beyond the range of the node redshifts.
    """
    if node_redshifts is None:
        node_redshifts = inputs.node_redshifts
    node_redshifts = sorted(node_redshifts, reverse=True)

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
    for box in coevals:
        # Get lightcone slices
        if prev_box is not None:
            for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                box, prev_box
            ):
                if this_lc is not None:
                    lightcone.lightcones[quantity][..., idx] = this_lc

        prev_box = box

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
    """
    boxes = [read_output_struct(fl) for fl in filelist]
    inputs = boxes[0].inputs
    boxes = sorted(boxes, key=lambda b: b.redshift, reverse=True)

    def as_coevals() -> Iterator[SimpleNamespace]:
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
            yield _box

    return construct_lightcone_from_coevals(
        as_coevals(),
        lightconer,
        inputs,
        node_redshifts=[b.redshift for b in boxes],
    )
