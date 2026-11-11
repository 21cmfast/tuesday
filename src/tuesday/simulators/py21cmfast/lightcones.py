"""Generate lightcones from 21cmFAST output cache."""

import hashlib
import itertools
import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import attrs
import h5py
import numpy as np
from astropy import units
from astropy.units import Quantity
from py21cmfast.drivers.lightcone import LightCone, setup_lightcone_instance
from py21cmfast.io import read_output_struct
from py21cmfast.io.caching import RunCache
from py21cmfast.lightconers import AngularLightconer, Lightconer
from py21cmfast.rsds import include_dvdr_in_tau21
from py21cmfast.wrapper import outputs
from py21cmfast.wrapper.inputs import InputParameters
from scipy.spatial.transform import Rotation

from tuesday.core.lightcones import LightconeStore

logger = logging.getLogger(__name__)


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
    apply_rsds
        Whether to add a ``<quantity>_with_rsds`` field with redshift-space
        distortions for every quantity, as :func:`py21cmfast.run_lightcone` does.
        The lightconer is extended to the distances the RSDs need, and the result is
        trimmed back to the requested distances.
    include_dvdr_in_tau21
        Whether to apply the velocity-gradient (dv/dr) correction to
        ``brightness_temp``, as :func:`py21cmfast.run_lightcone` does by default.
        The ``los_velocity`` (and, with ``USE_TS_FLUCT``, ``tau_21``) lightcones it
        needs are added to the output.
    n_rsd_subcells
        The number of sub-cells per cell used to apply the RSDs.

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

    # The requested distances, before validate_options extends them for RSDs.
    lc_distances = lightconer.lc_distances.copy()
    # validate_options returns the lightconer to use: it adds the los_velocity and
    # tau_21 quantities that the dv/dr correction and the RSDs need.
    lightconer = lightconer.validate_options(
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
                    coeval.ionized_box.log10_mturn_ave_acg
                )
            elif quantity == "log10_mturn_mcg":
                lightcone.global_quantities[quantity][iz] = (
                    coeval.ionized_box.log10_mturn_ave_mcg
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

    # Apply the dv/dr correction and RSDs, and trim the RSD buffer, with the method
    # that run_lightcone calls at its last node (private in 21cmFAST 4.3).
    return lightcone._finalize_lightcone_at_last_redshift(
        include_dvdr_in_tau21=include_dvdr_in_tau21,
        apply_rsds=apply_rsds,
        n_rsd_subcells=n_rsd_subcells,
        inputs=inputs,
        lightconer=lightconer,
        lc_distances=lc_distances,
        lowz_buffer_pixels=0,
        highz_buffer_pixels=0,
        lightcone_filename=None,
    )


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


#: The output structs whose arrays are coeval fields, in the order in which
#: :class:`py21cmfast.Coeval` looks a field up.
_COEVAL_STRUCTS = (
    "PerturbedField",
    "IonizedBox",
    "BrightnessTemp",
    "TsBox",
    "EmissivityFields",
)

#: Options of the lightconers that change how fast the slices are made, but not the
#: slices (beyond rounding). A store can be resumed with other values of these.
_SPEED_OPTIONS = ("tile_first",)

#: Arrays of a lightconer's description with more values than this are recorded by
#: their SHA-256 hash.
_MAX_STORED_VALUES = 64


class CachedFields:
    """The coeval fields of a 21cmFAST run in its cache, read one field at a time.

    :func:`make_lightcones_on_disk` reads the boxes it needs from this. Unlike
    :meth:`py21cmfast.io.caching.RunCache.get_coeval_at_z`, it reads one field of one
    output struct at a time, and it can tell which fields are cached before the run is
    complete.

    Parameters
    ----------
    cache
        The cache of the run.
    """

    def __init__(self, cache: RunCache):
        self.cache = cache
        self.inputs = cache.inputs
        kinds = {}
        z = self.inputs.node_redshifts[0]
        for kind in _COEVAL_STRUCTS:
            if getattr(cache, kind) is None:
                continue
            struct = getattr(outputs, kind).new(self.inputs, redshift=z)
            for name in struct.arrays:
                kinds.setdefault(name, kind)
        self.kinds: dict[str, str] = kinds
        """The output struct that holds each coeval field."""

    @property
    def node_redshifts(self) -> list[float]:
        """The node redshifts of the run, in decreasing order."""
        return sorted(self.inputs.node_redshifts, reverse=True)

    def path(self, redshift: float, field: str) -> Path:
        """Return the cache file that holds a field at a node redshift.

        Parameters
        ----------
        redshift
            The node redshift.
        field
            The name of the field.

        Returns
        -------
        Path
            The file, which may not exist yet.
        """
        if field not in self.kinds:
            raise KeyError(
                f"{field!r} is not a coeval field of this run; the fields are "
                f"{sorted(self.kinds)}"
            )
        files = getattr(self.cache, self.kinds[field])
        if redshift not in files:
            raise KeyError(f"{redshift} is not a node redshift of this run")
        return Path(files[redshift])

    def has(self, redshift: float, fields: Iterable[str]) -> bool:
        """Whether some fields are cached at a node redshift.

        Parameters
        ----------
        redshift
            The node redshift.
        fields
            The names of the fields.

        Returns
        -------
        bool
            Whether the files of all the fields exist.
        """
        return all(self.path(redshift, field).exists() for field in fields)

    def read(self, redshift: float, field: str) -> np.ndarray:
        """Read one field at a node redshift.

        Parameters
        ----------
        redshift
            The node redshift.
        field
            The name of the field.

        Returns
        -------
        np.ndarray
            The field.
        """
        return read_output_struct(self.path(redshift, field)).get(field)


def _serialise(value: Any) -> Any:
    """Return a value of a lightconer as a value that HDF5 can store as an attribute."""
    if isinstance(value, Rotation):
        value = value.as_matrix()
    elif isinstance(value, Quantity):
        value = value.to_value(value.unit)
    elif isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    elif isinstance(value, bool | int | float | str | np.number | np.bool_):
        return value
    elif value is None:
        return "None"
    value = np.asarray(value)
    if value.dtype.kind not in "biuf":
        return repr(value.tolist())
    if value.size > _MAX_STORED_VALUES:
        data = np.ascontiguousarray(value, dtype=float)
        return "sha256:" + hashlib.sha256(data.tobytes()).hexdigest()
    return value


def lightconer_metadata(lightconer: Lightconer) -> dict[str, Any]:
    """Describe a lightconer by values that HDF5 can store as attributes.

    The description holds the class of the lightconer and each of its options that
    the slices depend on, except its quantities, its distances and its cosmology. A
    :class:`~tuesday.core.lightcones.store.LightconeStore` records the distances and
    redshifts of the slices, which depend on the cosmology. Arrays of more than 64
    values, such as the directions of an angular lightcone, are described by their
    SHA-256 hash, and units are dropped.

    Parameters
    ----------
    lightconer
        The lightconer.

    Returns
    -------
    dict
        The description.
    """
    kind = type(lightconer)
    out = {"lightconer": f"{kind.__module__}.{kind.__qualname__}"}
    for field in attrs.fields(kind):
        name = field.name.lstrip("_")
        if not field.eq or name in ("quantities", "lc_distances", "cosmo"):
            continue
        if name in _SPEED_OPTIONS:
            continue
        out[name] = _serialise(getattr(lightconer, field.name))
    return out


def _with_quantities(lightconer: Lightconer, quantities: Sequence[str]) -> Lightconer:
    """Return the lightconer, making only the given quantities.

    21cmFAST makes the ``los_velocity`` slices together with those of the first
    quantity, so ``los_velocity`` cannot be first. If it is the only quantity,
    ``velocity_z`` is made first.
    """
    quantities = list(quantities)
    if quantities[0] == "los_velocity":
        if len(quantities) == 1:
            quantities.insert(0, "velocity_z")
        else:
            quantities.append(quantities.pop(0))
    return attrs.evolve(lightconer, quantities=tuple(quantities))


def _fields_of(lightconer: Lightconer) -> list[str]:
    """Return the coeval fields that a lightconer's slices are made from."""
    fields = [q for q in lightconer.quantities if q != "los_velocity"]
    if "los_velocity" in lightconer.quantities:
        if isinstance(lightconer, AngularLightconer):
            fields += ["velocity_x", "velocity_y", "velocity_z"]
        else:
            fields.append("velocity_z")
    return list(dict.fromkeys(fields))


def _slices_by_interval(
    lightconer: Lightconer, inputs: InputParameters, node_redshifts: Sequence[float]
) -> list[np.ndarray]:
    """Return the indices of the slices between each pair of consecutive nodes.

    These are the slices that ``lightconer.make_lightcone_slices`` makes from the
    boxes at the two node redshifts.
    """
    cell = inputs.simulation_options.cell_size
    pixeleq = units.pixel_scale(cell / units.pixel)
    nodes = inputs.cosmo_params.cosmo.comoving_distance(np.asarray(node_redshifts))
    nodes = nodes.to_value(units.pixel, equivalencies=pixeleq)
    slices = lightconer.get_lc_distances_in_pixels(cell).to_value(units.pixel)
    out = []
    for d1, d2 in itertools.pairwise(nodes):
        dmin, dmax = min(d1, d2), max(d1, d2)
        out.append(np.nonzero((slices >= dmin * (1 - 1e-6)) & (slices < dmax))[0])
    return out


def _to_fill(
    slices: Sequence[np.ndarray], paths: Iterable[Path], *, clobber: bool
) -> list[int]:
    """Return the intervals that have slices, and are not filled in all the stores.

    If a store does not exist, or ``clobber`` is True, every interval with slices is
    to be filled.
    """
    filled = np.ones(len(slices), dtype=bool)
    for path in paths:
        if clobber or not path.exists():
            filled[:] = False
            break
        with LightconeStore.open(path) as store:
            done = store.done
        filled &= done if done.shape == filled.shape else False
    return [i for i, index in enumerate(slices) if index.size and not filled[i]]


def _read_node(
    source: CachedFields,
    nodes: Sequence[float],
    node_index: int,
    fields: Sequence[str],
    stores: Mapping[str, Mapping[str, LightconeStore]],
) -> SimpleNamespace:
    """Read the fields at a node redshift, and record their means in the stores."""
    inputs = source.inputs
    redshift = nodes[node_index]
    box = {f: source.read(redshift, f) for f in fields}
    for by_quantity in stores.values():
        for q, store in by_quantity.items():
            if q in box:
                store.set_global_mean(node_index, float(np.mean(box[q])))
    return SimpleNamespace(
        redshift=redshift,
        simulation_options=inputs.simulation_options,
        cosmo_params=inputs.cosmo_params,
        **box,
    )


def _runs(index: Iterable[int]) -> Iterator[tuple[int, int]]:
    """Yield the ``(start, stop)`` of each run of consecutive integers."""
    index = np.array(sorted(index), dtype=int)
    if not index.size:
        return
    breaks = np.nonzero(np.diff(index) != 1)[0] + 1
    for run in np.split(index, breaks):
        yield int(run[0]), int(run[-1]) + 1


def _fill_interval(
    lightconer: Lightconer,
    stores: Mapping[str, LightconeStore],
    low: SimpleNamespace,
    high: SimpleNamespace,
) -> None:
    """Write the slices between two boxes into the stores of each quantity."""
    values = {q: {} for q in stores}
    for q, idx, this_lc in lightconer.make_lightcone_slices(low, high):
        if this_lc is not None and q in values:
            values[q][int(idx)] = this_lc
    for q, by_index in values.items():
        for start, stop in _runs(by_index):
            block = np.stack([by_index[k] for k in range(start, stop)], axis=-1)
            stores[q].write(slice(start, stop), block)


def make_lightcones_on_disk(
    lightconers: Mapping[str, Lightconer],
    quantities: str | Sequence[str],
    source: CachedFields | RunCache,
    out_dir: str | Path,
    *,
    partial: bool = False,
    clobber: bool = False,
) -> dict[str, dict[str, Path]]:
    """Make lightcones from the cached boxes of a run, in files that can be resumed.

    The lightcones are made in one pass over the node redshifts, from high to low
    redshift, as :func:`py21cmfast.run_lightcone` makes them. At each node redshift,
    only the fields that the lightcones need are read. The slices between two node
    redshifts are made by each lightconer's ``make_lightcone_slices``, so they are
    those of :func:`py21cmfast.run_lightcone` (before the dv/dr correction and RSDs).

    Each lightcone of each quantity is written to its own file,
    ``out_dir/<name>/<quantity>.h5``, a
    :class:`~tuesday.core.lightcones.store.LightconeStore`. ``<name>`` is the key of
    the lightconer in ``lightconers``, chosen freely, e.g. ``"angular"`` for an
    :class:`py21cmfast.AngularLightconer` and ``"rectilinear"`` for a
    :class:`py21cmfast.RectilinearLightconer`. Lightconers of all kinds are filled
    in the same pass over the node redshifts. The store records which
    intervals between node redshifts are filled. Calling this again with the same
    arguments fills the intervals that are not filled yet. The store also records the
    mean of each quantity's box at the node redshifts that were read.

    Parameters
    ----------
    lightconers
        The lightconers, by name. The name is the directory of their files in
        ``out_dir``. Their quantities are replaced by ``quantities``.
    quantities
        The quantities to make lightcones of: coeval fields, or ``"los_velocity"``,
        the velocity along the line of sight that 21cmFAST uses for the dv/dr
        correction and the RSDs.
    source
        The cache of the run, or the fields of its boxes (:class:`CachedFields`).
        Any object with the attributes and methods ``inputs``, ``node_redshifts``
        (decreasing), ``has`` and ``read`` of :class:`CachedFields` can be used.
    out_dir
        The directory of the files. It should hold the lightcones of one run only.
    partial
        Whether to skip the intervals whose boxes are not cached yet, and fill the
        others. By default, this raises an error before writing anything if a box is
        missing.
    clobber
        Whether to make the lightcones again from the start, if their files exist.

    Returns
    -------
    dict[str, dict[str, Path]]
        The file of each lightcone, by name and quantity.

    Raises
    ------
    FileNotFoundError
        If a box that is needed is not cached, and ``partial`` is False.
    ValueError
        If a file exists with another lightconer, and ``clobber`` is False, or if a
        lightcone extends beyond the node redshifts.
    """
    if isinstance(source, RunCache):
        source = CachedFields(source)
    quantities = (quantities,) if isinstance(quantities, str) else tuple(quantities)
    if not quantities or len(set(quantities)) != len(quantities):
        raise ValueError("quantities must be distinct, and there must be at least one")
    inputs = source.inputs
    nodes = list(source.node_redshifts)
    if np.any(np.diff(nodes) >= 0):
        raise ValueError("the node redshifts of the source must decrease")
    paths = {
        name: {q: Path(out_dir) / name / f"{q}.h5" for q in quantities}
        for name in lightconers
    }

    makers = {}
    for name, lightconer in lightconers.items():
        makers[name] = _with_quantities(lightconer, quantities)
        makers[name].validate_options(
            inputs, include_dvdr_in_tau21=False, apply_rsds=False
        )
    fields = list(dict.fromkeys(f for lc in makers.values() for f in _fields_of(lc)))
    for field in fields:
        source.has(nodes[0], [field])  # an unknown field raises a KeyError
    slices = {n: _slices_by_interval(lc, inputs, nodes) for n, lc in makers.items()}

    # The intervals left to fill, and whether their boxes are cached, are known
    # before any file is written.
    todo = {}
    for name in makers:
        for interval in _to_fill(slices[name], paths[name].values(), clobber=clobber):
            todo.setdefault(interval, []).append(name)
    cached = {
        node_index: source.has(nodes[node_index], fields)
        for interval in todo
        for node_index in (interval, interval + 1)
    }
    if missing := sorted({nodes[n] for n, ok in cached.items() if not ok}):
        if not partial:
            raise FileNotFoundError(
                f"{fields} are not cached at {len(missing)} node redshifts "
                f"({missing[-1]} to {missing[0]}); use partial=True to fill the "
                "intervals whose boxes are cached"
            )
        logger.info(
            "Skipping the intervals next to the %d node redshifts whose boxes are "
            "not cached (z = %.3f to %.3f).",
            len(missing),
            missing[0],
            missing[-1],
        )
        todo = {
            interval: names
            for interval, names in todo.items()
            if cached[interval] and cached[interval + 1]
        }

    stores = {name: {} for name in makers}
    try:
        for name, lightconer in makers.items():
            empty = [
                interval
                for interval, index in enumerate(slices[name])
                if not index.size
            ]
            for q, path in paths[name].items():
                stores[name][q] = LightconeStore.open_or_create(
                    path,
                    quantity=q,
                    shape=lightconer.get_shape(inputs.simulation_options),
                    distances=lightconer.lc_distances,
                    redshifts=lightconer.lc_redshifts,
                    node_redshifts=nodes,
                    metadata=lightconer_metadata(lightconer),
                    clobber=clobber,
                )
                if empty:
                    stores[name][q].mark_done(empty)

        # The boxes at the two ends of the interval, by node index. The box shared
        # with the previous interval is kept.
        boxes = {}
        for interval, names in sorted(todo.items()):
            ends = (interval, interval + 1)
            boxes = {n: box for n, box in boxes.items() if n in ends}
            for node_index in ends:
                if node_index not in boxes:
                    boxes[node_index] = _read_node(
                        source, nodes, node_index, fields, stores
                    )
            for name in names:
                _fill_interval(
                    makers[name], stores[name], boxes[interval + 1], boxes[interval]
                )
                for store in stores[name].values():
                    store.mark_done(interval)
            logger.info(
                "Filled the slices between z = %.3f and %.3f of %s.",
                nodes[interval],
                nodes[interval + 1],
                ", ".join(names),
            )
    finally:
        for by_quantity in stores.values():
            for store in by_quantity.values():
                store.close()

    return paths


# --- Finishing the lightcones on disk ------------------------------------------------

#: The name of the brightness-temperature lightcone with the dv/dr correction.
DVDR = "brightness_temp_dvdr"

#: About how many bytes 21cmFAST's dv/dr correction uses per value of the lightcone
#: (the inputs, the velocity gradient and the result, in float64).
_DVDR_BYTES_PER_VALUE = 64


def _complete(path: Path) -> bool:
    """Whether a file is a complete lightcone store."""
    if not path.exists():
        return False
    try:
        with LightconeStore.open(path) as store:
            return store.complete
    except (OSError, ValueError):
        return False


def _temporary_path(path: Path) -> Path:
    """Return the temporary path at which to write a file before renaming it."""
    return path.with_name(f".{path.name}.tmp")


def include_dvdr_on_disk(
    directory: str | Path,
    inputs: InputParameters,
    *,
    max_memory: float = 2e9,
    clobber: bool = False,
) -> Path:
    """Apply 21cmFAST's dv/dr correction to a brightness-temperature lightcone on disk.

    This applies :func:`py21cmfast.rsds.include_dvdr_in_tau21` to the stores
    ``brightness_temp.h5``, ``los_velocity.h5`` and (with ``USE_TS_FLUCT``)
    ``tau_21.h5`` in ``directory``, as made by :func:`make_lightcones_on_disk`. The
    result is what :func:`py21cmfast.run_lightcone` returns as ``brightness_temp``
    with ``include_dvdr_in_tau21=True`` (its default). It is written to
    ``brightness_temp_dvdr.h5`` in the same directory.

    The correction differentiates the velocity along each line of sight, so the
    lightcone is corrected in blocks of lines of sight that fit in ``max_memory``.
    These are blocks of whole chunks if ``max_memory`` allows it: with the default
    chunks of the stores, the lines of sight of one chunk need about 1 GiB. The file
    is written under a temporary name and renamed when it is complete.

    Parameters
    ----------
    directory
        The directory of the stores of one lightconer.
    inputs
        The input parameters of the run.
    max_memory
        About how many bytes of memory to use.
    clobber
        Whether to make the corrected lightcone again if it exists.

    Returns
    -------
    Path
        The file of the corrected lightcone.

    Raises
    ------
    ValueError
        If a store that is needed is missing or not complete, or if the slices are
        not one cell apart (which the correction assumes).
    """
    directory = Path(directory)
    names = ["brightness_temp", "los_velocity"]
    if inputs.astro_options.USE_TS_FLUCT:
        names.append("tau_21")
    paths = [directory / f"{name}.h5" for name in names]
    if missing := [p.name for p in paths if not _complete(p)]:
        raise ValueError(f"{missing} in {directory} are missing or not complete")
    out = directory / f"{DVDR}.h5"
    if _complete(out) and not clobber:
        return out

    stores = [LightconeStore.open(p) for p in paths]
    try:
        temperature, velocity, *tau = stores
        spacing = np.diff(temperature.distances)
        if not np.allclose(spacing, inputs.simulation_options.cell_size, rtol=1e-6):
            raise ValueError(
                "21cmFAST's dv/dr correction assumes slices one cell apart; the slices "
                f"in {directory} are not"
            )
        shape = temperature.shape
        # blocks of whole chunks, if a chunk fits in memory
        rows = max(1, int(max_memory // (_DVDR_BYTES_PER_VALUE * np.prod(shape[1:]))))
        chunk = temperature.lightcone.chunks[0]
        if rows > chunk:
            rows = rows // chunk * chunk

        tmp = _temporary_path(out)
        with LightconeStore.open_or_create(
            tmp,
            quantity=DVDR,
            shape=shape,
            distances=temperature.distances,
            redshifts=temperature.redshifts,
            node_redshifts=temperature.node_redshifts,
            metadata=temperature.metadata,
            chunks=temperature.lightcone.chunks,
            clobber=True,
        ) as result:
            for start in range(0, shape[0], rows):
                block = slice(start, start + rows)
                corrected = include_dvdr_in_tau21(
                    brightness_temp=temperature.lightcone[block],
                    los_velocity=velocity.lightcone[block],
                    redshifts=temperature.redshifts,
                    inputs=inputs,
                    periodic=False,
                    tau_21=tau[0].lightcone[block] if tau else None,
                )
                result.lightcone[block] = np.asarray(corrected, dtype=np.float32)
            result.mark_done(range(result.done.size))
    finally:
        for store in stores:
            store.close()
    tmp.replace(out)
    return out


def link_lightcone_file(
    directory: str | Path, inputs: InputParameters, path: str | Path | None = None
) -> Path:
    """Write a 21cmFAST lightcone file that links to the lightcones on disk.

    The file has the layout of the files of :class:`py21cmfast.LightCone`, so
    :meth:`py21cmfast.LightCone.from_file` (or
    :meth:`py21cmfast.AngularLightcone.from_file`) can read it. Its lightcones are
    HDF5 external links to the complete stores in ``directory``, so nothing is
    copied, and h5py can read parts of them. If the dv/dr-corrected lightcone
    (:func:`include_dvdr_on_disk`) is there, ``brightness_temp`` links to it and
    ``brightness_temp_no_dvdr`` to the lightcone without the correction, as in the
    output of :func:`py21cmfast.run_lightcone`. The global quantities are the means
    recorded in the stores, in the order of ``inputs.node_redshifts``, and NaN at the
    node redshifts that were not read.

    The file is written under a temporary name and renamed when it is complete.

    Parameters
    ----------
    directory
        The directory of the stores of one lightconer.
    inputs
        The input parameters of the run.
    path
        The file to write. ``<directory>.h5`` by default.

    Returns
    -------
    Path
        The file.

    Raises
    ------
    ValueError
        If there is no complete store in ``directory``, or if the stores do not have
        the same distances.
    """
    directory = Path(directory)
    path = Path(path) if path is not None else directory.with_suffix(".h5")
    stores = {p.stem: p for p in sorted(directory.glob("*.h5")) if _complete(p)}
    if not stores:
        raise ValueError(f"there is no complete lightcone in {directory}")
    links = dict(stores)
    if DVDR in links:
        links["brightness_temp"] = links.pop(DVDR)
        if "brightness_temp" in stores:
            links["brightness_temp_no_dvdr"] = stores["brightness_temp"]

    node_redshifts = np.asarray(inputs.node_redshifts, dtype=float)
    distances = None
    global_quantities = {}
    for name, store_path in stores.items():
        with LightconeStore.open(store_path) as store:
            if distances is None:
                distances = store.distances
            elif not np.allclose(store.distances, distances, rtol=1e-10):
                raise ValueError(f"the lightcones in {directory} have other distances")
            means = store.global_mean
            stored = store.node_redshifts
        if name != DVDR and np.isfinite(means).any():
            order = [int(np.argmin(np.abs(stored - z))) for z in node_redshifts]
            global_quantities[name] = means[order]

    tmp = _temporary_path(path)
    LightCone(
        lightcone_distances=distances,
        inputs=inputs,
        lightcones={},
        global_quantities=global_quantities,
        last_completed_node=len(node_redshifts) - 1,
        last_completed_lcidx=0,
    ).save(tmp, clobber=True)
    with h5py.File(tmp, "a") as file:
        for name, target in sorted(links.items()):
            relative = os.path.relpath(target, path.parent)
            file["lightcones"][name] = h5py.ExternalLink(relative, "/lightcone")
    tmp.replace(path)
    return path
