"""Generate lightcones from 21cmFAST output cache."""

import hashlib
import itertools
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import attrs
import numpy as np
from astropy import units
from astropy.units import Quantity
from py21cmfast.drivers.lightcone import LightCone, setup_lightcone_instance
from py21cmfast.io import read_output_struct
from py21cmfast.io.caching import RunCache
from py21cmfast.lightconers import AngularLightconer, Lightconer
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


# --- Lightcones on disk, filled from the cache one node interval at a time ----------

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


def _describe(value: Any) -> Any:
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
        out[name] = _describe(getattr(lightconer, field.name))
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
    j: int,
    fields: Sequence[str],
    stores: Mapping[str, Mapping[str, LightconeStore]],
) -> SimpleNamespace:
    """Read the fields at a node redshift, and record their means in the stores."""
    inputs = source.inputs
    box = {f: source.read(nodes[j], f) for f in fields}
    for by_quantity in stores.values():
        for q, store in by_quantity.items():
            if q in box:
                store.set_global_mean(j, float(np.mean(box[q])))
    return SimpleNamespace(
        redshift=nodes[j],
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
    :class:`~tuesday.core.lightcones.store.LightconeStore`. The store records which
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
        for i in _to_fill(slices[name], paths[name].values(), clobber=clobber):
            todo.setdefault(i, []).append(name)
    cached = {j: source.has(nodes[j], fields) for i in todo for j in (i, i + 1)}
    if missing := sorted({nodes[j] for j, ok in cached.items() if not ok}):
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
        todo = {i: names for i, names in todo.items() if cached[i] and cached[i + 1]}

    stores = {name: {} for name in makers}
    try:
        for name, lightconer in makers.items():
            empty = [i for i, index in enumerate(slices[name]) if not index.size]
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

        boxes = {}
        for i, names in sorted(todo.items()):
            boxes = {j: box for j, box in boxes.items() if j in (i, i + 1)}
            for j in (i, i + 1):
                if j not in boxes:
                    boxes[j] = _read_node(source, nodes, j, fields, stores)
            for name in names:
                _fill_interval(makers[name], stores[name], boxes[i + 1], boxes[i])
                for store in stores[name].values():
                    store.mark_done(i)
            logger.info(
                "Filled the slices between z = %.3f and %.3f of %s.",
                nodes[i],
                nodes[i + 1],
                ", ".join(names),
            )
    finally:
        for by_quantity in stores.values():
            for store in by_quantity.values():
                store.close()

    return paths
