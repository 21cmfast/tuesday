"""Lightconers: objects that make lightcones from the boxes of a simulation.

A simulation gives the fields of the universe at a number of redshifts: one
*snapshot* per redshift, each with a box per field. A *lightconer* makes lightcones
out of them. It is given the snapshots one at a time, from high to low redshift, and
returns the lightcone piece by piece.

Nothing here depends on any simulator. The lightconers follow the protocol of
21cmFAST (:class:`py21cmfast.lightconers.LightconerProtocol`), so they can be passed
to :func:`py21cmfast.run_lightcone`, and :func:`build_lightcone` runs the same loop on
snapshots from anywhere else.

There are two layers:

* What a driver uses: a lightconer has ``quantities`` and ``lc_distances``, and
  ``start(context)`` returns a *session*, the state of one run. The driver calls
  ``session.ingest(snapshot)`` for each snapshot, and gets :class:`Chunk` objects
  back. How many snapshots a session keeps, and what it caches, is its own business.

* What is used to write a lightconer: :class:`BracketingLightconer` is a base class
  for lightconers that make each slice of the lightcone from the two snapshots
  around it. A subclass says how one slice is made (:meth:`~BracketingLightconer.
  make_field_slice`), and may set up something for each slice first
  (:meth:`~BracketingLightconer.prepare_slice`), which is then shared by all the
  quantities. :class:`RectilinearLightconer` and :class:`AngularLightconer` are
  written that way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property, partial
from typing import Any, ClassVar, Literal, NamedTuple, Self

import attrs
import numpy as np
from astropy import units
from astropy.cosmology import FLRW, Planck18, z_at_value
from astropy.units import Mpc, Quantity, pixel, pixel_scale
from cosmotile import (
    make_lightcone_slice_interpolator,
    make_lightcone_slice_vector_field,
)
from scipy.spatial.transform import Rotation

from .healpix import subpixel_lonlat
from .slices import _plain, samples_first, shell_slice

__all__ = [
    "AngularLightconer",
    "BracketingLightconer",
    "BracketingSession",
    "Chunk",
    "LightconeContext",
    "RectilinearLightconer",
    "SessionCache",
    "Slice",
    "Snapshot",
    "build_lightcone",
    "interpolate_in_distance",
]


# ----------------------------------------------------------------------------------
# What a driver and a lightconer pass to each other
# ----------------------------------------------------------------------------------
class Chunk(NamedTuple):
    """A piece of a lightcone, returned by a session.

    It is stored with ``lightcones[quantity][index] = values``.
    """

    #: The name of the quantity.
    quantity: str
    #: Where the values go in the lightcone of that quantity: ``(..., i)`` for the
    #: whole slice at index ``i`` along the line of sight.
    index: Any
    #: The values.
    values: np.ndarray


@attrs.frozen(kw_only=True)
class LightconeContext:
    """What a lightconer is told about a simulation, when a run starts."""

    #: The cosmology of the simulation.
    cosmo: FLRW = attrs.field()
    #: The number of cells of the boxes along each axis.
    box_shape: tuple[int, int, int] = attrs.field(
        converter=lambda x: tuple(int(n) for n in x)
    )
    #: The comoving size of a cell of the boxes.
    cell_size: Quantity = attrs.field()
    #: The redshifts of the snapshots that will be passed to the session, in order.
    node_redshifts: tuple[float, ...] = attrs.field(
        default=(), converter=lambda x: tuple(float(z) for z in x)
    )


class Snapshot:
    """The fields of a simulation at one redshift.

    Parameters
    ----------
    redshift
        The redshift.
    fields
        The box of each field, by name. A box can also be given as a function
        without arguments that returns it, which is then only called when the field
        is first asked for (e.g. to read it from disk).

    Examples
    --------
    >>> import numpy as np
    >>> snapshot = Snapshot(8.0, {"density": np.zeros((4, 4, 4))})
    >>> snapshot["density"].shape
    (4, 4, 4)
    """

    def __init__(
        self,
        redshift: float,
        fields: Mapping[str, np.ndarray | Callable[[], np.ndarray]],
    ):
        self.redshift = float(redshift)
        self._fields = dict(fields)

    def __getitem__(self, field: str) -> np.ndarray:
        """Return the box of a field."""
        box = self._fields[field]
        if callable(box):
            box = self._fields[field] = box()
        return box

    def __contains__(self, field: str) -> bool:
        """Whether the snapshot has a field."""
        return field in self._fields

    def __repr__(self) -> str:
        """Return a short description."""
        return f"Snapshot(redshift={self.redshift}, fields={sorted(self._fields)})"


def interpolate_in_distance(
    distance: float,
    box_a: np.ndarray,
    box_b: np.ndarray,
    distance_a: float,
    distance_b: float,
    kind: str = "mean",
) -> np.ndarray:
    """Interpolate two boxes of a field linearly in comoving distance.

    Parameters
    ----------
    distance
        The comoving distance to interpolate to.
    box_a, box_b
        The field at the two distances. They must have the same shape.
    distance_a, distance_b
        The comoving distances of the two boxes, in the same units as ``distance``.
    kind
        Either "mean", for the linear interpolation, or "mean_max", which is the
        same, except where the two boxes have opposite signs, where the larger of
        the two values is taken. This is meant for fields in which a negative value
        means "not yet" (like the redshift of reionization of 21cmFAST).

    Returns
    -------
    np.ndarray
        The interpolated field.
    """
    if box_a.shape != box_b.shape:
        raise ValueError("the two boxes must have the same shape")

    out = (
        np.abs(distance_b - distance) * box_a + np.abs(distance_a - distance) * box_b
    ) / np.abs(distance_a - distance_b)

    if kind == "mean_max":
        flag = box_a * box_b < 0
        out[flag] = np.maximum(box_a, box_b)[flag]
    elif kind != "mean":
        raise ValueError("kind must be 'mean' or 'mean_max'")

    return out


# ----------------------------------------------------------------------------------
# Lightconers that make each slice from the two snapshots around it
# ----------------------------------------------------------------------------------
@attrs.define
class SessionCache:
    """Where a lightconer keeps what it wants to use again during a run.

    There is one dictionary per lifetime. A lightconer puts whatever it likes in
    them (from :meth:`BracketingLightconer.prepare_slice`, or while making a slice),
    and the session empties each one when its time is over.
    """

    #: Kept for the whole run. For what does not depend on the slice, e.g. the
    #: directions of the pixels.
    run: dict = attrs.field(factory=dict)
    #: Kept while the slices between one pair of snapshots are made. For what
    #: depends on the two snapshots, e.g. a box computed from both.
    interval: dict = attrs.field(factory=dict)
    #: Kept while the quantities of one slice are made. For what depends on the
    #: distance of the slice, and is the same for all quantities.
    slice: dict = attrs.field(factory=dict)


@attrs.frozen(kw_only=True)
class Slice:
    """One slice of a lightcone, and the two snapshots it is made from."""

    #: The index of the slice along the line of sight.
    index: int = attrs.field()
    #: The comoving distance of the slice, in cells.
    distance: np.float64 = attrs.field()
    #: The snapshot on the near (low-redshift) side of the slice_.
    near: Any = attrs.field()
    #: The snapshot on the far (high-redshift) side of the slice_.
    far: Any = attrs.field()
    #: The comoving distance of the near snapshot, in cells.
    near_distance: np.float64 = attrs.field()
    #: The comoving distance of the far snapshot, in cells.
    far_distance: np.float64 = attrs.field()
    #: What :meth:`BracketingLightconer.prepare_slice` returned for this slice_.
    state: Any = attrs.field(default=None)

    def interpolate(
        self, near: np.ndarray, far: np.ndarray, kind: str = "mean"
    ) -> np.ndarray:
        """Interpolate values at the two snapshots to the distance of the slice_.

        See :func:`interpolate_in_distance`.
        """
        return interpolate_in_distance(
            self.distance, near, far, self.near_distance, self.far_distance, kind=kind
        )


@attrs.define(kw_only=True, slots=False)
class BracketingLightconer:
    """Base class of lightconers that make each slice from the two snapshots around it.

    A subclass defines :meth:`lightcone_shape` and :meth:`make_field_slice`, and usually
    :meth:`make_los_velocity_slice` and :meth:`prepare_slice`. The session
    (:class:`BracketingSession`) then finds which slices lie between each pair of
    consecutive snapshots, and makes each quantity on each of them.

    Parameters
    ----------
    lc_distances
        The comoving distances to the lightcone slices. Either this or
        ``lc_redshifts`` must be provided.
    lc_redshifts
        The redshifts of the lightcone slices. Either this or ``lc_distances`` must
        be provided.
    cosmo
        The cosmology used to convert between redshifts and distances. It must be
        that of the simulation.
    quantities
        The quantities to make lightcones of: names of fields of the simulation, or
        ``"los_velocity"``, the velocity along the line of sight.
    interp_kinds
        The kind of interpolation in distance of each quantity ("mean" or
        "mean_max", see :func:`interpolate_in_distance`). Quantities that are not
        given use "mean". By default ``{"z_reion": "mean_max"}``.
    """

    #: The cosmology that is used if none is given.
    default_cosmo: ClassVar[FLRW] = Planck18

    cosmo: FLRW = attrs.field(validator=attrs.validators.instance_of(FLRW))

    _lc_redshifts: np.ndarray = attrs.field(default=None, eq=False)
    lc_distances: Quantity = attrs.field(
        eq=attrs.cmp_using(eq=partial(np.allclose, rtol=1e-5, atol=0))
    )

    quantities: tuple[str, ...] = attrs.field(
        default=("brightness_temp",),
        converter=tuple,
        validator=attrs.validators.deep_iterable(attrs.validators.instance_of(str)),
    )
    interp_kinds: dict[str, str] = attrs.field()

    #: The fields that the quantity ``"los_velocity"`` is made from.
    los_velocity_fields: ClassVar[tuple[str, ...]] = ("velocity_z",)

    #: The ratio of the rms of ``"los_velocity"`` to the rms of the full velocity.
    #: 21cmFAST uses it to choose how far to extend the lightcone for redshift-space
    #: distortions.
    los_velocity_rms_factor: ClassVar[float] = 1.0

    @cosmo.default
    def _cosmo_default(self):
        return self.default_cosmo

    @lc_distances.default
    def _lcd_default(self):
        if self._lc_redshifts is None:
            raise ValueError("Either lc_distances or lc_redshifts must be provided")
        return self.cosmo.comoving_distance(self._lc_redshifts)

    @lc_distances.validator
    def _lcd_vld(self, attribute, value):
        if np.any(value < 0):
            raise ValueError("lc_distances must be non-negative")

    @_lc_redshifts.validator
    def _lcz_vld(self, attribute, value):
        if value is not None and np.any(value < 0):
            raise ValueError("lc_redshifts must be non-negative")

    @interp_kinds.default
    def _interp_kinds_def(self):
        return {"z_reion": "mean_max"}

    @cached_property
    def lc_redshifts(self) -> np.ndarray:
        """The redshifts of the lightcone slices."""
        if self._lc_redshifts is not None:
            return self._lc_redshifts

        d = self.lc_distances
        zmin = z_at_value(self.cosmo.comoving_distance, d.min()).value
        zmax = z_at_value(self.cosmo.comoving_distance, d.max()).value

        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        dgrid = self.cosmo.comoving_distance(zgrid)
        return np.interp(d.value, dgrid.value, zgrid)

    def get_lc_distances_in_pixels(self, resolution: Quantity) -> Quantity:
        """Get the lightcone distances in pixels (cells), given the size of a cell."""
        return self.lc_distances.to(pixel, pixel_scale(resolution / pixel))

    @classmethod
    def between_redshifts(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: Quantity,
        cosmo: FLRW | None = None,
        **kw,
    ) -> Self:
        """Make a lightconer with slices regular in comoving distance between two z's.

        Parameters
        ----------
        min_redshift, max_redshift
            The redshifts of the first and (about) the last slice_.
        resolution
            The comoving distance between slices, usually the size of a cell.
        cosmo
            The cosmology. The default of the class if None.

        Other Parameters
        ----------------
        All other parameters are passed to the constructor.
        """
        if cosmo is None:
            cosmo = cls.default_cosmo
        d_at_redshift = cosmo.comoving_distance(min_redshift).to_value(Mpc)
        dmax = cosmo.comoving_distance(max_redshift).to_value(Mpc)
        res = resolution.to_value(Mpc)

        lc_distances = np.arange(d_at_redshift, dmax + res, res)

        return cls(lc_distances=lc_distances * Mpc, cosmo=cosmo, **kw)

    # ------------------------------------------------------------------
    # What a driver uses
    # ------------------------------------------------------------------
    def evolve(self, **changes: Any) -> Self:
        """Return a copy of the lightconer with some attributes changed.

        If ``lc_distances`` is changed, the redshifts are those of the new distances
        (unless ``lc_redshifts`` is given as well).
        """
        if "lc_distances" in changes:
            changes.setdefault("lc_redshifts", None)
        return attrs.evolve(self, **changes)

    def required_fields(self) -> tuple[str, ...]:
        """Return the names of the fields that are needed to make the quantities."""
        fields = [f for q in self.quantities for f in self.fields_for(q)]
        return tuple(dict.fromkeys(fields))

    def start(self, context: LightconeContext) -> BracketingSession:
        """Start a run, and return the session that will make the lightcone."""
        return BracketingSession(self, context)

    # ------------------------------------------------------------------
    # What a subclass defines
    # ------------------------------------------------------------------
    def lightcone_shape(self, context: LightconeContext) -> tuple[int, ...]:
        """Return the shape of the lightcone of each quantity (line of sight last)."""
        raise NotImplementedError

    def fields_for(self, quantity: str) -> tuple[str, ...]:
        """Return the names of the fields that a quantity is made from."""
        if quantity == "los_velocity":
            return self.los_velocity_fields
        return (quantity,)

    def prepare_slice(self, slice_: Slice, session: BracketingSession) -> Any:
        """Set up what is needed to make the quantities of one slice_.

        Called once for each slice, before its quantities are made. What is
        returned is passed on as ``slice_.state``, so it is shared by all the
        quantities of the slice (an interpolator onto the slice, for instance).
        Things that are worth keeping for longer than one slice can be put in
        ``session.cache``.
        """
        return None

    def make_slice(
        self, quantity: str, slice_: Slice, session: BracketingSession
    ) -> np.ndarray:
        """Make one quantity on one slice of the lightcone.

        By default, ``"los_velocity"`` is made by :meth:`make_los_velocity_slice`,
        and any other quantity is the field of that name, made by
        :meth:`make_field_slice`. A subclass can define other quantities here (and
        the fields they need in :meth:`fields_for`).
        """
        if quantity == "los_velocity":
            return self.make_los_velocity_slice(slice_, session)
        return self.make_field_slice(
            slice_.near[quantity],
            slice_.far[quantity],
            slice_,
            session,
            kind=self.interp_kinds.get(quantity, "mean"),
        )

    def make_field_slice(
        self,
        near: np.ndarray,
        far: np.ndarray,
        slice_: Slice,
        session: BracketingSession,
        kind: str = "mean",
    ) -> np.ndarray:
        """Make one slice of the lightcone from the boxes of a field around it.

        Parameters
        ----------
        near, far
            The boxes of the field in the snapshots on the near and far sides.
        slice_
            The slice (its distance, the distances of the snapshots, and what
            :meth:`prepare_slice` returned).
        session
            The session, for its ``context`` and ``cache``.
        kind
            The kind of interpolation in distance.
        """
        raise NotImplementedError

    def make_los_velocity_slice(
        self, slice_: Slice, session: BracketingSession
    ) -> np.ndarray:
        """Make one slice of the velocity along the line of sight.

        By default, the one field in ``los_velocity_fields``, made into a slice like
        any other field.
        """
        (field,) = self.los_velocity_fields
        return self.make_field_slice(
            slice_.near[field],
            slice_.far[field],
            slice_,
            session,
            kind=self.interp_kinds.get(field, "mean"),
        )


class BracketingSession:
    """The making of one lightcone, slice by slice, from consecutive snapshots.

    Parameters
    ----------
    lightconer
        The lightconer.
    context
        The cosmology and the boxes of the simulation.

    Attributes
    ----------
    shape
        The shape of the lightcone of each quantity.
    cache
        Where the lightconer keeps what it uses again (:class:`SessionCache`).
    slice_distances
        The comoving distances of the slices, in cells.
    """

    #: A new session only needs the latest snapshot to carry on after it.
    resume_depth = 1

    def __init__(self, lightconer: BracketingLightconer, context: LightconeContext):
        if lightconer.cosmo != context.cosmo:
            raise ValueError(
                "The cosmology of the lightconer is not that of the simulation. "
                "Give the lightconer the cosmology of the simulation (cosmo=...)."
            )
        self.lightconer = lightconer
        self.context = context
        self.cache = SessionCache()
        self.shape = tuple(lightconer.lightcone_shape(context))

        self._to_cells = pixel_scale(context.cell_size / pixel)
        self.slice_distances = lightconer.lc_distances.to_value(pixel, self._to_cells)

        self._previous = None
        self._previous_distance = None

    def distance(self, redshift: float) -> np.float64:
        """Return the comoving distance to a redshift, in cells."""
        distance = self.context.cosmo.comoving_distance(redshift)
        return np.float64(distance.to_value(pixel, self._to_cells))

    def slices_between(self, distance_a: float, distance_b: float) -> np.ndarray:
        """Return the indices of the slices between two distances (in cells).

        A slice at the smaller distance is included, one at the larger is not, so
        that every slice belongs to one pair of consecutive snapshots.
        """
        dmin, dmax = min(distance_a, distance_b), max(distance_a, distance_b)
        # Some tolerance at the near end: the first slice is often *exactly* at the
        # last snapshot, up to the rounding of the conversion from redshift.
        d = self.slice_distances
        return np.nonzero((d >= dmin * (1 - 1e-6)) & (d < dmax))[0]

    def ingest(self, snapshot: Any) -> Iterator[Chunk]:
        """Make the slices that lie between this snapshot and the one before it."""
        distance = self.distance(snapshot.redshift)
        far, far_distance = self._previous, self._previous_distance
        self._previous, self._previous_distance = snapshot, distance

        if far is None:
            # Nothing can be made from the first snapshot alone.
            return

        lightconer = self.lightconer
        self.cache.interval.clear()
        for i in self.slices_between(distance, far_distance):
            self.cache.slice.clear()
            this = Slice(
                index=int(i),
                distance=self.slice_distances[i],
                near=snapshot,
                far=far,
                near_distance=distance,
                far_distance=far_distance,
            )
            this = attrs.evolve(this, state=lightconer.prepare_slice(this, self))
            for quantity in lightconer.quantities:
                values = lightconer.make_slice(quantity, this, self)
                yield Chunk(quantity, (..., int(i)), values)

        self.cache.slice.clear()
        self.cache.interval.clear()

    def finish(self) -> tuple[Chunk, ...]:
        """Return nothing: every slice was returned by ``ingest``."""
        self._previous = None
        self.cache.run.clear()
        return ()


def build_lightcone(
    lightconer: Any,
    snapshots: Iterable[Any],
    context: LightconeContext,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Make lightcones from the snapshots of a simulation.

    Parameters
    ----------
    lightconer
        The lightconer.
    snapshots
        The snapshots (e.g. :class:`Snapshot`), from high to low redshift. They are
        used one at a time, so this can be a generator that makes (or reads) each
        snapshot when it is asked for.
    context
        The cosmology and the boxes of the simulation.
    dtype
        The type of the lightcones.

    Returns
    -------
    dict
        The lightcone of each quantity of the lightconer.
    """
    session = lightconer.start(context)
    lightcones = {
        q: np.zeros(session.shape, dtype=dtype) for q in lightconer.quantities
    }

    for snapshot in snapshots:
        for quantity, index, values in session.ingest(snapshot):
            lightcones[quantity][index] = values
    for quantity, index, values in session.finish():
        lightcones[quantity][index] = values

    return lightcones


# ----------------------------------------------------------------------------------
# Rectilinear lightcones
# ----------------------------------------------------------------------------------
@attrs.define(kw_only=True, slots=False)
class RectilinearLightconer(BracketingLightconer):
    """A lightconer whose lightcones are boxes: two transverse axes, and distance.

    Each slice of the lightcone is a slice of the (periodic) box, perpendicular to
    its last axis, interpolated linearly in comoving distance between the two
    snapshots around it. A slice that is one cell closer uses the slice of the box
    that is one cell closer, so that structures are continuous along the line of
    sight.

    Parameters
    ----------
    index_offset
        The index of the slice of the box that is used for the last (most distant)
        slice of the lightcone, plus one. By default, the number of slices of the
        lightcone.

    Other Parameters
    ----------------
    All the parameters of :class:`BracketingLightconer`.
    """

    index_offset: int = attrs.field()

    #: One component of an isotropic velocity field.
    los_velocity_rms_factor: ClassVar[float] = 1.0 / np.sqrt(3.0)

    @index_offset.default
    def _index_offset_default(self) -> int:
        return len(self.lc_distances)

    def lightcone_shape(self, context: LightconeContext) -> tuple[int, int, int]:
        """Return the shape of the lightcone: the box across, the slices along."""
        return (*context.box_shape[:2], len(self.lc_distances))

    def prepare_slice(self, slice_: Slice, session: BracketingSession) -> int:
        """Return the index of the slice of the box that is used for this slice_."""
        from_back = int(session.slice_distances.max() - slice_.distance + 1)
        return self.index_offset - from_back

    def make_field_slice(
        self,
        near: np.ndarray,
        far: np.ndarray,
        slice_: Slice,
        session: BracketingSession,
        kind: str = "mean",
    ) -> np.ndarray:
        """Interpolate one slice of the box between the two snapshots."""
        return slice_.interpolate(
            near.take(slice_.state, axis=2, mode="wrap"),
            far.take(slice_.state, axis=2, mode="wrap"),
            kind=kind,
        )


# ----------------------------------------------------------------------------------
# Angular lightcones
# ----------------------------------------------------------------------------------
def _rotation_eq(x, y):
    """Compare two rotations."""
    if x is None or y is None:
        return x is None and y is None

    return np.allclose(x.as_matrix(), y.as_matrix())


@attrs.define(kw_only=True, slots=False)
class AngularLightconer(BracketingLightconer):
    """A lightconer whose slices are sets of directions on the sky.

    Each slice of the lightcone is the field on a spherical shell around the
    observer, at the directions given by ``latitude`` and ``longitude``, so the
    lightcone has shape ``(n_pixels, n_slices)``. The (periodic) box is tiled to
    fill the shell, and sampled on it with cosmotile.

    Each slice is made from the two snapshots around it in one of two ways, which
    give the same result (to rounding): by sampling each box at the directions, and
    interpolating the samples in distance, or by interpolating the boxes in
    distance, cell by cell, and sampling the result. By default the cheaper way is
    used (:meth:`samples_first`).

    Each pixel can be the mean over several directions (sub-samples), so that
    pixels larger than the cells average the field instead of aliasing it. Then
    ``latitude`` and ``longitude`` hold ``n_subsamples * n_pixels`` directions,
    ordered sub-sample by sub-sample: the first ``n_pixels`` are the first direction
    of each pixel, and so on (``np.ravel`` of an array of shape
    ``(n_subsamples, n_pixels)``). The velocity along the line of sight is projected
    on each direction before the average. :meth:`healpix` makes a lightconer whose
    slices are HEALPix maps, with sub-sampled pixels if asked.

    Parameters
    ----------
    latitude, longitude
        The directions of the pixels, in radians (1D arrays of the same length).
        The longitude is between 0 and 2 pi.
    interpolation_order
        The order of the interpolation of the box on the shell (0, 1, 3 or 5).
    origin
        The position of the observer in the box, in cells. The corner of the box by
        default.
    rotation
        A rotation applied to the directions.
    tile_first
        True to sample (tile) the boxes before interpolating them in distance, False
        to interpolate them first, None (the default) to choose the cheaper for each
        slice_. It does not change the result, only the time taken.
    n_subsamples
        The number of directions averaged in each pixel.

    Other Parameters
    ----------------
    All the parameters of :class:`BracketingLightconer`.

    Notes
    -----
    The "mean_max" interpolation in distance is not linear in the boxes, so when
    the boxes are sampled first it is applied at the cells around each direction
    before they are weighted (or, for ``interpolation_order`` above 1, to the boxes
    first), which gives the same result as sampling the interpolated box.

    With sub-samples, a driver that applies corrections that are not linear in the
    fields (such as the velocity-gradient correction and the redshift-space
    distortions of 21cmFAST) applies them to the averaged pixels, not to each
    sub-sample.
    """

    latitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    longitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    interpolation_order: int = attrs.field(
        default=1, converter=int, validator=attrs.validators.in_([0, 1, 3, 5])
    )
    origin: Quantity = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    rotation: Rotation = attrs.field(
        default=None,
        eq=attrs.cmp_using(eq=_rotation_eq),
        validator=attrs.validators.optional(attrs.validators.instance_of(Rotation)),
    )
    tile_first: bool | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(bool)),
    )
    n_subsamples: int = attrs.field(default=1, converter=int)

    los_velocity_fields: ClassVar[tuple[str, ...]] = (
        "velocity_x",
        "velocity_y",
        "velocity_z",
    )

    @longitude.validator
    def _longitude_validator(self, attribute, value):
        if value.ndim != 1:
            raise ValueError("longitude must be 1-dimensional")
        if np.any(value < 0) or np.any(value > 2 * np.pi):
            raise ValueError("longitude must be in the range [0, 2pi]")
        if value.shape != self.latitude.shape:
            raise ValueError("longitude and latitude must have the same shape")

    @origin.default
    def _origin_default(self):
        return np.zeros(3) * pixel

    @n_subsamples.validator
    def _n_subsamples_validator(self, attribute, value):
        if value < 1:
            raise ValueError("n_subsamples must be at least 1")
        if self.longitude.size % value:
            raise ValueError(
                f"the {self.longitude.size} directions cannot be split into "
                f"{value} sub-samples per pixel"
            )

    @classmethod
    def like_rectilinear(
        cls,
        n_cells: int,
        cell_size: Quantity,
        match_at_z: float,
        cosmo: FLRW | None = None,
        **kw,
    ) -> Self:
        """Create an angular lightconer with the same pixel size as a rectilinear one.

        This is useful for comparing the two lightconer types.

        Parameters
        ----------
        n_cells
            The number of cells of the box along each transverse axis.
        cell_size
            The comoving size of a cell.
        match_at_z
            The redshift at which the angular lightconer should match the
            rectilinear one. It is also the lowest redshift of the lightcone.
        cosmo
            The cosmology to use. The default of the class if None.

        Other Parameters
        ----------------
        All other parameters are passed to :meth:`between_redshifts` (which needs at
        least ``max_redshift``).
        """
        if cosmo is None:
            cosmo = cls.default_cosmo
        box_len = (n_cells * cell_size).to_value(Mpc)
        box_size_radians = box_len / cosmo.comoving_distance(match_at_z).value

        lon = np.linspace(0, box_size_radians, n_cells)
        # This makes the X-values increasing from 0.
        lat = np.linspace(0, box_size_radians, n_cells)[::-1]

        lon, lat = (x.flatten() for x in np.meshgrid(lon, lat))

        origin_offset = -cosmo.comoving_distance(match_at_z).to(
            pixel, pixel_scale(cell_size / pixel)
        )
        origin = np.array([0, 0, origin_offset.value]) * origin_offset.unit
        rot = Rotation.from_euler("Y", -np.pi / 2)

        return cls.between_redshifts(
            min_redshift=match_at_z,
            resolution=cell_size,
            latitude=lat,
            longitude=lon,
            origin=origin,
            rotation=rot,
            cosmo=cosmo,
            **kw,
        )

    @classmethod
    def healpix(
        cls,
        nside: int,
        *,
        order: Literal["ring", "nested"] = "ring",
        pixels: Sequence[int] | np.ndarray | None = None,
        subsample_level: int = 0,
        **kw,
    ) -> Self:
        r"""Create an angular lightconer whose slices are HEALPix maps.

        Each slice is a map of the field on the pixels of a HEALPix grid, on a
        spherical shell around ``origin`` (the corner of the box by default). The
        lightcone has shape ``(n_pixels, n_slices)``, so ``lightcone[:, i]`` is the
        map of slice ``i``, in the pixel order given by ``order``. It can be used
        directly with healpy and other HEALPix codes.

        Parameters
        ----------
        nside
            The Nside of the maps (a power of 2).
        order
            The ordering of the pixels, "ring" or "nested".
        pixels
            The distinct indices (in ``order``) of the pixels to include, for a map of
            part of the sky (e.g. :func:`tuesday.core.lightcones.disc_pixels`). All
            ``12 * nside**2`` pixels by default.
        subsample_level
            If 0 (the default), each pixel takes the value of the field at its
            centre. If ``k > 0``, each pixel is the mean of the field at the centres
            of its ``4**k`` sub-pixels
            (:func:`tuesday.core.lightcones.subpixel_lonlat`). Pixels larger than the
            cells then average the field instead of aliasing it. The cost grows as
            ``4**k``.

        Other Parameters
        ----------------
        kw
            If ``min_redshift`` is given (and not None), all other parameters are
            passed to :meth:`between_redshifts` (e.g. ``min_redshift``,
            ``max_redshift``, ``resolution``, ``origin``, ``quantities``), otherwise
            to the constructor (with ``lc_distances`` or ``lc_redshifts``).

        Notes
        -----
        The box is tiled periodically. A shell at comoving distance :math:`\chi`
        crosses a box of side :math:`L` about :math:`\chi / L` times along each axis,
        so the same structures appear every :math:`L / \chi` radians on the sky. A
        patch smaller than that has no repeated structures.

        To match ``nside`` and ``subsample_level`` to the cells of the box, use
        :func:`tuesday.core.lightcones.nside_for_cell_size` and
        :func:`tuesday.core.lightcones.subsample_level_for_cell_size`.
        """
        longitude, latitude = subpixel_lonlat(
            nside, order=order, subsample_level=subsample_level, pixels=pixels
        )
        directions = {
            "latitude": latitude.to_value(units.rad).ravel(),
            "longitude": longitude.to_value(units.rad).ravel(),
            "n_subsamples": 4**subsample_level,
        }
        if kw.get("min_redshift") is not None:
            return cls.between_redshifts(**directions, **kw)
        kw.pop("min_redshift", None)
        return cls(**directions, **kw)

    @property
    def n_pixels(self) -> int:
        """The number of pixels of each slice_."""
        return self.longitude.size // self.n_subsamples

    def lightcone_shape(
        self, context: LightconeContext | None = None
    ) -> tuple[int, int]:
        """Return the shape of the lightcone: ``(n_pixels, n_slices)``."""
        return (self.n_pixels, len(self.lc_distances))

    def samples_first(self, n_cells: int) -> bool:
        """Whether a slice samples the boxes first, or interpolates them first.

        Parameters
        ----------
        n_cells
            The number of cells of each box.

        Returns
        -------
        bool
            ``tile_first`` if it is set, otherwise whether sampling first is cheaper
            for boxes of ``n_cells`` cells and all directions of a slice
            (:func:`tuesday.core.lightcones.samples_first`).
        """
        if self.tile_first is not None:
            return self.tile_first
        return samples_first(
            n_cells, self.longitude.size, order=self.interpolation_order
        )

    def slice_interpolator(self, distance: Quantity | float) -> partial:
        """Return cosmotile's interpolator onto the slice at a distance.

        Parameters
        ----------
        distance
            The comoving distance of the slice, in cells.

        Returns
        -------
        partial
            The interpolator (:func:`cosmotile.make_lightcone_slice_interpolator`),
            which averages each pixel over its sub-samples.
        """
        latitude, longitude = self.latitude, self.longitude
        if self.n_subsamples > 1:
            # cosmotile averages over the first axis.
            shape = (self.n_subsamples, self.n_pixels)
            latitude, longitude = latitude.reshape(shape), longitude.reshape(shape)

        return make_lightcone_slice_interpolator(
            latitude=latitude,
            longitude=longitude,
            distance_to_shell=Quantity(distance, pixel),
            interpolation_order=self.interpolation_order,
            origin=self.origin,
            rotation=self.rotation,
        )

    def prepare_slice(self, slice_: Slice, session: BracketingSession) -> partial:
        """Make the interpolator onto the slice, used for all its quantities."""
        return self.slice_interpolator(slice_.distance)

    def make_field_slice(
        self,
        near: np.ndarray,
        far: np.ndarray,
        slice_: Slice,
        session: BracketingSession,
        kind: str = "mean",
    ) -> np.ndarray:
        """Make one slice from the boxes of a field in the two snapshots around it."""
        return shell_slice(
            lambda a, b: _plain(slice_.interpolate(a, b, kind=kind)),
            near,
            far,
            slice_.state,
            linear=kind == "mean",
            sample_first=self.samples_first(np.size(near)),
        )

    def make_los_velocity_slice(
        self, slice_: Slice, session: BracketingSession
    ) -> np.ndarray:
        """Make one slice of the velocity along the line of sight.

        The three components of the velocity are projected on the direction of each
        pixel (of each sub-sample, with sub-samples). If the slice samples the boxes
        first, this is done for each snapshot, and the two are interpolated in
        distance. Otherwise the boxes are interpolated in distance first.
        """
        kind = self.interp_kinds.get("velocity", "mean")
        interpolator = slice_.state
        velocities = [
            [snapshot[field] for field in self.los_velocity_fields]
            for snapshot in (slice_.near, slice_.far)
        ]

        if kind == "mean" and self.samples_first(np.size(velocities[0][0])):
            # cosmotile needs Quantities here, since the pixel coordinates are.
            los_near, los_far = make_lightcone_slice_vector_field(
                [
                    [box << units.dimensionless_unscaled for box in boxes]
                    for boxes in velocities
                ],
                interpolator,
            )
            return _plain(slice_.interpolate(los_near, los_far, kind=kind))

        boxes = [
            slice_.interpolate(near, far, kind=kind) << units.dimensionless_unscaled
            for near, far in zip(*velocities, strict=True)
        ]
        return _plain(next(make_lightcone_slice_vector_field([boxes], interpolator)))
