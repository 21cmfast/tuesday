"""An angular lightconer for 21cmFAST that chooses how to make each slice.

21cmFAST 4.3's :class:`py21cmfast.AngularLightconer` makes each slice of an angular
lightcone by interpolating the two coeval boxes around the slice in redshift, cell by
cell, and then sampling the interpolated box at the slice's directions with cosmotile.
For a large box and few directions (e.g. a ring of directions through a 1500^3 box),
almost all of that work is the interpolation of cells that are never sampled.
Sampling each box at the slice's directions first and interpolating the samples gives
the same slice, to rounding, at a cost set by the number of directions.

:class:`FastAngularLightconer` makes each slice in whichever of the two ways is
cheaper (:func:`tuesday.core.lightcones.samples_first`), unless ``tile_first`` says
which to use. Both ways sample the boxes with cosmotile's slice interpolator
(:func:`cosmotile.make_lightcone_slice_interpolator`), as 21cmFAST does.

The lightconer can also average each pixel over several directions (sub-samples), so
that pixels larger than the cells average the field instead of aliasing it. cosmotile
(0.3 and later) does the average, and projects the velocity on each direction before
it.
:func:`healpix_lightconer` makes one whose slices are HEALPix maps, of the full sky or
of a patch, with each pixel averaged over its sub-pixels if asked. The same is proposed
for 21cmFAST itself (branch ``feat/healpix-lightconer``).

The loop over slices and quantities, the redshift interpolation and the
line-of-sight convention are 21cmFAST's own: the lightconer only changes the steps of
:meth:`py21cmfast.lightconers.Lightconer.make_lightcone_slices` that it calls for each
slice. It works with :func:`py21cmfast.run_lightcone` and with
:func:`tuesday.simulators.py21cmfast.lightcones.construct_lightcone_from_cache`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import attrs
import numpy as np
from astropy import units
from astropy.units import Quantity
from cosmotile import (
    make_lightcone_slice_interpolator,
    make_lightcone_slice_vector_field,
)
from py21cmfast.lightconers import AngularLightconer
from py21cmfast.wrapper.inputs import SimulationOptions

from tuesday.core.lightcones import (
    sample_then_interpolate,
    samples_first,
    subpixel_lonlat,
)
from tuesday.core.lightcones.slices import _plain


class _Deferred:
    """A coeval box whose slice is made by sampling it first."""

    __slots__ = ("box",)

    def __init__(self, box: np.ndarray):
        self.box = box


class _Pair:
    """Two coeval boxes and the interpolation between them, to be sampled first."""

    __slots__ = ("box_a", "box_b", "interpolate", "linear")

    def __init__(
        self,
        box_a: np.ndarray,
        box_b: np.ndarray,
        interpolate: Callable[[np.ndarray, np.ndarray], np.ndarray],
        linear: bool,
    ):
        self.box_a = box_a
        self.box_b = box_b
        self.interpolate = interpolate
        self.linear = linear


@attrs.define(kw_only=True, slots=False)
class FastAngularLightconer(AngularLightconer):
    """An angular lightconer that makes each slice in the cheaper of two ways.

    It takes the parameters of :class:`py21cmfast.AngularLightconer` (``latitude``,
    ``longitude``, ``origin``, ``rotation``, ``interpolation_order``, and those of the
    lightcone's distances and quantities) and makes the same lightcones, to rounding.
    Each slice is made from the two coeval boxes around it either by sampling the
    boxes at the slice's directions and interpolating the samples in redshift, or by
    interpolating the boxes and sampling the result. By default it takes the cheaper
    way for the size of the boxes (:meth:`samples_first`). Both ways sample with
    cosmotile's slice interpolator.

    Each pixel can be the mean over several directions (sub-samples). Then
    ``latitude`` and ``longitude`` hold ``n_subsamples * n_pixels`` directions,
    ordered sub-sample by sub-sample: the first ``n_pixels`` are the first direction
    of each pixel, and so on (``np.ravel`` of an array of shape
    ``(n_subsamples, n_pixels)``, which is what cosmotile takes). The line-of-sight
    velocity is projected on each direction before the average.

    Parameters
    ----------
    n_subsamples
        The number of directions averaged in each pixel.
    tile_first
        True to sample (tile) the boxes before interpolating them, False to
        interpolate them first, None (the default) to choose the cheaper for each slice
        (:meth:`samples_first`). It does not change the result, only the time taken.
        The name is that of 21cmFAST's proposed ``AngularLightconer.tile_first``.

    Notes
    -----
    With ``interpolation_order`` 3 or 5, slices interpolate the boxes first unless
    ``tile_first=True``, since sampling first is not cheaper for splines
    (:func:`tuesday.core.lightcones.samples_first`).

    With sub-samples, :func:`py21cmfast.run_lightcone` applies the velocity-gradient
    (dv/dr) correction and the RSDs to the averaged pixels, not to each sub-sample.
    For the brightness temperature, which is not linear in the velocity gradient,
    that differs from averaging the corrected sub-samples.

    The lightconer is registered with 21cmFAST under its class name, so a lightcone
    saved with it can be read back once this module is imported.
    """

    n_subsamples: int = attrs.field(default=1, converter=int)
    tile_first: bool | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(bool)),
    )

    @n_subsamples.validator
    def _n_subsamples_validator(self, attribute, value):
        if value < 1:
            raise ValueError("n_subsamples must be at least 1")
        if self.longitude.size % value:
            raise ValueError(
                f"the {self.longitude.size} directions cannot be split into "
                f"{value} sub-samples per pixel"
            )

    @property
    def n_pixels(self) -> int:
        """The number of pixels of each slice."""
        return self.longitude.size // self.n_subsamples

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int]:
        """Get the shape of the lightcone: ``(n_pixels, n_slices)``."""
        return (self.n_pixels, len(self.lc_redshifts))

    def samples_first(self, n_cells: int) -> bool:
        """Whether a slice samples the boxes first, or interpolates them first.

        Parameters
        ----------
        n_cells
            The number of cells of each coeval box.

        Returns
        -------
        bool
            ``tile_first`` if it is set, otherwise whether sampling first is cheaper
            for boxes of ``n_cells`` cells and all directions of a slice (sub-samples
            included; :func:`tuesday.core.lightcones.samples_first`).
        """
        if self.tile_first is not None:
            return self.tile_first
        return samples_first(
            n_cells, self.longitude.size, order=self.interpolation_order
        )

    def slice_interpolator(self, lcd: Quantity) -> partial:
        """Return cosmotile's interpolator onto the slice at a distance.

        The interpolator of the last distance is kept, since 21cmFAST asks for it once
        per quantity of each slice.

        Parameters
        ----------
        lcd
            The comoving distance of the slice, in pixels.

        Returns
        -------
        partial
            The interpolator (:func:`cosmotile.make_lightcone_slice_interpolator`).
        """
        cached = getattr(self, "_slice_interpolator", None)
        if cached is not None and cached[0] == lcd:
            return cached[1]
        shape = (self.n_subsamples, self.n_pixels)
        interpolator = make_lightcone_slice_interpolator(
            latitude=self.latitude.reshape(shape),
            longitude=self.longitude.reshape(shape),
            distance_to_shell=lcd,
            interpolation_order=self.interpolation_order,
            origin=self.origin,
            rotation=self.rotation,
        )
        self._slice_interpolator = (lcd, interpolator)
        return interpolator

    # The four steps of 21cmFAST's Lightconer.make_lightcone_slices, for each slice.

    def coeval_subselect(
        self, lcd: Quantity, coeval: np.ndarray, coeval_res: Quantity
    ) -> np.ndarray | _Deferred:
        """Select the part of a coeval box needed for a slice.

        If the slice samples the boxes first (:meth:`samples_first`), the box is passed
        on unchanged, to be sampled in :meth:`construct_lightcone`.
        """
        if self.samples_first(np.size(coeval)):
            return _Deferred(coeval)
        return super().coeval_subselect(lcd, coeval, coeval_res)

    def redshift_interpolation(
        self,
        dc: Quantity,
        coeval_a: np.ndarray | _Deferred,
        coeval_b: np.ndarray | _Deferred,
        dc_a: Quantity,
        dc_b: Quantity,
        kind: str = "mean",
    ) -> np.ndarray | _Pair:
        """Interpolate two coeval boxes to the distance of a slice.

        If the slice samples the boxes first, the interpolation (21cmFAST's
        :meth:`~py21cmfast.lightconers.Lightconer.redshift_interpolation`) is kept with
        the two boxes, and applied to their samples in :meth:`construct_lightcone`.
        """
        if not isinstance(coeval_a, _Deferred):
            return super().redshift_interpolation(
                dc, coeval_a, coeval_b, dc_a, dc_b, kind=kind
            )
        interpolate = super().redshift_interpolation

        def interpolate_samples(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _plain(interpolate(dc, a, b, dc_a, dc_b, kind=kind))

        return _Pair(coeval_a.box, coeval_b.box, interpolate_samples, kind == "mean")

    def construct_lightcone(self, lcd: Quantity, box: np.ndarray | _Pair) -> np.ndarray:
        """Make a slice from an interpolated box, or from two boxes, sampling first.

        Two boxes are sampled with cosmotile and their samples interpolated
        (:func:`tuesday.core.lightcones.sample_then_interpolate`). The "mean_max"
        interpolation (used for ``z_reion``) is not linear in the boxes, so it is then
        applied at the cells around each direction before they are weighted (or, for
        splines, to the boxes first), which gives the same result as sampling the
        interpolated box.
        """
        interpolator = self.slice_interpolator(lcd)
        if isinstance(box, _Pair):
            return sample_then_interpolate(
                box.interpolate, box.box_a, box.box_b, interpolator, linear=box.linear
            )
        return _plain(interpolator(box))

    def construct_los_velocity_lightcone(
        self, lcd: Quantity, velocities: Sequence[np.ndarray | _Pair]
    ) -> np.ndarray:
        """Make a slice of the line-of-sight velocity from the three components.

        cosmotile projects the velocity on the line of sight of each direction
        (:func:`cosmotile.make_lightcone_slice_vector_field`). When sampling first,
        this is done for each box, and the two projections are interpolated.
        """
        interpolator = self.slice_interpolator(lcd)
        if not isinstance(velocities[0], _Pair):
            return _plain(
                next(make_lightcone_slice_vector_field([velocities], interpolator))
            )
        if not all(v.linear for v in velocities):
            raise ValueError("the velocity can only be sampled first with kind 'mean'")
        # cosmotile multiplies the samples by the coordinates, which are Quantities
        los_a, los_b = make_lightcone_slice_vector_field(
            [
                [v.box_a << units.dimensionless_unscaled for v in velocities],
                [v.box_b << units.dimensionless_unscaled for v in velocities],
            ],
            interpolator,
        )
        return velocities[0].interpolate(_plain(los_a), _plain(los_b))


def healpix_lightconer(
    nside: int,
    *,
    order: Literal["ring", "nested"] = "ring",
    pixels: Sequence[int] | np.ndarray | None = None,
    subsample_level: int = 0,
    **kw,
) -> FastAngularLightconer:
    r"""Create an angular lightconer whose slices are HEALPix maps.

    Each slice is a map of the field on the pixels of a HEALPix grid, on a spherical
    shell around ``origin`` (the corner of the box by default). The lightcone has shape
    ``(n_pixels, n_slices)``, so ``lightcone[:, i]`` is the map of slice ``i``, in the
    pixel order given by ``order``. It can be used directly with healpy and other
    HEALPix codes.

    Parameters
    ----------
    nside
        The Nside of the maps (a power of 2).
    order
        The ordering of the pixels, "ring" or "nested".
    pixels
        The distinct indices (in ``order``) of the pixels to include, for a map of part
        of the sky (e.g. :func:`tuesday.core.lightcones.disc_pixels`). All
        ``12 * nside**2`` pixels by default.
    subsample_level
        If 0 (the default), each pixel takes the value of the field at its centre. If
        ``k > 0``, each pixel is the mean of the field at the centres of its ``4**k``
        sub-pixels (:func:`tuesday.core.lightcones.subpixel_lonlat`). Pixels larger
        than the cells then average the field instead of aliasing it. The cost grows
        as ``4**k``.

    Other Parameters
    ----------------
    kw
        If ``min_redshift`` is given (and not None), all other parameters are passed to
        :meth:`~py21cmfast.lightconers.Lightconer.between_redshifts` (e.g.
        ``min_redshift``, ``max_redshift``, ``resolution``, ``origin``,
        ``quantities``), otherwise to
        :class:`FastAngularLightconer` (with ``lc_distances`` or
        ``lc_redshifts``).

    Returns
    -------
    FastAngularLightconer
        The lightconer.

    Notes
    -----
    The box is tiled periodically. A shell at comoving distance :math:`\chi` crosses a
    box of side :math:`L` about :math:`\chi / L` times along each axis, so the same
    structures appear every :math:`L / \chi` radians on the sky. A patch smaller than
    that has no repeated structures.

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
        return FastAngularLightconer.between_redshifts(**directions, **kw)
    kw.pop("min_redshift", None)
    return FastAngularLightconer(**directions, **kw)
