"""An angular lightconer for 21cmFAST that chooses how to make each slice.

21cmFAST 4.3's :class:`py21cmfast.AngularLightconer` makes each slice of an angular
lightcone by interpolating the two coeval boxes around the slice in redshift, cell by
cell, and then sampling the interpolated box at the slice's directions. For a large
box and few directions (e.g. a ring of directions through a 1500^3 box), almost all
of that work is the interpolation of cells that are never sampled. Sampling each box
at the slice's directions first and interpolating the samples gives the same slice,
to rounding, at a cost set by the number of directions.

:class:`FastAngularLightconer` makes each slice in whichever of the two ways is
cheaper, with :func:`tuesday.core.lightcones.shell_slice`
(:func:`~tuesday.core.lightcones.sample_then_interpolate` or
:func:`~tuesday.core.lightcones.interpolate_then_sample`), unless ``tile_first`` says
which to use. The same is proposed for 21cmFAST itself (branch
``perf/angular-lightcone-sample-first``, with the same ``tile_first`` parameter).

The lightconer can also average each pixel over several directions (sub-samples), so
that pixels larger than the cells average the field instead of aliasing it.
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

from collections.abc import Sequence
from functools import cached_property
from typing import Literal

import attrs
import numpy as np
from astropy import units
from astropy.units import Quantity
from cosmotile import transform_to_pixel_coords
from py21cmfast.lightconers import AngularLightconer
from py21cmfast.wrapper.inputs import SimulationOptions

from tuesday.core.lightcones import (
    average_subsamples,
    line_of_sight_component,
    samples_first,
    shell_slice,
    subpixel_lonlat,
)


class _Deferred:
    """A coeval box whose slice is made in ``redshift_interpolation``."""

    __slots__ = ("box",)

    def __init__(self, box: np.ndarray):
        self.box = box


def _plain(values: np.ndarray | Quantity) -> np.ndarray:
    """Return a dimensionless array or Quantity as a plain array."""
    return Quantity(values, units.dimensionless_unscaled).value


@attrs.define(kw_only=True, slots=False)
class FastAngularLightconer(AngularLightconer):
    """An angular lightconer that makes each slice in the cheaper of two ways.

    It takes the parameters of :class:`py21cmfast.AngularLightconer` (``latitude``,
    ``longitude``, ``origin``, ``rotation``, ``interpolation_order``, and those of the
    lightcone's distances and quantities) and makes the same lightcones, to rounding.
    Each slice is made from the two coeval boxes around it either by sampling the
    boxes at the slice's directions and interpolating the samples in redshift, or by
    interpolating the boxes and sampling the result. By default it takes the cheaper
    way for the size of the boxes (:meth:`samples_first`).

    Each pixel can be the mean over several directions (sub-samples). Then
    ``latitude`` and ``longitude`` hold ``n_subsamples * n_pixels`` directions,
    ordered sub-sample by sub-sample: the first ``n_pixels`` are the first direction
    of each pixel, and so on (``np.ravel`` of an array of shape
    ``(n_subsamples, n_pixels)``). The line-of-sight velocity is projected on each
    direction before the average.

    Parameters
    ----------
    n_subsamples
        The number of directions averaged in each pixel.
    tile_first
        True to sample (tile) the boxes before interpolating them, False to
        interpolate them first, None (the default) to choose the cheaper for each slice
        (:meth:`samples_first`). It does not change the result, only the time taken.
        The name is that of 21cmFAST's proposed ``AngularLightconer.tile_first``.
    threads
        The number of threads that sample the boxes.
    chunk_size
        The number of directions sampled at a time, which sets the memory taken by
        their coordinates.

    Notes
    -----
    With ``interpolation_order`` 3 or 5, slices are made by 21cmFAST's own
    ``AngularLightconer``, which interpolates the boxes first, since spline
    interpolation pre-filters the whole box.

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
    threads: int = attrs.field(default=1, converter=int, eq=False)
    chunk_size: int = attrs.field(default=2**20, converter=int, eq=False)

    @n_subsamples.validator
    def _n_subsamples_validator(self, attribute, value):
        if value < 1:
            raise ValueError("n_subsamples must be at least 1")
        if self.longitude.size % value:
            raise ValueError(
                f"the {self.longitude.size} directions cannot be split into "
                f"{value} sub-samples per pixel"
            )

    @threads.validator
    def _threads_validator(self, attribute, value):
        if value < 1:
            raise ValueError("threads must be at least 1")

    @chunk_size.validator
    def _chunk_size_validator(self, attribute, value):
        if value < 1:
            raise ValueError("chunk_size must be at least 1")

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
            included; :func:`tuesday.core.lightcones.samples_first`). False for
            ``interpolation_order`` above 1.
        """
        if self.interpolation_order > 1:
            return False
        if self.tile_first is not None:
            return self.tile_first
        return samples_first(
            n_cells, self.longitude.size, order=self.interpolation_order
        )

    @cached_property
    def _directions(self) -> np.ndarray:
        """Unit vectors of the (rotated) directions, of shape ``(3, n)``."""
        return transform_to_pixel_coords(
            comoving_radius=1.0,
            latitude=self.latitude,
            longitude=self.longitude,
            origin=None,
            rotation=self.rotation,
        )

    # The four steps of 21cmFAST's Lightconer.make_lightcone_slices, for each slice.

    def coeval_subselect(
        self, lcd: Quantity, coeval: np.ndarray, coeval_res: Quantity
    ) -> np.ndarray | _Deferred:
        """Select the part of a coeval box needed for a slice.

        For ``interpolation_order`` 0 and 1, the box is passed on unchanged, to be
        made into the slice in :meth:`redshift_interpolation`.
        """
        if self.interpolation_order <= 1:
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
    ) -> np.ndarray:
        """Interpolate two coeval boxes to the distance of a slice, on the slice.

        For ``interpolation_order`` 0 and 1, this returns the slice itself, made by
        :func:`tuesday.core.lightcones.shell_slice` with 21cmFAST's
        :meth:`~py21cmfast.lightconers.Lightconer.redshift_interpolation`, either
        sampling first or interpolating first (:meth:`samples_first`). The "mean_max"
        interpolation (used for ``z_reion``) is not linear in the boxes, so when
        sampling first with order 1 it is applied at the eight cells around each
        direction before they are weighted, which gives the same result as sampling
        the interpolated box.
        """
        if not isinstance(coeval_a, _Deferred):
            return super().redshift_interpolation(
                dc, coeval_a, coeval_b, dc_a, dc_b, kind=kind
            )

        interpolate = super().redshift_interpolation
        return shell_slice(
            lambda a, b: _plain(interpolate(dc, a, b, dc_a, dc_b, kind=kind)),
            coeval_a.box,
            coeval_b.box,
            self._directions,
            Quantity(dc, units.pixel).value,
            origin=Quantity(self.origin, units.pixel).value,
            order=self.interpolation_order,
            linear=kind == "mean",
            sample_first=self.samples_first(np.size(coeval_a.box)),
            threads=self.threads,
            chunk_size=self.chunk_size,
        )

    def construct_lightcone(self, lcd: Quantity, box: np.ndarray) -> np.ndarray:
        """Make a slice from its samples, or from an interpolated box.

        Samples (1D) are averaged over the sub-samples of each pixel. A box (3D,
        ``interpolation_order`` above 1) is first sampled by 21cmFAST.
        """
        if np.ndim(box) != 1:
            box = super().construct_lightcone(lcd, box)
        return average_subsamples(_plain(box), self.n_subsamples)

    def construct_los_velocity_lightcone(
        self, lcd: Quantity, velocities: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Make a slice of the line-of-sight velocity from the three components.

        Samples (1D) are projected on the line of sight of each direction
        (:func:`tuesday.core.lightcones.line_of_sight_component`). Boxes (3D,
        ``interpolation_order`` above 1) are sampled and projected by 21cmFAST. The
        result is then averaged over the sub-samples of each pixel.
        """
        if np.ndim(velocities[0]) == 1:
            los = line_of_sight_component(
                np.array([_plain(v) for v in velocities]), self._directions
            )
        else:
            los = super().construct_los_velocity_lightcone(lcd, velocities)
        return average_subsamples(_plain(los), self.n_subsamples)


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
        If ``min_redshift`` is given, all other parameters are passed to
        :meth:`~py21cmfast.lightconers.Lightconer.between_redshifts` (e.g.
        ``min_redshift``, ``max_redshift``, ``resolution``, ``origin``,
        ``quantities``, ``threads``), otherwise to
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
    if "min_redshift" in kw:
        return FastAngularLightconer.between_redshifts(**directions, **kw)
    return FastAngularLightconer(**directions, **kw)
