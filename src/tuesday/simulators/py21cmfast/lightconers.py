"""An angular lightconer for 21cmFAST whose pixels can average several directions.

21cmFAST's :class:`py21cmfast.AngularLightconer` (4.3.2 and later) makes each slice of
an angular lightcone in whichever of two ways is cheaper. Both give the same slice, to
rounding: interpolating the two coeval boxes around the slice in redshift, cell by
cell, and sampling the result at the slice's directions with cosmotile, or sampling
each box at the directions first and interpolating the samples (``tile_first``).

:class:`FastAngularLightconer` makes the same choice
(:func:`tuesday.core.lightcones.samples_first`) and samples first with
:func:`tuesday.core.lightcones.sample_then_interpolate`. In addition, each pixel can
be the mean over several directions (sub-samples), so that pixels larger than the cells
average the field instead of aliasing it, and ``tile_first=True`` samples first at
any interpolation order. cosmotile (0.3 and later) does the average, and projects the
velocity on each direction before it.
:meth:`FastAngularLightconer.healpix` makes one whose slices are HEALPix maps, of the
full sky or of a patch, with each pixel averaged over its sub-pixels if asked.

The loop over slices and quantities, the redshift interpolation and the
line-of-sight convention are 21cmFAST's own: the lightconer only changes how each
slice is made (:meth:`~py21cmfast.lightconers.Lightconer.interpolate_slice`) and the
interpolator of each slice. It works with :func:`py21cmfast.run_lightcone` and with
:func:`tuesday.simulators.py21cmfast.lightcones.construct_lightcone_from_cache`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal

import attrs
import numpy as np
from astropy import units
from astropy.units import Quantity
from cosmotile import make_lightcone_slice_interpolator
from py21cmfast.lightconers import AngularLightconer
from py21cmfast.wrapper.inputs import SimulationOptions

from tuesday.core.lightcones import (
    sample_then_interpolate,
    samples_first,
    subpixel_lonlat,
)
from tuesday.core.lightcones.slices import _plain


@attrs.define(kw_only=True, slots=False)
class FastAngularLightconer(AngularLightconer):
    """An angular lightconer whose pixels can average several directions.

    It takes the parameters of :class:`py21cmfast.AngularLightconer` (``latitude``,
    ``longitude``, ``origin``, ``rotation``, ``interpolation_order``, ``tile_first``,
    and those of the lightcone's distances and quantities) and makes the same
    lightcones, to rounding. Each slice is made from the two coeval boxes around it
    either by sampling the boxes at the slice's directions and interpolating the
    samples in redshift, or by interpolating the boxes and sampling the result. By
    default it takes the cheaper way for the size of the boxes (:meth:`samples_first`).
    Both ways sample with cosmotile's slice interpolator.

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

    Notes
    -----
    ``tile_first`` is 21cmFAST's: True to sample (tile) the boxes before interpolating
    them, False to interpolate them first, None (the default) to choose the cheaper for
    each slice. It does not change the result, only the time taken. Unlike in
    21cmFAST, ``tile_first=True`` also samples first for ``interpolation_order`` 3 and
    5. Without it, those orders interpolate first, since sampling first is not cheaper
    for splines (:func:`tuesday.core.lightcones.samples_first`).

    With sub-samples, :func:`py21cmfast.run_lightcone` applies the velocity-gradient
    (dv/dr) correction and the RSDs to the averaged pixels, not to each sub-sample.
    For the brightness temperature, which is not linear in the velocity gradient,
    that differs from averaging the corrected sub-samples.

    The lightconer is registered with 21cmFAST under its class name, so a lightcone
    saved with it can be read back once this module is imported.
    """

    n_subsamples: int = attrs.field(default=1, converter=int)

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
    def healpix(
        cls,
        nside: int,
        *,
        order: Literal["ring", "nested"] = "ring",
        pixels: Sequence[int] | np.ndarray | None = None,
        subsample_level: int = 0,
        **kw,
    ) -> FastAngularLightconer:
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
            passed to :meth:`~py21cmfast.lightconers.Lightconer.between_redshifts`
            (e.g. ``min_redshift``, ``max_redshift``, ``resolution``, ``origin``,
            ``quantities``), otherwise to the constructor (with ``lc_distances`` or
            ``lc_redshifts``).

        Returns
        -------
        FastAngularLightconer
            The lightconer.

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
        """The number of pixels of each slice."""
        return self.longitude.size // self.n_subsamples

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int]:
        """Get the shape of the lightcone: ``(n_pixels, n_slices)``."""
        return (self.n_pixels, len(self.lc_redshifts))

    def samples_first(self, n_cells: int, interpolator: partial | None = None) -> bool:
        """Whether a slice samples the boxes first, or interpolates them first.

        Parameters
        ----------
        n_cells
            The number of cells of each coeval box.
        interpolator
            The interpolator of the slice, which 21cmFAST passes. The number of
            directions is that of the lightconer, sub-samples included.

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

    def slice_interpolator(self, lcd: Quantity) -> partial:
        """Return cosmotile's interpolator onto the slice at a distance.

        Parameters
        ----------
        lcd
            The comoving distance of the slice, in pixels.

        Returns
        -------
        partial
            The interpolator (:func:`cosmotile.make_lightcone_slice_interpolator`),
            which averages each pixel over its sub-samples.
        """
        return self._get_interpolator(lcd)

    def _get_interpolator(self, lcd: Quantity) -> partial:
        # 21cmFAST asks for the interpolator of a slice with this private method, once
        # per quantity, and makes it from the flat directions. Here cosmotile gets them
        # as an array of shape (n_subsamples, n_pixels), so that it averages each pixel
        # over its sub-samples. The interpolator of the last distance is kept.
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

    def interpolate_slice(
        self,
        lcd: Quantity,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: Quantity,
        dc_b: Quantity,
        coeval_res: Quantity,
        kind: str = "mean",
    ) -> np.ndarray:
        """Make one slice of a field from the two coeval boxes around it.

        If the slice samples the boxes first (:meth:`samples_first`), the boxes are
        sampled with cosmotile and the samples interpolated in redshift
        (:func:`tuesday.core.lightcones.sample_then_interpolate`). The "mean_max"
        interpolation (used for ``z_reion``) is not linear in the boxes, so it is then
        applied at the cells around each direction before they are weighted (or, for
        splines, to the boxes first), which gives the same result as sampling the
        interpolated box. Otherwise the boxes are interpolated first, as in 21cmFAST.
        The line-of-sight velocity is made by 21cmFAST's
        :meth:`~py21cmfast.lightconers.AngularLightconer.interpolate_los_velocity_slice`,
        which uses :meth:`samples_first` and the interpolator of this lightconer.
        """
        if not self.samples_first(np.size(coeval_a)):
            return super().interpolate_slice(
                lcd, coeval_a, coeval_b, dc_a, dc_b, coeval_res, kind=kind
            )

        def interpolate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _plain(self.redshift_interpolation(lcd, a, b, dc_a, dc_b, kind=kind))

        return sample_then_interpolate(
            interpolate,
            coeval_a,
            coeval_b,
            self._get_interpolator(lcd),
            linear=kind == "mean",
        )
