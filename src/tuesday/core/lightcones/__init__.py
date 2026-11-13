"""Algorithms for building lightcones from simulation boxes."""

__all__ = [
    "CELLS_PER_SAMPLE",
    "degrade",
    "disc_pixels",
    "interpolate_then_sample",
    "n_samples",
    "nside_for_cell_size",
    "sample_cellwise",
    "sample_then_interpolate",
    "samples_first",
    "shell_slice",
    "subpixel_lonlat",
    "subsample_level_for_cell_size",
]

from .healpix import (
    degrade,
    disc_pixels,
    nside_for_cell_size,
    subpixel_lonlat,
    subsample_level_for_cell_size,
)
from .slices import (
    CELLS_PER_SAMPLE,
    interpolate_then_sample,
    n_samples,
    sample_cellwise,
    sample_then_interpolate,
    samples_first,
    shell_slice,
)
