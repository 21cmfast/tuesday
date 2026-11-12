"""A sub-package for typical plotting routines such as power spectra."""

__all__ = [
    "eor_colormap",
    "plot_1d_power_spectrum_k",
    "plot_1d_power_spectrum_z",
    "plot_2d_power_spectrum",
    "plot_angular_lightcone_slice",
    "plot_coeval_slice",
    "plot_healpix_map",
    "plot_lightcone_slice",
    "plot_pdf",
    "plot_power_spectrum",
]

from .angular import (
    eor_colormap,
    plot_angular_lightcone_slice,
    plot_healpix_map,
)
from .powerspectra import (
    plot_1d_power_spectrum_k,
    plot_1d_power_spectrum_z,
    plot_2d_power_spectrum,
    plot_power_spectrum,
)
from .sliceplots import plot_coeval_slice, plot_lightcone_slice, plot_pdf
