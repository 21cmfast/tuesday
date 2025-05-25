"""Module for LC and coeval sliceplots."""
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology.units import littleh
from astropy import units as un
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

def plot_slice(
    slice: un.Quantity,
    xaxis: un.Quantity,
    yaxis: un.Quantity,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    log: tuple[bool, bool, bool] = (False, False, False),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot a 2D slice of the data."""
    if ax is None:
        _, ax = plt.subplots()
    cmap_kwargs = {}
    if vmin is None:
        if log[2]:
            cmap_kwargs["vmin"] = np.nanpercentile(np.log10(slice.value), 5)
        else:
            cmap_kwargs["vmin"] = np.nanpercentile(slice.value, 5)
    if vmax is None:
        if log[2]:
            cmap_kwargs["vmax"] = np.nanpercentile(np.log10(slice.value), 95)
        else:
            cmap_kwargs["vmax"] = np.nanpercentile(slice.value, 95)
    if log[2]:
        cmap_kwargs = {}
        cmap_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(xaxis,yaxis,slice.T, cmap=cmap, shading='auto')

    if log[0]:
        ax.set_xscale('log')
    if log[1]:
        ax.set_yscale('log')
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    cbar = plt.colorbar(im, ax=ax, label=clabel)

    return ax

def get_slice_index(
    zmin: float | None = None,
    zmax: float | None = None,
    idx: int | None = 0,
) -> un.Quantity:
    """Get the slice index for a given redshift range."""
    def slice_index(lc: un.Quantity, redshift: np.ndarray | un.Quantity) -> un.Quantity:
        """Get the slice index for a given redshift range."""
        if zmin is None:
            idx_min = 0
        else:
            idx_min = np.argmin(np.abs(redshift - zmin))
        if zmax is None:
            idx_max = lc.shape[-1]
        else:
            idx_max = np.argmin(np.abs(redshift - zmax)) + 1
        
        return lc[idx,:,idx_min:idx_max]
    return slice_index


def plot_lightcone_slice(
    lightcone: un.Quantity,
    box_length: un.Quantity,
    redshift: np.ndarray | un.Quantity,
    title: str | None = None,
    xlabel: str | None = "Redshift",
    ylabel: str | None = "Distance",
    clabel: str | None = None,
    cmap: str = "eor",
    logx: bool = False,
    logy: bool = False,
    logc: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    smooth: bool | float = False,
    slicing_fnc: callable | None = None,
) -> plt.Axes:
    """Plot a slice from a lightcone of shape (HII_DIM, HII_DIM, n_z)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(20,4))
    if slicing_fnc is not None:
        lightcone = slicing_fnc(lightcone, redshift)
    else:
        lightcone = get_slice_index(zmin=zmin, zmax=zmax, idx=0)(lightcone, redshift)
    if smooth:
        if isinstance(smooth, bool):
            smooth = 1.0
        lightcone = gaussian_filter(lightcone.value, sigma=smooth) * lightcone.unit
    yaxis = np.linspace(0, box_length, lightcone.shape[0])

    if clabel is None:
        if lightcone.unit.physical_type == un.get_physical_type("temperature"):
            clabel = f"Brightness Temperature [{lightcone.unit}]"
        elif lightcone.unit.is_equivalent(un.dimensionless_unscaled):
            clabel = "Density Contrast"
        else:
            clabel = f"{lightcone.unit.physical_type} [{lightcone.unit}]"
    return plot_slice(lightcone, 
                      redshift, 
                      yaxis,
                      vmin=vmin,
                      vmax=vmax,
                      log = [logx,logy,logc],
                      title=title, 
                      xlabel="Redshift" if xlabel is None else xlabel, 
                      ylabel=ylabel + " [" + str(box_length.unit) + "]" if ylabel is None else ylabel,
                      clabel=clabel, 
                      cmap=cmap,
                      ax=ax)

def plot_coeval_slice(
    coeval: un.Quantity,
    box_length: un.Quantity,
    title: str | None = None,
    xlabel: str | None = "Distance",
    ylabel: str | None = "Distance",
    clabel: str | None = None,
    cmap: str = "eor",
    logx: bool = False,
    logy: bool = False,
    logc: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
    smooth: bool | float = False,
    slicing_fnc: callable | None = None,
) -> plt.Axes:
    """Plot a slice from a coeval of shape (HII_DIM, HII_DIM, HII_DIM)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if slicing_fnc is not None:
        coeval = slicing_fnc(coeval)
    else:
        coeval = get_slice_index(zmin=zmin, zmax=zmax, idx=0)(coeval)
    if smooth:
        if isinstance(smooth, bool):
            smooth = 1.0
        coeval = gaussian_filter(coeval.value, sigma=smooth) * coeval.unit
    xaxis = np.linspace(0, box_length, coeval.shape[0])
    yaxis = np.linspace(0, box_length, coeval.shape[1])

    if clabel is None:
        if coeval.unit.physical_type == un.get_physical_type("temperature"):
            clabel = f"Brightness Temperature [{coeval.unit}]"
        elif coeval.unit.is_equivalent(un.dimensionless_unscaled):
            clabel = "Density Contrast"
        else:
            clabel = f"{coeval.unit.physical_type} [{coeval.unit}]"
    return plot_slice(coeval,
                      xaxis,
                      yaxis,
                      vmin=vmin,
                      vmax=vmax,
                      log=[logx, logy, logc],
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel + " [" + str(box_length.unit) + "]" if ylabel is None else ylabel,
                      clabel=clabel,
                      cmap=cmap,
                      ax=ax)
