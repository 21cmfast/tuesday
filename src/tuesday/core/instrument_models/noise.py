"""A module to add thermal noise to lightcones."""

import warnings
from collections.abc import Callable

import astropy.units as un
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18
from astropy.cosmology.units import littleh
from py21cmsense import Observation
from py21cmsense.conversions import dk_du, f2z
from scipy.signal import windows


def grid_baselines(uvws, freq, boxlength: un.Quantity, lc_shape, weights):
    r"""Grid baselines in uv space.

    Parameters
    ----------
    uvws : np.ndarray
        Baselines in uv space with shape (N bls, N time offsets, 3).
    freq : un.Quantity
        Frequency at which the baselines are projected.
    boxlength : un.Quantity
        Length of the box in which the lightcone is defined.
    lc_shape : tuple
        Shape of the lightcone (Nx, Ny, Nz).
        We assume that Nx = Ny.
    weights : np.ndarray
        Weights for each baseline group with shape (N bls).

    Returns
    -------
    uvsum : np.ndarray
        2D histogram of uv counts for one day
        of observation with shape (Nu=Nx, Nv=Nx).

    """
    dx = float(boxlength.value) / float(lc_shape[0])
    ugrid_edges = (
        np.fft.fftshift(np.fft.fftfreq(lc_shape[0], d=dx))
        * 2
        * np.pi
        / Planck18.h
        * littleh
        / un.Mpc
    )  # h/Mpc

    ugrid_edges = np.append(
        ugrid_edges, ugrid_edges[-1:] + ugrid_edges[-1:] - ugrid_edges[-2], axis=0
    )

    ugrid_edges /= dk_du(f2z(freq))

    weights = np.repeat(weights, uvws.shape[1])
    uvws = uvws.reshape((uvws.shape[0] * uvws.shape[1], -1))
    uvsum = np.histogram2d(uvws[:, 0], uvws[:, 1], bins=ugrid_edges, weights=weights)[0]

    # add mirrored baselines to uv grid
    uvsum += np.flip(uvsum)
    # but they're not independent measurements
    uvsum /= 2.0

    return uvsum


def thermal_noise(
    observation: Observation,
    freqs: np.ndarray,
    boxlen: float,
    lc_shape: tuple,
    a_eff: un.Quantity | None = None,
):
    r"""
    Calculate thermal noise RMS per integration.

    Eqn 3 from Prelogovic+22 2107.00018 without the last sqrt term
    That eqn comes from converting Eqn 9 in Ghara+16 1511.07448
    that's a flux density [Jy] to temperature [mK],
    but without the assumption of a circular symmetry of antenna distribution.

    Parameters
    ----------
    observation : py21cmsense.Observation
        Instance of `Observation`.
    freqs : astropy.units.Quantity
        Frequencies at which the noise is calculated.
    boxlen : astropy.units.Quantity
        Length of the box in which the lightcone is defined.
    lc_shape : tuple
        Shape of the lightcone (Nx, Ny, Nz).
    a_eff : astropy.units.Quantity, optional
        Effective area of the antenna with shape (Nfreqs,).
        If provided, we use
        $\Omega_{\rm beam} = \lambda^2/A_{\rm eff}$.
        If not provided, we use
        $\Omega_{\rm beam} = 0.004 \cdot (\nu/150 MHz)^{-2} \cdot rad^2$,
        which comes from assuming that the effective area is approximately 1000 m$^2$.
    """
    if not hasattr(freqs, "__len__"):
        freqs = np.array([freqs])
    sig_uv = np.zeros(len(freqs))
    for i, nu in enumerate(freqs):
        obs = observation.clone(
            observatory=observation.observatory.clone(
                beam=observation.observatory.beam.clone(frequency=nu)
            )
        )
        tsys = obs.Tsys.to(un.mK)
        if a_eff is None:
            f0 = 150.0 * un.MHz
            # =\lambda^2/A_{eff}, this approx assumes A_{eff} ~ 1000 m^2
            omega_beam = 0.004 * (nu.to(un.MHz) / f0) ** (-2.0) * un.rad**2
        else:
            wavelength = (c / nu.to("Hz")).to(un.m)
            omega_beam = (wavelength**2 / a_eff[i].to(un.m**2)) * un.rad**2

        d = Planck18.comoving_distance(f2z(nu)).to(un.Mpc)  # Mpc
        theta_box = (boxlen.to(un.Mpc) / d) * un.rad
        omega_pix = theta_box**2 / np.prod(lc_shape[:2])

        sqrt = np.sqrt(2.0 * observation.bandwidth.to("Hz") * obs.integration_time).to(
            un.dimensionless_unscaled
        )
        # I need this 1e6 to get the same numbers as tools...
        sig_uv[i] = tsys.value * omega_beam / omega_pix / sqrt / 1e6
    return sig_uv * tsys.unit


def blackmanharris(n: int):
    r"""Blackman-Harris window function for a 2D grid.

    Parameters
    ----------
    n : int
        Size of the window function, assumed to be square.

    Returns
    -------
    wf : np.ndarray
        2D Blackman-Harris window function with shape (n, n)

    """
    wf = np.abs(windows.blackmanharris(n))
    return np.sqrt(np.outer(wf, wf))


def sample_lc_noise(
    rms_noise: un.Quantity,
    seed: int | None = None,
    nsamples: int = 1,
    window_fnc: Callable | None = None,
):
    """Sample noise for a lightcone slice given the corresponding rms noise in uv space.

    Parameters
    ----------
    rms_noise : astropy.units.Quantity
        RMS noise in uv space, shape (Nx, Ny, Nfreqs).
    nsamples : int, optional
        Number of noise realisations to sample, by default 1.
    window_fnc : Callable, optional
        Window function to be applied to the noise sampled in uv space,
        by default windows.blackmanharris.

    Returns
    -------
    lc_noise : un.Quantity
        Noise sampled in real space, shape (nsamples, Nx, Ny, Nfreqs

    """
    if len(rms_noise.shape) == 2:
        rms_noise = rms_noise[..., None]
    if seed is None:
        seed = np.random.Generator().integers(0, 2**31 - 1)
        warnings.warn("Setting random seed to", seed, stacklevel=2)
    rng = np.random.default_rng(seed)
    

    lc_noise = np.zeros((nsamples, *rms_noise.shape))
    if window_fnc is None:
        window_fnc = blackmanharris(rms_noise.shape[0])
    for i in range(rms_noise.shape[-1]):
        noise = (
            rng.normal(size=lc_noise.shape[:-1])
            + 1j * rng.normal(size=lc_noise.shape[:-1])
        ) * rms_noise[..., i].value[None, ...]

        noise *= window_fnc[None, ...]
        noise = (noise + np.conj(noise)) / 2.0
        noise = np.fft.ifft2(np.fft.ifftshift(noise), axes=(1, 2))

        lc_noise[..., i] = noise.real
    return lc_noise * rms_noise.unit
