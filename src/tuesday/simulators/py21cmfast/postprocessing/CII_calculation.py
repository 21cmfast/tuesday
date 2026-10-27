"""
Author: Yilong Zhang
Email: yilong.zhang@sns.it
Updated: 2026-10-02
Version: 21cmFAST 4.1.2.dev112+g424ee2b7d.d20260601
Description: calculating the CII intensity from PerturbHaloField
"""

import numpy as np
from astropy import cosmology as cosmo
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import Planck18
from py21cmfast.wrapper.cfuncs import convert_halo_properties

# set the cosmo model the same as the cosmo_params used in simulation
# from .globals import GlobalConfig
# nu_CII = GlobalConfig.nu_CII.value

# 1 generate PerturbHaloField (HaloCatalog)


# 2 calculate CII intensity from SFR for each halo
def calculate_CII(pt_halos, model):  # pt_halos refers to the perturbhalofield
    # calculate the luminosity and intensity for each halo
    z = pt_halos.redshift
    Dlum = Planck18.luminosity_distance(z)  # Mpc

    H = Planck18.H(z) * u.km / u.s / (u.Mpc.to(u.km)) / u.km
    t_h = 1 / H.value  # Hubble time in unit of sec

    nu_CII = 1900.5369
    mu = 1  #'lensing magnification'
    nu_CII_obs = nu_CII / (1 + z)

    BOX_LEN = pt_halos.simulation_options.BOX_LEN  # unit Mpc
    HII_DIM = pt_halos.simulation_options.HII_DIM  # resolution
    cell_size = BOX_LEN / HII_DIM  # unit Mpc

    halo_props = convert_halo_properties(
        redshift=pt_halos.redshift,
        inputs=pt_halos.inputs,
        halo_masses=pt_halos.get("halo_masses"),
        halo_coords=pt_halos.get("halo_coords"),
        star_rng=pt_halos.get("star_rng"),
        sfr_rng=pt_halos.get("sfr_rng"),
        xray_rng=pt_halos.get("xray_rng"),
    )

    halo_mass, Mstar, sfr, metallicity = (
        halo_props["halo_mass"],
        halo_props["halo_stars"],
        halo_props["halo_sfr"],
        halo_props["metallicity"],
    )

    SFR_yr = sfr * u.yr.to(u.s)
    logSFR_yr = np.log10(SFR_yr)
    logZ = np.log10(metallicity)
    logOH12 = logZ + 8.69
    logSM = np.log10(Mstar)
    logHM = np.log10(halo_mass)

    logOH12 = logZ + 8.69

    if (
        model == "Vallini15"
    ):  # for redshift 6.6 with varying SFR(0.1-100) and gas metallicity(0.05-1 Zsun)
        sigma_metal = 0.054  # 0.054 dex (for log10-normal distribution) in Curti+20 FMR
        ln_sigma_metal = sigma_metal * np.log(10)  # Convert the dex of log10 to ln
        Z_sample = metallicity * np.exp(
            -ln_sigma_metal * ln_sigma_metal / 2
            + np.random.normal(0, 1, len(metallicity)) * ln_sigma_metal
        )

        log_L_CII = (
            7.0
            + 1.2 * np.log10(SFR_yr)
            + 0.021 * np.log10(Z_sample)
            + 0.012 * np.log10(SFR_yr) * np.log10(Z_sample)
            - 0.74 * np.log10(Z_sample) ** 2
        )

    elif model == "Horlaville24":  # for z=0-8 galaxies with stellar mass 1e5-1e11
        M0_HI = 1.9e9
        M_min = 2e10
        alpha_HI = 0.74
        M_HI = (
            M0_HI
            * (halo_mass / M_min) ** alpha_HI
            * np.exp(-((M_min / halo_mass) ** 0.35))
        )

        sigma_metal = 0.4  # 0.4 dex (for log10-normal distribution) in Horlaville+24
        ln_sigma_metal = sigma_metal * np.log(10)  # Convert the dex of log10 to ln
        Z_sample = metallicity * np.exp(
            -ln_sigma_metal * ln_sigma_metal / 2
            + np.random.normal(0, 1, len(metallicity)) * ln_sigma_metal
        )

        log_L_CII = np.log10(0.024) + np.log10(M_HI) + np.log10(Z_sample)
        # Lum_CII = 0.024 * M_HI * metallicity

    elif model == "deLooze14":  # for local galaxies
        log_L_CII = 7.10 + 1.05 * np.log10(SFR_yr)
        scatter = 0.4  # dex
        log_L_CII = log_L_CII + np.random.normal(0, 1, len(log_L_CII)) * scatter

    elif model == "Lagache18":  # for high redshift z=4-8, LCII>1e7, SFR<1e3
        log_L_CII = np.log10(SFR_yr) * (1.4 - 0.07 * z) + (7.1 - 0.07 * z)
        scatter = 0.5  # dex
        log_L_CII = log_L_CII + np.random.normal(0, 1, len(log_L_CII)) * scatter

    elif model == "Casavecchia24":  # for redshift 6-12
        log_L_CII = 1.19 * np.log10(SFR_yr) - 0.12 * z + 7.58
        scatter = 0.5
        log_L_CII = log_L_CII + np.random.normal(0, 1, len(log_L_CII)) * scatter

    Lum_CII = 10**log_L_CII
    # calculate the intensity from luminosity (in unit of Jy km/s)
    Int_CII = mu * Lum_CII / 1.04e-3 / Dlum.value**2 / nu_CII_obs

    return Lum_CII, Int_CII


# 3. calculate the CII surface brightness density (after form into grids)
def calculate_Bvoxel(lightcone):
    print("Calculating the surface brightness density ...", end="")
    z = lightcone.lightcone_redshifts
    nu_CII = 1900.5369  # *u.GHz
    nu_CII_obs = nu_CII / (1 + z)
    CII_Intensity = lightcone.lightcones["CII_IntensityBox"]
    BOX_LEN = lightcone.simulation_options.BOX_LEN  # unit Mpc
    HII_DIM = lightcone.simulation_options.HII_DIM
    cell_size = BOX_LEN / HII_DIM * u.Mpc  # comoving cell size
    Dcomv = lightcone.lightcone_distances  # unit Mpc
    Dlum = Dcomv * (1 + z)
    Dagl = Dcomv / (1 + z)

    solid_angle = (cell_size / Dcomv) ** 2  # u.sr
    Dcomv_front = np.zeros(Dcomv.shape) * u.Mpc
    Dcomv_back = np.zeros(Dcomv.shape) * u.Mpc
    z_front = np.zeros(Dcomv.shape)
    z_back = np.zeros(Dcomv.shape)
    nu_front = np.zeros(Dcomv.shape)
    nu_back = np.zeros(Dcomv.shape)
    delta_nu = np.zeros(Dcomv.shape)
    B_voxel = np.zeros(CII_Intensity.shape)

    for i in range(len(Dcomv)):
        Dcomv_front[i] = Dcomv[i] - cell_size / 2
        Dcomv_back[i] = Dcomv[i] + cell_size / 2
        z_front[i] = cosmo.z_at_value(Planck18.comoving_distance, Dcomv_front[i])
        z_back[i] = cosmo.z_at_value(Planck18.comoving_distance, Dcomv_back[i])
        nu_front[i] = nu_CII / (1 + z_front[i])
        nu_back[i] = nu_CII / (1 + z_back[i])
        delta_nu[i] = nu_front[i] - nu_back[i]

    for k in range(CII_Intensity.shape[0]):
        for j in range(CII_Intensity.shape[1]):
            B_voxel[k, j, :] = (
                nu_CII_obs
                / (c.to(u.km / u.s).value * delta_nu * solid_angle)
                * CII_Intensity[k, j, :]
            )

    print("done")
    return B_voxel
