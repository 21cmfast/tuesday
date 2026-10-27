"""
Author: Yilong Zhang
Email: yilong.zhang@sns.it
Updated: 2026-10-02
Version: 21cmFAST 4.1.2.dev112+g424ee2b7d.d20260601
Description: making a CII lightcone from PerturbHaloField
"""

import os
import numpy as np
import pandas as pd
import py21cmfast as p21c
from astropy import units as u
from astropy import cosmology as cosmo
from astropy.cosmology import Planck18
from astropy.constants import c

from py21cmfast.io.h5 import read_output_struct
from py21cmfast.io.caching import OutputCache

from attrs import define, field
from CII_calculation import calculate_CII, calculate_Bvoxel

#1 define a new coeval box for cii, under the definition of coeval in 21cmfast
@define(slots=False)
class cii_coeval(p21c.Coeval):
    CII_IntensityBox: np.ndarray = field(default=None)

#2 define a new lightcone for cii, under the definition of lightcone in 21cmfast
@define(slots=False)
class cii_lightcone(p21c.LightCone):
    """A LightCone class extended with CII-specific properties."""
    CII_BrightnessDensity: np.ndarray = field(default=None)  # New attribute
    CII_observation: np.ndarray = field(default=None)  # New attribute
        

#3 match the cache file according to redshifts
#4 generate the Coeval box with CII intensity for each redshifts according to the cache files
def gen_coeval(z, model, directory_path):
    output_cache = OutputCache(direc=directory_path)
    cache_path = output_cache.list_datasets(redshift=z)#,inputs=lc.inputs,kind=None)
    cache_obj = {}
    for cache_file in cache_path:
        name = cache_file.stem
        cache_obj[name] = read_output_struct(cache_file)

    #calculate cii from perturbed halo field
    PerturbHaloField = cache_obj['HaloCatalog']
    lumi, intens = calculate_CII(PerturbHaloField, model)#='Vallini15')#'deLooze14'

    #form halos (with cii) into grids
    coordinates_Mpc = PerturbHaloField.get('halo_coords') #the physical (x,y,z)Mpc coordinates for each halo
    box_len = PerturbHaloField.simulation_options.BOX_LEN   #coeval box lenth
    resolution = PerturbHaloField.simulation_options.HII_DIM   #number of bins
    cell_size = box_len/resolution
    coordinates = coordinates_Mpc/cell_size
    cii_intens, edges = np.histogramdd(coordinates, bins=resolution, weights=intens, 
                                           range=[[0,resolution], [0,resolution], [0,resolution]])

    coeval = cii_coeval(
                initial_conditions=cache_obj['InitialConditions'],
                perturbed_field=cache_obj['PerturbedField'],
                ionized_box=cache_obj['IonizedBox'],
                brightness_temperature=cache_obj['BrightnessTemp'],
                halobox=cache_obj['HaloBox'],
                CII_IntensityBox=cii_intens,
                )
    
    #add CII attribute to the Coeval
    coeval.CII_IntensityBox = cii_intens
    return coeval

#5 generate the coeval list according to the redshift list
def gen_coeval_list(lc, model, directory_path):
    coeval_list  = []
    node_redshifts = lc.inputs.node_redshifts
    for i,z in enumerate(node_redshifts):
        print(z)
        coeval = gen_coeval(z, model, directory_path)
        coeval_list.append(coeval)
    return coeval_list

#6 making lightconer for interpolation
#7 generate the Coeval slides between redshift-nodes and make lightcone
def gen_lc(lc, lcn, model, directory_path):
    inputs = lc.inputs
    lcn.quantities = ["CII_IntensityBox",'brightness_temp']
    #in order to interpolate the cii quantity in coeval, the lightcone should have this quantity first
    lc.lightcones['CII_IntensityBox']= np.zeros(lc.lightcones['brightness_temp'].shape)
    
    coeval_list = gen_coeval_list(lc, model, directory_path)
    for i in range(len(coeval_list)-1):
        coeval1 = coeval_list[i]
        coeval2 = coeval_list[i + 1]
        slices = lcn.make_lightcone_slices(c1=coeval1, c2=coeval2)
        #use a list put in all of the slides
        #organize slices into a lightcone
        for scalar_field, lcidx, slice_data in slices:
            #print(scalar_field, lcidx, np.mean(slice_data))
            if scalar_field in lc.lightcones and lcidx < lc.lightcones[f'{scalar_field}'].shape[-1]:
                #the lcidx is for coeval box, which might be larger than the redshift slices that stored for lightcones 
                lc.lightcones[f'{scalar_field}'][..., lcidx] = slice_data
    
    cii_lc = cii_lightcone(
        lightcone_distances=lc.lightcone_distances,
        inputs=lc.inputs,
        lightcones=lc.lightcones,
        global_quantities=lc.global_quantities,
        photon_nonconservation_data={},
    )

    B_voxel = calculate_Bvoxel(lc)
    cii_lc.lightcones['CII_BrightnessDensity'] = B_voxel
    
    return cii_lc

