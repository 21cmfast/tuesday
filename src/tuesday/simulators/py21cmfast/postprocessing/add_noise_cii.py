#this script is for adding the instrumental noise to the CII surface brightness density map
#based on AtLAST

import numpy as np
from tqdm import tqdm
import astropy.units as u
from astropy import constants as c
from astropy.io import fits
import astropy.constants as cst
from astropy.cosmology import Planck18 as cosmo
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_gaussian
from astropy.stats import gaussian_fwhm_to_sigma

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_tpix(nu_obs,
                   Diam,
                   surv_area,
                   t_surv,
                   N_feed):
    
    #nu_cii = 1900.54 * u.GHz
    #nu_obs = nu_cii/(1+z)

    fwhm_beam = 1.22 * u.rad * cst.c / nu_obs / Diam
    sigma_beam = fwhm_beam * gaussian_fwhm_to_sigma
    beam_area = 2.0 * np.pi * sigma_beam ** 2
    beam_area = beam_area.to(u.sr)
    #print('Omega_beam',beam_area)
    #print('fwhm_beam',fwhm_beam.to(u.deg))
    
    Omega_surv = surv_area
    Omega_pix = beam_area

    t_vox = t_surv/(Omega_surv/Omega_pix)
    #print('t_vox =',t_vox.to(u.s))
    t_pix = t_vox*N_feed
    #print('t_pix =',t_pix.to(u.s))
    
    return t_pix.to(u.s)


def calculate_noise(nu_obs,
                    noise_temp=None,
                    noisefile=None,
                    R=None,
                    Diam=None,
                    f_min=None,
                    f_max=None,
                    val_sens=None,
                    N_feed=None,
                    t_surv=None,
                    surv_area=None,
                    t_pix=None):
    
    #for atlast we directly assume a survey chould give t_pix=1 hour
    if noise_temp == 'atlast':
        N_feed = 1
        t_surv = 3000*u.hour
        surv_area = 600*u.deg**2
        
        if noisefile is not None:
            with fits.open(noisefile) as f_in:
                R=1000
                Diam = 50*u.m
                data = f_in[1].data
                nu_sens = data['nu'] * u.GHz
                #bw_sens = data['bandwidth'] * u.GHz
                octiles = (np.arange(1,8) * 12.5)
                octile_sel = 0
                list_sens = [data['RMS_noise [octile {0:.1f}]'.format(octile)] for octile in octiles][octile_sel] *u.Jy*(u.hour**0.5) 
                t_pix = 1000 *u.hour #in unit of Jy for 1h  #需要1000 hour来达到SNR=3
                #fwhm_sens = data['FWHM_beam'] * u.deg
                f_min = 125 * u.GHz
                f_max = 500 * u.GHz
                #use nu_lc to interpolate nu_sens(x) and obtain the corresponding noise curve
                val_sens = np.interp(nu_obs, nu_sens, list_sens)
                
        elif val_sens is not None:
            val_sens = val_sens
        else:
            t_pix = calculate_tpix(nu_obs,Diam,surv_area,t_surv,N_feed)
            from atlast_sc.calculator import Calculator
            input_data = {'t_int': {'value': t_pix.value, 'unit': f'{t_pix.unit}'},
                          'bandwidth': {'value': bw_sens, 'unit': f'{bw_sens.unit}'},
                          'obs_freq':  {'value': nu_sens, 'unit': f'{nu_sens.unit}'}}
            calculator = Calculator(input_data)
            val_sens = calculator.calculate_sensitivity()
        
    elif noise_temp == 'time':
        R = 120                                 #1812.08135v2 for R=150
        Diam = 12*u.m                           #2012.09160v2
        f_min = 183 *u.GHz
        f_max = 326 *u.GHz
        if nu_obs >= 230*u.GHz:
            bw_sens  = 1.9 *u.GHz                   #1812.08135v2
            val_sens = 11 *u.MJy*(u.s)**0.5 /u.sr   #NEI, 1812.08135v2
        else:
            bw_sens  = 1.5 *u.GHz                   #1812.08135v2
            val_sens = 5.2 *u.MJy*(u.s)**0.5 /u.sr  #NEI, 1812.08135v2
        
        N_feed = 32                             #1812.08135v2
        t_surv = 1000*u.hour                    #1812.08135v2
        surv_area = 78*u.arcmin * 0.5*u.arcmin  #1812.08135v2
        
    elif noise_temp == 'time2':
        R = 120                                 #2012.09160v2
        Diam = 10*u.m                           #2012.09160v2
        f_min = 183 *u.GHz
        f_max = 326 *u.GHz
        if nu_obs >= 230*u.GHz:
            bw_sens  = 1.9 *u.GHz
            val_sens = 5 *u.MJy*(u.s)**0.5 /u.sr  #NEI, 2012.09160v2
        else:
            bw_sens  = 1.5 *u.GHz
            val_sens = 2.5 *u.MJy*(u.s)**0.5 /u.sr  #NEI, 2012.09160v2
        
        N_feed = 32                              #2012.09160v2
        t_surv = 3000*u.hour                     #2012.09160v2
        surv_area = 78*u.arcmin * 0.5*u.arcmin

    elif noise_temp == 'ccatp':
        R = 100
        Diam = 6*u.m
        f_min = 210 *u.GHz                     #CCAT-prime Collaboration 2023
        f_max = 420 *u.GHz                     #CCAT-prime Collaboration 2023
        val_sens = 72.5 *u.mJy*(u.s)**0.5      #NEFD (Jy s1/2 /beam), 2309.15733v3
        
        N_feed = 120 #5040/42                  #CCAT-prime Collaboration 2023
        t_surv = 2000*u.hour                   #CCAT-prime Collaboration 2023
        surv_area = 4*u.deg**2                 #CCAT-prime Collaboration 2023
    
    elif noise_temp == 'ccatp2':
        R = 100
        Diam = 6*u.m
        f_min = 210 *u.GHz
        f_max = 420 *u.GHz
        val_sens = 72.5 *u.mJy*(u.s)**0.5      #NEFD, 2309.15733v3
        
        N_feed = 240
        t_surv = 8000*u.hour
        surv_area = 8*u.deg**2
        t_pix = 38*u.hour                      #2309.15733v3
    
    else:
        print('Input a correct instrument template.')
        
    nu_sens = nu_obs
    bw_sens = nu_sens/R
    NU_CII = 1900.54 * u.GHz
    z_sens = NU_CII/nu_sens - 1
    z_min, z_max = (f_max/nu_sens-1), (f_min/nu_sens-1)
    
    fwhm_sens = 1.22 * u.rad * cst.c *(1/ nu_sens).to(u.s, equivalencies=u.spectral()) / Diam
    beam_sigma = fwhm_sens * gaussian_fwhm_to_sigma
    beam_area = 2.0 * np.pi * beam_sigma ** 2 #solid angle for the instrument beam
    beam_area = beam_area.to(u.sr)
    
    if t_pix is None:
        t_vox = t_surv/(surv_area/beam_area)
        t_pix = (t_vox*N_feed).to(u.s)
    else:
        t_pix = t_pix.to(u.s)
        
    if u.sr in val_sens.unit.bases:
        val_sens = val_sens.to(u.Jy * (u.s)**0.5 / u.sr)  #nei in unit Jy h^1/2 /sr
        rms = val_sens/np.sqrt(t_pix)   #in unit Jy/sr
    elif u.s in val_sens.unit.bases or u.hour in val_sens.unit.bases:
        val_sens = val_sens.to(u.Jy * (u.s)**0.5) #nefd in unit Jy h^1/2 /beam
        rms = val_sens/beam_area/np.sqrt(t_pix)   #in unit Jy/sr
    else:
        val_sens = val_sens.to(u.Jy)   #in unit Jy /beam
        rms = val_sens/beam_area   #in unit Jy/sr
        
    noise_info = {'diameter': Diam, "frequency resolution":R, 'nu_sens': nu_sens, 'bw_sens': bw_sens,
                  "f_min": f_min, "f_max": f_max, "z_min": z_min, "z_max": z_max, 
                  'N_feed': N_feed, 't_surv': t_surv, 'surv_area': surv_area, 't_pix': t_pix,
                  'fwhm': fwhm_sens, 'beam_area': beam_area, "NEFD or NEI":val_sens, "RMS_sigma": rms}
    
    return noise_info


def add_cii_noise(lc_obj,noise_temp=None,noisefile=None):
    NU_CII = 1900.54 * u.GHz
    nu_lc = NU_CII / (1+lc_obj.lightcone_redshifts)
    B_cii = lc_obj.lightcones['CII_BrightnessDensity'] #*u.Jy/u.sr   #B_cii should be in unit of Jy/sr
    
    seed = np.random.randint(0, 2**16)
    RState = np.random.RandomState(seed=seed)
    
    noise   = np.zeros(B_cii.shape)
    B_real  = np.zeros(B_cii.shape)
    B_conv  = np.zeros(B_cii.shape)
    B_noise = np.zeros(B_cii.shape)
    
    #print(f'nu range in LC {f_min},{f_max}')
    #print(f'noisefile covers redshifts {NU_CII/f_max-1},{NU_CII/f_min-1}')
    print(f'adding {noise_temp} noise ...')
    
    for i in range(nu_lc.size):
        nu_sens = nu_lc[i]
        noise_info = calculate_noise(nu_obs=nu_sens, noise_temp=noise_temp, noisefile=noisefile)
        
        rms = noise_info['RMS_sigma'].to(u.Jy/u.sr)
        fwhm_sens = noise_info['fwhm']
        
        z_sens = NU_CII/nu_sens - 1
        #angular size of cell in bands
        d_a_sens = lc_obj.cosmo_params.cosmo.angular_diameter_distance(z_sens)
        ang_size_sim = ((lc_obj.simulation_options.BOX_LEN/lc_obj.simulation_options.HII_DIM) / (1+z_sens) * u.Mpc / d_a_sens) * u.rad
        #solid_ang_sim = ang_size_sens**2 #u.sr
        #solid_ang_sim = 2 * np.pi * (1 - np.cos(ang_size_sim))  #solid angle for the simulation cell size

        beam_sigma = fwhm_sens * gaussian_fwhm_to_sigma
        beam_sigma_cell = (beam_sigma / ang_size_sim).to('').value #beam_sigma per cell units
        beam_area = 2.0 * np.pi * beam_sigma ** 2 #solid angle for the instrument beam
        beam_area = beam_area.to(u.sr)
        
        bw = noise_info['bw_sens']
        nu_min = (nu_sens - bw/2)
        nu_max = (nu_sens + bw/2)

        sel = (nu_lc > nu_min) & (nu_lc < nu_max)
        
        B_real_2d = B_cii[...,sel].sum(axis=-1)
        noise_2d   = RState.normal(loc=0.0, scale=1.0, size=B_real_2d.shape) * rms.value
        B_conv_2d  = gaussian_filter(B_real_2d,sigma=beam_sigma_cell)
        B_noise_2d = B_conv_2d + noise_2d
        
        noise[...,i]   = noise_2d   #noise
        B_real[...,i]  = B_real_2d  #after integration in f interval
        B_conv[...,i]  = B_conv_2d  #after beam filter
        B_noise[...,i] = B_noise_2d #after noise and beam filter
        
    lc_obj.lightcones['noise_'+noise_temp] = noise
    lc_obj.lightcones['Bcii_'+noise_temp]  = B_conv
    lc_obj.lightcones['Bcii_'+noise_temp+'_noise'] = B_noise
    
    # print(f'nu {nu_sens.min():.2e} to {nu_sens.max():.2e}: sens mean {val_sens.mean()} fhwm mean {beam_sigma.mean()}')
    return lc_obj #noise, B_real, B_conv, B_noise, noise_info #output is in unit of Jy/sr


def chunk_lc_field(lc,kind,redshift,subtract_mean=False):
    
    ncells = lc.simulation_options.HII_DIM
    boxlen = lc.simulation_options.BOX_LEN
    cellsize = boxlen / ncells
    z_lc = lc.lightcone_redshifts
    field = lc.lightcones[kind]
    dim = lc.shape
    
    #Remove mean of each frequency slice
    if subtract_mean:
        field = field - field.mean()
    
    #找红移序号
    #find slices centred at desired redshifts
    slice_idx = np.argmin(np.fabs(lc.lightcone_redshifts - redshift))
    z_slice = lc.lightcone_redshifts[slice_idx]

    #if dim[0] is odd, the centre is on slice_idx
    #if dim[0] is even, the centre is between slice_idx and slice_idx + 1
    slmin = slice_idx - dim[0]//2
    slmax = slice_idx + dim[0]//2 + dim[0]%2

    chunk_box = field[:,:,slmin:slmax]
    boxlen  = cellsize * np.array(chunk_box.shape)
    #mean = field[:,:,slmin:slmax].mean() 
    
    return chunk_box, boxlen
