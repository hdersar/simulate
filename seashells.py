#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate spectral data with Poisson noise
"""
#%%
import flexData
import flexProject
import flexUtil
import flexModel
import flexSpectrum

import numpy as np
#import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

# Create volume and forward project:
data_path = '/export/scratch1/sarkissi/data/seashell/Phantom/segmentation/'
res_path = '/export/scratch2/sarkissi/Data/seashell/Simulations/'
# Initialize images:    
slice_nb = 485
ydim = 340
xdim = 460
proj_nb = 1000
phantom_dim = 460
#det_dim_x = det_dim_y = 700
det_dim_x = det_dim_y = 600
vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
proj = np.zeros([det_dim_y, proj_nb, det_dim_x], dtype = 'float32')


# Create phantom (150 micron wide, 15 micron wall thickness):
#vol = flexModel.phantom(vol.shape, 'ball', [150,])
phantom = np.zeros([slice_nb, phantom_dim, phantom_dim], dtype = 'float32')
sn = 'seg_{0:05d}.tiff'
for s in range(0,485):
    vol[s,...] = misc.imread(data_path + sn.format(s))  
    
# Extend vol to square image
phantom[:,-ydim:,:] = vol
flexUtil.display_slice(phantom)

# Define a simple projection geometry:
src2obj = 61.850     # mm
det2obj = 209.610 - src2obj    # mm
det_pixel = 0.0087   # mm (8.7 micron)
magnification = (det2obj + src2obj) / src2obj
im_pix = det_pixel / magnification
fdk_scale_factor = det_pixel*im_pix*im_pix*im_pix

vol = np.zeros_like(phantom, dtype = 'float32')
geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, 360], proj_nb, endpoint=False)

run_nb = 1

res_nopsf_path = res_path + 'no_psf/'
res_psf_path = res_path + 'psf/'

#%% Try different reconstructions of the volume
proj = np.zeros([det_dim_y, proj_nb, det_dim_x], dtype = 'float32')
flexProject.forwardproject(proj, phantom, geometry)
vol = np.zeros((400,400,400), dtype=np.float32)
counts_poly = -np.log(counts / flat_field)
flexProject.FDK(proj, vol, geometry)
vol /= fdk_scale_factor
flexData.write_raw(res_path + 'test/FDK/', 'volume', vol, dim=0)
flexData.write_raw(res_path + 'test/', 'proj', proj, dim=1)


vol = np.zeros((485,460,460), dtype=np.float32)
#counts_poly = -np.log(counts / flat_field)
flexProject.SIRT_CPU(proj, vol, geometry, iterations=100)
flexUtil.display_slice(vol)
#flexData.write_raw(res_path + 'test/SIRT/', 'volume', vol, dim=0)


#%%
for run_id in range(1,run_nb+1):
    proj = np.zeros([det_dim_y, proj_nb, det_dim_x], dtype = 'float32')
    # Define random rotation parameters
    vol_rot = np.random.rand(3)*np.pi
    #geometry['vol_rot'] = vol_rot
    
    flexProject.forwardproject(proj, phantom, geometry)
    
    
    # Simulate spectrum:
    energy_bins = 20
    e_min = 0.1
    e_max = 60
    energy = np.linspace(e_min, e_max, energy_bins)
    
    # Tube:
    spectrum = flexSpectrum.bremsstrahlung(energy, 60) 
    spectrum[2] = 0.4
    spectrum[3] = 1
    spectrum[4] = 0.4
    spectrum[5] = 0.3
    spectrum[6] = 0.3
    spectrum[7] = 0.4
    j = 7
    c = spectrum[j]
    n = energy_bins-1
    a = c/(j-n)
    b = -a*n
    for i in (range(8,energy_bins)):
        spectrum[i] = a*i+b
    
    # Filter:
    #spectrum *= flexSpectrum.total_transmission(energy, 'Cu', 8, 0.1)
    
    # Detector:
    #spectrum *= flexSpectrum.scintillator_efficiency(energy, rho = 5, thickness = 1)
    # Normalize:
    #spectrum /= (energy*spectrum).sum()
    #spectrum /= spectrum.sum()
    
    # Get the material refraction index:
    #mu = flexSpectrum.linear_attenuation(energy, 'Al', 2.7)
    mu = flexSpectrum.linear_attenuation(energy, 'CaCO3', flexSpectrum.find_nist_name('Calcium Carbonate')['density'])
    #mu = np.array([mu])
    #mu_air = flexSpectrum.linear_attenuation(energy, 'air', 1.225e-3)
    
    # Display:
    #flexUtil.plot(energy, spectrum, 'Spectrum')
    #flexUtil.plot(energy, mu, 'Linear attenuation')
    
    #
    #flexProject.FDK(proj, vol, geometry)
    #flexUtil.display_slice(vol, title = 'Uncorrected polytechromatic FDK')
    
    
    
    # Model data:
        
    # Simulate intensity images:
    counts = np.zeros_like(proj, dtype=np.float32)
    flat_field = np.zeros((det_dim_x,1,det_dim_y), dtype=np.float32)
    
    
    ## Detector gain, spatial and energy
    # Spatial
    det_sensi = np.ones((det_dim_x, 1, det_dim_y))
    #det_sensi = 
    
    # Energy
    k_energy_factor = 1
    #det_energy_response = k_energy_factor*energy
    det_energy_response = k_energy_factor * np.ones_like(energy)
    
    
    ## Simulate the polyenergetic projections
    # TODO: Add the sample holder to the data ?
    print('Simulate polychromatic projection for energy bin...')
    for ii in range(len(energy)):
        print('energy bin', ii+1, '/', energy_bins)
        # Add monochromatic components weighted with energy response
        counts += det_sensi * det_energy_response[ii] * spectrum[ii] * np.exp(-proj * mu[ii])
        
        # Do the same for the flat field
        flat_field += det_sensi * det_energy_response[ii] * spectrum[ii]
    
    print(flat_field.max())
    n_phot = 1e5 / flat_field.max()
    counts *= n_phot
    flat_field *= n_phot
    print('Simulate polychromatic projection... done')
    
    counts_noiseless = np.array(counts.copy(), dtype=np.float32)
    ff_noiseless = np.asarray(flat_field.copy(), dtype=np.float32)
    
    # Simulate Poisson noise AFTER PSF
    print('Simulate Poisson noise...')
    counts = flexModel.apply_noise(counts, 'poisson', 1.0)
    flat_field = flexModel.apply_noise(flat_field, 'poisson', 1.0)
    print('Simulate Poisson noise... done')
    
    # Save psf_free projections
#    directory = res_nopsf_path + str(run_id) + '/'
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    flexData.write_raw(directory, 'proj', np.asarray(counts, dtype=np.float32))
#    flexData.write_raw(directory, 'flat', np.asarray(flat_field, dtype=np.float32))
#    flexData.write_meta(directory+'/log.txt', geometry)
#    
#    sigmas = [0.5,1,1.5,2]
#    for s in sigmas:
#        counts = counts_noiseless.copy()
#        flat_field = ff_noiseless.copy()
#        # Simulate detector blurring:
#        print('Simulate PSF...')
#        ctf = flexModel.get_ctf(counts.shape[::2], 'gaussian', [det_pixel, s*det_pixel])
#        counts = flexModel.apply_ctf(counts, ctf)
#        ctf = flexModel.get_ctf(flat_field.shape, 'gaussian', [det_pixel, s*det_pixel])
#        flat_field = flexModel.apply_ctf(flat_field, ctf)
#        print('Simulate PSF... done')
#    
#    
#        # Simulate Poisson noise AFTER PSF
#        print('Simulate Poisson noise...')
#        counts = flexModel.apply_noise(counts, 'poisson', 1.0)
#        flat_field = flexModel.apply_noise(flat_field, 'poisson', 1.0)
#        print('Simulate Poisson noise... done')
#
#        counts = np.asarray(counts, dtype=np.float32)
#    
#        # Save psf-ed projections
#        directory = os.path.join(res_psf_path, str(s),str(run_id))
#        if not os.path.exists(directory):
#            os.makedirs(directory)
#        flexData.write_raw(directory, 'proj', np.asarray(counts, dtype=np.float32))
#        flexData.write_raw(directory, 'flat', np.asarray(flat_field, dtype=np.float32))
#        flexData.write_meta(directory+'/log.txt', geometry)
#        # Simulate electronic noise after detection noise
        #noise_sigma = 1e2
        #counts += flexModel.apply_noise(np.zeros_like(counts), noise_sigma)
        #flat_field += flexModel.apply_noise(np.zeros_like(flat_field), noise_sigma)
        
        # Display:
        #flexUtil.display_slice(counts, title = 'Modeled sinogram') 
        
        # Correct for flat field
    vol = np.zeros((600,600,600), dtype=np.float32)
    #counts_poly = -np.log(counts / flat_field)
    counts_poly = -np.log(counts / 1e5)
    counts_poly = np.asarray(counts_poly, dtype=np.float32)
    flexProject.FDK(counts_poly, vol, geometry)
    vol /= fdk_scale_factor
    flexData.write_raw(res_path + 'test/FDK/', 'FDK', vol, dim=0)
    
    vol = np.zeros((600,600,600), dtype=np.float32)
    #counts_poly = -np.log(counts / flat_field)
    flexProject.SIRT_CPU(counts_poly, vol, geometry,iterations=500)
    #flexProject.SIRT_CPU(counts_poly, vol, geometry,iterations=1)
    flexData.write_raw(res_path + 'test/FDK/', 'SIRT', vol, dim=0)
        
#%% Reconstruct volumes as monochromatic with FDK / SIRT
import flexData
import flexProject
import flexUtil
import flexModel
import flexSpectrum

import numpy as np
#import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

# Create volume and forward project:
data_path = '/export/scratch2/sarkissi/Data/seashell/Simulations/'

# Initialize images:    
slice_nb = 485
ydim = 460
xdim = 460
slice_nb = 485
ydim = 600
xdim = 600
proj_nb = 700
phantom_dim = 460
det_dim_x = 700
det_dim_y = 700

# Define the projection geometry:
src2obj = 61.850     # mm
det2obj = 209.610 - src2obj    # mm
det_pixel = 0.0087   # mm (100 micron)
magnification = (det2obj + src2obj) / src2obj
im_pix = det_pixel / magnification
fdk_scale_factor = det_pixel*im_pix*im_pix*im_pix

geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, 360], proj_nb, endpoint=False)

nb_run = 10
do_no_psf = False
do_psf = True

psf = [0.5,1,1.5,2]

for run_id in range(1,nb_run+1):
    
    if do_no_psf:
        # Data without PSF
        datapath_no_psf = os.path.join(data_path, 'no_psf', str(run_id))
        vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
        proj = flexData.read_raw(datapath_no_psf, 'proj')
        #ff = flexData.read_raw(datapath_no_psf, 'flat')
        
        # Flat field and log
        proj = -np.log(proj / 1e5)
        proj = np.transpose(proj, (1,0,2))
        
        # Save FDK reconstructions
        flexProject.FDK(proj, vol, geometry)
        vol /= fdk_scale_factor
        directory = os.path.join(datapath_no_psf, 'FDK')
        if not os.path.exists(directory):
            os.makedirs(directory)
        flexData.write_raw(directory, 'recon', np.asarray(vol, dtype=np.float32), dim=0)
        
        
        # Save SIRT reconstructions
        vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
        flexProject.SIRT_CPU(proj, vol, geometry, iterations=300)
        
        directory = os.path.join(datapath_no_psf, 'SIRT')
        if not os.path.exists(directory):
            os.makedirs(directory)
        flexData.write_raw(directory, 'recon', np.asarray(vol, dtype=np.float32), dim=0)
    
    if do_psf:
        
        # Data with PSF
        datapath_psf = os.path.join(data_path, 'psf')
        for p in psf:
            datapath_psf = os.path.join(datapath_psf, str(p), str(run_id))
                                            vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
            proj = flexData.read_raw(datapath_psf, 'proj')
            #ff = flexData.read_raw(datapath_no_psf, 'flat')
            
            # Flat field and log
            proj = -np.log(proj / 1e5)
            proj = np.transpose(proj, (1,0,2))
            
            # Save FDK reconstructions
            flexProject.FDK(proj, vol, geometry)
            vol /= fdk_scale_factor
            directory = os.path.join(datapath_psf, 'FDK')
            if not os.path.exists(directory):
                os.makedirs(directory)
            flexData.write_raw(directory, 'recon', np.asarray(vol, dtype=np.float32), dim=0)
            
            
            # Save SIRT reconstructions
            vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
            flexProject.SIRT_CPU(proj, vol, geometry, iterations=300)
            
            directory = os.path.join(datapath_psf, 'SIRT')
            if not os.path.exists(directory):
                os.makedirs(directory)
            flexData.write_raw(directory, 'recon', np.asarray(vol, dtype=np.float32), dim=0)
            
            # Save SIRT reconstructions with PSF
            ctf = flexModel.get_ctf(proj.shape, 'gaussian', [det_pixel, p*det_pixel])
            psf_sp = np.fft.ifft2()
            
            vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
            flexProject.SIRT_CPU(proj, vol, geometry, iterations=300, psf=psf_sp)
            
            directory = os.path.join(datapath_psf, 'SIRT_PSF')
            if not os.path.exists(directory):
                os.makedirs(directory)
            flexData.write_raw(directory, 'recon', np.asarray(vol, dtype=np.float32), dim=0)
    
        
#%% Reconstruct the polychromatic data (no PSF):
#vol_rec_poly = np.zeros_like(vol)
#proj_0 = -np.log(counts_poly)
#
#vol_rot = [0,0,0]
#geometry['vol_rot'] = vol_rot
#
#flexProject.FDK(proj_0, vol_rec_poly, geometry)
#vol_rec_poly /= fdk_scale_factor
##flexUtil.display_slice(vol_rec_poly, title = 'Uncorrected polytechromatic FDK')
#
#
#
#
## Reconstruct polychromatic data with PSF:
#vol_rec_psf = np.zeros_like(vol)
#proj_0 = -np.log(counts)
#
#flexProject.FDK(proj_0, vol_rec_psf, geometry)
#vol_rec_psf /= fdk_scale_factor
##flexUtil.display_slice(vol_rec_psf, title = 'Uncorrected polychromatic FDK with PSF')
#
#
#
#
#
##%% Beam hardening correction: 
#proj_0 = -np.log(counts)
#
#energy_next, spectrum_next = flexSpectrum.calibrate_spectrum(proj_0, vol_rec, geometry, compound = 'Al', density = 2.7, n_bin = 100, iterations=1000)   
#proj_0 = flexSpectrum.equivalent_density(proj_0, geometry, energy_next, spectrum_next, compound = 'Al', density = 2.7)
#
#vol_rec_next = np.zeros_like(vol_rec)
#flexProject.FDK(proj_0, vol_rec_next, geometry)
#flexUtil.display_slice(vol_rec_next, title = 'Corrected FDK')
