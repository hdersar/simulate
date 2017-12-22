#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2017
@author: kostenko

This module some wrappers around ASTRA to make lives of people slightly less horrible.
"""

''' * Imports * '''

import numpy
import numpy as np
import misc
import astra
import astra.experimental as asex 
import sys
import matplotlib.pyplot as plt

import flexUtil
import flexData

''' * Methods * '''

def _backproject_block_(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', operation = '+'):
    """
    Use this internal function to compute backprojection of a single block of data.
    """           
    try:
        
        if (operation == '+'):
            volume_ = volume
            
        elif (operation == '*') | (operation == '/'):
            volume_ = np.zeros_like(volume)
            
        else: ValueError('Unknown operation type!')
                    
        sin_id = astra.data3d.link('-sino', proj_geom, projections)        
        vol_id = astra.data3d.link('-vol', vol_geom, volume_)    
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    
        if algorithm == 'BP3D_CUDA':
            asex.accumulate_BP(projector_id, vol_id, sin_id)
            
        elif algorithm == 'FDK_CUDA':
            asex.accumulate_FDK(projector_id, vol_id, sin_id)
            
        else:
            raise ValueError('Unknown ASTRA algorithm type.')
        
        if (operation == '*'):
             volume *= volume_
        elif (operation == '/'):
             volume_[volume_ < 1e-10] = numpy.inf
             volume /= volume_
             
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)                 
            
def _forwardproject_block_(projections, volume, proj_geom, vol_geom, operation = '+'):
    """
    Use this internal function to compute backprojection of a single block of data.
    """           
    try:
        
        if (operation == '+'):
            projections_ = projections
            
        elif (operation == '*') | (operation == '/'):
            projections_ = np.zeros_like(projections)
            
        else: ValueError('Unknown operation type!')    
        
        sin_id = astra.data3d.link('-sino', proj_geom, projections_)        
        vol_id = astra.data3d.link('-vol', vol_geom, volume)    
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        
        asex.accumulate_FP(projector_id, vol_id, sin_id)
        
        if (operation == '*'):
             projections *= projections_
        elif (operation == '/'):
            
             projections_[projections_ < 1e-10] = numpy.inf        
             projections /= projections_
             
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)            
            
def backproject(projections, volume, geometry, algorithm = 'BP3D_CUDA', operation = '+', psf=None):
    """
    Backproject useing standard ASTRA functionality
    """
    # If the data is not memmap:        
    if not isinstance(projections, numpy.memmap):    
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])
        
        # If PSF is present, convolve with it before backprojection
        # TODO: Reverse the PSF first
        # TODO: It would be better not to alter the existing projections
        if psf is not None:
            for i in range(projections.shape[1]):
                projections[:,i,:] = scipy.signal.fftconvolve(projections[:,i,:], psf, 'same')
                
        _backproject_block_(projections, volume, proj_geom, vol_geom, algorithm, operation)
        
    else:
        # Decide on the size of the block:
        n = projections.shape[1]    
        l = n // 10
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        
        # Loop over blocks:
        for ii in range(n // l):
            
            i0 = (ii * l)
            i1 = min((ii * l + l), n)
            
            # Extract a block:
            proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2], i0, i1)    
            
            block = numpy.ascontiguousarray(projections[:, i0:i1,:])

            _backproject_block_(block, volume, proj_geom, vol_geom, algorithm, operation)  

import scipy
def forwardproject(projections, volume, geometry, operation='+', psf=None):
    """
    Forwardproject
    """
    # If the data is not memmap:        
    if not isinstance(volume, numpy.memmap):   
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])
        
        _forwardproject_block_(projections, volume, proj_geom, vol_geom, operation)
        if psf is not None:
            for i in range(projections.shape[1]):
                projections[:,i,:] = scipy.signal.fftconvolve(projections[:,i,:], psf, 'same')
        
    else:
        # Decide on the size of the block:
        n = volume.shape[0]    
        l = n // 10
        
        # Initialize ASTRA geometries:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])    
        
        # Loop over blocks:
        for ii in range(n // l):
            
            i0 = (ii * l)
            i1 = min((ii * l + l), n)
            
            # Extract a block:
            vol_geom = flexData.astra_vol_geom(geometry, volume.shape, i0, i1)
            
            block = numpy.ascontiguousarray(projections[i0:i1, :, :])

            _forwardproject_block_(block, volume, proj_geom, vol_geom, operation)  



def FDK(projections, volume, geometry):
    """
    FDK
    """
    # Make sure array is contiguous (if not memmap):
    if not isinstance(projections, numpy.memmap):
        projections = numpy.ascontiguousarray(projections) 
        
    misc.progress_bar(0)    
    
    # Yeeey!
    backproject(projections, volume, geometry, 'FDK_CUDA')
    
    misc.progress_bar(1)    
    
def SIRT(projections, volume, geometry, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None}):
    """
    SIRT
    """ 
    # Make sure array is contiguous (if not memmap):
    if not isinstance(projections, numpy.memmap):
        projections = numpy.ascontiguousarray(projections)        
    
    # We will use quick and dirty scaling coefficient instead of proper calculation of weights
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    prj_weight = 2 / (projections.shape[1] * geometry['det_pixel'] ** 4 * max(volume.shape) / m)
                    
    # Initialize L2:
    l2 = []

    print('Doing SIRT`y things...')
    
    misc.progress_bar(0)
        
    for ii in range(iterations):
    
        projections_ = projections.copy()
        
        forwardproject(projections_, -volume, geometry)
        
        # Take into account Poisson:
        if options.get('poisson_weight'):
            # Some formula representing the effect of photon starvation...
            projections_ *= numpy.sqrt(numpy.exp(-projections))
            
        projections_ *= prj_weight    
        
        backproject(projections_, volume, geometry, 'BP3D_CUDA')    
        
        # Apply bounds
        if options.get('bounds') is not None:
            numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 

        # L2 norm:
        if options.get('l2_update'):
            l2.append(numpy.sqrt((projections_ ** 2).mean()))
            
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
            
        misc.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):   
         plt.figure(15)
         plt.plot(l2)
         plt.title('Residual L2')
         
def SIRT_GPU(projections, volume, geometry, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None}):
    """
    SIRT GPU by native ASTRA implementation
    """ 
    # Make sure array is contiguous (if not memmap):
    if not isinstance(projections, numpy.memmap):
        projections = numpy.ascontiguousarray(projections)        
    
    # We will use quick and dirty scaling coefficient instead of proper calculation of weights
    #m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    #prj_weight = 2 / (projections.shape[1] * geometry['det_pixel'] ** 4 * max(volume.shape) / m)
                    
    # Initialize L2:
    data_shape = projections.shape[::2]
    vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
    proj_geom = flexData.astra_proj_geom(geometry, data_shape)
    
    sin_id = astra.data3d.link('-sino', proj_geom, projections)        
    vol_id = astra.data3d.link('-vol', vol_geom, volume)    
    
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = sin_id
    cfg['ReconstructionDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)

    print('Native ASTRA SIRT on GPU...')
    astra.algorithm.run(alg_id, iterations=iterations)
    
    astra.data3d.delete(vol_id)
    astra.data3d.delete(sin_id)
    astra.algorithm.delete(alg_id)


def SIRT_CPU(projections, volume, geometry, iterations=10, relaxation=1,options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None}, psf=None):
    """
    SIRT on CPU with or without detector PSF
    """ 
    # Make sure array is contiguous (if not memmap):
    if not isinstance(projections, numpy.memmap):
        projections = numpy.ascontiguousarray(projections)        
    
    # We will NOT use quick and dirty scaling coefficient instead of proper calculation of weights
    #m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    #prj_weight = 2 / (projections.shape[1] * geometry['det_pixel'] ** 4 * max(volume.shape) / m) 
    
    # Create the forward and the backward weights
    fwd_weights = np.zeros_like(projections)
    forwardproject(fwd_weights, np.ones_like(volume, dtype=np.float32), geometry, psf=psf)
    tol = 1e-6
    fwd_weights[fwd_weights < tol] = np.Inf
    fwd_weights = 1 / fwd_weights
    
    bwd_weights = np.zeros_like(volume, dtype=np.float32)
    backproject(np.ones_like(projections, dtype=np.float32), bwd_weights, geometry, psf=psf)
    bwd_weights[bwd_weights < tol] = np.Inf
    bwd_weights = relaxation / bwd_weights
    
    #cur_proj = np.zeros_like(projections,dtype=np.float32)
    
    
    print('Doing SIRT with CPU/GPU iterations...')
        
    misc.progress_bar(0)

    for ii in range(iterations):
        #vol_update = bwd_weights.copy()
        vol_update = np.zeros_like(volume)
        cur_proj = projections.copy()
        forwardproject(cur_proj, -volume, geometry, operation='+')
        cur_proj *= fwd_weights
        
        backproject(cur_proj, vol_update, geometry,operation='+')
        volume += bwd_weights*vol_update
        volume += vol_update
        misc.progress_bar((ii+1) / iterations)
        
         
    
def EM(projections, volume, geometry, iterations, options = {'preview':False, 'bounds':None}):
    """
    Expectation Maximization
    """ 
    # Make sure array is contiguous (if not memmap):
    if not isinstance(projections, numpy.memmap):
        projections = numpy.ascontiguousarray(projections)    
        
    # Make sure that the volume is positive:
    if volume.max() <= 0: volume += 1
    elif volume.min() < 0: volume[volume < 0] = 0

    projections[projections < 0] = 0
            
    print('Em Emm Emmmm...')
    
    misc.progress_bar(0)
        
    for ii in range(iterations):

        # Temp projection data
        forwardproject(projections, volume, geometry, operation = '/')
                
        # Temp reconstruction volume        
        backproject(projections, volume, geometry, 'BP3D_CUDA', operation = '*')    
                        
        # Apply bounds
        if options.get('bounds') is not None:
            numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 
           
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
            
        misc.progress_bar((ii+1) / iterations)
        
        