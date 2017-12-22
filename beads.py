#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:42:30 2017

@author: sarkissi
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


#%% Load the dataset:
proj_nb = 64
data_path = '/export/scratch2/sarkissi/Data/SophiaBeads/'
data_path = data_path + 'SophiaBeads_' + str(proj_nb) + '_averaged/'
fname = 'SophiaBeads_' + str(proj_nb) + '_averaged_0'

# Initialize variables and read projections    
slice_nb = 200 # 2000 for full field
ydim = xdim = 2000
#ydim = xdim = 1564 # 2000 for full field
det_dim_x = det_dim_y = 2000
proj = flexData.read_raw(data_path,name=fname)
proj = proj[:-1,...]

# Apply this if you want to crop projections
crop_proj = False
if crop_proj:
    ydim = xdim = 1564
    det_dim_x = xdim
    det_dim_y = slice_nb
    offsety = int((2000 - det_dim_y) / 2)
    offsetx = int((2000 - det_dim_x) / 2)
    proj = proj[:,offsety:-offsety,offsetx:-offsetx]

# Create reconstruction volume
vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')


# Transpose them with astra geometry and put scan geometry in
proj = np.transpose(proj, (1,0,2))
proj = -np.log(proj/65535)
src2obj = 80.6392412185669
src2det = 1007.003
det_pixel = 0.2
magnification = src2det / src2obj
im_pixel = det_pixel / magnification
fdk_factor = im_pixel**3*det_pixel
det2obj = src2det - src2obj

#%%
vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, -360], proj_nb-1, endpoint=False)

#corrections = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
#corrections = [-3*det_pixel]
#corrections = [-0.2375]
corrections = [-0.2269]
#corrections = [0]
#cor = -0.2375/det_pixel
#cor = 0
#cor = -2#/im_pixel
for cor in corrections:
    vol = np.zeros([slice_nb, ydim, xdim], dtype = 'float32')
    geometry['axs_tra'] = [cor,0]
    flexProject.FDK(proj, vol, geometry)
    vol /= fdk_factor
    flexUtil.display_slice(vol, index=100)
    #flexData.write_raw('/export/scratch1/sarkissi/beads/', 'vol', vol,dim=0)
