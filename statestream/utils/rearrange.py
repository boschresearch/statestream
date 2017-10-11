# -*- coding: utf-8 -*-
# Copyright (c) 2017 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/VolkerFischer/statestream
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numpy as np



def rearrange_3D_to_2D (dat, img, shape, sub_shape):
    """Function rearranges a 3d feature map (features, dimX, dimY) into a 2d image.
    
    Parameters
    ----------
    dat : 3D np-array
        3D input data of shape [features, dimX, dimY]
    img : 2D np-array
        Resulting 2D image.
    shape : list
        shape of result image (sub_shape[0] * dimX, sub_shape[1] * dimY)
    sub_shape : list
        Number of subwindows in X and Y dimension.
        
    Returns
    -------
    see img
    """
    # Loop over all features.
    for f in range(dat.shape[0]):
        # determine sub index
        sX = f % sub_shape[0]
        sY = f // sub_shape[0]
        # copy feature map
        img[sX * dat.shape[1]:(sX + 1) * dat.shape[1], \
            sY * dat.shape[2]:(sY + 1) * dat.shape[2]] = dat[f,:,:]



def rearrange_4D_to_2D (dat, img, shape):
    '''Function rearranges a 4d weight map (tgt_c, src_c, dimX, dimY) into a 2d image.
    
    Parameters
    ----------
    dat : 4D np-array
        4D input data of shape (tgt_c, src_c, dimX, dimY)
    img : 2D np-array
        Resulting 2D image.
    shape : list
        shape of result image (sub_shape[0] * dimX, sub_shape[1] * dimY)
        
    Returns
    -------
    see img
    '''
    for tgt_c in range(dat.shape[0]):
        for src_c in range(dat.shape[1]):
            img[tgt_c * dat.shape[2]:(tgt_c + 1) * dat.shape[2], \
                src_c * dat.shape[3]:(src_c + 1) * dat.shape[3]] = dat[tgt_c,src_c,:,:]
