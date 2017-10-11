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



""" Example of wrapping a C library function that accepts a C double array as
    input using the numpy.ctypeslib.
"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_float32

# Define some default variables for c-functions.
array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags="CONTIGUOUS")

# Load the compiled c-library, using numpy mechanisms.
libcd = npct.load_library("ccpp/lib/libchebbian", ".")

# Setup the return types and argument types.

void chebb_raw(float* src,
                float* tgt,
                float* upd,
                float tgt_eps,
                int agents,
                int src_X,
                int src_Y,
                int src_C,
                int tgt_X,
                int tgt_Y,
                int tgt_C,
                int rf_X,
                int rf_Y,
                int dil_X,
                int dil_Y,
                int ignore_border)

libcd.chebb_raw.restype = None
libcd.chebb_raw.argtypes = [array_1d_float,
                            array_1d_float,
                            array_1d_float,
                            c_float32,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int,
                            c_int]

# Function for fast blitting and color conversion of images.
def chebb_raw2D(source, target, update, eps=1e-2, dilation=(1,1), ignore_border=True):
    """Wrapper for colorcode c-function.

    This function expects a grey-scaled image in 'source' and colorcodes it
    with the given colormap 'cm'. The colorcoded map is directly written into
    source again and overwrites all data there.

    Parameters:
    -----------
    source : 1D array of doubles
        This array hold the pixels of the image that should be colorcoded.
    w : int
        The width of the image in pixels.
    h : int
        The height of the image in pixels.
    cm : [255,3] shaped int8 array
        A RGB colormap.
    colorcorrect : boolean
        Boolean that specifies if colorcorrection is needed.
    """
    return libcd.cgraphics_colorcode(source,
                                     w,
                                     h, 
                                     np.float32(cm[:,0]),
                                     np.float32(cm[:,1]),
                                     np.float32(cm[:,2]),
                                     colorcorrect)

# Function for fast computation of repulsive forces between nps.
def cgraphics_np_force(item_pos_X, item_pos_Y, item_force_X, item_force_Y):
    """Wrapper for np_force c-function.

    This function computes pair-wise repulsive forces between neuron-pools.

    Parameters:
    item_pos_X : 1D double array
        This array contains x-coordinates of all neuron-pool items on the screen.
    item_pos_Y : 1D double array
        This array contains y-coordinates of all neuron-pool items on the screen.
    item_force_X : 1D double array
        In this array the resulting cummulated x-coordinate of the force is stored.
    item_force_Y : 1D double array
        In this array the resulting cummulated y-coordinate of the force is stored.
    """
    return libcd.cgraphics_np_force(item_pos_X,
                                     item_pos_Y,
                                     len(item_pos_X), 
                                     item_force_X,
                                     item_force_Y)
