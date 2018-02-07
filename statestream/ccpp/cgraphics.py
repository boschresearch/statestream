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
import numpy.ctypeslib as npct
from ctypes import c_int



# Define some default variables for c-functions.
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=c_int, ndim=1, flags="CONTIGUOUS")

# Load the compiled c-library.
libcd = npct.load_library("ccpp/lib/libcgraphics", ".")

# Setup the return and argument types.
libcd.cgraphics_colorcode.restype = None
libcd.cgraphics_colorcode.argtypes = [array_1d_double,
                                      c_int, 
                                      c_int, 
                                      array_1d_float,
                                      array_1d_float, 
                                      array_1d_float,
                                      c_int]
libcd.cgraphics_vec_to_RGBangle.restype = None
libcd.cgraphics_vec_to_RGBangle.argtypes = [array_1d_double,
                                      c_int, 
                                      c_int, 
                                      array_1d_float,
                                      array_1d_float, 
                                      array_1d_float,
                                      c_int]
libcd.cgraphics_np_force.restype = None
libcd.cgraphics_np_force.argtypes = [array_1d_double,
                                     array_1d_double,
                                     array_1d_int,
                                     c_int, 
                                     c_int, 
                                     c_int,
                                     array_1d_double,
                                     array_1d_double]
libcd.cgraphics_tensor_dist.restype = None
libcd.cgraphics_tensor_dist.argtypes = [array_1d_float,
                                        array_1d_float, 
                                        array_1d_float, 
                                        c_int,
                                        c_int,
                                        c_int,
                                        c_int,
                                        c_int,
                                        c_int]



def cgraphics_colorcode(source, w, h, cm, colorcorrect):
    """Wrapper for colorcode c-function.

    This function expects a grey-scaled image in 'source' and colorcodes it
    with the given colormap 'cm'. The colorcoded map is directly written into
    source again and overwrites all data there.

    Parameters
    ----------
    source : 1D array of np.float32
        This array hold the pixels of the image that should be colorcoded.
    w : int
        The width of the image in pixels.
    h : int
        The height of the image in pixels.
    cm : [255,3] shaped int8 array
        A RGB colormap.
    colorcorrect : boolean
        Boolean that specifies if colorcorrection is needed.

    Returns
    -------
    The colorcorrect is performed in-place, hence source holds the corrected
    values.
    """
    return libcd.cgraphics_colorcode(np.double(source),
                                     w,
                                     h, 
                                     np.float32(cm[:,0]),
                                     np.float32(cm[:,1]),
                                     np.float32(cm[:,2]),
                                     colorcorrect)

def cgraphics_vec_to_RGBangle(source, w, h, cm, colorcorrect):
    """Pixelwise conversion of a 2D vectors to RGB angular image.

    This is especially useful for optic-flow visualization.

    Parameter
    ---------
    source : numpy array [2 * dim_x * dim_y]
        The 2D vector image of resolution dim_x x dim_y.
    w, h : int
        Width and height of the 2D vector image.
    cm : [255,3] shaped int8 array
        A RGB colormap.
    colorcorrect : bool
        Color correction flag.

    Return
    ------
    The color-coded angle / amplitude is stored in the first dimension of
    the source.

    """
    return libcd.cgraphics_vec_to_RGBangle(np.double(source),
                                           w,
                                           h, 
                                           np.float32(cm[:,0]),
                                           np.float32(cm[:,1]),
                                           np.float32(cm[:,2]),
                                           colorcorrect)



def cgraphics_np_force(item_pos_X, 
                       item_pos_Y, 
                       conn_mat,
                       item_number, 
                       min_dist, 
                       max_dist,
                       item_force_X, 
                       item_force_Y):
    """Wrapper for np_force c-function.

    This function computes pair-wise repulsive forces between neuron-pools.

    Parameters
    ----------
    item_pos_X : 1D double array
        This array contains x-coordinates of all neuron-pool items on the screen.
    item_pos_Y : 1D double array
        This array contains y-coordinates of all neuron-pool items on the screen.
    conn_mat : 1D int32 array
        Array of size nps**2 with [src_np + tgt_np + no_np] providing the minimum
        path length from src np to tgt np.
    item_force_X : 1D double array
        In this array the resulting cummulated x-coordinate of the force is stored.
    item_force_Y : 1D double array
        In this array the resulting cummulated y-coordinate of the force is stored.

    Returns
    -------
    The computed forces can be found in item_force_X/Y.
    """
    return libcd.cgraphics_np_force(item_pos_X,
                                    item_pos_Y,
                                    conn_mat,
                                    item_number, 
                                    min_dist,
                                    max_dist,
                                    item_force_X,
                                    item_force_Y)


def cgraphics_tensor_dist(x, 
                          y, 
                          d,
                          a, 
                          fx, 
                          fy,
                          sx, 
                          sy,
                          m):
    """Wrapper for tensor_dist c-function.

    This function computes pair-wise tensor distances for
    two tensors x and y of the dimensions:
        [agents, fx, sx, sy] and [agents, fx, sx, sy]
    respectively. The output d is of dimension [fx, fy].
    The distance will be computed as the mean distance
    over all agents for each feature fx X fy combination.

    Parameters
    ----------
        x : 1D float32 np.array
            The first tensor of implicit form:
            [agents, fx, sx, sy]
        y : 1D float32 np.array
            The second tensor of implicit form:
            [agents, fy, sx, sy]
        d : 1D float32 np.array
            The resulting distance matrix of implicit form:
            [fx, fy]
        a : int
            The number of agents (a.k.a. batchsize).
        fx : int
            Feature dimension of x.
        fy : int
            Feature dimension of y.
        sx : int
            First spatial dimension.
        sy : int
            Second spatial dimension.
        m : int
            Modus specifying the used distance measure:
            -1: inf-norm
            0:  0-norm
            1:  1-norm
            2:  2-norm
            3:  dot product
            4:  cosine distance

        Returns
        -------
        The computed distances can be found in d.
    """
    return libcd.cgraphics_tensor_dist(x,
                                       y,
                                       d,
                                       a,
                                       fx,
                                       fy,
                                       sx,
                                       sy,
                                       m)
