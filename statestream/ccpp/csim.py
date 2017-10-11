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
from ctypes import c_int, c_float



# Define some default variables for c-functions.
array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')

# Load the compiled c-library.
libcd = npct.load_library("ccpp/lib/libcsim", ".")

# Setup the return and argument types.
libcd.my_cdistPLS.restype = None
libcd.my_cdistPLS.argtypes = [c_float, c_float, c_float, c_float,
			                        c_float, c_float,
			                        array_1d_float]
libcd.csim.restype = None
libcd.csim.argtypes = [array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float, array_1d_float, array_1d_float,
                          array_1d_float,
                          c_int,
                          c_int,
                          c_int,
                          c_int,
                          c_int,
                          c_int,
                          c_int,
                          c_int,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float,
                          array_1d_float]



def my_cdistPLS(ls_x1, ls_y1, ls_x2, ls_y2, p_x, p_y, dist):
    """Function for fast computation of distance between point and line segment.
    """
    return libcd.my_cdistPLS(ls_x1, ls_y1, ls_x2, ls_y2, p_x, p_y, dist)



def csim_func(world_params,
             ls_x1, ls_y1, ls_x2, ls_y2,
             ls_R, ls_G, ls_B,
             m_x1, m_y1, m_x2, m_y2,
             c_x, c_y, c_r, c_a,
             c_dx, c_dy, c_da,
             c_Fx, c_Fy, c_Fa,
             c_R, c_G, c_B,
             a_x, a_y, a_a, a_r,
             a_dx, a_dy, a_da,
             a_Fx, a_Fy, a_Fa,
             a_R, a_G, a_B,
             a_lookat, a_fF, a_pF,
             a_motor,
             no_f, no_p,
             a_ddx, a_ddy, a_dda,
             a_sensor,
             f_R, f_G, f_B, f_D,
             p_R, p_G, p_B, p_D):
    """Main world function.
    """
    return libcd.csim(world_params,
                         ls_x1, ls_y1, ls_x2, ls_y2,
                         ls_R, ls_G, ls_B,
                         m_x1, m_y1, m_x2, m_y2,
                         c_x, c_y, c_r, c_a,
                         c_dx, c_dy, c_da,
                         c_Fx, c_Fy, c_Fa,
                         c_R, c_G, c_B,
                         a_x, a_y, a_a, a_r,
                         a_dx, a_dy, a_da,
                         a_Fx, a_Fy, a_Fa,
                         a_R, a_G, a_B,
                         a_lookat, a_fF, a_pF,
                         a_motor,
                         len(ls_x1),
                         len(m_x1),
                         len(c_x),
                         len(a_x),
                         no_f,
                         no_p,
                         int(len(a_motor) / len(a_x)),
                         int(len(a_sensor) / len(a_x)),
                         a_ddx, a_ddy, a_dda,
                         a_sensor,
                         f_R, f_G, f_B, f_D,
                         p_R, p_G, p_B, p_D)
