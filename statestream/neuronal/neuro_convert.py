# -*- coding: utf-8 -*-
# Copyright (c) 2017 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/statestream
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
import copy


# =============================================================================
# =============================================================================
# =============================================================================

def float_2_np(par, val, shape):
    """Convert a float value into an array representation.
    """
    # Create empty numpy array of [agents, features] size.
    if len(shape) == 3:
        arr = np.zeros(shape, dtype=np.float32)
        if par["func_x"] == "linear" and par["func_y"] == "gauss":
            if shape[0] > 1:
                dim = 0
            elif shape[1] > 1:
                dim = 1
            elif shape[2] > 1:
                dim = 2
            dx = (par["func_x_max"] - par["func_x_min"]) / (shape[dim] - 1)
            val_loc = copy.copy(val)
            if par["type"] == "line":
                val_loc = min(max(val, par["func_x_min"]), par["func_x_max"])
            for xi in range(shape[dim]):
                dist = abs(par["func_x_min"] + xi * dx - val_loc)
                if par["type"] == "modulo":
                    while dist > (par["func_x_max"] - par["func_x_min"]) / 2.0:
                        dist -= (par["func_x_max"] - par["func_x_min"]) / 2.0
                if dim == 0:
                    arr[xi,0,0] = par["func_y_amp"] \
                                       * np.exp(-dist**2 / (2 * par["func_y_sigma"]**2))
                elif dim == 1:
                    arr[0,xi,0] = par["func_y_amp"] \
                                       * np.exp(-dist**2 / (2 * par["func_y_sigma"]**2))
                elif dim == 2:
                    arr[0,0,xi] = par["func_y_amp"] \
                                       * np.exp(-dist**2 / (2 * par["func_y_sigma"]**2))
                else:
                    arr[0,0,0] = val

        return arr
    return None

# =============================================================================
# =============================================================================
# =============================================================================
