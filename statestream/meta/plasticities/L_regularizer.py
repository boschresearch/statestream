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
import importlib

from statestream.meta.neuron_pool import np_shm_layout
from statestream.meta.synapse_pool import sp_shm_layout

from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL



def plast_init(net, name, dat_name, dat_layout, mode=None):
    """Return valid value for plasticity parameter / variable.
    """
    # Default return is None.
    dat_value = None

    # Return initialized value.
    return dat_value



def plast_shm_layout(name, net, param):
    """Function returns shm layout for L*-regularizer plasticity.
    """
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}
    # Get / set local dictionary.
    p = net["plasticities"][name]

    # Optimizer parameters and variables.
    # -------------------------------------------------------------------------
    # Determine optimizer.
    optimizer = p.get("optimizer", "grad_desc")
    # Get layout defining function for optimizer.
    opt_shm_layout_fct = getattr(importlib.import_module("statestream.meta.optimizer"),
                                 optimizer + "_shm_layout")
    opt_shm_layout = opt_shm_layout_fct(p, net, param)
    # Get / set optimizer dependent layout.
    for T in ["parameter", "variables"]:
        for d,d_l in opt_shm_layout[T].items():
            shm_layout[T][d] = d_l

    # Parameter for regularizer.    
    # -------------------------------------------------------------------------
    shm_layout["parameter"]["L1"] = ShmL("backend", (), dtype, 0.0, 0.0, None)
    shm_layout["parameter"]["L2"] = ShmL("backend", (), dtype, 0.0, 0.0, None)
        
    # Variables for regularizer.
    # -------------------------------------------------------------------------
    shm_layout["variables"]["loss"] = ShmL("backend", (), dtype, 0.0)

    # Adding difference, which we will call gradient for simplicity as variable.
    # -------------------------------------------------------------------------    
    for par in range(len(p["parameter"])):
        P = p["parameter"][par]
        if P[0] == "np":
            vs_tmp = np_shm_layout(P[1], net, param)
        elif P[0] == "sp":
            vs_tmp = sp_shm_layout(P[1], net, param)
        shm_layout["variables"]["grad." + P[0] + "." + P[1] + "." + P[2]] = vs_tmp["parameter"][P[2]]

    # Return layout.
    return shm_layout


