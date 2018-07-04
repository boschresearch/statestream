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

from statestream.meta.neuron_pool import np_shm_layout
from statestream.meta.synapse_pool import sp_shm_layout

from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL



def grad_desc_shm_layout(name, net, param):
    """Return shared memory layout for SGD optimizer.

    Parameters:
    -----------
    name : str
        The unique string identifier for the plasticity using this optimizer.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    Returns:
    --------
        shm_layout : dict
            Dictionary containing shared memory layout for all pars / vars of the optimizer.
    """
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}
    # Add learning rate.
    shm_layout["parameter"]["lr"] = ShmL("backend", (), dtype, 1e-3, 0.0, None)
    # Return layout.
    return shm_layout



def adam_shm_layout(name, net, param):
    """Return shared memory layout for Adam optimizer.

    Parameters:
    -----------
    name : str
        The unique string identifier for the plasticity using this optimizer.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    Returns:
    --------
        shm_layout : dict
            Dictionary containing shared memory layout for all pars / vars of the optimizer.
    """
    # Get local plasticity dictionary.
    p = net["plasticities"][name]
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}
    # Add parameters.
    shm_layout["parameter"]["lr"] \
        = ShmL("backend", (), dtype, 1e-3, 0.0, None)
    shm_layout["parameter"]["decay"] \
        = ShmL("backend", (), dtype, 0.99, 0.0, None)
    shm_layout["parameter"]["momentum"] \
        = ShmL("backend", (), dtype, 0.999, 0.0, None)
    # Add variables.
    shm_layout["variables"]["t"] = ShmL("backend", (), dtype, 0, 0, None)
    # Need moments + varianc for all to be updated parameters.
    for par in range(len(p["parameter"])):
        P = p["parameter"][par]
        if P[0] == "np":
            vs_tmp = np_shm_layout(P[1], net, param)
        elif P[0] == "sp":
            vs_tmp = sp_shm_layout(P[1], net, param)
        shm_layout["variables"]["moments." + P[0] + "." + P[1] + "." + P[2]] \
            = vs_tmp["parameter"][P[2]]
        shm_layout["variables"]["varianc." + P[0] + "." + P[1] + "." + P[2]] \
            = vs_tmp["parameter"][P[2]]

    # Return layout.
    return shm_layout



def rmsprop_shm_layout(name, net, param):
    """Return shared memory layout for RMSprop optimizer.

    Parameters:
    -----------
    name : str
        The unique string identifier for the plasticity using this optimizer.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    Returns:
    --------
        shm_layout : dict
            Dictionary containing shared memory layout for all pars / vars of the optimizer.
    """
    # Get local plasticity dictionary.
    p = net["plasticities"][name]
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}
    # Add parameters.
    shm_layout["parameter"]["lr"] \
        = ShmL("backend", (), dtype, 1e-3, 0.0, None)
    shm_layout["parameter"]["rho"] \
        = ShmL("backend", (), dtype, 0.9, 0.0, None)
    # Add variables.
    shm_layout["variables"]["t"] = ShmL("backend", (), dtype, 0, 0, None)
    # Need moments + varianc for all to be updated parameters.
    for par in range(len(p["parameter"])):
        P = p["parameter"][par]
        if P[0] == "np":
            vs_tmp = np_shm_layout(P[1], net, param)
        elif P[0] == "sp":
            vs_tmp = sp_shm_layout(P[1], net, param)
        shm_layout["variables"]["accumulation." + P[0] + "." + P[1] + "." + P[2]] \
            = vs_tmp["parameter"][P[2]]

    # Return layout.
    return shm_layout

