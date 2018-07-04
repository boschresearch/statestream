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

from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL



def np_state_shape(net, name):
    """Determine neuron-pool"s 4D state shape.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string identifier for the neuron-pool.
    """
    # Start with empty.
    np_shape = None
    # Get neuron pools dictionary.
    p = net["neuron_pools"][name]
    # If explicit shape given, take this.
    if "shape" in p:
        np_shape = [net["agents"]] + p["shape"]
    else:
        # Begin with agents (= batchsize).
        np_shape = [net["agents"], 1, 1, 1]
        # Add features.
        np_shape[1] = np_shape[1] * p.get("features", 1)
        # Add samples.
        np_shape[1] = np_shape[1] * p.get("samples", 1)
        # Add space.
        if "spatial" in p:
            if len(p["spatial"]) == 1:
                np_shape[2] = np_shape[2] * p["spatial"][0]
            elif len(p["spatial"]) == 2:
                np_shape[2] = np_shape[2] * p["spatial"][0]
                np_shape[3] = np_shape[3] * p["spatial"][1]
    # Return shape.
    return np_shape



def np_init(net, name, dat_name, dat_layout, mode=None):
    """Return valid value for neuron-pool parameter / variable.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string identifier for the neuron-pool.
    dat_name : str
        Unique identifier for parameter / variable to be initialized.
    dat_layout : SharedMemoryLayout
        The shared memory layout for this parameter / variable.
    mode : None, str, int, float
        The mode specifies how the parameter / variable should be initialized (e.g. 'xavier', 0.0)

    Returns:
    --------
    dat_value : np.array
        The numpy array with the same layout as in dat_layout but initialized according to mode.
    """
    # Get local dictionary.
    p = net["neuron_pools"][name]
    # Default return is None.
    dat_value = None
    if mode is None:
        # Try as specified, otherwise use default.
        init_as = p.get("init " + dat_name, None)
        if init_as is None:
            init_as = dat_layout.default
    else:
        init_as = mode
    # Dependent on init_as initialize.
    if dat_name[0] in ["b", "g"]:
        if isinstance(init_as, str):
            if init_as == "normal":
                dat_value = np.random.normal(loc=0.0, 
                                             scale=1.0, 
                                             size=dat_layout.shape).astype(dat_layout.dtype)
        else:
            dat_value = (float(init_as) * np.ones(dat_layout.shape, dtype=dat_layout.dtype)).astype(dat_layout.dtype)

    # Return initialized value.
    return dat_value



def np_shm_layout(name, net, param):
    """This function returns the entire shared memory layout for a neuron-pool.

    Parameters:
    -----------
    name : str
        The unique string identifier for the neuron-pool.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    Returns:
    --------
    shm_layout : dict
        Dictionary containig shared memory layouts for all vars / pars of the neuron-pool.
    """
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    
    np_shape = np_state_shape(net, name)

    # Add layout for all parameter.
    # -------------------------------------------------------------------------
    shm_layout["parameter"] = {}
    p = net["neuron_pools"][name]
    # Get bias and gain shape.
    bias_shape = p.get("bias_shape", "feature")
    gain_shape = p.get("gain_shape", False)
    # Check for un-shared for bias / gain.
    bias_unshared = False
    gain_unshared = False
    bias_preshape = []
    gain_preshape = []
    if "unshare" in p:
        if "b" in p["unshare"]:
            bias_unshared = True
            bias_preshape = [net["agents"]]
        if "g" in p["unshare"]:
            gain_unshared = True
            gain_preshape = [net["agents"]]

    # Set bias parameter.
    if bias_shape == "full":
        shm_layout["parameter"]["b"] = ShmL("backend", 
                                            bias_preshape + np_shape[1:], 
                                            dtype, 
                                            0.0)
    elif bias_shape == "feature":
        shm_layout["parameter"]["b"] = ShmL("backend", 
                                            bias_preshape + [np_shape[1],], 
                                            dtype, 
                                            0.0)
    elif bias_shape == "spatial":
        shm_layout["parameter"]["b"] = ShmL("backend", 
                                            bias_preshape + [np_shape[2], np_shape[3]], 
                                            dtype, 
                                            0.0)
    elif bias_shape == "scalar":
        shm_layout["parameter"]["b"] = ShmL("backend", 
                                            bias_preshape + [1,], 
                                            dtype, 
                                            0.0)
    else:
        # No gain parameter needed.
        pass

    # Set gain parameter.
    if gain_shape == "full":
        shm_layout["parameter"]["g"] = ShmL("backend", 
                                            gain_preshape + np_shape[1:], 
                                            dtype, 
                                            1.0)
    elif gain_shape == "feature":
        shm_layout["parameter"]["g"] = ShmL("backend", 
                                            gain_preshape + [np_shape[1],], 
                                            dtype, 
                                            1.0)
    elif gain_shape == "spatial":
        shm_layout["parameter"]["g"] = ShmL("backend", 
                                            gain_preshape + [np_shape[2], np_shape[3]], 
                                            dtype, 
                                            1.0)
    elif gain_shape == "scalar":
        shm_layout["parameter"]["g"] = ShmL("backend", 
                                            gain_preshape + [1,], 
                                            dtype, 
                                            1.0)
    else:
        # No gain parameter needed.
        pass

    # Set noise parameters.
    if "noise" in p:
        if p["noise"] == "normal":
            shm_layout["parameter"]["noise_mean"] \
                = ShmL("backend", (), dtype, 0.0)
            shm_layout["parameter"]["noise_std"] \
                = ShmL("backend", (), dtype, 1.0, 0.0, None)
        elif p["noise"] == "uniform":
            shm_layout["parameter"]["noise_min"] = ShmL("backend", (), dtype, -1.0)
            shm_layout["parameter"]["noise_max"] = ShmL("backend", (), dtype, 1.0)
    # Check for dropout.
    if "dropout" in p:
        shm_layout["parameter"]["dropout"] \
            = ShmL("backend", (), dtype, 1e-5, 0.0, 1.0)
    # Check for zoneout.
    if "zoneout" in p:
        shm_layout["parameter"]["zoneout"] \
            = ShmL("backend", (), dtype, 1e-5, 0.0, 1.0)

    # Add layout for variables.
    # -------------------------------------------------------------------------
    shm_layout["variables"] = {}

    # Add layout for state.
    # -------------------------------------------------------------------------
    shm_layout["state"] = ShmL("backend", np_shape, dtype, 0.0)

    # Return final layout.
    return shm_layout



def np_needs_rebuild(orig_net, new_net, np_id):
    """Determines if a neuron-pool needs rebuild for new network dictionary.
    
    Parameters:
    orig_net : dict
        Dictionary containing the original network.
    new_net : dict
        Dictionary containing the new edited network.
    np_id : str
        String specifying the neuron-pool under consideration.
    """
    from statestream.meta.synapse_pool import sp_needs_rebuild

    needs_rebuild = False
    # Get neuron-pool sub-dict.
    orig_p = orig_net["neuron_pools"][np_id]
    new_p = new_net["neuron_pools"][np_id]
    # Check if bias shape is different.
    orig_bias_shape = orig_p.get("bias_shape", "feature")
    new_bias_shape = new_p.get("bias_shape", "feature")
    if orig_bias_shape != new_bias_shape:
        needs_rebuild = True
    # Check if activation function is different.
    orig_act = orig_p.get("act", "Id")
    new_act = new_p.get("act", "Id")
    if orig_act != new_act:
        needs_rebuild = True
    # Also rebuild if any of the incomming sp needs rebuild.
    for s, S in new_net["synapse_pools"].items():
        if S["target"] == np_id:
            if sp_needs_rebuild(orig_net, new_net, s):
                needs_rebuild = True
                break

    # Return rebuild flag.
    return needs_rebuild

