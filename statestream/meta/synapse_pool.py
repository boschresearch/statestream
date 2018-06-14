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

from statestream.meta.neuron_pool import np_state_shape
from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.utils.helper import is_scalar_shape



def sp_get_dict(sub_dict, param, default):
    """This method yields a standard interface to list of list params.

    Parameters:
    -----------
    sub_dict : dict
        Items sub-dictionary of the network dictionary.
    param : str
        String specifying the parameter to extract.
    default : None, int, etc.
        The default value for the parameter if none given.
    """
    # Determine if param is present in specification.
    value = sub_dict.get(param, None)
    # Get list of all sources.
    sources = [src for src_list in sub_dict["source"] for src in src_list]
    # Dependent on sources layout specify parameter value.
    if len(sources) == 1:
        if value is None:
            value = [[default]]
        else:
            if not isinstance(value, list):
                value = [[value]]
    else:
        if value is None:
            value = []
            for src_list in range(len(sub_dict["source"])):
                value.append([])
                for s in sub_dict["source"][src_list]:
                    value[-1].append(default)
    return value



def sp_shm_layout(name, net, param):
    """This function returns the entire shared memory layout for a synapse-pool.

    Parameters:
    -----------
    name : str
        The unique string identifier for the synapse-pool.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    Returns:
    --------
    shm_layout : dict
        Dictionary containig shared memory layouts for all vars / pars of the synapse-pool.
    """
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    
    # Add layout for all parameter
    # -------------------------------------------------------------------------
    shm_layout["parameter"] = {}
    p = net["synapse_pools"][name]
    # Set noise parameters.
    if "noise" in p:
        if p["noise"] == "normal":
            shm_layout["parameter"]["noise_mean"] \
                = ShmL("backend", (), dtype, 0.0)
            shm_layout["parameter"]["noise_std"] \
                = ShmL("backend", (), dtype, 1.0, 0.0, None)
        elif p["noise"] == "uniform":
            shm_layout["parameter"]["noise_min"] = ShmL("backend", (), dtype, 0.0)
            shm_layout["parameter"]["noise_max"] = ShmL("backend", (), dtype, 0.0)
    # Dependent on number of factors determine layout.
    no_factors = len(p["source"])

    # Get factor shape (all concat inputs of a factor will be treated 
    # with this shape).
    factor_shapes = p.get("factor_shapes", ["full" for f in range(no_factors)])
    # Get / set target_shapes for all inputs.
    if "target_shapes" in p:
        target_shapes = p["target_shapes"]
    else:
        target_shapes = []
        for f in range(no_factors):
            target_shapes.append([])
            for i in range(len(p["source"][f])):
                target_shapes[-1].append(factor_shapes[f])
    # Get / set weight_shapes for all inputs.
    weight_shapes = sp_get_dict(p, "weight_shapes", "full")
    # Get / set bias shapes for each factor.
    src_nps = [s for src_list in p["source"] for s in src_list]
    if "bias_shapes" in p:
        bias_shapes = p["bias_shapes"]
    else:
        # For simple case of single input do not use bias_shapes
        # by default.
        if len(src_nps) == 1:
            bias_shapes = None
        else:
            bias_shapes = []
            for f in range(no_factors):
                if factor_shapes[f] in ["full", "feature"]:
                    bias_shapes.append("feature")
                else:
                    bias_shapes.append(factor_shapes[f])
    # get source and target
    snps = p["source"]
    tnp = p["target"]
    snps_shape = []
    # Determine source neuron_pool"s shapes.
    for f in range(no_factors):
        snps_shape.append([])
        for i in range(len(snps[f])):
            snps_shape[-1].append(np_state_shape(net, snps[f][i]))
    # Get target neuron_pools shape.
    tnp_shape = np_state_shape(net, tnp)
    # Get target shape factor for avg/max-out.
    avgout = p.get("avgout", 1)
    maxout = p.get("maxout", 1)
    # Get rf size (as list of lists of rfs, one for each input).
    rf_size = sp_get_dict(p, "rf", 0)
    # Get preprocessing projection dimensions.
    ppp_dims = sp_get_dict(p, "ppp", 0)
    # Loop over all factors.
    for f in range(no_factors):
        # Add weights for each factor / input.
        for i in range(len(snps[f])):
            # Specify parameter names.
            W_name = "W_" + str(f) + "_" + str(i)
            P_name = "P_" + str(f) + "_" + str(i)
            # Get unshare flag for all inputs.
            unshared = False
            W_preshape = []
            P_preshape = []
            # Check for unshared sp parameters.
            if "unshare" in p:
                if W_name in p["unshare"]:
                    unshared = True
                    W_preshape = [net["agents"]]
                if P_name in p["unshare"]:
                    unshared = True
                    P_preshape = [net["agents"]]
            # Determine number of target features from target shapes.
            if target_shapes[f][i] in ["full", "feature"]:
                target_features = tnp_shape[1] * avgout * maxout
            elif target_shapes[f][i] in ["scalar", "spatial"]:
                target_features = 1
            # Proceed dependent on given rf size.
            # Dependent on target-shapes set rf_size_x/y.
            if target_shapes[f][i] in ["full", "spatial"]:
                # Set filter shape dependent on 1D or 2D input.
                if rf_size[f][i] == 0:
                    rf_size_x = snps_shape[f][i][2]
                    rf_size_y = snps_shape[f][i][3]
                else:
                    if snps_shape[f][i][3] == 1:
                        rf_size_x = rf_size[f][i]
                        rf_size_y = 1
                    else:
                        rf_size_x = rf_size[f][i]
                        rf_size_y = rf_size[f][i]
            elif target_shapes[f][i] in ["feature", "scalar"]:
                # Set filter shape dependent on 1D or 2D input.
                if rf_size[f][i] == 0:
                    rf_size_x = snps_shape[f][i][2]
                    rf_size_y = snps_shape[f][i][3]
                else:
                    if snps_shape[f][i][3] == 1:
                        rf_size_x = rf_size[f][i]
                        rf_size_y = 1
                    else:
                        rf_size_x = rf_size[f][i]
                        rf_size_y = rf_size[f][i]
            # Set ppp shapes where needed.
            source_feat_shape = snps_shape[f][i][1]
            if ppp_dims[f][i] > 0:
                source_feat_shape = ppp_dims[f][i]
                shm_layout["parameter"][P_name] \
                    = ShmL("backend",
                           P_preshape + [ppp_dims[f][i],
                           snps_shape[f][i][1],
                           1,
                           1],
                           dtype,
                           "xavier")
            # Finally set shape of W_f_i.
            if weight_shapes[f][i] == "full":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [target_features,
                            source_feat_shape, 
                            rf_size_x, 
                            rf_size_y], 
                           dtype, 
                           "xavier")
            elif weight_shapes[f][i] == "spatial":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [1,
                            1, 
                            rf_size_x, 
                            rf_size_y], 
                           dtype, 
                           "xavier",
                           broadcastable=[True, True, False, False])
            elif weight_shapes[f][i] == "feature":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [target_features,
                            source_feat_shape, 
                            1, 
                            1], 
                           dtype, 
                           "xavier",
                           broadcastable=[False, False, True, True])
            elif weight_shapes[f][i] == "src_feature":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [1,
                            source_feat_shape, 
                            1, 
                            1], 
                           dtype, 
                           "xavier",
                           broadcastable=[True, False, True, True])
            elif weight_shapes[f][i] == "tgt_feature":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [target_features,
                            1, 
                            1, 
                            1], 
                           dtype, 
                           "xavier",
                           broadcastable=[False, True, True, True])
            elif weight_shapes[f][i] == "src_spatial":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [1,
                            source_feat_shape, 
                            rf_size_x, 
                            rf_size_y], 
                           dtype, 
                           "xavier",
                           broadcastable=[True, False, False, False])
            elif weight_shapes[f][i] == "tgt_spatial":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [target_features,
                            1, 
                            rf_size_x, 
                            rf_size_y], 
                           dtype, 
                           "xavier",
                           broadcastable=[False, True, False, False])
            elif weight_shapes[f][i] == "scalar":
                shm_layout["parameter"][W_name] \
                    = ShmL("backend", 
                           W_preshape + [1,
                            1, 
                            1, 
                            1], 
                           dtype, 
                           "xavier",
                           broadcastable=[False, False, False, False])

        # For each factor add a bias parameter.
        if bias_shapes is not None:
            if bias_shapes[f] == "full":
                shm_layout["parameter"]["b_" + str(f)] \
                    = ShmL("backend", 
                           [tnp_shape[1], tnp_shape[2], tnp_shape[3]], 
                           dtype, 
                           0.0)
            elif bias_shapes[f] == "feature":
                shm_layout["parameter"]["b_" + str(f)] \
                    = ShmL("backend", 
                           [tnp_shape[1],], 
                           dtype, 
                           0.0)
            elif bias_shapes[f] == "spatial":
                shm_layout["parameter"]["b_" + str(f)] \
                    = ShmL("backend", 
                           [tnp_shape[2], tnp_shape[3]], 
                           dtype, 
                           0.0)
            elif bias_shapes[f] == "scalar":
                shm_layout["parameter"]["b_" + str(f)] \
                    = ShmL("backend", 
                           [1,], 
                           dtype,
                           0.0)

    # Add layout for all variables
    # -------------------------------------------------------------------------
    shm_layout["variables"] = {}

    # Return layout for synapse pool.
    return shm_layout





def sp_init(net, name, dat_name, dat_layout, mode=None):
    """Initialize / set synapse-pool data (parameter, variables).

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string identifier for the synapse-pool.
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
    p = net["synapse_pools"][name]
    # Default return is None.
    dat_value = None
    # Set all scalars to their default value.
    if is_scalar_shape(dat_layout.shape):
        dat_value = p.get(dat_name, dat_layout.default)
    else:
        if mode is None:
            # Try as specified, otherwise use default.
            init_as = p.get("init " + dat_name, None)
            if init_as is None:
                init_as = dat_layout.default
        else:
            init_as = mode
            
        # Dependent on init_as initialize.
        if dat_name[0] in ["W", "P"]:
            if isinstance(init_as, str):
                if init_as == "xavier":
                    # Compute fan_in / fan_out and weight bounds.
                    fan_in  = dat_layout.shape[1] * dat_layout.shape[2] \
                              * dat_layout.shape[3]
                    fan_out = dat_layout.shape[0] * dat_layout.shape[2] \
                              * dat_layout.shape[3]
                    W_bound = np.sqrt(6. / (fan_in + fan_out))
                    dat_value \
                        = np.asarray(np.random.RandomState(42).uniform(low=-W_bound,
                                                                       high=W_bound,
                                                                       size=dat_layout.shape), 
                                     dtype=dat_layout.dtype)
                elif init_as.startswith("xavier_"):
                    # Determine parameter.
                    try:
                        x_par = float(init_as.split("_")[1])
                    except:
                        x_par = 1.0
                    # Compute fan_in / fan_out and weight bounds.
                    fan_in  = dat_layout.shape[1] * dat_layout.shape[2] \
                              * dat_layout.shape[3]
                    fan_out = dat_layout.shape[0] * dat_layout.shape[2] \
                              * dat_layout.shape[3]
                    W_bound = x_par * np.sqrt(6. / (fan_in + fan_out))
                    dat_value \
                        = np.asarray(np.random.RandomState(42).uniform(low=-W_bound,
                                                                       high=W_bound,
                                                                       size=dat_layout.shape), 
                                     dtype=dat_layout.dtype)
                elif init_as in ["id", "Id"]:
                    dat_value = np.zeros(dat_layout.shape, dtype=dat_layout.dtype)
                    # Simple identity for source features equal target features.
                    if dat_layout.shape[0] == dat_layout.shape[1]:
                        # Loop over all target features.
                        for f in range(dat_layout.shape[0]):
                            dat_value[f, f, dat_layout.shape[2] // 2, dat_layout.shape[3] // 2] \
                                = 1.0
                    # In case target target dim is multiple of source dim.
                    elif dat_layout.shape[0] > dat_layout.shape[1] and dat_layout.shape[0] % dat_layout.shape[1] == 0:
                        _factor = dat_layout.shape[0] // dat_layout.shape[1]
                        eye = np.eye(dat_layout.shape[1])
                        one = np.ones([_factor,1])
                        if init_as == "id":
                            kron = np.kron(eye, one)
                        else:
                            kron = np.kron(one, eye)
                        for f_tgt in range(dat_layout.shape[0]):
                            for f_src in range(dat_layout.shape[1]):
                                dat_value[f_tgt, f_src, dat_layout.shape[2] // 2, dat_layout.shape[3] // 2] \
                                    = kron[f_tgt, f_src]
                elif init_as.startswith("id_") or init_as.startswith("Id_"):
                    # Determine parameter.
                    try:
                        x_par = float(init_as.split("_")[1])
                    except:
                        x_par = 1.0
                    dat_value = np.zeros(dat_layout.shape, dtype=dat_layout.dtype)
                    # Assert source features equal target features.
                    assert (dat_layout.shape[0] == dat_layout.shape[1]), \
                        "ERROR during init of sp " + str(name) \
                        + ": init as id requires same source and target features."
                    # Loop over all target features.
                    for f in range(dat_layout.shape[0]):
                        dat_value[f, f, dat_layout.shape[2] // 2, dat_layout.shape[3] // 2] \
                            = x_par
                elif init_as in ["-id", "-Id"]:
                    dat_value = np.zeros(dat_layout.shape, dtype=dat_layout.dtype)
                    # Assert source features equal target features.
                    assert (dat_layout.shape[0] == dat_layout.shape[1]), \
                        "ERROR during init of sp " + str(name) \
                        + ": init as id requires same source and target features."
                    # Loop over all target features.
                    for f in range(dat_layout.shape[0]):
                        dat_value[f, f, dat_layout.shape[2] // 2, dat_layout.shape[3] // 2] \
                            = -1.0
                elif init_as == "loc_inh":
                    inhibition_factor \
                        = -1 / float(dat_layout.shape[1] \
                                     * dat_layout.shape[2] \
                                     * dat_layout.shape[3])
                    dat_value = inhibition_factor * np.ones(dat_layout.shape, 
                                                            dtype=dat_layout.dtype)
                    # Assert source features par_shape target features.
                    assert (dat_layout.shape[0] == dat_layout.shape[1]), \
                        "ERROR during init of sp " + str(name) \
                        + ": init as id requires same source and target features."
                    # Loop over all target features.
                    for f in range(dat_layout.shape[0]):
                        dat_value[f, f, dat_layout.shape[2] // 2, dat_layout.shape[3] // 2] \
                            = 0.5
                elif init_as == "normal":
                    dat_value = np.random.normal(loc=0.0, 
                                                 scale=1.0, 
                                                 size=dat_layout.shape).astype(dat_layout.dtype)
                elif init_as.startswith("normal_"):
                    mu = float(init_as.split("_")[1])
                    std = float(init_as.split("_")[2])
                    dat_value = np.random.normal(loc=mu, 
                                                 scale=std, 
                                                 size=dat_layout.shape).astype(dat_layout.dtype)
                elif init_as == "bilin":
                    dat_value = np.zeros(dat_layout.shape, dtype=dat_layout.dtype)
                    # Assert source features equal target features.
                    assert (dat_layout.shape[0] == dat_layout.shape[1]), \
                        "ERROR during init of sp " + str(name) \
                        + ": init as id requires same source and target features."
                    # Distinguish 1D / 2D bilin-interpolation.
                    if dat_layout.shape[0] > 1 and dat_layout.shape[1] > 1:
                        # 2D bilin-interpolation.
                        # center of rf
                        rf_x = dat_layout.shape[2]
                        rf_y = dat_layout.shape[3]
                        for i_x in np.arange(-(rf_x // 2), rf_x // 2 + 1):
                            for i_y in np.arange(-(rf_y // 2), rf_y // 2 + 1):
                                # Compute "inverse" dist from center.
                                if i_x == 0 and i_y == 0:
                                    D = 1.0
                                else:
                                    D = 1.0 / (2.0 * (np.abs(float(i_x)) \
                                                      + np.abs(float(i_y))))
                                # Loop over all target features.
                                for f in range(dat_layout.shape[0]):
                                    dat_value[f, f, i_x + rf_x // 2, i_y + rf_y // 2] \
                                        = D
            else:
                dat_value = (float(init_as) * np.ones(dat_layout.shape, dtype=dat_layout.dtype)).astype(dat_layout.dtype)

        else:
            if init_as in ["zero", 0]:
                dat_value = np.zeros(dat_layout.shape, dtype=dat_layout.dtype)
            elif init_as in ["one", 1]:
                dat_value = np.ones(dat_layout.shape, dtype=dat_layout.dtype)
    # Return initialized value.
    return dat_value



def sp_needs_rebuild(orig_net, new_net, sp_id):
    """Determines if a synapse-pool needs rebuild for new network dictionary.
    
    Parameters:
    orig_net : dict
        Dictionary containing the original network.
    new_net : dict
        Dictionary containing the new edited network.
    sp_id : str
        String specifying the synapse-pool under consideration.
    """
    needs_rebuild = False
    # Get synapse-pool sub-dict.
    orig_p = orig_net["synapse_pools"][sp_id]
    new_p = new_net["synapse_pools"][sp_id]
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

    # Return rebuild flag.
    return needs_rebuild
