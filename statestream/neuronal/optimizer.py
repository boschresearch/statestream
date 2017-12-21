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





# =============================================================================
# =============================================================================
# =============================================================================

def grad_desc(params, params_id, grads, dat, B):
    """Standard (stochastic) gradient descent with fixed stepwidth lr.

    # Parameters:
        params : list of np/th/etc. variables
            This list contains all parameters to be updated by the plasticity.
        params_id : list of str
            This list holds the 'item type + item name + param name' for all
            parameters to be updated by the plasticity.
        grads : list of np/th/etc. variables
            This list contains all gradients for parameters to be updated by the
            plasticity. In shapes / dims it is identical to params.
        dat : dict
            This dictionary holds the local (for this process) data copy from
            shared memory for this plasticities variables / parameter (not
            to be confused with the parameters to be updated by this plast). 
            It is created using the structure given by the plast_shm_layout() function that has
            to be provided by each plasticity.

    # Returns:
        updates : list of np/th/etc. variables
            This is the list containing all parameter updates of the plasticity
            as well as all updates for plasticity internal variables.
    """
    updates = []
    for param_i, grad_i, p_id in zip(params, grads, params_id):
        updates.append(B.update(param_i, B.clip(-dat["parameter"]["lr"] * grad_i, -1.0, 1.0)))
    # Update gradients to variables.
    for p in range(len(params)):
        updates.append(B.update(dat["variables"]["grad." + params_id[p]], grads[p]))
    return updates



# =============================================================================
# =============================================================================
# =============================================================================



# params_id = "np/sp name par" for all params (or grads)
def adam(params, params_id, grads, dat, B):
    """The Adam optimizer.
    
    # Parameters:
        params : list of np/th/etc. variables
            This list contains all parameters to be updated by the plasticity.
        params_id : list of str
            This list holds the 'item type + item name + param name' for all
            parameters to be updated by the plasticity.
        grads : list of np/th/etc. variables
            This list contains all gradients for parameters to be updated by the
            plasticity. In shapes / dims it is identical to params.
        dat : dict
            This dictionary holds the local (for this process) data copy from
            shared memory for this plasticities variables / parameter (not
            to be confused with the parameters to be updated by this plast). 
            It is created using the structure given by the plast_shm_layout() function that has
            to be provided by each plasticity.

    # Returns:
        updates : list of np/th/etc. variables
            This is the list containing all parameter updates of the plasticity
            as well as all updates for plasticity internal variables.

    # Reference:
        Diederik Kingma, Jimmy Ba
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """
    # Begin with empty update list.
    updates = []
    # Update time and add it to updates.
    t = dat["variables"]["t"] + 1
    updates.append(B.update(dat["variables"]["t"], t))
    # Collect moments and variances in lists.
    _moments = []
    _varianc = []
    for p in range(len(params)):
        _moments.append(dat["variables"]["moments." + params_id[p]])
        _varianc.append(dat["variables"]["varianc." + params_id[p]])
    # Update moments.
    for grad_i, mom_i in zip(grads, _moments):
        updates.append(B.update(mom_i, dat["parameter"]["momentum"] * mom_i + \
                                (1 - dat["parameter"]["momentum"]) * grad_i))
    # Compute updates for variances.
    for grad_i, var_i in zip(grads, _varianc):
        updates.append(B.update(var_i, dat["parameter"]["decay"] * var_i + \
                                (1 - dat["parameter"]["decay"]) * grad_i**2))
    # Compute parameter updates.
    for p_i, grad_i, mom_i, var_i in zip(params, grads, _moments, _varianc):
        updates.append(B.update(p_i, -(dat["parameter"]["lr"] \
                                * B.sqrt(1 - dat["parameter"]["decay"]**t) \
                                / (1 - dat["parameter"]["momentum"]**t)) \
                                * (mom_i / (B.sqrt(var_i) + 1e-6))))
    # Update gradients to variables.
    for p in range(len(params)):
        updates.append(B.update(dat["variables"]["grad." + params_id[p]], grads[p]))
    # Finally return updates.
    return updates

    
    
# =============================================================================
# =============================================================================
# =============================================================================



# params_id = "np/sp name par" for all params (or grads)
def rmsprop(params, params_id, grads, dat, B):
    """The RMSprop optimizer.
    
    # Parameters:
        params : list of np/th/etc. variables
            This list contains all parameters to be updated by the plasticity.
        params_id : list of str
            This list holds the 'item type + item name + param name' for all
            parameters to be updated by the plasticity.
        grads : list of np/th/etc. variables
            This list contains all gradients for parameters to be updated by the
            plasticity. In shapes / dims it is identical to params.
        dat : dict
            This dictionary holds the local (for this process) data copy from
            shared memory for this plasticities variables / parameter (not
            to be confused with the parameters to be updated by this plast). 
            It is created using the structure given by the plast_shm_layout() function that has
            to be provided by each plasticity.

    # Returns:
        updates : list of np/th/etc. variables
            This is the list containing all parameter updates of the plasticity
            as well as all updates for plasticity internal variables.

    # Reference:
        [RMSprop]
        Divide the gradient by a running average of its recent magnitude.
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    # Begin with empty update list.
    updates = []
    # Update time and add it to updates.
    t = dat["variables"]["t"] + 1
    updates.append(B.update(dat["variables"]["t"], t))
    # Collect accumulations in lists.
    _accumulations = []
    for p in range(len(params)):
        _accumulations.append(dat["variables"]["accumulation." + params_id[p]])
    # Update accumulations.
    for p_i, grad_i, acc_i in zip(params, grads, _accumulations):
        next_acc = dat["parameter"]["rho"] * acc_i \
                   + (1 - dat["parameter"]["rho"]) * grad_i**2
        updates.append(B.update(acc_i, next_acc))
        updates.append(B.update(p_i, - dat["parameter"]["lr"] * grad_i \
                            /   (B.sqrt(next_acc) + 1e-7)))

    # Update gradients to variables.
    for p in range(len(params)):
        updates.append(B.update(dat["variables"]["grad." + params_id[p]], grads[p]))
    # Finally return updates.
    return updates

    
    
# =============================================================================
# =============================================================================
# =============================================================================


