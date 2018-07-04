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

from statestream.meta.variable import MetaVariable
from statestream.meta.network import shortest_path
from statestream.utils.pygame_import import pg
from statestream.visualization.graphics import cc, plot_circle



# =============================================================================

def client_shm_layout(param, net, mv_param, selected_values):
    """Function returns individualized shm layout for system clients.

    For a client, parameters could hold some information how online
    computation the client performs is parameterized.
    Variables store output results of the client.
    """
    # Get / set default dtype.
    dtype = eval(param["core"]["dtype_default"])
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}

    # Get source np.
    source = None
    for p in mv_param:
        if p['name'] == 'source':
            source = p['value']

    source_shape = net["neuron_pools"][source]['shape']

    # Add variables.
    # ------------------------------------------------------------------------
    # Gradient of magic(target) with respect to source.
    shm_layout['variables']['grads'] = {}
    shm_layout['variables']['grads']['shape'] = [1] + source_shape
    # Average-pooled gradient of magig(target) with respect to source.
    shm_layout['variables']['avgpool_grads'] = {}
    shm_layout['variables']['avgpool_grads']['shape'] = [1, source_shape[0], 1, 1]
    # Weighted feature maps (source) using average-pooled gradients.
    shm_layout['variables']['grad_weighted_maps'] = {}
    shm_layout['variables']['grad_weighted_maps']['shape'] = [1] + source_shape
    # Mean over all weighted source maps.
    shm_layout['variables']['grad_weighted_map'] = {}
    shm_layout['variables']['grad_weighted_map']['shape'] = [1, 1, source_shape[1], source_shape[2]]

    # Return layout.
    return shm_layout

# =============================================================================

def get_parameter(net, children):
    """Return list of parameter dicts for this type of meta-variable.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    children : list of str
        This list contains the item names of all child items this meta-variable
        is derived from.

    Returns:
    --------
    parameter : list of dict
        List of all parameters for this meta-variables. This list is used to 
        get the parameters at meta-variable instantiation from the user via
        the GUI and is then used for parameterized calculation of the meta-
        variable.
    """
    # Determine source and target np.
    path_01 = shortest_path(net, children[0], children[1])
    path_10 = shortest_path(net, children[1], children[0])
    if len(path_01) > 1 and len(path_10) > 1:
        if len(path_01) < len(path_10):
            source = children[0]
            target = children[1]
        else:
            source = children[1]
            target = children[0]
    elif len(path_01) > 1 and len(path_10) in [0, 1]:
        source = children[0]
        target = children[1]
    elif len(path_01) in [0, 1] and len(path_10) > 1:
        source = children[1]
        target = children[0]
    parameter = []
    # Add window size over which to compute the mean confusion matrix.
    parameter.append({
        "name": "name",
        "type": "string",
        "default": "name"
    })
    parameter.append({
        "name": "source",
        "type": "string",
        "default": source
    })
    parameter.append({
        "name": "target",
        "type": "string",
        "default": target
    })
    parameter.append({
        "name": "color",
        "type": "string",
        "min": None,
        "max": None,
        "default": "(200,80,180)"
    })
    parameter.append({
        "name": "magic",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "T.max(#)"
    })
    parameter.append({
        "name": "device",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "cpu"
    })
    # Return list of parameter.
    return parameter

# =============================================================================

def type_exists_for(net, children, shm):
    """This function checks if this type of meta-varialbe is possible for the selected children.
    """
    exists = True
    if len(children) != 2:
        exists = False
    for c in children:
        if c not in net["neuron_pools"]:
            exists = False
            break
    # Check for path between nps.
    if exists:
        path_01 = shortest_path(net, children[0], children[1])
        path_10 = shortest_path(net, children[1], children[0])
        if len(path_01) == 0 and len(path_10) == 0:
            exists = False
    return exists

# =============================================================================

def exists_for(net, children, shm):
    """This function checks which instances of this meta-variable exist.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    children : list of str
        This list contains the item names of all child items this meta-variable
        is derived from.
    shm : class SharedMemory 
        This is a pointer to the entire shared memory. We use this here, because the
        SharedMemory has a representation of the entire internal structure of 
        variables / parameters / etc. for all items.
    Returns:
    --------
    exists : boolean
        This flag specifies if this type of meta-variable actually exists for the selected
        child items.
    """

    # At most 2 children, and they have to be nps.
    exists = True
    if len(children) != 2:
        exists = False
    for c in children:
        if c not in net["neuron_pools"]:
            exists = False
            break
    # Check for path between nps.
    if exists:
        path_01 = shortest_path(net, children[0], children[1])
        path_10 = shortest_path(net, children[1], children[0])
        if len(path_01) == 0 and len(path_10) == 0:
            exists = False
    if exists:
        return ['np layer']
    else:
        return []

# =============================================================================

class saliency_grad(MetaVariable):
    """Provide gradient based saliency visualization for two tensors.

    References:
    -----------
    [Selvaraju et al. 2017]

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.
    mv_param : list of dict
        Dictionary containing parameters of this meta-variable. Each meta-variable
        has to provide this structure by a function get_parameter(net, children).
    children : list of str
        This list contains the item names of all child items this meta-variable
        is derived from.
    """
    def __init__(self, net, param, mv_param=[], client_param={}):
        # Initialize parent class.
        MetaVariable.__init__(self, net, param, mv_param, client_param)

        self.type = "saliency_grad"

        # Get parameters.
        self.parameter = {}
        for p in range(len(mv_param)):
            self.parameter[mv_param[p]["name"]] = mv_param[p]["value"]

        # Meta-var is not itemable.
        self.itemable = 0
        self.itemized = 0

        # Begin with empty subwindow.
        self.SW = []
        self.shm_layout = client_shm_layout(param, net, mv_param, client_param['selected_values'])

        # Set meta variable types (selected item parameters / variables / etc.).
        self.sv = copy.deepcopy(client_param['selected_values'])
        
        # Generate local statistics.
        self.stats = None

        self.current_frame = 0
        
        # Get color.
        try:
            self.col = eval(self.parameter["color"])
        except:
            self.col = (255, 255, 255)



    def adjust_subwin(self):
        """This method adjusts the last added sub-window for this meta-variable.
        """
        pass



    def plot(self, shm, screen, viz_brain, ccol):
        """Plot meta-variable to screen.

        Parameters:
        -----------
        shm : SharedMemory class
            This is a class holding references to the entire shared memory of this ST session.
        screen : pygame.Surface
            The pygame surface to draw the meta-variable onto.
        viz_brain : dict
            The brain dictionary containing visualization information about all items
            (see also the statestream.visualization.visualization.brain).
        ccol : boolean
            Flag for color correction.

        Returns:
        --------
            plotted : boolean
                Flag if the meta-variable was actually drawn here. If not a fallback drawing can
                be tried in the visualization.
        """
        return False


        
    # Pretty print for meta variable, returns a nice list of lines.
    def pprint(self):
        """Method returns specialized list of lines for pretty print.
        """
        pp_lines = []
        pp_lines.append(self.name + " (saliency_grad):")
        pp_lines.append("      source: " + self.parameter['source'])
        pp_lines.append("      target: " + self.parameter['target'])
        pp_lines.append("       magic: " + self.parameter["magic"])
        return pp_lines
    