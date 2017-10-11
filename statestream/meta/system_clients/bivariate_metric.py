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
import copy

from statestream.meta.variable import MetaVariable
from statestream.utils.pygame_import import pg
from statestream.visualization.graphics import cc, plot_circle
from statestream.utils.properties import np_feature_metric



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

    if len(selected_values) == 1:
        np0 = selected_values[0][0]
        np1 = selected_values[0][0]
    else:
        np0 = selected_values[0][0]
        np1 = selected_values[1][0]
    shape = [net["neuron_pools"][np0]['shape'][0],
             net["neuron_pools"][np1]['shape'][0]]

    # Add statistics as variable.
    shm_layout['variables']['stats'] = {}
    shm_layout['variables']['stats']['shape'] = shape

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
    parameter = []
    # Add window size over which to compute the mean confusion matrix.
    parameter.append({
        "name": "name",
        "type": "string",
        "default": "name"
    })
    parameter.append({
        "name": "window",
        "type": "int",
        "min": 1,
        "max": 2048,
        "default": 8
    })
    parameter.append({
        "name": "samples",
        "type": "int",
        "min": 1,
        "max": net['agents'],
        "default": 1
    })
    parameter.append({
        "name": "color",
        "type": "string",
        "min": None,
        "max": None,
        "default": "(200,80,180)"
    })
    parameter.append({
        "name": "metric",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "cos"
    })
    parameter.append({
        "name": "device",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "cpu"
    })
    parameter.append({
        "name": "backend",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "c"
    })
    # Return list of parameter.
    return parameter

# =============================================================================

def type_exists_for(net, children, shm):
    """This function checks if this type of meta-varialbe is possible for the selected children.
    """
    exists = True
    if len(children) > 2:
        exists = False
    for c in children:
        if c not in net["neuron_pools"]:
            exists = False
            break
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
    if len(children) > 2:
        exists = False
    for c in children:
        if c not in net["neuron_pools"]:
            exists = False
            break
    if exists:
        return ['np state']
    else:
        return []

# =============================================================================

class bivariate_metric(MetaVariable):
    """Provide visualization of a metric between two tensors.

    For now we only allow metrics between two (maybe same) nps.

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

        self.type = "bivariate_metric"

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
        
        # Begin with empty statistics.
        self.stats = None
        if len(self.selected_values) == 1:
            self.np0 = self.selected_values[0][0]
            self.np1 = self.selected_values[0][0]
        else:
            self.np0 = self.selected_values[0][0]
            self.np1 = self.selected_values[1][0]
        self.shape = [self.net["neuron_pools"][self.np0]['shape'][0],
                      self.net["neuron_pools"][self.np1]['shape'][0]]

        self.window = None
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
        pp_lines.append(self.name + " (bivariate_metric):")
        pp_lines.append("      values: " + self.np0 + "  " + self.np1)
        pp_lines.append("      metric: " + self.parameter["metric"])
        pp_lines.append("      window: " + str(self.parameter["window"]))
        return pp_lines
    