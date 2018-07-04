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
from statestream.utils.pygame_import import pg
from statestream.visualization.graphics import cc, plot_circle
from statestream.utils.properties import array_property



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

    nodes = 1
    for p,P in enumerate(mv_param):
        if P['name'] == 'nodes':
            nodes = P['value']

    # Add statistics as variable.
    shm_layout['variables']['stats'] = {}
    shm_layout['variables']['stats']['shape'] = [3, len(selected_values), nodes]

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
        "name": "nodes",
        "type": "int",
        "min": 4,
        "max": 2048,
        "default": 8
    })
    parameter.append({
        "name": "color",
        "type": "string",
        "min": None,
        "max": None,
        "default": "(200,80,180)"
    })
    parameter.append({
        "name": "stat x",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "mean"
    })
    parameter.append({
        "name": "stat y",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "var"
    })
    parameter.append({
        "name": "stat size",
        "type": "string",
        "min": 0,
        "max": 0,
        "default": "max"
    })
    # Return list of parameter.
    return parameter

# =============================================================================

def type_exists_for(net, children, shm):
    """This function checks if this type of meta-varialbe is possible for the selected children.
    """
    return True

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
    exists = []
    all_nps = True
    all_sssps = True
    all_plasts = True
    all_ifs = True
    # All children have to be of the same type for this meta-variable.
    for c in children:
        if c not in net["neuron_pools"]:
            all_nps = False
        if c not in net["synapse_pools"]:
            all_sssps = False
        if c not in net["plasticities"]:
            all_plasts = False
        if c not in net["interfaces"]:
            all_ifs = False
        if c in net["synapse_pools"]:
            # Check for single-source sps.
            if len(net["synapse_pools"][c]["source"]) != 1 \
                    or len(net["synapse_pools"][c]["source"][0]) != 1:
                all_sssps = False
    if all_nps:
        exists += ["np state"]
        # Check for consistent bias term.
        if "bias_shape" in net["neuron_pools"][children[0]]:
            all_bias = net["neuron_pools"][children[0]]
        else:
            all_bias = "feature"
        for c in children[1:]:
            if "bias_shape" in net["neuron_pools"][c]:
                if all_bias != net["neuron_pools"][c]["bias_shape"]:
                    all_bias = False
                    break
            else:
                if all_bias != "feature":
                    all_bias = False
                    break
        if all_bias is not False:
            exists += ["np par b"]
        # Check for consistent gain term.
        if "gain_shape" in net["neuron_pools"][children[0]]:
            all_gain = net["neuron_pools"][children[0]]
        else:
            all_gain = False
        for c in children[1:]:
            if "gain_shape" in net["neuron_pools"][c]:
                if all_gain != net["neuron_pools"][c]["gain_shape"]:
                    all_gain = False
                    break
            else:
                all_gain = False
                break
        if all_gain is not False:
            exists += ["np par g"]
    if all_sssps:
        exists += ["sp par W_0_0"]
    if all_plasts:
        # Go through all variables and parameters.
        for PV in ["variables", "parameter"]:
            for pv in shm.dat[children[0]][PV]:
                all_pv = True
                for c in children[1:]:
                    if pv not in shm.dat[c][PV]:
                        all_pv = False
                        break
                if all_pv:
                    exists += ["plast " + PV[0:3] + " " + pv]
    if all_ifs:
        # Go through all variables and parameters.
        for PV in ["variables", "parameter"]:
            for pv in shm.dat[children[0]][PV]:
                all_pv = True
                for c in children[1:]:
                    if pv not in shm.dat[c][PV]:
                        all_pv = False
                        break
                if all_pv:
                    exists += ["if " + PV[0:3] + " " + pv]

    # Return result.
    return exists

# =============================================================================

class scalar_three_stats(MetaVariable):
    """Provide parallel visualization for three scalar statistics.

    For a (selected) subset of items (= children) which share a same data type, 
    e.g. all neuron-pools have states, or plasticities having a loss variable,
    this meta-variable computes a scalar statistic, such as mean or variance,
    for each item for the specified data type. Then this meta-variables 
    visualizes these three scalar statistic for all (selected) items.

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

        self.type = "scalar_three_stats"

        # Get parameters.
        self.parameter = {}
        for p in range(len(mv_param)):
            self.parameter[mv_param[p]["name"]] = mv_param[p]["value"]

        # Set meta variable types (selected item parameters / variables / etc.).
        self.sv = copy.deepcopy(client_param['selected_values'])

        # Activation statistics are never itemizable.
        self.itemable = 0
        self.itemized = 0

        # Begin with empty subwindow.
        self.SW = []
        self.shm_layout = client_shm_layout(param, net, mv_param, client_param['selected_values'])

        # Begin with empty statistics.
        self.stat_list = ['x', 'y', 'size']
        self.stats = None
        self.shape = [3, len(self.sv), self.parameter["nodes"]]
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
        if self.SW[-1]['mode'] == 'scatter':
            self.SW[-1]['colormap'] = 'hsv'
            self.SW[-1]['legend right'] \
                = [str(self.sv[i][0]) + " " + str(self.sv[i][1].rstrip()) for i in range(len(self.sv))]
            self.SW[-1]['legend right relative'] = False
            self.SW[-1]['legend right color'] = True
            self.SW[-1]['axis x'] = self.parameter['stat x']
            self.SW[-1]['axis y'] = self.parameter['stat y']



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
        pp_lines.append(self.name + " (scalar_three_stats):")
        pp_lines.append("      values: " + str(len(self.sv)))
        pp_lines.append("       nodes: " + str(self.parameter["nodes"]))
        pp_lines.append("      window: " + str(self.parameter["window"]))
        pp_lines.append("      stat x: " + str(self.parameter["stat x"]))
        pp_lines.append("      stat y: " + str(self.parameter["stat y"]))
        pp_lines.append("   stat size: " + str(self.parameter["stat size"]))
        return pp_lines
    