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



import sys
import numpy as np
import gzip

if sys.version[0] == "2":
    import cPickle as pckl
elif sys.version[0] == "3":
    import pickle as pckl

from statestream.interfaces.process_if import ProcessIf
from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.meta.neuron_pool import np_state_shape



def if_interfaces():
    """Returns the specific interfaces as strings of the interface.
    Parameters:
    -----------
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string name of this interface.
    """
    # Specify sub-interfaces.
    return {"out": ["random"],
            "in": []
           }



def if_init(net, name, dat_name, dat_layout, mode=None):
    """Return value for interface parameter / variable.

    Parameters:
    -----------
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string name of this interface.
    dat_name : str
        The unique name for the parameter to initialize.
    dat_layout : SharedMemoryLayout
        The layout (see SharedMemoryLayout) of the parameter to be initialized.
    mode : None, value (float, int), str
        The mode determines how a parameter has to be initialized (e.g. 'xavier' or 0.0)
    """
    # Default return is None.
    dat_value = None

    # Return initialized value.
    return dat_value



def if_shm_layout(name, net, param):
    """Return shared memory layout for the random interface.

    Parameters:
    -----------
    name : str
        The unique interface name.
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    """
    # Get interface dictionary.
    p = net["interfaces"][name]
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}

    # Add parameter.
    # -------------------------------------------------------------------------
    # Add mode parameter.
    shm_layout["parameter"]["mode"] \
        = ShmL("np", (), np.int32, p.get("mode", 0), 1, None)

    if p["distribution"] == "bernoulli":
        shm_layout["parameter"]["p"] \
            = ShmL("np", (), np.float32, p.get("p", 0.5), 1, None)
    elif p["distribution"] == "normal":
        shm_layout["parameter"]["mean"] \
            = ShmL("np", (), np.float32, p.get("mean", 0.0), 1, None)
        shm_layout["parameter"]["std"] \
            = ShmL("np", (), np.float32, p.get("std", 1.0), 1, None)
    elif p["distribution"] == "uniform":
        shm_layout["parameter"]["min"] \
            = ShmL("np", (), np.float32, p.get("min", 0.0), 1, None)
        shm_layout["parameter"]["max"] \
            = ShmL("np", (), np.float32, p.get("max", 1.0), 1, None)

    # Add variables.
    # -------------------------------------------------------------------------
    # Add all outputs as variables.
    for o in p["out"]:
        tmp_target = o
        # Consider remapping.
        if "remap" in p:
            if o in p["remap"]:
                tmp_target = p["remap"][o]
        # Set layout.
        shm_layout["variables"][o] \
            = ShmL("np", np_state_shape(net, tmp_target), np.float32, 0)

    # Return layout.
    return shm_layout

    

class ProcessIf_random(ProcessIf):
    """Interface class providing basic random activations.

        To use this interface no external dataset is needed.

        Parameters:
        -----------
        name : str
            Unique interface identifier.
        ident : int
            Unique id for the interface's process.
        net : dict
            The network dictionary containing all nps, sps, plasts, ifs.
        param : dict
            Dictionary of core parameters.

        Interface parameters:
        ---------------------
        distribuation : string
            Specifies the used random distribution used. Available:
            - bernoulli: Bernoullie distribution.
                - p : float32
                    Specifies the probability for an event.
            - normal: Normal distribution.
                - mean: float32
                - std: Standard deviation.
            - uniform: Uniform distribution.
                - min: float32
                - max: float32

        Inputs:
        -------

        Outputs:
        --------
        random : np.array, shape [agents, c np, x np, y np]
            The random output neuron-pool.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)



    def initialize(self):
        """Method to initialize the random interface.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        if self.p["distribution"] == "bernoulli":
            self.dat["parameter"]["p"] = self.p.get("p", 0.5)
        elif self.p["distribution"] == "normal":
            self.dat["parameter"]["mean"] = self.p.get("mean", 0.0)
            self.dat["parameter"]["std"] = self.p.get("std", 1.0)
        elif self.p["distribution"] == "uniform":
            self.dat["parameter"]["min"] = self.p.get("min", 0.0)
            self.dat["parameter"]["max"] = self.p.get("max", 1.0)

        # Get shapes of important neuron-pools.
        self.np_shape = {}
        for o in self.p["out"] + self.p["in"]:
            tmp_target = o
            # Consider remapping.
            if "remap" in self.p:
                if o in self.p["remap"]:
                    tmp_target = self.p["remap"][o]
            if o not in self.np_shape:
                self.np_shape[o] = np_state_shape(self.net, tmp_target)

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
 
        # Initialize global normal filter.
        # ---------------------------------------------------------------------
 


    def draw_sample(self, sample):
        """Draw a new random sample.
        """
        pass



    def update_sample(self, sample):
        """Update a sample.
        """
        pass



    def update_frame_writeout(self):
        """Method to update the experimental state of the random interface.
        """
        # Draw from distribution.
        if self.p["distribution"] == "bernoulli":
            self.dat["variables"]["random"][:,:,:,:] \
                = np.random.binomial(n=1, 
                                     p=self.dat["parameter"]["p"],
                                     size=self.np_shape["random"])[:,:,:,:]
        elif self.p["distribution"] == "normal":
            self.dat["variables"]["random"][:,:,:,:] \
                = np.random.normal(loc=self.dat["parameter"]["mean"],
                                   scale=self.dat["parameter"]["std"],
                                   size=self.np_shape["random"])[:,:,:,:]
        elif self.p["distribution"] == "uniform":
            self.dat["variables"]["random"][:,:,:,:] \
                = np.random.uniform(low=self.dat["parameter"]["min"],
                                   high=self.dat["parameter"]["max"],
                                   size=self.np_shape["random"])[:,:,:,:]

