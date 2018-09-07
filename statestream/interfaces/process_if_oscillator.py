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
    return {"out": ["osci"],
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
    """Return shared memory layout for the oscillator interface.

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
        = ShmL("np", (), np.int32, p.get("mode", 0))

    if p["osci_type"] == "sin":
        shm_layout["parameter"]["amplitude"] \
            = ShmL("np", (), np.float32, p.get("amplitude", 0.5), 1, None)
        shm_layout["parameter"]["frequency"] \
            = ShmL("np", (), np.float32, p.get("frequency", 0.5), 1, None)
        shm_layout["parameter"]["phase"] \
            = ShmL("np", (), np.float32, p.get("phase", 0.5), 1, None)

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

    

class ProcessIf_oscillator(ProcessIf):
    """Interface class providing basic oscillator activations.

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
        osci_type : string
            Specifies the used oscillator used. Available:
            - sin: Sinus oscillation.
                - amplitude : float32
                    Specifies the amplitude of the sinus.
                - frequency : float32
                    Specifies the frequency in frames.
                - phase : float32
                    Specifies the phase in frames.

        Inputs:
        -------

        Outputs:
        --------
        osci : np.array, shape [agents, c np, x np, y np]
            The oscillator output neuron-pool.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)



    def initialize(self):
        """Method to initialize the oscillator interface.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        if self.p["osci_type"] == "sin":
            self.dat["parameter"]["amplitude"] = self.p.get("amplitude", 1.0)
            self.dat["parameter"]["frequency"] = self.p.get("frequency", 0.1)
            self.dat["parameter"]["phase"] = self.p.get("phase", 0.0)

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
        """Method to update the experimental state of the oscillator interface.
        """
        # Draw from distribution.
        if self.p["osci_type"] == "sin":
            t = 2 * np.pi * (self.frame_cntr + self.dat["parameter"]["phase"])
            t *= self.dat["parameter"]["frequency"]
            amp = self.dat["parameter"]["amplitude"] * np.sin(t)
            self.dat["variables"]["osci"][:,:,:,:] \
                = amp * np.ones(self.np_shape["osci"])[:,:,:,:]
