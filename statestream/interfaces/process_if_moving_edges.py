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
    return {"out": ["medge_board", 
                    "medge_dxdy"],
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
    """Return shared memory layout for moving edges interface.

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
    # Add durations as numpy parameters.
    shm_layout["parameter"]["minmax_velocity"] \
        = ShmL("np", (), np.float32, p.get("minmax_velocity", 1), 1, None)
    shm_layout["parameter"]["min_period"] \
        = ShmL("np", (), np.int32, p.get("min_period", 4), 1, None)
    shm_layout["parameter"]["max_period"] \
        = ShmL("np", (), np.int32, p.get("max_period", 8), 1, None)
    shm_layout["parameter"]["min_duration"] \
        = ShmL("np", (), np.int32, p.get("min_duration", 8), 1, None)
    shm_layout["parameter"]["max_duration"] \
        = ShmL("np", (), np.int32, p.get("max_duration", 16), 1, None)
    shm_layout["parameter"]["sigma"] \
        = ShmL("np", (), np.int32, p.get("sigma", 64), 1, None)

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

    

class ProcessIf_moving_edges(ProcessIf):
    """Interface class providing basic moving edges.

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
        minmax_velocity : float32
            The min/max velocity in pixels for the moving edges.
        min_period : int
            Minimal period of a moving bar in pixels.
        max_period : int
            Maximal period of a moving bar in pixels.
        min_duration : int
            The minimum duration a mnist number will be presented in frames.
            The actual duration will be drawn uniformly between min_duration
            and max_duration.
        max_duration : int
            The maximum duration a mnist number will be presented in frames.

        Inputs:
        -------

        Outputs:
        --------
        medge_board : np.array, shape [agents, 1, source np, source np]
            The grey-scale board of moving mnist images.
        medge_dxdy : np.array, shape [agents, 2, 1, 1]
            The global velocity in pixels.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)



    def initialize(self):
        """Method to initialize (load) the mnist interface class.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        self.dat["parameter"]["minmax_velocity"] = self.p.get("minmax_velocity", 1)
        self.dat["parameter"]["min_duration"] = self.p.get("min_duration", 8)
        self.dat["parameter"]["max_duration"] = self.p.get("max_duration", 16)
        self.dat["parameter"]["min_period"] = self.p.get("min_period", 4)
        self.dat["parameter"]["max_period"] = self.p.get("max_period", 8)
        self.dat["parameter"]["sigma"] = self.p.get("sigma", 64)

        # Set border which is needed for the mask.
        self.border = self.dat["parameter"]["max_period"]

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
        self.current_period = [0 for i in range(self.net["agents"])]
        self.current_vel = [0 for i in range(self.net["agents"])]
        self.current_duration = [0 for i in range(self.net["agents"])]
        self.current_elapsed = [0 for i in range(self.net["agents"])]
        self.current_orient = [0 for i in range(self.net["agents"])]
        for a in range(self.net["agents"]):
            self.draw_sample(a)

        # Initialize global normal filter.
        # ---------------------------------------------------------------------
        self.x_grid = np.linspace(-self.np_shape["medge_board"][2] / 2,
                                   self.np_shape["medge_board"][2] / 2,
                                   self.np_shape["medge_board"][2])
        self.y_grid = np.linspace(-self.np_shape["medge_board"][3] / 2,
                                   self.np_shape["medge_board"][3] / 2,
                                   self.np_shape["medge_board"][3])
        self.xv, self.yv = np.meshgrid(self.x_grid, self.y_grid)


    def draw_sample(self, sample):
        """Draw a new random sample.
        """
        # Set back elapsed.
        self.current_elapsed[sample] = 0
        # Draw new duration.
        self.current_duration[sample] \
            = np.random.randint(self.dat["parameter"]["min_duration"],
                                self.dat["parameter"]["max_duration"] - 1)
        # Draw new period.
        if self.dat["parameter"]["min_period"] \
                == self.dat["parameter"]["max_period"]:
            self.current_period[sample] = self.dat["parameter"]["min_period"]
        else:
            self.current_period[sample] \
                = self.dat["parameter"]["min_period"] \
                  + np.random.rand() \
                  * (self.dat["parameter"]["max_period"] - self.dat["parameter"]["min_period"])
        # Draw new velocity in pixels.
        self.current_vel[sample] = np.random.rand() * self.dat["parameter"]["minmax_velocity"]
        # Draw new orientation.
        self.current_orient[sample] = 2 * np.pi * np.random.rand()



    def update_sample(self, sample):
        """Update a sample.
        """
        self.current_elapsed[sample] += 1
        # Check duration.
        if self.current_elapsed[sample] > self.current_duration[sample]:
            self.draw_sample(sample)



    def update_frame_writeout(self):
        """Method to update the experimental state of the mnist interface.
        """
        # Tabula rasa.
        self.dat["variables"]["medge_board"][:,:,:,:] \
            = 0.0 * self.dat["variables"]["medge_board"][:,:,:,:]
        # Update experimental state for all agents.
        for a in range(self.net["agents"]):
            # Update current experimental state.
            # -----------------------------------------------------------------
            self.update_sample(a)

        # Update internal data (= variables).
        # -----------------------------------------------------------------
        # Update board.
        if "medge_board" in self.p["out"]:
            for a in range(self.net["agents"]):
                # Rotate grid.
                angle = self.current_orient[a]
                rot_x = self.xv * np.cos(angle) \
                        - self.yv * np.sin(angle)
                rot_y = self.xv * np.sin(angle) \
                        + self.yv * np.cos(angle)
                # Transform velocity to phase (everything in pixels).
                phase = self.current_elapsed[a] * self.current_vel[a]
                # Compute sine wave.
                waves = (np.sin(2 * np.pi * (rot_x / self.current_period[a] + phase)) + 1) / 2
                waves = waves ** 2
                # Compute gauss-filter.
                normal = np.exp(-(self.xv**2 + self.yv**2) / (2 * self.dat["parameter"]["sigma"])**2)
                self.dat["variables"]["medge_board"][a,0,:,:] \
                    = waves * normal
        if "medge_dxdy" in self.p["out"]:
            for a in range(self.net["agents"]):
                self.dat["variables"]["medge_dxdy"][a,0,0,0] \
                    = self.current_vel[a] * np.cos(self.current_orient[a])
                self.dat["variables"]["medge_dxdy"][a,1,0,0] \
                    = self.current_vel[a] * np.sin(self.current_orient[a])
