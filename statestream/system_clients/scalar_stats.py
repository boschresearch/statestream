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

from statestream.utils.properties import array_property



# =============================================================================

class scalar_stats(object):
    """Provide scalar statistics for set of items all of type np or sp.

    For a (selected) subset of items (= children) which share a same data type, 
    e.g. all neuron-pools have states, or plasticities having a loss variable,
    this meta-variable computes a scalar statistic, such as mean or variance,
    for each item for the specified data type. Then this meta-variables 
    visualizes this scalar statistic for all (selected) items.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : string
        Instance name of this system client.
    client_param : dict
        Dictionary containing system clients parameters.
    """
    def __init__(self, name, net, client_param):
        self.type = "scalar_stats"
        self.name = client_param['name']
        self.p = copy.deepcopy(client_param)
        self.dat = {}
        self.sv = copy.deepcopy(client_param['selected_values'])

        self.parameter = {}
        for p,P in enumerate(client_param['params']):
            self.parameter[P['name']] = P['value']

        # Begin with empty statistics.
        self.shape = [self.parameter["nodes"], len(self.sv)]
        self.window = np.zeros([self.parameter["window"], len(self.sv)],
                               dtype=np.float32)
        self.stats = np.zeros([len(self.sv)], dtype=np.float32)
        self.stats_hist = np.zeros(self.shape, dtype=np.float32)
        self.current_frame = 0
        


    def initialize(self, shm):
        """Initialize this client.
        """
        # Generate client side structure to hold necessary network data.
        self.sv_dat = {}
        for v in range(len(self.sv)):
            C = self.sv[v][0]
            V = self.sv[v][1]
            if C not in self.sv_dat:
                self.sv_dat[C] = {}
            if V == "np state":
                self.sv_dat[C][V] = np.zeros(shm.dat[C]["state"].shape)
            for s in ['np par ', 'sp par ', 'plast par ', 'if par']:
                if V.startswith(s):
                    par = V[len(s):]
                    self.sv_dat[C][V] = np.zeros(shm.dat[C]["parameter"][par].shape)
            for s in ['if var ', 'plast var ']:
                if V.startswith(s):
                    var = V[len(s):]
                    self.sv_dat[C][V] = np.zeros(shm.dat[C]["variables"][var].shape)



    def update_frame_readin(self, shm):
        """System client dependent read in.
        """
        # Read all selected values.
        for v in range(len(self.sv)):
            C = self.sv[v][0]
            V = self.sv[v][1]
            if V == "np state":
                self.sv_dat[C][V][:] = shm.dat[C]["state"]
            for s in ['np par ', 'sp par ', 'plast par ', 'if par']:
                if V.startswith(s):
                    par = V[len(s):]
                    self.sv_dat[C][V][:] = shm.dat[C]["parameter"][par]
            for s in ['if var ', 'plast var ']:
                if V.startswith(s):
                    var = V[len(s):]
                    self.sv_dat[C][V][:] = shm.dat[C]["variables"][var]



    def update_frame_writeout(self):
        """Method to compute activation statistics for all child nps.
        """
        # Compute current statistics.
        for v in range(len(self.sv)):
            C = self.sv[v][0]
            V = self.sv[v][1]
            self.stats[v] = array_property(self.sv_dat[C][V], self.parameter["stat"])
        # Write current statistics to window and compute mean for full window.
        self.current_frame = (self.current_frame + 1) % (self.parameter["nodes"] * self.parameter["window"])
        self.window[self.current_frame % self.parameter["window"],:] \
            = self.stats[:]
        if self.current_frame % self.parameter["window"] == 0:
            # Update local stats history.
            self.stats_hist[self.current_frame // self.parameter["window"], :] \
                = np.mean(self.window, axis=0)[:]
            I = (self.current_frame \
                // self.parameter["window"] - 1) \
                % self.parameter["nodes"]
            self.dat['variables']['stats'][:] \
                = np.roll(self.stats_hist, self.parameter["nodes"] - I - 2, axis=0)
            self.window *= 0.0
        
