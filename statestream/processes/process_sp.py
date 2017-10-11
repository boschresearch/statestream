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

from statestream.processes.process import STProcess
from statestream.utils.helper import is_scalar_shape



class ProcessSp(STProcess):
    def __init__(self, name, ident, net, param):
        # Initialize process wrapper.
        STProcess.__init__(self, name, ident, net, param)

    def initialize(self):
        """Initialization of process for synapse-pool.
        """
        # Start with empty updates.
        self.current_update = {
            "value": [],
            "source": []
        }
        # Check if parameters are shared from somewhere.        
        if "share params" in self.p:
            # Do nothing, because the shared source will handle the updates.
            pass
        else:
            # Allocate structure to store updates between read / write phases.
            # Collect updates for this sp from all plasticities (incl. shared).
            for p, P in self.net["plasticities"].items():
                # par[0]: np or sp
                # par[1]: np/sp name
                # par[2]: parameter name
                for par in P["parameter"]:
                    # Only consider sp parameter.
                    if par[0] == "sp":
                        # Check if self is the source for this sp,
                        if "share params" in self.net["synapse_pools"][par[1]]:
                            # Loop over all shared params of this sp.
                            for t, T in self.net["synapse_pools"][par[1]]["share params"].items():
                                # If self is source of this shared parameter.
                                if T[0] == self.name:
                                    # Add update.
                                    self.current_update["source"].append([p, par[0], par[1], par[2]])
                        # or if self is this sp.
                        elif par[1] == self.name:
                            self.current_update["source"].append([p, par[0], par[1], par[2]])
            # Finally allocate memory.
            for u in range(len(self.current_update["source"])):
                par = self.current_update["source"][u]
                shml = self.shm.layout[par[2]]["parameter"][par[3]]
                if is_scalar_shape(shml.shape):
                    self.current_update["value"].append(np.zeros([1,], dtype=shml.dtype))
                else:
                    self.current_update["value"].append(np.zeros(shml.shape, dtype=shml.dtype))



    def update_frame_readin(self):
        """Read phase for synapse pool process.
        """
        # Read updates from all plasticities for this sp.
        for u in range(len(self.current_update["source"])):
            par = self.current_update["source"][u]
            self.current_update["value"][u] = np.copy(self.shm.dat[par[0]]["updates"][par[2]][par[3]])



    def update_frame_writeout(self):
        """Write phase for synapse pool process.
        """
        # Cummulate updates to shared memory.
        for u in range(len(self.current_update["source"])):
            par = self.current_update["source"][u]
            new_value = self.shm.dat[self.name]["parameter"][par[3]] + self.current_update["value"][u]
            self.shm.set_shm([self.name, "parameter", par[3]], new_value)
