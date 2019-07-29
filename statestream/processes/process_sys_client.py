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



import copy
import importlib
import numpy as np
import os

from statestream.processes.process import STProcess
from statestream.utils.helper import is_scalar_shape



class ProcessSysClient(STProcess):
    """This is generic API for parallel clients synchronized with the core.
    
    In contrast to core clients, system clients run in a separate process
    instead inside the core. As np/sp/plast/if processes they respect the
    reading / writing phases triggered by the core and may have own 
    parameters / variables. 
    They may however not write parameters from other processes (e.g. nps)
    due to interference with plasts that might write to them.
    """
    def __init__(self, name, ident, metanet, param, client_param):
        STProcess.__init__(self, name, ident, metanet, param)
        self.type = client_param['type']
        # For system clients we also have to set the client dict.
        self.p = copy.copy(client_param)



    def initialize(self):
        """System client dependent initialization.
        """
        # Initialize system client dependent on its type.
        SysClient = getattr(importlib.import_module("statestream.system_clients." + self.type), self.type)
        self.sys_client = SysClient(self.name, self.net, self.p)

        # Set client side structure for parameters and variables.
        self.sys_client.dat['parameter'] = {}
        self.sys_client.dat['variables'] = {}
        for T in self.sys_client.dat:
            if T in self.p:
                for p,P in self.p[T].items():
                    self.sys_client.dat[T][p] = np.zeros(P['shape'])

        # Call client dependent initialization.
        self.sys_client.initialize(self.shm)

        # Get / set nps, sps.
        if hasattr(self.sys_client, 'nps'):
            self.nps = self.sys_client.nps
        else:
            self.nps = {}
        if hasattr(self.sys_client, 'sps'):
            self.sps = self.sys_client.sps
        else:
            self.nps = {}

        self.shm.update_sys_client()


    def update_frame_readin(self):
        """System client dependent read update.
        """
        if int(self.IPC_PROC["pause"][self.id]) == 0 \
                and int(self.frame_cntr) % int(self.IPC_PROC["period"][self.id]) == int(self.IPC_PROC["period offset"][self.id]):
            # Read system client's parameters.
            for p,P in self.p['parameter'].items():
                self.sys_client.dat['parameter'][p] = np.copy(self.shm.dat[self.p['name']]['parameter'][p])

            # Read necessary (all in-comming) np states / parameters.
            for n in self.nps:
                #self.nps[n].state[0].set_value(np.ascontiguousarray(self.shm.dat[n]["state"]))
                self.nps[n].state[0].set_value(np.require(self.shm.dat[n]["state"],
                                                          requirements='C')[0:self.sys_client.samples,:,:,:])
                if n == self.name:
                    for par in self.shm.dat[n]["parameter"]:
                        if is_scalar_shape(self.shm.layout[n]["parameter"][par].shape):
                            self.nps[n].dat["parameter"][par].set_value(self.shm.dat[n]["parameter"][par][0])
                        else:
                            self.nps[n].dat["parameter"][par].set_value(self.shm.dat[n]["parameter"][par])
            # Read necessary (all in-comming) sp parameters.
            for s in self.sps:
                if self.IPC_PROC["pause"][self.shm.proc_id[s][0]] == 0:
                    for par in self.shm.dat[s]["parameter"]:
                        # Check for parameter sharing.
                        source_sp = s
                        source_par = par
                        if "share params" in self.net["synapse_pools"][s]:
                            if par in self.net["synapse_pools"][s]["share params"]:
                                source_sp = self.net["synapse_pools"][s]["share params"][par][0]
                                source_par = self.net["synapse_pools"][s]["share params"][par][1]
                        if is_scalar_shape(self.shm.layout[s]["parameter"][par].shape):
                            self.sps[s].dat["parameter"][par].set_value(self.shm.dat[source_sp]["parameter"][source_par][0])
                        else:
                            self.sps[s].dat["parameter"][par].set_value(self.shm.dat[source_sp]["parameter"][source_par])
                else:
                    for par in self.shm.dat[s]["parameter"]:
                        if is_scalar_shape(self.shm.layout[s]["parameter"][par].shape):
                            self.sps[s].dat["parameter"][par].set_value(0.0 * self.shm.dat[s]["parameter"][par][0])
                        else:
                            self.sps[s].dat["parameter"][par].set_value(0.0 * self.shm.dat[s]["parameter"][par])

            # Execute system client specific read phase.
            self.sys_client.update_frame_readin(self.shm)



    def update_frame_writeout(self):
        """System client dependent write update.
        """
        # Only execute client if not in pause and if period/offset is met.
        if self.IPC_PROC["pause"][self.id] == 0 and \
                self.frame_cntr % self.IPC_PROC["period"][self.id] == self.IPC_PROC["period offset"][self.id]:
            # Execute write phase.
            self.sys_client.update_frame_writeout()
            # Write variables.
            for v,V in self.p['variables'].items():
                self.shm.set_shm([self.p['name'], 'variables', v], self.sys_client.dat['variables'][v])

