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



import os
import numpy as np
import importlib

from statestream.processes.process import STProcess
from statestream.utils.helper import is_scalar_shape
from statestream.backends.backends import import_backend



class ProcessPlast(STProcess):
    def __init__(self, name, ident, net, param):
        STProcess.__init__(self, name, ident, net, param)
        # Get / set type.
        self.type = self.p["type"]

        # Determine if split should be ignored.
        self.ignore_split = self.p.get("ignore_split", False)



    def initialize(self):
        """Plasticity dependent initialization.
        """
        # Import backend.
        self.B = import_backend(self.net, self.param, self.name)
        from statestream.neuronal.neuron_pool import NeuronPool
        from statestream.neuronal.synapse_pool import SynapsePool
        # Define random seed.
#        self.srng = self.B.randomstream(np.random.RandomState(self.param["core"]["random_seed"]).randint(999999))

        # Build nps / sps.
        # ---------------------------------------------------------------------
        self.nps = {}
        self.sps = {}
        self.all_nps = list(set([n for np_list in self.mn.net_plast_nps[self.name] for n in np_list]))
        self.all_sps = list(set([s for sp_list in self.mn.net_plast_sps[self.name] for s in sp_list]))
        # Create all necessary neuron pools.
        for n in self.all_nps:
            if n not in self.nps:
                self.nps[n] = NeuronPool(n, self.net, self.param, self.mn)
        # Create all necessary synapse pools.
        for s in self.all_sps:
            S = self.net["synapse_pools"][s]
            # List of sources.
            source_np_list = []
            for I in range(len(S["source"])):
                source_np_list.append([])
                for i in range(len(S["source"][I])):
                    source_np_list[-1].append(self.nps[S["source"][I][i]])
            self.sps[s] = SynapsePool(s,
                                      self.net,
                                      self.param,
                                      self.mn,
                                      source_np_list,
                                      self.nps[S["target"]])

        # Rollout network to specified depth.
        # ---------------------------------------------------------------------
        # Rollout network.
        for depth in range(self.mn.net_plast_depth[self.name]):
            # Post synaptic has to come BEFORE next state.
            for s in self.all_sps:
                if s in self.mn.net_plast_sps[self.name][depth + 1]:
                    self.sps[s].compute_post_synaptic(as_empty=False)
                else:
                    self.sps[s].compute_post_synaptic(as_empty=True)
            # Now update next state.
            for n in self.all_nps:
                if n in self.mn.net_plast_nps[self.name][depth + 1]:
                    self.nps[n].compute_algebraic_next_state(as_empty=False)
                else:
                    self.nps[n].compute_algebraic_next_state(as_empty=True)

        # Initialize plasticity.
        class_name = "Plasticity_" + self.type
        Plasticity = getattr(importlib.import_module("statestream.neuronal.plasticities." + self.type), class_name)
        self.plasticity = Plasticity(self.name, self.net, self.param, self.mn, self.nps, self.sps)



    def update_frame_readin(self):
        """Plasticity dependent read update.
        """
        if self.IPC_PROC["pause"][self.id].value == 0 and self.frame_cntr % self.IPC_PROC["period"][self.id].value == self.IPC_PROC["period offset"][self.id].value:
            # Read necessary np states / parameters.
            for n in self.plasticity.nps:
                self.B.set_value(self.plasticity.nps[n].state[0],
                                 self.shm.dat[n]["state"])
                for par in self.shm.dat[n]["parameter"]:
                    if is_scalar_shape(self.shm.layout[n]["parameter"][par].shape):
                        self.B.set_value(self.plasticity.nps[n].dat["parameter"][par],
                                         self.shm.dat[n]["parameter"][par][0])
                    else:
                        self.B.set_value(self.plasticity.nps[n].dat["parameter"][par],
                                         self.shm.dat[n]["parameter"][par])
            # Read necessary sp parameters.
            for s in self.plasticity.sps:
                if self.IPC_PROC["pause"][self.shm.proc_id[s][0]].value == 0:
                    for par in self.shm.dat[s]["parameter"]:
                        # Check for parameter sharing.
                        source_sp = s
                        source_par = par
                        if "share params" in self.net["synapse_pools"][s]:
                            if par in self.net["synapse_pools"][s]["share params"]:
                                source_sp = self.net["synapse_pools"][s]["share params"][par][0]
                                source_par = self.net["synapse_pools"][s]["share params"][par][1]
                        if is_scalar_shape(self.shm.layout[s]["parameter"][par].shape):
                            self.B.set_value(self.plasticity.sps[s].dat["parameter"][par],
                                             self.shm.dat[source_sp]["parameter"][source_par][0])
                        else:
                            self.B.set_value(self.plasticity.sps[s].dat["parameter"][par],
                                             self.shm.dat[source_sp]["parameter"][source_par])
                else:
                    for par in self.shm.dat[s]["parameter"]:
                        if is_scalar_shape(self.shm.layout[s]["parameter"][par].shape):
                            self.B.set_value(self.plasticity.sps[s].dat["parameter"][par],
                                             0.0)
                        else:
                            self.B.set_value(self.plasticity.sps[s].dat["parameter"][par],
                                             0.0 * self.shm.dat[s]["parameter"][par])
            # Read necessary plast parameters.
            for par in self.shm.dat[self.name]["parameter"]:
                if is_scalar_shape(self.shm.layout[self.name]["parameter"][par].shape):
                    self.B.set_value(self.plasticity.dat["parameter"][par],
                                     self.shm.dat[self.name]["parameter"][par][0])
                else:
                    self.B.set_value(self.plasticity.dat["parameter"][par],
                                     self.shm.dat[self.name]["parameter"][par])
            # Read train/test split parameter.
            split = np.ones([self.net['agents'],], dtype=np.float32)
            if self.IPC_PROC['plast split'].value > 0 and not self.ignore_split:
                split[0:int(self.IPC_PROC['plast split'].value)] = 0.0
            self.B.set_value(self.plasticity.split,
                             split)



    def update_frame_writeout(self):
        """Plasticity dependent write update.
        """
        # Only execute plasticity if not in pause and if period/offset is met.
        if self.IPC_PROC["pause"][self.id].value == 0 and self.frame_cntr % self.IPC_PROC["period"][self.id].value == self.IPC_PROC["period offset"][self.id].value:
            if self.plasticity.startframe < self.frame_cntr:
                # Note: after this the params contain only their updates.
                self.plasticity.update_parameter()
            
                # Write plast variables to shared memory.
                for var in self.shm.dat[self.name]["variables"]:
                    if self.shm.layout[self.name]["variables"][var].type == "backend":
                        value = self.B.get_value(self.plasticity.dat["variables"][var])
                    elif self.shm.layout[self.name]["variables"][var].type == "np":
                        value = self.plasticity.dat["variables"][var]
                    # Check for NaNs.
                    if np.isnan(np.sum(value)):
                        print("\nWarning: Plasticity " + self.name \
                               + " detected NaN in variable " + var + ".\n")
                        value.fill(0)
                        # Set system into pause mode.
                        self.IPC_PROC["break"].value = 1
                    # Assign value.
                    self.shm.set_shm([self.name, "variables", var], 
                                      value)
            
                # Write update for parameters to shared memory.
                try:
                    for param, param_id in zip(self.plasticity.params, self.plasticity.params_id):
                        par = param_id.split(".")
                        if self.shm.layout[par[1]]["parameter"][par[2]].type == "backend":
                            value = self.B.get_value(param)
                        elif self.shm.layout[par[1]]["parameter"][par[2]].type == "np":
                            value = param
                        # Check for NaNs.
                        if np.isnan(np.sum(value)):
                            print("\nWarning: Plasticity " + self.name \
                                   + " detected NaN in updates " + str(par[2]) + ".\n")
                            value.fill(0)
                            # Set system into pause mode.
                            self.IPC_PROC["break"].value = 1
                        # Assign value.
                        self.shm.set_shm([self.name, "updates", par[1], par[2]],
                                          value)
                except:
                    print("Error: Copying theano -> shared mem failed for plast updates: " + self.name)
                    for param, param_id in zip(self.plasticity.params, self.plasticity.params_id):
                        par = param_id.split(".")
                        print("param: " + str(param_id))
                        print("    IPC: " + str(self.shm.dat[self.name]["updates"][par[1]][par[2]].shape))
                        print("    T  : " + str(self.B.get_value(param).shape))
        else:
            # In off or pause mode only write zeros to update IPC.
            try:
                for param, param_id in zip(self.plasticity.params, self.plasticity.params_id):
                    par = param_id.split(".")
                    self.shm.dat[self.name]["updates"][par[1]][par[2]].fill(0.0)
            except:
                print("Error: Copying theano -> shared mem failed for plast updates: " + self.name)
                for param, param_id in zip(self.plasticity.params, self.plasticity.params_id):
                    par = param_id.split(".")
                    print("param: " + str(param_id))
                    print("    IPC: " + str(self.shm.dat[self.name]["updates"][par[1]][par[2]].shape))
                    print("    T  : " + str(self.B.get_value(param).shape))


