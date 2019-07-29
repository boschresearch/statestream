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
import os

from statestream.processes.process import STProcess
from statestream.utils.helper import is_scalar_shape
from statestream.backends.backends import import_backend



class ProcessNp(STProcess):
    def __init__(self, name, ident, metanet, param):
        # Initialize process wrapper.
        STProcess.__init__(self, name, ident, metanet, param)



    def initialize(self):
        """Initialization of process for neuron-pool.
        """
        if self.net.get("backend", "theano") == "theano":
            # Define session / process dependent compile directory.
            base_compiledir = os.path.expanduser("~") + "/.statestream/compiledirs/" + str(self.id)
            if not os.path.isdir(base_compiledir):
                os.makedirs(base_compiledir)
            # Set GPU and compile directory.
            os.environ["THEANO_FLAGS"] = self.param["core"]["THEANO_FLAGS"] \
                                         + ", device=" + self.device \
                                         + ", base_compiledir=" + base_compiledir
        # Import backend.
        self.B = import_backend(self.net, self.param, self.name)
        from statestream.neuronal.neuron_pool import NeuronPool
        from statestream.neuronal.synapse_pool import SynapsePool
        # Define random seed.
        #self.srng = self.B.randomstream(np.random.RandomState(self.param["core"]["random_seed"]).randint(999999))

        # Build neuron / synapse pools.
        # ---------------------------------------------------------------------
        # Build neuron pools.
        for n in self.mn.net_np_nps[self.name]:
            self.nps[n] = NeuronPool(n, self.net, self.param, self.mn)
        # Build synapse pools.
        for s in self.mn.net_np_sps[self.name]:
            S = self.net["synapse_pools"][s]
            # Get list of sources for this sp.
            source_np_list = []
            for I in range(len(S["source"])):
                source_np_list.append([])
                for i in range(len(S["source"][I])):
                    source_np_list[-1].append(self.nps[S["source"][I][i]])
            # Get sp instance.
            self.sps[s] = SynapsePool(s, 
                                      self.net,
                                      self.param,
                                      self.mn,
                                      source_np_list,
                                      self.nps[S["target"]])

        # Rollout network to depth 1.
        # ---------------------------------------------------------------------
        # Post synaptic has to come BEFORE next state.
        for s in self.sps:
            self.sps[s].compute_post_synaptic()
#            for n in self.nps:
                # Dependent on (is input) compute algebraic next state with different activation functions.
#                if n != self.name:
#                    self.nps[n].compute_algebraic_next_state(is_input=True)
#                else:
#                    self.nps[n].compute_algebraic_next_state()
        self.nps[self.name].compute_algebraic_next_state()

        # Define central one-step function.
        updates = [(self.nps[self.name].state[0], self.nps[self.name].state[1])]
        self.update_np_state = self.B.function([], [], updates=updates)

        # Allocate structure to store updates between read / write phases.
        # ---------------------------------------------------------------------
        # Start with empty updates.
        self.current_update = {
            "value": [],
            "source": []
        }
        # Collect updates for this np from all plasticities (incl. shared).
        for p,P in self.net["plasticities"].items():
            # par[0]: np or sp
            # par[1]: np/sp name
            # par[2]: parameter name
            for par in P["parameter"]:
                # Only consider sp parameter.
                if par[0] == "np":
                    # Check if self is the source for this sp,
                    if "share params" in self.net["neuron_pools"][par[1]]:
                        # Loop over all shared params of this sp.
                        for t, tv in self.net["neuron_pools"][par[1]]["neuron_pools"].items():
                            # If self is source of this shared parameter.
                            if tv[0] == self.name:
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

        # Assume no incomming interfaces.
        self.prior_state_flag = False
        self.prior_state_value = None


    def update_frame_readin(self):
        """Read phase for neuron pool process.
        """
        # Read updates from all plasticities for this np.
        for u in range(len(self.current_update["source"])):
            par = self.current_update["source"][u]
            self.current_update["value"][u] = np.copy(self.shm.dat[par[0]]["updates"][par[2]][par[3]])
            # Check for NaNs.
            if np.isnan(np.sum(self.current_update["value"][u])):
                print("\nWarning: Neuron pool " + par[2] + " detected NaN in parameter " + par[3] + " from plasticity " + par[0] + ".\n")
                self.current_update["value"][u].fill(0)
                # Set system into pause mode.
                self.IPC_PROC["break"].value = 1

        # If not in pause mode, read.
        if self.IPC_PROC["pause"][self.id] == 0 and self.frame_cntr % self.IPC_PROC["period"][self.id] == self.IPC_PROC["period offset"][self.id]:
            # Read input from incomming interfaces.
            self.prior_state_flag = False
            self.prior_state_value = None
            for i,I in self.net["interfaces"].items():
                for n in I["out"]:
                    tmp_target = n
                    if "remap" in I:
                        if n in I["remap"]:
                            tmp_target = I["remap"][n]
                    if self.name == tmp_target:
                        # Update sum of all interface inputs.
                        if self.prior_state_flag:
                            self.prior_state_value += self.shm.dat[i]["variables"][n]
                        else:
                            self.prior_state_value = np.copy(self.shm.dat[i]["variables"][n])
                            self.prior_state_flag = True

            # Read necessary (all in-comming) np states / parameters.
            for n in self.nps:
                #self.nps[n].state[0].set_value(np.ascontiguousarray(self.shm.dat[n]["state"]))
                self.B.set_value(self.nps[n].state[0], 
                                 np.require(self.shm.dat[n]["state"], requirements='C'))
                if n == self.name:
                    for par in self.shm.dat[n]["parameter"]:
                        if is_scalar_shape(self.shm.layout[n]["parameter"][par].shape):
                            self.B.set_value(self.nps[n].dat["parameter"][par],
                                             self.shm.dat[n]["parameter"][par][0])
                        else:
                            self.B.set_value(self.nps[n].dat["parameter"][par],
                                             self.shm.dat[n]["parameter"][par])
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
                            self.B.set_value(self.sps[s].dat["parameter"][par],
                                             self.shm.dat[source_sp]["parameter"][source_par][0])
                        else:
                            self.B.set_value(self.sps[s].dat["parameter"][par],
                                             self.shm.dat[source_sp]["parameter"][source_par])
                else:
                    for par in self.shm.dat[s]["parameter"]:
                        if is_scalar_shape(self.shm.layout[s]["parameter"][par].shape):
                            self.B.set_value(self.sps[s].dat["parameter"][par],
                                             0.0 * self.shm.dat[s]["parameter"][par][0])
                        else:
                            self.B.set_value(self.sps[s].dat["parameter"][par],
                                             0.0 * self.shm.dat[s]["parameter"][par])



    def update_frame_writeout(self):
        """Write phase for neuron pool process.
        """
        # Cummulate updates to shared memory.
        for u in range(len(self.current_update["source"])):
            par = self.current_update["source"][u]
            new_value = self.shm.dat[self.name]["parameter"][par[3]] + self.current_update["value"][u]
            self.shm.set_shm([self.name, "parameter", par[3]], new_value)

        # If not in pause mode, write.
        if self.IPC_PROC["pause"][self.id] == 0 \
                and self.frame_cntr % self.IPC_PROC["period"][self.id] == self.IPC_PROC["period offset"][self.id]:
            # Compute np one-step update.
#            try:
#                self.update_np_state()
#            except:
#                print("Error: update_np_state() failed for np: " + self.name)
            self.update_np_state()
            # Write update for np state to shared memory.
            try:
                # Write to shared memory.
                if self.prior_state_flag:
                    self.shm.dat[self.name]["state"][:,:,:,:] = self.B.get_value(self.nps[self.name].state[0])[:,:,:,:] \
                                                                + self.prior_state_value[:,:,:,:]
                else:
                    self.shm.dat[self.name]["state"][:,:,:,:] = self.B.get_value(self.nps[self.name].state[0])[:,:,:,:]
            except:
                print("Error: Copying theano -> shared mem failed for np: " + self.name)
                print("    SHAPE(net)   : " + str(self.shm.layout[self.name]["state"].shape))
                print("    SHAPE(theano): " + str(self.B.get_value(self.nps[self.name].state[0]).shape))
            # Write np variables to shared memory.
            try:
                for var in self.shm.dat[self.name]["variables"]:
                    self.shm.set_shm([self.name, "variables", var], 
                                     self.B.get_value(self.nps[self.name].dat["variables"][var]))
            except:
                print("Error: Copying theano -> shared mem failed for np variable: " + self.name)
                for var in self.shm.dat[self.name]["variables"]:
                    print("variable: " + str(var))
                    print("    IPC: " + str(self.shm.layout[self.name]["variables"][var].shape))
                    print("    T  : " + str(self.B.get_value(self.nps[self.name].dat["variables"][var]).shape))
        elif self.IPC_PROC["pause"][self.id] == 2:
            # Real "off" flag for np, only write zero state.
            self.shm.dat[self.name]["state"].fill(0)

