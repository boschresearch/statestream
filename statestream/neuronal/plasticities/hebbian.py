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
import importlib

from statestream.neuronal.plasticity import Plasticity



class Plasticity_hebbian(Plasticity):
    """Constructor for hebbian based plasticity ((anti-)hebbian, dt-hebbian).

    Parameters
    ----------
    
    """
    def __init__(self, name, net, param, mn, nps, sps):
        # Initialize parent Plasticity class.
        Plasticity.__init__(self, name, net, param, mn, nps, sps)
        # Import theano.
        import theano
        import theano.tensor as T

        # For hebbian we need only:
        #    one parameter of a SP
        #    source np state
        #    target np state
        # TODO: Note: for now hebbian targets the np states, but it should be free to
        #       target the post-synapses (of target NP)
        # TODO: Note: at the moment hebbian only available for raw feature reps. [*,*,1,1]
        # define parameters (== one W of one SP)
        # Get sp id.
        self.hebbian_sp_id = self.p["parameter"][0][1]
        # Get source and target shape.
        self.source_np_shape = self.mn.np_shape[self.p["source"]]
        self.target_np_shape = self.mn.np_shape[self.p["target"]]
        # Determine if weights are unshared.
        self.unshared = False
        if "unshare" in self.net["synapse_pools"][self.hebbian_sp_id]:
          if self.p["parameter"][0][2] in self.net["synapse_pools"][self.hebbian_sp_id]["unshare"]:
              self.unshared = True

        # Assert that source and target are raw feature reps.
        if self.target_np_shape[2] != 1 or self.target_np_shape[3] != 1:
            print("Error: Init plasticity of type hebbian: \
                  Expect raw feature np representations for nps: " \
                  + self.p["source"] + "   " + self.p["target"])

        # Despite hebbian / anti-hebbian is gradient free, we use the gradient
        # construct to leverage sophisticated optimizers like adam.
        if self.p["modus"].startswith("dt"):
            target = self.dat["variables"]["target(now)"] - self.dat["variables"]["target(now-1)"]
        else:
            target = self.dat["variables"]["target(now)"]

        # Split target accoring to split parameter.
#        target *= self.split.dimshuffle(0, 'x', 'x', 'x')

        if self.unshared:
            # TODO: May be accomplished much nicer using the SCAN function.
            single_grads = []
            for a in range(self.net["agents"]):
                single_grads.append(T.dot(target[a,:,0,0], self.dat["variables"]["source(now-1)"][a,:,0,0]))
            grads = T.concatenate(single_grads)
        else:
            grads = T.tensordot(target[:,:,0,0], 
                                self.dat["variables"]["source(now-1)"], 
                                axes=[[0], [0]])

        # Consider anti-hebbian.
        if self.p["modus"] in ["dt hebbian", "hebbian"]:
            self.grads = [-grads]
        elif self.p["modus"] in ["dt anti_hebbian", "anti_hebbian"]:
            self.grads = [grads]


        # Define updates on parameters.
        optimizer = getattr(importlib.import_module("statestream.neuronal.optimizer"), 
                                                    self.p.get("optimizer", "grad_desc"))
        self.updates = optimizer(self.params, 
                                 self.params_id,
                                 self.grads, 
                                 self.dat)

        # Time shift of source activation is also needed for hebbian plasts.
        self.updates.append((self.dat["variables"]["source(now-1)"], 
                             self.dat["variables"]["source(now)"]))
        self.updates.append((self.dat["variables"]["source(now)"], 
                             self.nps[self.p["source"]].state[0]))
        self.updates.append((self.dat["variables"]["target(now-1)"], 
                             self.dat["variables"]["target(now)"]))
        self.updates.append((self.dat["variables"]["target(now)"], 
                             self.nps[self.p["target"]].state[0]))

        # Sanity shape check for updates.
        for u in self.updates:
            if len(u[0].shape.eval()) != len(u[1].shape.eval()):
                print("Error: inconsistant shape for plast update: " \
                      + str(u[0].shape.eval()) + " and " + str(u[1].shape.eval()))

        # Define function for sp/np parameter update (e.g. optimization).
        self.update_parameter = theano.function([], [], updates=self.updates)

        
