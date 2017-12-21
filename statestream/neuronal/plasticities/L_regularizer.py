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



import importlib

from statestream.neuronal.plasticity import Plasticity
from statestream.backends.backends import import_backend


"""
All plasticities must have an .update member for their parameter.
NOTE: the update of the parameters is 
    NOT the updated version of the parameters,
    BUT only the update relative to the old parameters (so to say delta(parameters))
    HENCE in each update step the parameters themself are overwritten
        by only their updates and the new parameters have to be re-read each time!
"""
class Plasticity_L_regularizer(Plasticity):
    """Constructor for L*-norm weight regularizers.
    """
    def __init__(self, name, net, param, mn, nps, sps):
        # Initialize parent ProcessIf class.
        Plasticity.__init__(self, name, net, param, mn, nps, sps)
        
        # Import backend.
        self.B = import_backend(None, None, name)

        # Define loss_function.
        # ---------------------------------------------------------------------
        # Begin with first parameter under regularization.
        p = self.p["parameter"][0]
        if p[0] == "np":
            self.loss = self.dat["parameter"]["L1"] * self.B.sum(self.B.abs(self.nps[p[1]].dat["parameter"][p[2]])) \
                        + self.dat["parameter"]["L2"] * self.B.sum(self.nps[p[1]].dat["parameter"][p[2]]**2)
        elif p[0] == "sp":
            self.loss = self.dat["parameter"]["L1"] * self.B.sum(self.B.abs(self.sps[p[1]].dat["parameter"][p[2]])) \
                        + self.dat["parameter"]["L2"] * self.B.sum(self.sps[p[1]].dat["parameter"][p[2]]**2)
        # Now loop over rest of to be regularized parameter.
        for p_idx in range(len(self.p["parameter"]) - 1):
            p = self.p["parameter"][p_idx + 1]
            if p[0] == "np":
                self.loss += self.dat["parameter"]["L1"] * self.B.sum(self.B.abs(self.nps[p[1]].dat["parameter"][p[2]])) \
                             + self.dat["parameter"]["L2"] * self.B.sum(self.nps[p[1]].dat["parameter"][p[2]]**2)
            elif p[0] == "sp":
                self.loss += self.dat["parameter"]["L1"] * self.B.sum(self.B.abs(self.sps[p[1]].dat["parameter"][p[2]])) \
                             + self.dat["parameter"]["L2"] * self.B.sum(self.sps[p[1]].dat["parameter"][p[2]]**2)
        # Compute gradient.
        self.grads = self.B.grad(self.loss, self.params)
        # Define updates on parameters using the specified optimizer.
        optimizer = getattr(importlib.import_module("statestream.neuronal.optimizer"), self.p.get("optimizer", "grad_desc"))
        self.updates = optimizer(self.params, self.params_id, self.grads, self.dat, self.B)
        # Append variables to updates.
        self.updates.append(self.B.update(self.dat["variables"]["loss"], self.loss))

        # Sanity shape check for updates.
        for u in self.updates:
            assert (len(u[0].shape.eval()) == len(u[1].shape.eval())), "Error: Inconsistant shape for plast update: " + str(u[0].shape.eval()) + " and " + str(u[1].shape.eval())                                    
        
        # Define function for sp/np parameter update (e.g. optimization).
        self.update_parameter = self.B.function([], [], updates=self.updates)

