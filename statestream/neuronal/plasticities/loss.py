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



from statestream.neuronal.plasticity import Plasticity
from statestream.meta.losses import has_target
from statestream.backends.backends import import_backend

import importlib
import numpy as np



"""
All plasticities must have an .update member for their parameter.
NOTE: the update of the parameters is 
    NOT the updated version of the parameters,
    BUT only the update relative to the old parameters (so to say delta(parameters))
    HENCE in each update step the parameters themself are overwritten
        by only their updates and the new parameters have to be re-read each time!
"""
class Plasticity_loss(Plasticity):
    """Constructor for Loss based plasticity (autoencoder, classification, etc.).
    """
    def __init__(self, name, net, param, mn, nps, sps):
        # Initialize parent ProcessIf class.
        Plasticity.__init__(self, name, net, param, mn, nps, sps)

        # Import backend.
        self.B = import_backend(None, None, name)
        # Set source (= prediction).
        self.source = self.p["source"]
        self.target = None
        # Get ground truth and prediction.
        x = self.nps[self.p["source"]].state[self.p["source_t"]]
        y = None
        # Get error function.
        loss_fnc = "self.B." + self.p["loss_function"]
        # Compute error.
        if has_target(self.p["loss_function"]):
            # Set target (= ground truth).
            self.target = self.p["target"]
            y = self.nps[self.p["target"]].state[self.p["target_t"]]
            error = eval(loss_fnc)(x, y)
        else:
            error = eval(loss_fnc)(x)

        # Mask out ignored pixels.
        if "mask" in self.p:
            if y is not None:
                mask = self.B.dimshuffle(self.nps[self.p["mask"]].state[self.p["target_t"]][:,0,:,:], (0, 'x', 1, 2))
            else:
                mask = self.B.dimshuffle(self.nps[self.p["mask"]].state[self.p["source_t"]][:,0,:,:], (0, 'x', 1, 2))
            error = error * mask

        # Split error.
        split = self.B.dimshuffle(self.split, (0, 'x', 'x', 'x'))
        error0 = error * (1 - split)
        error1 = error * split

        # Convert error to loss.
        self.loss0 = self.B.mean(error0) * np.float32(self.net['agents']) / self.B.maximum(np.float32(self.net['agents']) - self.B.sum(self.split), 1)
        self.loss1 = self.B.mean(error1) * np.float32(self.net['agents']) / (self.B.sum(self.split) + 1e-6)


        #self.sample_loss1 = [T.mean(error1[i,:,:,:]) for i in range(self.net['agents'])]
        #self.sample_grads = [T.grad(self.sample_loss1[i], self.params) for i in range(self.net['agents'])]


#        # Add confidence penalty and label smoothing.
#        if "confidence_penalty" in self.p:
#            self.loss += self.dat["parameter"]["confidence_penalty"] \
#                         * T.dot(x.flatten(), T.log(T.maximum(x.flatten(), 1e-6)))
#        # Add confidence penalty and label smoothing.
#        if "label_smoothing" in self.p:
#            no_classes = np.asarray([self.nps[self.p["source"]].shape[1]],
#                                     dtype=np.float32)
#            self.loss += self.dat["parameter"]["label_smoothing"] \
#                         * T.sum(T.log(T.maximum(x.flatten(), 1e-6))) \
#                         / no_classes[0]

        # Print np information.
#        for n in self.nps:
#            print("\n " + n + "   " + str(self.nps[n].state))


        # Define gradient.
        self.grads = self.B.grad(self.loss1, self.params)
        # Define updates on parameters.
        optimizer = getattr(importlib.import_module("statestream.neuronal.optimizer"),
                            self.p.get("optimizer", "grad_desc"))
        self.updates = optimizer(self.params, self.params_id, self.grads, self.dat, self.B)
        # Append variables to updates.
        self.updates.append(self.B.update(self.dat["variables"]["loss0"], self.loss0))
        self.updates.append(self.B.update(self.dat["variables"]["loss1"], self.loss1))

        # Sanity shape check for updates.
        if False:
            for u in self.updates:
                shape_0 = self.B.shape(u[0])
                shape_1 = self.B.shape(u[1])
                print("Shape for plast " + self.name + " update " + str(u[0].name) + " : " + str(shape_0) + " and " + str(shape_1))                                 
        
        # Define function for sp/np parameter update (e.g. optimization).
        self.update_parameter = self.B.function([], [], updates=self.updates)

