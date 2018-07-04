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
import importlib

from statestream.utils.helper import is_scalar_shape
from statestream.backends.backends import import_backend



class Plasticity(object):
    """Wrapper class for plasticities.

    Parameters:
    -----------
    name : str
        The unique string identifier for the neuron-pool.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.
    mn : MetaNetwork
        Deprecated.   
    nps : list of NeuronPool
        This list contains instances of all neuron-pools that are needed to 
        compute the parameter updates of this plasticity. This may heavily
        vary between plasticity types (e.g. hebbian only needs the source
        and target np, where loss plasts may need many nps for local network
        rollout).
    sps : list of SynapsePool
        This list contains instances of all synapse-pools that are needed
        to compute the parameter updates of this plasticity (see np above).
    """
    def __init__(self, name, net, param, mn, nps, sps):
        # Import backend.
        self.B = import_backend(None, None, name)
        # Get / set global specifications.
        self.name = name
        self.net = net
        self.param = param
        self.mn = mn
        self.p = self.net["plasticities"][self.name]
        self.nps = nps
        self.sps = sps

        # Get start frame.
        self.startframe = self.p.get("startframe", 2)
        # Get plasticity type.
        self.type = self.p["type"]

        # Create list of all to be updated parameters.
        self.params = []
        self.params_id = []
        for p in self.p["parameter"]:
            self.params_id.append(p[0] + "." + p[1] + "." + p[2])
            if p[0] == "sp":
                try:
                    self.params.append(self.sps[p[1]].dat["parameter"][p[2]])
                except:
                    print("\nError: In plast: " + self.name + " collecting parameter: " \
                          + str(p[2]) + " for sp: " + str(p[1] \
                          + ". This may most certainly be caused by an inpropper depth " \
                          + "rollout for this plasticity."))
            if p[0] == "np":
                try:
                    self.params.append(self.nps[p[1]].dat["parameter"][p[2]])
                except:
                    print("\nError: In plast: " + self.name + " collecting parameter: " \
                          + str(p[2]) + " for np: " + str(p[1]))


        # Get shm layout for this plasticity.
        # ---------------------------------------------------------------------
        plast_shm_layout_fct \
            = getattr(importlib.import_module("statestream.meta.plasticities." \
                                              + self.type), "plast_shm_layout")
        self.dat_layout = plast_shm_layout_fct(self.name, self.net, self.param)
        self.dat = {}
        for t in ["parameter", "variables"]:
            settable = True
            self.dat[t] = {}
            for i, i_l in self.dat_layout[t].items():
                if i_l.type == "backend":
                    if is_scalar_shape(i_l.shape):
                        self.dat[t][i] = self.B.scalar(0.0, 
                                                       dtype=np.float32, 
                                                       name=name + "." + i,
                                                       settable=settable)
                    else:
                        self.dat[t][i] = self.B.variable(np.zeros(i_l.shape, dtype=i_l.dtype),
                                                         borrow=True,
                                                         name=name + "." + i,
                                                         settable=settable)
                elif i_l.type == "np":
                    if is_scalar_shape(i_l.shape):
                        self.dat[t][i] = np.array([1,], dtype=i_l.dtype)
                    else:
                        self.dat[t][i] = np.array(i_l.shape, dtype=i_l.dtype)

        # First initialization of plasticity variables (mostly from optimizer).
        for i, i_l in self.dat_layout["variables"].items():
            if i_l.type == "backend":
                value = None
                if is_scalar_shape(i_l.shape):
                    value = 0.0
                else:
                    value = np.zeros(i_l.shape, dtype=i_l.dtype)
                self.B.set_value(self.dat[t][i], value)

        # Instantiate split parameter.
        self.split = self.B.variable(np.zeros([self.net['agents'],], dtype=np.float32),
                                     borrow=True,
                                     name=name + '.' + 'split')
        # Define function for sp/np parameter update (e.g. optimization).
        self.update_parameter = lambda: None

