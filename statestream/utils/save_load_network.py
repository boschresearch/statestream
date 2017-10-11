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
import copy
from time import strftime, gmtime


try:
    import pickle as pckl
except:
    try:
        import cPickle as pckl
    except:
        pckl = None


import statestream.meta.network as mn


def save_network(save_file, net, shm):
    """Function to save an entire network to a file.
    """
    shm.update_net(net)
    
    net['info'] = {}
    net['info']['save timestamp'] = strftime("%a, %d %b %Y %H:%M:%S", gmtime())

    saveList = [[net], [], [], [], [], [], []]
    # Store np states.
    saveList.append([])
    for n in net["neuron_pools"]:
        saveList[1].append(shm.dat[n]["state"][:,:,:,:])
        saveList[2].append(n)
    par_var = ["parameter", "variables"]
    for I in ["np", "sp", "plast", "if"]:
        for i in net[mn.S2L(I)]:
            for mod in par_var:
                mod_idx = par_var.index(mod)
                for par in shm.dat[i][mod]:
                    saveList[2 * mod_idx + 3].append(shm.dat[i][mod][par])
                    saveList[2 * mod_idx + 4].append([i, mod, par, I])
    # Open file, write and close.
    with open(save_file, "wb") as f:
        pckl.dump(saveList, f, protocol=pckl.HIGHEST_PROTOCOL)
