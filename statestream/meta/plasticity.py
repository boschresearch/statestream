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

from statestream.meta.network import MetaNetwork
from statestream.meta.neuron_pool import np_needs_rebuild
from statestream.meta.synapse_pool import sp_needs_rebuild



def plast_needs_rebuild(orig_net, new_net, plast_id):
    """Determines if a plasticity needs rebuild for new network dictionary.
    
    Parameters:
    orig_net : dict
        Dictionary containing the original network.
    new_net : dict
        Dictionary containing the new edited network.
    plast_id : str
        String specifying the plasticity under consideration.
    """
    needs_rebuild = False
    # Get meta-network.
    new_mn = MetaNetwork(new_net)
    # Check if any needed np/sp has to be rebuild.
    all_nps = list(set([n for np_list in new_mn.net_plast_nps[plast_id] for n in np_list]))
    all_sps = list(set([s for sp_list in new_mn.net_plast_sps[plast_id] for s in sp_list]))
    for n in all_nps:
        if np_needs_rebuild(orig_net, new_net, n):
            needs_rebuild = True
            break
    for s in all_sps:
        if sp_needs_rebuild(orig_net, new_net, s):
            needs_rebuild = True
            break
    # Return rebuild flag.
    return needs_rebuild

