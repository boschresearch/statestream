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
import networkx as nx
import copy
import time



def generate_graph(net, lod='np-np'):
    """Generate a networkx graph from a network dictionary.
    
    Parameters
    ----------
    net : dict
        The network dictionary to be converted.
    mode : string
        Specifies the level of detail the graph is generated.
        'np-np': All neuron-pools are generated as nodes,
            all synapse-pools are generated as edges.
        'np-sp-np': All nps and sps are generated as nodes.
        'np-f-sp-np': All nps, sps and sp's factors are
            generated as nodes.
    Returns 
    -------
    nx.MultiDiGraph
        The converted graph.
    """
    nx_graph = nx.MultiDiGraph()
    nx_graph.lod = copy(lod)

    # Add all neuron-pools as nodes of type np.
    for n,N in net['neuron_pools'].items():
        nx_graph.add_node(n, node_type='np')

    # Dependent on lod add synapse-pools.
    if lod == 'np-np':
        # Add synapse-pools as edges.
        for s,S in net['synapse_pools'].items():
            for factor in S['source']:
                for src in factor:
                    nx_graph.add_edge(src, S['target'],
                                      edge_name=s)
    elif lod == 'np-sp-np':
        # Add synapse-pools as nodes.
        for s,S in net['synapse_pools'].items():
            nx_graph.add_node(s, node_type='sp')
            nx_graph.add_edge(s, S['target'])
            # Add all sources as edges.
            for factor in S['source']:
                for src in factor:
                    nx_graph.add_edge(src, s)
    elif lod == 'np-f-sp-np':
        # Add synapse-pool's factors as nodes.
        for s,S in net['synapse_pools'].items():
            nx_graph.add_node(s, node_type='sp')
            nx_graph.add_edge(s, S['target'])
            for f,F in enumerate(S['source']):
                node_name = s + " " + str(f)
                nx_graph.add_node(node_name,
                                  node_type='factor')
                nx_graph.add_edge(node_name, s)
                for src in F:
                    nx_graph.add_edge(src, node_name)

    return nx_graph
