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



import copy
import numpy as np
import os



from statestream.meta.network import network_rollback, \
                                     shortest_path, \
                                     MetaNetwork



# =============================================================================

class saliency_grad(object):
    """Provide saliency using grad CAM visualization.

    For now we only allow metrics between two (maybe same) nps.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : string
        Instance name of this system client.
    client_param : dict
        Dictionary containing system clients parameters.
    """
    def __init__(self, name, net, client_param):
        self.name = copy.copy(name)
        self.type = "saliency_grad"
        self.p = copy.deepcopy(client_param)
        self.net = copy.deepcopy(net)
        self.param = copy.deepcopy(client_param['param'])
        self.dat = {}
        self.sv = copy.deepcopy(client_param['selected_values'])
        self.mn = MetaNetwork(net)
        
        self.parameter = {}
        for p,P in enumerate(client_param['params']):
            self.parameter[P['name']] = P['value']

        # Number of samples evaluated (only want the first).
        self.samples = 1

        self.stats = {}
        for v,V in client_param['variables'].items():
            self.stats[v] = np.zeros(V['shape'], dtype=np.float32)

        self.current_frame = 0
        


    def initialize(self, shm):
        """Initialize this client.
        """
        # Check / set device and backend.
        self.parameter['device'] = self.parameter.get('device', 'cpu')
        # Set GPU.
        os.environ["THEANO_FLAGS"] = 'floatX=float32, optimizer_including=cudnn, optimizer=fast_compile, mode=FAST_RUN, blas.ldflags="-lopenblas"' \
                                     + ",device=" + self.parameter['device']
        # Here now theano may be imported and everything dependent on it.
        import theano
        import theano.tensor as T
        from statestream.neuronal.neuron_pool import NeuronPool
        from statestream.neuronal.synapse_pool import SynapsePool
        # Generate theano variables.
        self.th_stats = {}
        for v,V in self.stats.items():
            self.th_stats[v] \
                = theano.shared(np.zeros(V.shape, dtype=theano.config.floatX),
                                borrow=True,
                                name='stats ' + v)

        # Build nps / sps.
        # ---------------------------------------------------------------------
        # Determine depth.
        state_stream = shortest_path(self.net, self.parameter['source'], self.parameter['target'])
        self.client_graph_depth = len(state_stream) - 1
        # Build client graph.
        self.client_graph_nps = [[] for i in range(self.client_graph_depth + 1)]
        self.client_graph_sps = [[] for i in range(self.client_graph_depth + 1)]
        # Initialize with target layer.
        self.client_graph_nps[self.client_graph_depth].append(self.parameter["target"])
        # Rollback (fill) graph.
        network_rollback(self.net, self.client_graph_depth, self.client_graph_nps, self.client_graph_sps)

        self.nps = {}
        self.sps = {}
        self.all_nps = list(set([n for np_list in self.client_graph_nps for n in np_list]))
        self.all_sps = list(set([s for sp_list in self.client_graph_sps for s in sp_list]))
        # Create all necessary neuron pools.
        loc_net = copy.deepcopy(self.net)
        loc_net['agents'] = self.samples
        loc_mn = MetaNetwork(loc_net)
        self.net['agents'] = 1
        for n in self.all_nps:
            if n not in self.nps:
                self.nps[n] = NeuronPool(n, loc_net, self.param, loc_mn)
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
                                      loc_net,
                                      self.param,
                                      loc_mn,
                                      source_np_list,
                                      self.nps[S["target"]])

        # Rollout network to specified depth.
        # ---------------------------------------------------------------------
        # Rollout network.
        for depth in range(self.client_graph_depth):
            # Post synaptic has to come BEFORE next state.
            for s in self.all_sps:
                if s in self.client_graph_sps[depth + 1]:
                    self.sps[s].compute_post_synaptic(as_empty=False)
                else:
                    self.sps[s].compute_post_synaptic(as_empty=True)
            # Now update next state.
            for n in self.all_nps:
                if n in self.client_graph_nps[depth + 1]:
                    self.nps[n].compute_algebraic_next_state(as_empty=False)
                else:
                    self.nps[n].compute_algebraic_next_state(as_empty=True)

        # Apply magic function on target.
        tgt_fcn = self.parameter['magic'].replace('#', 'self.nps[self.parameter["target"]].state[self.client_graph_depth]')
        self.target = eval(tgt_fcn)
        self.source = self.nps[self.parameter["source"]].state[0]
        # Define updates for grad-CAM visualization objects.
        self.updates = []
        grad = T.grad(self.target, self.source)
        avgpool_grad = T.mean(grad, axis=[2,3], keepdims=True)
        grad_weighted_maps = self.source * avgpool_grad
        grad_weighted_map = T.mean(grad_weighted_maps, axis=1, keepdims=True)
        self.updates.append((self.th_stats['grads'], grad))
        self.updates.append((self.th_stats['avgpool_grads'], T.unbroadcast(T.unbroadcast(avgpool_grad, 2), 3)))
        self.updates.append((self.th_stats['grad_weighted_maps'], grad_weighted_maps))
        self.updates.append((self.th_stats['grad_weighted_map'], T.unbroadcast(grad_weighted_map, 1)))
        # Define theano function for updates.
        self.update_client = theano.function([], [], updates=self.updates)



    def update_frame_readin(self, shm):
        """System client dependent read in.
        """
        pass



    def update_frame_writeout(self):
        """Method to compute activation statistics for all child nps.
        """
        self.current_frame += 1
        self.update_client()
        for v,V in self.th_stats.items():
            self.dat['variables'][v][:] = V.get_value()
        
