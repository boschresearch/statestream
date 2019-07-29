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

from statestream.utils.helper import is_scalar_shape
from statestream.meta.neuron_pool import np_shm_layout, np_state_shape, np_init
from statestream.backends.backends import import_backend



class NeuronPool(object):
    """Default constructor of neuron pool class.

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
    """
    def __init__(self, name, net, param, mn):
        # Import backend.
        self.B = import_backend(None, None, name)
        # Get global structure.
        self.name = name
        self.net = net
        self.param = param
        self.mn = mn
        # Get local representation of shared memory.
        self.dat_layout = np_shm_layout(self.name, self.net, self.param)
        self.dat = {}
        for t in ["parameter", "variables"]:
            settable = True
            self.dat[t] = {}
            for i,i_l in self.dat_layout[t].items():
                if i_l.type == "backend":
                    if is_scalar_shape(i_l.shape):
                        self.dat[t][i] = self.B.scalar(1.0, 
                                                       dtype=np.float32, 
                                                       name=name + "." + i,
                                                       settable=settable)
                    else:
                        if i_l.broadcastable is not None:
                            self.dat[t][i] = self.B.variable(np.ascontiguousarray(np.zeros(i_l.shape, dtype=np.float32)),
                                                                        borrow=True,
                                                                        name=name + "." + i,
                                                                        broadcastable=i_l.broadcastable,
                                                                        settable=settable)
                        else:
                            self.dat[t][i] = self.B.variable(np.ascontiguousarray(np.zeros(i_l.shape, dtype=np.float32)),
                                                                        borrow=True,
                                                                        name=name + "." + i,
                                                                        settable=settable)
                elif i_l.type == "np":
                    if is_scalar_shape(i_l.shape):
                        self.dat[t][i] = np.array([1,], dtype=i_l.dtype)
                    else:
                        self.dat[t][i] = np.array(i_l.shape, dtype=i_l.dtype)

        # Get np dictionary.
        self.p = self.net["neuron_pools"][self.name]
        # Get binary flag.
        self.binary = self.p.get("binary", False)
        # Get shape of neuron pool.
        self.shape = np_state_shape(self.net, self.name)
        # Get random seed.
        self.rng = np.random.RandomState(self.param["core"]["random_seed"])
        # Get bias and gain shape.
        self.bias_shape = self.p.get("bias_shape", "feature")
        self.gain_shape = self.p.get("gain_shape", False)
        # Check for un-shared for bias / gain.
        self.bias_unshare = False
        self.gain_unshare = False
        if "unshare" in self.p:
            if "b" in self.p["unshare"]:
                self.bias_unshare = True
            if "g" in self.p["unshare"]:
                self.gain_unshare = True
        # Get activation function.
        self.activation = self.p.get("act", "Id")
        # Get / set noise.
        self.noise = self.p.get("noise", None)
        if self.noise is not None:
            if self.noise == "Bernoulli":
                self.noise_dist = self.B.randomstream(np.random.RandomState(42).randint(99999), "binomial")
            else:
                self.noise_dist = self.B.randomstream(np.random.RandomState(42).randint(99999), self.noise)
        # Get np dropout (stochastic activation).
        self.dropout = self.p.get("dropout", None)
        if self.dropout is not None:
            # Random stream for dropout.
            self.dropout_srng = self.B.randomstream(np.random.RandomState(42).randint(99999), "binomial")
        # Get np zoneout.
        self.zoneout = self.p.get("zoneout", None)
        if self.zoneout is not None:
            # Random stream for zoneout.
            self.zoneout_srng = self.B.randomstream(np.random.RandomState(42).randint(99999), "binomial")
        # Get / set batch-norm specified or default parameter.
        self.batchnorm_mean = self.p.get("batchnorm_mean", False)
        self.batchnorm_std = self.p.get("batchnorm_std", False)
        # Get / set layernorm for mean / std correction.
        self.layernorm_mean = self.p.get("layernorm_mean", False)
        self.layernorm_std = self.p.get("layernorm_std", False)

        # Initialize algebraic next states with empty.
        self.state = []
        self.state_SUM = []
        self.state_AB = []
        self.state_SPA = []
        self.sparse_state = []
        # Define state of neuron-pool dependent on backend dim. ordering.
#        self.state.append(theano.shared(np.ascontiguousarray(np.zeros(self.shape, dtype=np.float32)),
#                                                             borrow=True,
#                                                             name=name + " state"))
        state = self.B.variable(np.zeros(self.shape, dtype=np.float32),
                                borrow=True,
                                name=name + ".state",
                                settable=True)

        self.state.append(state)

        # Begin with an empty sources (synapse pools) list.
        self.sources = []
        # Neuron pool state.
        self.running = True

    def compute_algebraic_next_state(self, is_input=False, as_empty=False):
        """This method computes the np update from one frame to the next.

        Cummulate all incomming post-synapic inputs and apply activation.

        Parameters:
        -----------
        is_input : boolean
            Flag to specify if this neuron-pool is an input (= output of an if). If so we
            do not want to apply the activation function again.
        as_empty : boolean
            Flag if this instance is to be initialized as empty (this is necessary for efficitne
            local rollout in e.g. loss based plasticities).
        """
        # Do nothing to activation if is input.
        # NOTE: Otherwise activation would have been applied twice.
        if is_input:
            self.state.append(self.state[-1])
        else:
            if as_empty:
                # Empty if not needed.
                self.state_SUM.append(None)
                self.state_AB.append(None)
                self.state_SPA.append(None)
            else:
                if len(self.sources) == 0:
                    self.state_SUM.append(0 * self.state[-1])
                else:
                    if self.binary:
                        # Sum post synaptic activations as OR over Bernoulli dists.
                        self.state_SUM.append(self.sources[0].post_synaptic[-1])
                        for sp in range(len(self.sources) - 1):
                            self.state_SUM[-1] += self.sources[sp + 1].post_synaptic[-1] * (1.0 - self.state_SUM[-1])
                    else:
                        # Sum post synaptic activations.
                        self.state_SUM.append(self.sources[0].post_synaptic[-1])
                        for sp in range(len(self.sources) - 1):
                            self.state_SUM[-1] += self.sources[sp + 1].post_synaptic[-1]
                # Add noise.
                if self.noise == "normal":
                    self.state_SUM[-1] += self.dat["parameter"]["noise_mean"] \
                                          + self.dat["parameter"]["noise_std"] \
                                          * self.noise_dist(self.shape)
                elif self.noise == "uniform":
                    self.state_SUM[-1] += (self.dat["parameter"]["noise_max"] - self.dat["parameter"]["noise_min"]) \
                                          * self.noise_dist(self.shape) \
                                          + self.dat["parameter"]["noise_min"]
                elif self.noise == "Bernoulli":
                    self.state_SUM[-1] += self.noise_dist(self.shape, p=self.dat["parameter"]["noise_prob"]) \
                    #self.state_SPA[-1] = self.state_SPA[-1] * self.B.cast(mask, self.B._DTYPE)

                # Apply batch normalization for mean.
                if self.batchnorm_mean:
                    if self.batchnorm_mean == "full":
                        mean = self.B.mean(self.state_SUM[-1], 
                                           axis=[0, 1, 2, 3])
                    elif self.batchnorm_mean == "spatial":
                        mean = self.B.mean(self.state_SUM[-1], 
                                           axis=[0, 2, 3])
                        mean = self.B.dimshuffle(mean, ('x', 0, 'x', 'x'))
                    elif self.batchnorm_mean == "feature":
                        mean = self.B.mean(self.state_SUM[-1], 
                                           axis=[0, 1])
                        mean = self.B.dimshuffle(mean, ('x', 'x', 0, 1))
                    elif self.batchnorm_mean == "scalar":
                        mean = self.B.mean(self.state_SUM[-1], axis=[0])
                        mean = self.B.dimshuffle(mean, ('x', 0, 1, 2))
                    self.state_SUM[-1] = self.state_SUM[-1] - mean

                # Apply batch normalization for std.
                if self.batchnorm_std:
                    # Compute mean.
                    if self.batchnorm_std == "full":
                        std = self.B.sqrt(self.B.var(self.state_SUM[-1], 
                                                     axis=[0, 1, 2, 3]) + 1e-6)
                    if self.batchnorm_std == "spatial":
                        std = self.B.dimshuffle(self.B.sqrt(self.B.var(self.state_SUM[-1], 
                                                                       axis=[0,2,3]) + 1e-6), 
                                                ('x', 0, 'x', 'x'))
                    if self.batchnorm_std == "feature":
                        std = self.B.dimshuffle(self.B.sqrt(self.B.var(self.state_SUM[-1], 
                                                                       axis=[0,1]) + 1e-6), 
                                                ('x', 'x', 0, 1))
                    if self.batchnorm_std == "scalar":
                        std = self.B.dimshuffle(self.B.sqrt(self.B.var(self.state_SUM[-1], axis=[0]) + 1e-6), 
                                                ('x', 0, 1, 2))
                    # Divide by std.
                    self.state_SUM[-1] = self.state_SUM[-1] / std
                
                # Apply layer normalization for mean.
                # ------------------------------------------------------------
                if self.layernorm_mean:
                    # Compute mean.
                    if self.layernorm_mean == "full":
                        mean = self.B.dimshuffle(self.B.mean(self.state_SUM[-1], 
                                                             axis=[1,2,3]), 
                                                 (0, 'x', 'x', 'x'))
                    elif self.layernorm_mean == "spatial":
                        mean = self.B.dimshuffle(self.B.mean(self.state_SUM[-1], 
                                                             axis=[2,3]), 
                                                 (0, 1, 'x', 'x'))
                    elif self.layernorm_mean == "feature":
                        mean = self.B.dimshuffle(self.B.mean(self.state_SUM[-1], 
                                                             axis=[1]), 
                                                 (0, 'x', 1, 2))
                    # Substract mean.
                    self.state_SUM[-1] = self.state_SUM[-1] - mean

                # Apply layer normalization for variance.
                # ------------------------------------------------------------
                if self.layernorm_std:
                    # Compute mean.
                    if self.layernorm_std == "full":
                        std = self.B.dimshuffle(self.B.sqrt(self.B.var(self.state_SUM[-1], 
                                                                       axis=[1,2,3]) + 1e-6), 
                                                (0, 'x', 'x', 'x'))
                    if self.layernorm_std == "spatial":
                        std = self.B.dimshuffle(self.B.sqrt(self.B.var(self.state_SUM[-1], 
                                                                       axis=[2,3]) + 1e-6), 
                                                (0, 1, 'x', 'x'))
                    if self.layernorm_std == "feature":
                        std = self.B.dimshuffle(self.B.sqrt(self.B.var(self.state_SUM[-1], 
                                                                       axis=[1]) + 1e-6), 
                                                (0, 'x', 1, 2))
                    # Divide by std.
                    self.state_SUM[-1] = self.state_SUM[-1] / std

                # Apply gain.
                # ------------------------------------------------------------
                if self.gain_shape == "full":
                    if self.gain_unshare:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"], (0, 1, 2, 3))
                    else:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"], 
                                                                                    ("x", 0, 1, 2))
                elif self.gain_shape == "feature":
                    if self.gain_unshare:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"], 
                                                                                    (0, 1, "x", "x"))
                    else:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"], 
                                                                                    ("x", 0, "x", "x"))
                elif self.gain_shape == "spatial":
                    if self.gain_unshare:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"], 
                                                                                    (0, "x", 1, 2))
                    else:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"], 
                                                                                    ("x", "x", 0, 1))
                elif self.gain_shape == "scalar":
                    if self.gain_unshare:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.B.dimshuffle(self.dat["parameter"]["g"][:,0], 
                                                                                    (0, "x", "x", "x"))
                    else:
                        self.state_SUM[-1] = self.state_SUM[-1] * self.dat["parameter"]["g"][0]
                else:
                    # Nothing to be done here.
                    pass

                # Apply activation and bias.
                # ------------------------------------------------------------
                if self.bias_shape == "full":
                    if self.bias_unshare:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"], (0, 1, 2, 3)))
                    else:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"], 
                                                ("x", 0, 1, 2)))
                elif self.bias_shape == "feature":
                    if self.bias_unshare:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"],
                                                (0, 1, "x", "x")))
                    else:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"], 
                                                ("x", 0, "x", "x")))
                elif self.bias_shape == "spatial":
                    if self.bias_unshare:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"], 
                                                (0, "x", 1, 2)))
                    else:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"], 
                                                ("x", "x", 0, 1)))
                elif self.bias_shape == "scalar":
                    if self.bias_unshare:
                        self.state_AB.append(self.state_SUM[-1] \
                            + self.B.dimshuffle(self.dat["parameter"]["b"][:,0], 
                                                (0, "x", "x", "x")))
                    else:
                        self.state_AB.append(self.state_SUM[-1] + self.dat["parameter"]["b"][0])
                else:
                    self.state_AB.append(self.state_SUM[-1])

                # Evaluate activation function (which may be arbitrary).
                # First, replace all functions with the local backend.
                for fnc in self.B._FUNCTIONS:
                    self.activation = self.activation.replace("B." + fnc, "self.B." + fnc)
                while self.activation.find("self.self.") != -1:
                    self.activation = self.activation.replace("self.self.", "self.")
                # Insert state variable.
                if self.activation.find('$') != -1:
                    try:
                        activation = self.activation.replace('$', 'self.state_AB[-1]')
                        self.state_AB[-1] = eval(activation)
                    except:
                        print("\nError: Unable to evaluate activation function: " + str(self.activation) + ". Missing B. before function or unknown function?")
                else:
                    self.state_AB[-1] = eval("self.B." + self.activation)(self.state_AB[-1])


                # Sparcify state no activation. (Theano version dependent)
                # if self.sparsify == 0:
                #     self.state_SPA.append(self.state_AB[-1])
                # else:
                #     self.state_SPA.append(T.signal.downsample.max_pool_2d_same_size(self.state_AB[-1], 
                #                                                                     (self.sparsify, self.sparsify)))
                self.state_SPA.append(self.state_AB[-1])

                # Apply dropout.
                if self.dropout is not None:
                    # Get mask for dropout.
                    mask = self.dropout_srng(self.shape, p=self.dat["parameter"]["dropout"])
                    
                    self.state_SPA[-1] = self.state_SPA[-1] * self.B.cast(mask, self.B._DTYPE)

                # Apply zoneout.
                if self.zoneout is not None \
                        and len(self.state) > 0 \
                        and self.state[-1] is not None:
                    # Get mask for zoneout.
                    mask = self.zoneout_srng(n=1, 
                                             p=1 - self.dat["parameter"]["zoneout"],
                                             size=self.shape)
                    
                    self.state_SPA[-1] = self.state[-1] * (1.0 - self.B.cast(mask, self.B._DTYPE)) \
                                         + self.state_SPA[-1] * self.B.cast(mask, self.B._DTYPE)

            # Set next state.
            if self.state_SPA[-1] is None:
                self.state.append(self.state_SPA[-1])
            else:
                self.state.append(self.B.cpu_contiguous(self.state_SPA[-1]))
            
            
            
    def set_parameter(self, par=None, val=None):
        """Deprecated.
        """
        if par is None and val is None:
            # Initialize all data with default.
            for p in self.dat["parameter"]:
                self.dat["parameter"][p].set_value(np_init(self.net, self.name, p, self.dat_layout["parameter"][p]))
        elif par != None and val is None:
            # Initialize only par with default.
            self.dat["parameter"][par].set_value(np_init(self.net, self.name, par, self.dat_layout["parameter"][par]))
        elif par != None and val != None:
            # Set par to val.
            assert(val.shape == self.dat_layout["parameter"][par].shape), "ERROR: neuron_pool::set_parameter(), wrong shape."
            self.dat["parameter"][par].set_value(val)
            
            
            
            