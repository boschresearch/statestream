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



from __future__ import print_function



import os
import sys
import numpy as np
import pickle
import copy
from ruamel_yaml import YAML
from time import strftime, gmtime
import scipy
from scipy import ndimage



import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.datasets.cifar import load_batch
from keras.datasets import mnist
from keras.engine import Layer
from keras.layers import Activation, \
                         Add, \
                         Concatenate, \
                         Conv2D, \
                         Dense, \
                         Dropout, \
                         Flatten, \
                         GaussianNoise, \
                         Input, \
                         Lambda, \
                         MaxPooling2D, \
                         UpSampling2D



import statestream.meta.network as mn
from statestream.utils.yaml_wrapper import load_yaml, dump_yaml



class MyGaussianNoise(Layer):
    """Noise layer to apply noise and clip output.

    Parameter:
    ----------
    stddev :: float
        The standard deviation used to create normal noise.
    """
    def __init__(self, stddev, **kwargs):
        super(MyGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return K.clip(inputs + K.random_normal(shape=K.shape(inputs), 
                                          mean=0., 
                                          stddev=self.stddev), 0.0, 1.0)
        return noised()

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(MyGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def temporal_cross_entropy(target, prediction):
    """Compute the (temporal) sum over cross entropies.
    """
    pred_clipped = K.clip(prediction, 1e-7, 1.0 - 1e-7)
    return - K.sum(target * K.log(pred_clipped), axis=1)







class StGraph_2_keras(object):
    """ Keras model builder for st_graph specifications.

    This class generates a Keras model from a st_graph specification file.
    Thereby the rollout window is explicitly generated, dependent on the
    specified rollout pattern in the st_graph file. The rollout window
    for the streaming rollout pattern is always generated without any
    further specifications. The rollout window for the non-streaming case
    is generated unrolling edges by default sequentially and only when 
    specified (tag "stream") in the streaming manner.

    Parameter:
    ----------
    stgraph :: dict
        Dictionary with statestream model specification.
    mode :: string
        A string specifying the rollout mode. Either "streaming" 
        or "sequential". In case of sequential, all synapse-pools
        with the tag "stream" will be rolled out with 
        temporal skip. 
    rollout_window :: int
        Rollout window size.
    noise_std :: float
        The standard deviation used to create normal noise on the input.
    """
    def __init__(self, stgraph, mode, rollout_window, noise_std):
        # Load stgraph specification file.
        self.mn = mn.MetaNetwork(stgraph)
        self.net = self.mn.net

        # Set mode and rollout window size.
        self.mode = copy.copy(mode)
        self.noise_std = noise_std
        self.first_response = None
        self.rollouts = None
        self.first_response = stgraph["first_response_" + mode]
        self.rollouts = rollout_window + self.first_response - 1

        # Determine batchsize.
        self.batchsize = self.net["agents"]

        # Initialize structures for results.
        self.history = []

        # Get all network inputs and their dimensions.
        self.input_shape = []
        self.input_name = []
        for i,I in self.net["interfaces"].items():
            for o in I["out"]:
                tmp_target = o
                # Consider remapping.
                if "remap" in I:
                    if o in I["remap"]:
                        tmp_target = I["remap"][o]
                self.input_name.append(copy.copy(tmp_target))
                # Add temporal (rollout window) dimension to channels.
                shape = self.net["neuron_pools"][tmp_target]["shape"]
                self.input_shape.append([shape[1], shape[2], shape[0] * (self.rollouts + 1)])

        # This list of all np names determines the order of nps during initialization.
        self.nps = []
        for n in self.net["neuron_pools"]:
            self.nps.append(n)

        # Define temporal slicing functions for stacked inputs.
        self.my_slice = {}
        for np_idx,np_name in enumerate(self.input_name):
            self.my_slice[np_name] = {}
            features = self.input_shape[np_idx][2] // (self.rollouts + 1)
            for r in range(2 * (self.first_response + self.rollouts + 1)):
                self.my_slice[np_name][str(r)] = \
                    lambda x: x[:,:,:,(features * r):(features + (features * r))]

        logits = []
        self.M = {}

        # Create input sequence layers for all network inputs.
        for i in range(len(self.input_name)):
            self.M["input " + self.input_name[i]] = \
                Input(shape=self.input_shape[i])
            # The noise layer is always active.
            self.M["noisy_input " + self.input_name[i]] = \
                MyGaussianNoise(self.noise_std)(self.M["input " + self.input_name[i]])

        # Helper variables to generate rolled-out keras graph.
        self.M["layer"] = {}
        for n in self.net["neuron_pools"]:
            self.M["layer"][n] = [{} for r in range(self.rollouts + 1)]
            for r in range(self.rollouts + 1):
                self.M["layer"][n][r]["state"] = None
                self.M["layer"][n][r]["updated"] = False

        # Generate all parameterized functions.
        self.M["func"] = {}
        for n,N in self.net["neuron_pools"].items():
            self.M["func"][n] = {}
            for s,S in self.net["synapse_pools"].items():
                if S["target"] == n:
                    src_nps = [src for srcs in S["source"] for src in srcs]
                    if "rf" in S:
                        if isinstance(S["rf"], list):
                            rf_nps = [rf for rfs in S["rf"] for rf in rfs]
                        else:
                            rf_nps = [S["rf"] for srcs in S["source"] for src in srcs]
                    else:
                        rf_nps = [1 for srcs in S["source"] for src in srcs]
                    for snp_idx, snp in enumerate(src_nps):
                        # If target has no space, always use dense, otherwise convolutions.
                        if N["shape"][2] == 1:
                            self.M["func"][n][snp] = Dense(N["shape"][0], name=snp + "__" + n)
                        else:
                            # We need to consider down-sampling here.
                            if N["shape"][2] < self.net["neuron_pools"][snp]["shape"][2]:
                                down_factor = self.net["neuron_pools"][snp]["shape"][2] // N["shape"][2]
                                self.M["func"][n][snp] = Conv2D(N["shape"][0], 
                                                                (rf_nps[snp_idx], rf_nps[snp_idx]),
                                                                padding="same",
                                                                strides=(down_factor,down_factor))
                            else:
                                self.M["func"][n][snp] = Conv2D(N["shape"][0], 
                                                                (rf_nps[snp_idx], rf_nps[snp_idx]),
                                                                padding="same")

        # Initialize input states.
        for n in self.net["neuron_pools"]:
            self.M["layer"][n][0]["updated"] = True

        # Initialize input layers.
        for r in range(self.rollouts + 1):
            # Assuming input sequence is of shape:
            #     [batchsize, X, Y, time = rollouts + 1]
            for n in self.input_name:
                self.M["layer"][n][r]["state"] = Lambda(self.my_slice[n][str(r)])(self.M["noisy_input " + n])
                self.M["layer"][n][r]["updated"] = True

        # Initial zero-th network frame (except network inputs), which will be ignored.
        for n in self.net["neuron_pools"]:
            if n not in self.input_name:
                self.M["layer"][n][0]["state"] = None
                self.M["layer"][n][0]["updated"] = True

        # Iteratively build rollout computation graph.
        layers_updated = self.rollouts * len(self.input_name)
        while layers_updated > 0:
            layers_updated = 0
            # Loop over rollout frames.
            for r in np.arange(1, self.rollouts + 1):
                # Loop over all nps of r-th frame.
                for n,N in self.net["neuron_pools"].items():
                    # Check if np at r already updated.
                    if not self.M["layer"][n][r]["updated"]:
                        # Check if neuron_pool n can be updated for rollout-step r.
                        update_able = True
                        # Determine all sources (nps) in the rollout (np_name, frame).
                        src_nodes = []
                        for s,S in self.net["synapse_pools"].items():
                            if S["target"] == n:
                                src_nps = [src for srcs in S["source"] for src in srcs]
                                for src_np in src_nps:
                                    if self.mode == "streaming":
                                        src_nodes.append([src_np, r - 1])
                                    elif self.mode == "sequential":
                                        if "stream" in self.net["synapse_pools"][s]["tags"]:
                                            src_nodes.append([src_np, r - 1])
                                        else:
                                            src_nodes.append([src_np, r])
                        # If all sources are updated, N can also be updated.
                        for src_np in src_nodes:
                            if not self.M["layer"][src_np[0]][src_np[1]]["updated"]:
                                update_able = False
                                break

                        if update_able:
                            layers_updated += 1
                            # Determine number of stateful inputs.
                            states = []
                            for s,S in enumerate(src_nodes):
                                if self.M["layer"][S[0]][S[1]]["state"] is not None:
                                    states.append(s)
                            # Dependent on the number of input states, update current node in rollout.
                            if len(states) == 0:
                                # Only dependent on empty-init states --> do not compute anything.
                                pass
                            elif len(states) == 1:
                                # Only one input state, so no need for a summation over inputs.
                                S = src_nodes[states[0]]
                                if self.net["neuron_pools"][S[0]]["shape"][2] >= N["shape"][2]:
                                    # Flatten in case target has space (1,1) and source has some space (X,Y)
                                    if N["shape"][2] == 1 and self.net["neuron_pools"][S[0]]["shape"][2] > 1:
                                        this_state = self.M["func"][n][S[0]](Flatten()(self.M["layer"][S[0]][S[1]]["state"]))
                                    else:
                                        this_state = self.M["func"][n][S[0]](self.M["layer"][S[0]][S[1]]["state"])
                                elif self.net["neuron_pools"][S[0]]["shape"][2] < N["shape"][2]:
                                    up_factor = N["shape"][2] // self.net["neuron_pools"][S[0]]["shape"][2]
                                    this_state = UpSampling2D(size=(up_factor, up_factor), data_format="channels_first")(self.M["layer"][S[0]][S[1]]["state"])
                                    this_state = self.M["func"][n][S[0]](this_state)
                            else:
                                # Several input states which need to be summed up.
                                single_inputs = []
                                for s in states:
                                    S = src_nodes[s]
                                    if self.net["neuron_pools"][S[0]]["shape"][2] >= N["shape"][2]:
                                        # Flatten in case target has space (1,1) and source has some space (X,Y)
                                        if N["shape"][2] == 1 and self.net["neuron_pools"][S[0]]["shape"][2] > 1:
                                            single_inputs.append(self.M["func"][n][S[0]](Flatten()(self.M["layer"][S[0]][S[1]]["state"])))
                                        else:
                                            single_inputs.append(self.M["func"][n][S[0]](self.M["layer"][S[0]][S[1]]["state"]))
                                    elif self.net["neuron_pools"][S[0]]["shape"][2] < N["shape"][2]:
                                        up_factor = N["shape"][2] // self.net["neuron_pools"][S[0]]["shape"][2]
                                        single_inputs.append(self.M["func"][n][S[0]](UpSampling2D(size=(up_factor, up_factor))(self.M["layer"][S[0]][S[1]]["state"])))
                                this_state = Add()(single_inputs)

                            # Dropout and activation function.
                            if len(states) > 0:
                                # Apply dropout.
                                if "dropout" in N:
                                    this_state = Dropout(N["dropout"])(this_state)

                                if "act" in N:
                                    self.M["layer"][n][r]["state"] = Activation(N['act'])(this_state)
                                else:
                                    self.M["layer"][n][r]["state"] = this_state

                            self.M["layer"][n][r]["updated"] = True

                            if self.M["layer"][n][r]["state"] is None:
                                print("UPCHECK for " + n + " / " + str(r) + \
                                          "  SRCS: " + str(src_nodes) + "  shape: empty None state")                       
                            else:
                                print("UPCHECK for " + n + " / " + str(r) + \
                                          "  SRCS: " + str(src_nodes) + "  shape: " + str(self.M["layer"][n][r]["state"].get_shape().as_list()))                        
    
        # Apply softmax on temporal logit sum.
        self.M["outputs"] = [Activation('softmax')(self.M["layer"]["prediction"][self.first_response + r]["state"]) for r in range(rollout_window)]
        if len(self.M["outputs"]) > 1:
            self.M["concat"] = Concatenate(axis=-1)(self.M["outputs"])
        else:
            self.M["concat"] = self.M["outputs"][0]

        # Define list of all model inputs (input data sequence).
        model_inputs = [self.M["input " + n] for n in self.input_name]

        # Function to return network outputs.
        self.M["output function"] = K.function(model_inputs + [K.learning_phase()], self.M["outputs"])

        # Define model function.
        self.M["model"] = Model(inputs=model_inputs, outputs=self.M["concat"])

        if True:
            self.M["model"].summary()

        # initiate RMSprop optimizer
        print("Define optimizer ...")
        self.opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # lr=0.0001

        # Let's train the model using RMSprop
        print("Compile model ...")
        self.M["model"].compile(loss=temporal_cross_entropy,
                                optimizer=self.opt,
                                metrics=['accuracy'])



    def train_epoch(self, DATAx, DATAy):
        """Train model for one epoch on provided data.
        """
        history = self.M["model"].fit(DATAx["train"], 
                                      DATAy["train"], 
                                      batch_size=self.batchsize,
                                      epochs=1,
                                      verbose=1,
                                      validation_data=(DATAx["valid"], 
                                                       DATAy["valid"]),
                                      shuffle=True)
        self.history.append(copy.deepcopy(history.history))



