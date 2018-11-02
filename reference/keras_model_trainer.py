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



import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
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
from keras.datasets.cifar import load_batch
from keras.datasets import mnist
from keras.engine import Layer

import os
import sys
import numpy as np
import pickle
import copy
from ruamel_yaml import YAML
from time import strftime, gmtime
import scipy
from scipy import ndimage



import keras_model_builder
from keras_model_builder import StGraph_2_keras
from stdatasets import Stdataset



def print_help():
    """Function to print help instructions to konsole.
    """
    print("\nTo train a model, the model name and mode has to be specified:")
    print("    python keras_model_trainer.py model_specifications/<model name>.st_graph <mode>")
    print("    Available models: \n")
    for s in available_specs:
        print("        model_specifications/" + s)
    print("    Modes: streaming, sequential")
    print("")

# Get available specification files.
available_specs = ["model_specifications" + os.sep + a for a in os.listdir("./model_specifications/")]

# Check input arguments.
if len(sys.argv) != 3:
    print_help()
    sys.exit()
if sys.argv[1] not in available_specs:
    print_help()
    sys.exit()
if sys.argv[2] not in ["streaming", "sequential"]:
    print_help()
    sys.exit()

# Get model name from parameters.
model_name = sys.argv[1].split(".")[0].split("/")[1]

# Flag for full (computational expensive) model evaluation during training.
full_eval = True

# Get meta data (training epochs, input noise, rollout window size).
if model_name.startswith("mnist"):
    epochs = 100
    noise_std = 2.0
    rollout_window = 8
    repetitions = 6
elif model_name.startswith("cifar"):
    epochs = 100
    noise_std = 1.0
    rollout_window = 8
    repetitions = 1
elif model_name.startswith("gtsrb"):
    epochs = 10         # 100
    noise_std = 0.5
    rollout_window = 8
    repetitions = 4     # 12
else:
    epochs = 100
    noise_std = 1.0
    rollout_window = 8
    repetitions = 1

# Load model specification to dictionary stg.
dataset = None
try:
    yaml = YAML()
    stg = yaml.load(open(sys.argv[1], "r"))
    dataset = stg["interfaces"]["data"]["type"]
except:
    print("\nError: Unable to load specification from " + str(sys.argv[1]))
    sys.exit()



# Load and prepare dataset.
DS = Stdataset(dataset)



# Repeat everything.
for rep in range(repetitions):
    # Generate keras model from specification.
    KM = StGraph_2_keras(stg, sys.argv[2], rollout_window, noise_std)
    # Shuffle dataset for every repetition.
    DS.shuffle()
    # Generate rolled out datasets for cifar10 and mnist.
    # For gtsrb we will sample tracks epoch-wise.
    DATAx = {}
    DATAy = {}
    if dataset in ["cifar10", "mnist"]:
        for d in ["train", "valid", "test"]:
            DATAx[d] = np.concatenate([DS.DATAX[d] for r in range(KM.rollouts + 1)], axis=3)
            DATAy[d] = np.concatenate([DS.DATAY[d] for r in range(len(KM.M["outputs"]))], axis=1)
    elif dataset == "gtsrb":
        for d in ["valid", "test"]:
            DATAx[d] = np.zeros([DS.DATAX[d].shape[0],
                                 DS.DATAX[d].shape[1],
                                 DS.DATAX[d].shape[2],
                                 3 * (KM.rollouts + 1)])
            rnd_startframe = np.random.randint(low=0, high=30 - KM.rollouts - 2, size=[DS.DATAX[d].shape[0],])
            for i in range(DS.DATAX[d].shape[0]):
                DATAx[d][i,:,:,:] = DS.DATAX[d][i,:,:,3 * rnd_startframe[i]:3 * (rnd_startframe[i] + KM.rollouts + 1)]
            # Set ground truth tensor.
            DATAy[d] = np.concatenate([DS.DATAY[d] for r in range(len(KM.M["outputs"]))], axis=1)

    # Dictionary to store results during training.
    results = {}
    # Dictionary to store epoch-wise accuracies.
    for d in ["train", "valid", "test"]:
        results[d + "_acc"] = []

    # Train and evaluate model for the specified number of epochs.
    for e in range(epochs):
        # For the GTSRB dataset we use tracks not repeated samples.
        # As "temporal augmentation" we sample a different starting
        # frame for every track.
        if dataset == "gtsrb" and e == 0:
            # For training we sample from the entire track.
            DATAx["train"] = np.zeros([DS.DATAX["train"].shape[0],
                                       DS.DATAX["train"].shape[1],
                                       DS.DATAX["train"].shape[2],
                                       3 * (KM.rollouts + 1)])
            rnd_startframe = np.random.randint(low=0, high=30 - KM.rollouts - 2, size=[DS.DATAX["train"].shape[0],])
            for i in range(DATAx["train"].shape[0]):
                DATAx["train"][i,:,:,:] = DS.DATAX["train"][i,:,:,rnd_startframe[i] * 3:(rnd_startframe[i] + KM.rollouts + 1) * 3]
            # Set ground truth tensor.
            DATAy["train"] = np.concatenate([DS.DATAY["train"] for r in range(len(KM.M["outputs"]))], axis=1)
        # Train the rollout window for one epoch.
        print("Training epoch " + str(e + 1) + " / " + str(epochs))
        KM.train_epoch(DATAx, DATAy)
        print("TRAIN SHAPES: " + str(DATAy["train"].shape) + "  " + str(DS.DATAY["train"].shape))

        # Evaluate model.
        accuracy = {}
        for a in ["train", "valid", "test"]:
            if (full_eval and a in ["train", "valid"]) or (a == "test" and e == epochs - 1):
                # Compute logits on dataset.
                current_logits = []
                # ["logits"][epoch][batch][rollouts][batchsize, 10]
                for b in range(DATAx[a].shape[0] // KM.batchsize):
                    batch = DATAx[a][b * KM.batchsize:(b + 1) * KM.batchsize]
                    outs = KM.M["output function"]([batch] + [0])
                    current_logits.append(outs)

                # Compute test accuracy of current epoch.
                # For every batch, stack rollouts.
                current_batch = []
                for b in range(len(current_logits)):
                    current_batch.append(np.stack(current_logits[b], axis=-1))
                current_epoch = np.concatenate(current_batch)
                current_classes = np.argmax(current_epoch, axis=1)

                # Compute accuracies and store results.
                accuracy[a] = np.zeros([current_classes.shape[1],])
                for s in range(current_epoch.shape[0]):
                    for r in range(current_classes.shape[1]):
                        if np.argmax(DS.DATAY[a][s,:]) == current_classes[s][r]:
                            accuracy[a][r] += 1.
                accuracy[a] /= current_epoch.shape[0]
                results[a + "_acc"].append(accuracy[a])

                # Print some evaluation information.
                print("        epoch " + str(e) + "/" + str(epochs) + "  " + a + " acc.: " + str((100 * accuracy[a]).astype(np.int32)))

    # Save trained model.
    results_file = "model_trained/" + model_name + "-" + sys.argv[2] + "-" + str(rep)
    KM.M["model"].save_weights(results_file + ".h5")
    M_json = KM.M["model"].to_json()
    with open(results_file + ".json", "w") as model_file:
        model_file.write(M_json)
    pickle.dump(results, open(results_file + ".results", "wb"))
