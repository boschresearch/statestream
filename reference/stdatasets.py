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



import os
import sys
import gzip
import copy
import csv
if sys.version[0] == "2":
    import cPickle as pckl
elif sys.version[0] == "3":
    import pickle as pckl
import numpy as np
import scipy
from scipy import ndimage
import scipy.misc
import matplotlib.pyplot as plt



from keras import backend as K
from keras.utils import to_categorical
from keras.datasets.cifar import load_batch








# Here, source files / folders must be specified accordingly. 
cifar10_folder = "/.../cifar-10-batches-py/"
mnist_file = "/.../mnist.pkl.gz"
gtsrb_folder = "/.../GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/"




class Stdataset(object):
    """A class to provide access to used datasets.


    The main variables of this class are:

    self.res :: int
        The image resolution for x and y (e.g. 32 for CIFAR10 or 28 for MNIST).
    self.channels :: int
        The number of image channels (e.g. 3 for CIFAR10 or 1 for MNIST).
    self.classes :: int
        The number of classes (e.g. 10 for CIFAR10 or 43 for GTSRB).
    self.DATAX :: dictionary
        Dictionary of datasets 'train', 'valid', and 'test'. Each item
        contains a 4D tensor: [samples, res, res, channels] for MNIST and CIFAR10.
        For GTSRB entire tracks of 30 images are stored in each sample:
            [tracks, res, res, channels * 30]
    self.DATAY :: dictionary
        Dictionary of ground-truth labels 'train', 'valid', and 'test'. Each item
        contains a 2D tensor of one-hot encoded class labels: 
            [samples / tracks, classes]


    Parameter:
    ----------
    dataset :: string
        Specifies the used dataset: 'mnist', 'cifar10', 'gtsrb'
    """
    def __init__(self, dataset):
        self.dataset = dataset


        # Load and store dataset.
        # ====================================================================
        if dataset == "cifar10":
        # ====================================================================
            self.res = 32
            self.channels = 3
            self.classes = 10
            # Load images from file.
            num_train_samples = 50000
            x = {}
            y = {}
            self.DATAX = {}
            self.DATAY = {}
            x["train"] = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
            y["train"] = np.empty((num_train_samples,), dtype='uint8')
            for i in range(1, 6):
                fpath = os.path.join(cifar10_folder, 'data_batch_' + str(i))
                (x["train"][(i - 1) * 10000: i * 10000, :, :, :], y["train"][(i - 1) * 10000: i * 10000]) = load_batch(fpath)
            # Load also the test images.
            fpath = os.path.join(cifar10_folder, 'test_batch')
            x["test"], y["test"] = load_batch(fpath)
            y["train"] = np.reshape(y["train"], (len(y["train"]), 1))
            y["test"] = np.reshape(y["test"], (len(y["test"]), 1))
            # Resort data dimensions.
            if K.image_data_format() == 'channels_last':
                x["train"] = x["train"].transpose(0, 2, 3, 1)
                x["test"] = x["test"].transpose(0, 2, 3, 1)
            # Permute training images to extract validation.
            p = np.random.permutation(range(x["train"].shape[0]))
            x["train"] = x["train"][p]
            y["train"] = y["train"][p]
            # Split training data in train / valid.
            valid_size = x["train"].shape[0] // 5
            x["valid"] = x["train"][0:valid_size]
            y["valid"] = y["train"][0:valid_size]
            x["train"] = x["train"][valid_size:-1]
            y["train"] = y["train"][valid_size:-1]
            # Convert / normalize images.
            for t in ["train", "valid", "test"]:
                x[t] = x[t].astype('float32')
                self.DATAX[t] = x[t] / 255
                # Convert class vectors to binary class matrices.
                self.DATAY[t] = to_categorical(y[t], self.classes)



        # ====================================================================
        elif dataset == "mnist":
        # ====================================================================
            self.res = 28
            self.channels = 1
            self.classes = 10
            self.DATAY = {}
            self.DATAX = {}
            # Load dataset from file.
            DATA = {}
            f = gzip.open(mnist_file, "rb")
            if sys.version[0] == "2":
                DATA["train"], DATA["valid"], DATA["test"] = pckl.load(f)
            elif sys.version[0] == "3":
                DATA["train"], DATA["valid"], DATA["test"] = pckl.load(f, encoding="latin1")
            f.close()
            # Convert to standardized format.
            for a in ["train", "valid", "test"]:
                samples = DATA[a][1].shape[0]
                self.DATAX[a] = np.reshape(DATA[a][0], [samples, self.res, self.res])
                self.DATAX[a] = self.DATAX[a].astype("float32")
                self.DATAX[a] = self.DATAX[a][:,:,:,np.newaxis]
                self.DATAY[a] = to_categorical(DATA[a][1], self.classes)



        # ====================================================================
        elif dataset == "gtsrb":
        # ====================================================================
            print("Preparing GTSRB dataset ...")
            self.res = 32
            self.channels = 3
            self.classes = 43
            self.DATAX = {}
            self.DATAY = {}
            # Extract dataset from files.
            images = [] # list of all images
            labels = [] # corresponding labels
            # Loop over all 43 classes and load files.
            for c in range(0, 43):
                prefix = gtsrb_folder + '/' + format(c, '05d') + '/' # subdirectory for class
                gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
                gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
                # Loop over all images in current annotations file
                for row_idx, row in enumerate(gtReader):
                    if row_idx != 0:
                        images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                        labels.append(row[7]) # the 8th column is the label
                gtFile.close()
            # Determine number of tracks for single datasets.
            tracks = len(images) // 30
            num_tracks = {
                "test": tracks // 10,
                "valid": tracks // 10,
                "train": 0
            }
            num_tracks["train"] = tracks - num_tracks["valid"] - num_tracks["test"]
            # Generate overall dataset in DATAX, DATAY.
            # We use the tracks as temporal input sequences.
            # All samples in a track (30) are stored in the feature dimension.
            # Hence, the number of samples = number of tracks.
            DATAX = np.empty((tracks, self.res, self.res, 30 * 3))
            DATAY = np.empty((tracks,), dtype="uint8")
            for i in range(len(images)):
                images[i] = images[i].astype('float32')
                images[i] /= 255.0
            # Resize all images to same (res, res) size.
            for i in range(tracks):
                for f in range(30):
                    img_square = scipy.misc.imresize(images[f + i * 30], size=(self.res, self.res))
                    DATAX[i,:,:,f * 3:(f + 1) * 3] = img_square[:,:,:] / 255.0
                DATAY[i] = labels[i * 30]
            # Permute dataset (tracks, not single images) before split.
            # This is necessary, because tracks are ordered by classes.
            p = np.random.permutation(range(DATAX.shape[0]))
            DATAX = DATAX[p]
            DATAY = DATAY[p]
            # Split dataset into train / valid / test.
            self.DATAX["test"] = DATAX[0:num_tracks["test"],:,:,:]
            self.DATAY["test"] = to_categorical(DATAY[0:num_tracks["test"]], self.classes)
            self.DATAX["valid"] = DATAX[num_tracks["test"]:num_tracks["test"] + num_tracks["valid"],:,:,:]
            self.DATAY["valid"] = to_categorical(DATAY[num_tracks["test"]:num_tracks["test"] + num_tracks["valid"]], self.classes)
            self.DATAX["train"] = DATAX[num_tracks["test"] + num_tracks["valid"]:-1,:,:,:]
            self.DATAY["train"] = to_categorical(DATAY[num_tracks["test"] + num_tracks["valid"]:-1], self.classes)



        # Print some general information about dataset.
        print('Info for dataset: ' + self.dataset)
        print('    x/y_train shape: ' + str(self.DATAX["train"].shape) + "   " + str(self.DATAY["train"].shape))
        print('    valid samples  : ', self.DATAX["valid"].shape[0])
        print('    test samples   : ', self.DATAX["test"].shape[0])



    def shuffle(self):
        """Shuffle dataset and redraw 'train', 'valid', and 'test' data.
        """
        sets = ["train", "valid", "test"]
        # Get number of samples.
        samples = {}
        for t in sets:
            samples[t] = self.DATAX[t].shape[0]
        # Aggregate datasets.
        DATAX = np.concatenate([self.DATAX[t] for t in sets], axis=0)
        DATAY = np.concatenate([self.DATAY[t] for t in sets], axis=0)
        # Permutate data.
        p = np.random.permutation(range(DATAX.shape[0]))
        DATAX = DATAX[p]
        DATAY = DATAY[p]
        # Split dataset into train / valid / test.
        self.DATAX["test"] = DATAX[0:samples["test"],:,:,:]
        self.DATAY["test"] = DATAY[0:samples["test"]]
        self.DATAX["valid"] = DATAX[samples["test"]:samples["test"] + samples["valid"],:,:,:]
        self.DATAY["valid"] = DATAY[samples["test"]:samples["test"] + samples["valid"]]
        self.DATAX["train"] = DATAX[samples["test"] + samples["valid"]:-1,:,:,:]
        self.DATAY["train"] = DATAY[samples["test"] + samples["valid"]:-1]
