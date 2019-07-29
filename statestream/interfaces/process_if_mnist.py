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



import sys
import numpy as np
import gzip
import copy

if sys.version[0] == "2":
    import cPickle as pckl
elif sys.version[0] == "3":
    import pickle as pckl

from statestream.interfaces.process_if import ProcessIf
from statestream.interfaces.utils.temporal_confusion_matrix import TemporalConfusionMatrix
from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.meta.neuron_pool import np_state_shape


def if_interfaces():
    """Returns the specific interfaces as strings of the interface.
    
    Parameters:
    -----------
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string name of this interface.
    """
    # Specify sub-interfaces.
    return {"in": ["mnist_image", "mnist_label"], 
            "out": ["mnist_pred"]
           }



def if_init(net, name, dat_name, dat_layout, mode=None):
    """Return value for interface parameter / variable.

    Parameters:
    -----------
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string name of this interface.
    dat_name : str
        The unique name for the parameter to initialize.
    dat_layout : SharedMemoryLayout
        The layout (see SharedMemoryLayout) of the parameter to be initialized.
    mode : None, value (float, int), str
        The mode determines how a parameter has to be initialized (e.g. 'xavier' or 0.0)
    """
    # Default return is None.
    dat_value = None

    # Return initialized value.
    return dat_value



def if_shm_layout(name, net, param):
    """Return shared memory layout for mnist interface.

    Due to this layout shared memory will be allocated for this interface.

    Parameters:
    -----------
    name : str
        The unique interface name.
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    """
    # Get interface dictionary.
    p = net["interfaces"][name]
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}

    # Add parameter.
    # -------------------------------------------------------------------------
    # Add mode parameter.
    shm_layout["parameter"]["mode"] \
        = ShmL("np", (), np.int32, p.get("mode", 0))
    # Add durations as numpy parameters.
    shm_layout["parameter"]["min_duration"] \
        = ShmL("np", (), np.int32, p.get("min_duration", 16), 1, None)
    shm_layout["parameter"]["max_duration"] \
        = ShmL("np", (), np.int32, p.get("max_duration", 24), 1, None)
    shm_layout["parameter"]["fading"] \
        = ShmL("np", (), np.int32, p.get("fading", 4), 0, None)


    # Add variables.
    # -------------------------------------------------------------------------
    # Add all outputs as variables.
    for o in p["out"]:
        tmp_target = o
        # Consider remapping.
        if "remap" in p:
            if o in p["remap"]:
                tmp_target = p["remap"][o]
        # Set layout.
        shm_layout["variables"][o] \
            = ShmL("np", np_state_shape(net, tmp_target), np.float32, 0)
    # We also add a simple trigger variable for new stimulus onsets.
    shm_layout["variables"]["_trigger_"] \
        = ShmL("np", [net["agents"],], np.float32, 0)
    shm_layout["variables"]["_epoch_trigger_"] \
        = ShmL("np", [3,], np.float32, 0)
    # If a prediction is given as input we compute a confusion matrix.
    if "mnist_pred" in p["in"]:
        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "mnist_pred"
        if "remap" in p:
            if "mnist_pred" in p["remap"]:
                tmp_target = p["remap"]["mnist_pred"]
        no_classes = np_state_shape(net, tmp_target)[1]
        # Create layout for conf-mat.
        shm_layout["variables"]["conf_mat_train"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)
        shm_layout["variables"]["conf_mat_valid"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)
        shm_layout["variables"]["acc_train"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    1,
                    1],
                   np.float32,
                   0)
        shm_layout["variables"]["acc_valid"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    1,
                    1],
                   np.float32,
                   0)

    # Return layout.
    return shm_layout

    

class ProcessIf_mnist(ProcessIf):
    """Interface class providing basic mnist data.

        To use this interface please download the pickled version of the mnist 
        dataset: mnist.pkl.gz The absolute path to this file must be given 
        (under 'source_file') in the st_graph file that want to use mnist.

        Parameters:
        -----------
        name : str
            Unique interface identifier.
        ident : int
            Unique id for the interface's process.
        net : dict
            The network dictionary containing all nps, sps, plasts, ifs.
        param : dict
            Dictionary of core parameters.

        Interface parameters
        --------------------
        source_file : str
            This has to be the global path to the mnist dataset file (mnist.pkl.gz).
        min_duration : int
            The minimum duration a mnist number will be presented in frames.
            The actual duration will be drawn uniformly between min_duration
            and max_duration.
        max_duration : int
            The maximum duration a mnist number will be presented in frames.
        fading : int
            The number of neuronal frames to fade in/out a mnist number.
        conf-mat window : int
            As a performance measure for a potential classifier a confusion
            matrix is computed over a certain window of delays. This parameter
            specifies the window size, e.g. confusion matrices will be computed
            for all delays up to this window size. Note: This should be larger
            than the shortest path in the network from input to classification.
        conf-mat mean over : int
            Confusion matrices will (over the conf-mat window) will be computed
            as the mean over the last 'conf-mat mean over' frames for temporal
            smoothing and a better approximation.

        Inputs:
        -------
        mnist_pred : np.array, shape [agents, 10, 1, 1]
            The mnist interface provides a delayed confusion matrix as performance
            measure. To compute this a classification result is needed to be compared
            with the ground-truth.

        Outputs:
        --------
        mnist_image : np.array, shape [agents, 1, 28, 28]
            The grey-scale mnist images.
        mnist_label : np.array, shape [agents, 10, *, *]
            The one-hot encoded ground-truth label for the current image. Spatial
            dimensions will be taken from the associated neuron-pool.
    """ 
    def __init__(self, name, ident, metanet, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, metanet, param)



    def initialize(self):
        """Method to initialize (load) the mnist interface class.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        self.dat["parameter"]["min_duration"] = self.p.get("min_duration", 16)
        self.dat["parameter"]["max_duration"] = self.p.get("max_duration", 24)
        self.dat["parameter"]["fading"] = self.p.get("fading", 1)

        # Load mnist dataset.
        # ---------------------------------------------------------------------
        self.image_shape = np.array([28, 28]).astype(np.int32)
        self.no_classes = 10
        # Load dataset.
        f = gzip.open(self.net["interfaces"][self.name]["source_file"], "rb")
        if sys.version[0] == "2":
            train_set, valid_set, test_set = pckl.load(f)
        elif sys.version[0] == "3":
            train_set, valid_set, test_set = pckl.load(f, encoding="latin1")
        f.close()

        self.mode = None
        self.mode_shuffle = {}
        self.mode_current = {}

        self.samples = {}
        self.samples['train'] = train_set[1].shape[0]
        self.samples['valid'] = valid_set[1].shape[0]
        self.samples['test'] = test_set[1].shape[0]
        for t in ['train', 'valid', 'test']:
            if t + ' samples' in self.p:
                self.samples[t] = min(self.p[t + ' samples'], self.samples[t])
        self.dataset = {}
        self.dataset['train'] = {}
        self.dataset['valid'] = {}
        self.dataset['test'] = {}
        self.dataset['train']["mnist_image"] \
            = np.swapaxes(np.reshape(train_set[0], 
                                     [train_set[1].shape[0],
                                      self.image_shape[0], -1]), 
                          1, 2).astype(np.float32)[:,np.newaxis,:,:]
        self.dataset['valid']["mnist_image"] \
            = np.swapaxes(np.reshape(valid_set[0], 
                                     [valid_set[1].shape[0],
                                      self.image_shape[0], -1]), 
                          1, 2).astype(np.float32)[:,np.newaxis,:,:]
        self.dataset['test']["mnist_image"] \
            = np.swapaxes(np.reshape(test_set[0], 
                                     [test_set[1].shape[0],
                                      self.image_shape[0], -1]), 
                          1, 2).astype(np.float32)[:,np.newaxis,:,:]
        self.dataset['train']["mnist_label"] \
            = np.asarray(train_set[1]).astype(np.float32)[:,np.newaxis]
        self.dataset['valid']["mnist_label"] \
            = np.asarray(valid_set[1]).astype(np.float32)[:,np.newaxis]
        self.dataset['test']["mnist_label"] \
            = np.asarray(test_set[1]).astype(np.float32)[:,np.newaxis]

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_duration = []
        self.current_elapsed = []
        self.current_sample = []
        self.current_col_bg = []
        self.current_col_fg = []
        for a in range(self.net["agents"]):
            self.current_duration \
                += [np.random.randint(self.dat["parameter"]["min_duration"],
                                      self.dat["parameter"]["max_duration"] + 1)]
            self.current_elapsed += [0]
            if a > self.split:
                self.current_sample += [np.random.randint(0, self.samples['train'] - 1)]
            else:
                self.current_sample += [np.random.randint(0, self.samples['valid'] - 1)]
            # Draw initial colors.
            self.current_col_bg += [np.random.rand(3)]
            self.current_col_fg += [np.random.rand(3)]
        for t in ['train', 'valid', 'test']:
            self.mode_shuffle[t] = np.random.permutation(self.samples[t])
            self.mode_current[t] = 0

        # Instantiate temporal confusion matrix.
        if "mnist_pred" in self.p["in"]:
            self.TCM_train = TemporalConfusionMatrix(self.net, self.name, "mnist_pred")
            self.TCM_valid = TemporalConfusionMatrix(self.net, self.name, "mnist_pred")
            for a in range(self.net["agents"]):
                self.TCM_train.trigger_history[a] = [-1, -1]
                self.TCM_valid.trigger_history[a] = [-1, -1]

            # For cummulative performances (over entire epoch, e.g. valid and test).
            self.cumm_conf_mat = np.zeros(self.TCM_valid.conf_mat.shape, dtype=np.float32)



    def update_frame_writeout(self):
        """Method to update the experimental state of the mnist interface.
        """
        # Update sampling mode.
        if self.mode is None:
            self.mode = self.dat["parameter"]["mode"]
        if self.mode != self.dat["parameter"]["mode"]:
            self.mode = self.dat["parameter"]["mode"]
            self.mode_shuffle = {}
            self.mode_current = {}
            if self.mode == 2:
                for t in ['train', 'valid']:
                    self.mode_shuffle[t] = np.random.permutation(self.samples[t])
                    self.mode_current[t] = 0
            elif self.mode == 3:
                self.mode_shuffle['test'] = np.random.permutation(self.samples['test'])
                self.mode_current['test'] = 0
            self.cumm_conf_mat = np.zeros(self.TCM_valid.conf_mat.shape, dtype=np.float32)
            # Force update of all samples / agents.
            for a in range(self.net["agents"]):
                self.current_elapsed[a] = self.dat["parameter"]["max_duration"]
        # Clean labels and trigger.
        if "mnist_label" in self.p["out"]:
            self.dat["variables"]["mnist_label"] *= 0.0
        if "mnist_pred" in self.p["in"]:
            self.dat["variables"]["_trigger_"] *= 0.0
        self.dat["variables"]["_epoch_trigger_"] *= 0
        # Check for split update.
        if self.split != self.old_split:
            self.old_split = np.copy(self.split)
            if self.mode == 2:
                for t in ['train', 'valid']:
                    self.mode_shuffle[t] = np.random.permutation(self.samples[t])
                    self.mode_current[t] = 0
            elif self.mode == 3:
                self.mode_shuffle['test'] = np.random.permutation(self.samples['test'])
                self.mode_current['test'] = 0
            self.cumm_conf_mat = np.zeros(self.TCM_valid.conf_mat.shape, dtype=np.float32)
            # Force update of all samples / agents.
            for a in range(self.net["agents"]):
                self.current_elapsed[a] = self.dat["parameter"]["max_duration"]
        # Update experimental state for all agents.
        for a in range(self.net["agents"]):
            # Update current experimental state.
            # -----------------------------------------------------------------
            self.current_elapsed[a] += 1
            # Check for end of frame.
            if self.current_elapsed[a] >= self.current_duration[a]:
                # Set back elapsed.
                self.current_elapsed[a] = 0
                # Draw new duration.
                self.current_duration[a] \
                    = np.random.randint(self.dat["parameter"]["min_duration"],
                                        self.dat["parameter"]["max_duration"] + 1)
                # Dependent on presentations of this next frame, init its parameters.
                if self.mode in [0, 1]:
                    if a > self.split:
                        self.current_sample[a] = np.random.randint(0, self.samples['train'] - 1)
                    else:
                        self.current_sample[a] = np.random.randint(0, self.samples['valid'] - 1)
                elif self.mode == 2:
                    if a > self.split:
                        self.current_sample[a] = self.mode_shuffle['train'][self.mode_current['train']]
                        self.mode_current['train'] += 1
                        if self.mode_current['train'] >= self.samples['train']:
                            self.mode_shuffle['train'] = np.random.permutation(self.samples['train'])
                            self.mode_current['train'] = 0
                            self.dat["variables"]["_epoch_trigger_"][0] = 1
                    else:
                        self.current_sample[a] = self.mode_shuffle['valid'][self.mode_current['valid']]
                        self.mode_current['valid'] += 1
                        if self.mode_current['valid'] >= self.samples['valid']:
                            self.mode_shuffle['valid'] = np.random.permutation(self.samples['valid'])
                            self.cumm_conf_mat *= 0
                            self.mode_current['valid'] = 0
                            self.dat["variables"]["_epoch_trigger_"][1] = 1
                elif self.mode == 3:
                    self.current_sample[a] = self.mode_shuffle['test'][self.mode_current['test']]
                    self.mode_current['test'] += 1
                    if self.mode_current['test'] >= self.samples['test']:
                        self.mode_shuffle['test'] = np.random.permutation(self.samples['test'])
                        self.cumm_conf_mat *= 0
                        self.mode_current['test'] = 0
                        self.dat["variables"]["_epoch_trigger_"][2] = 1
                # Draw new colors.
                self.current_col_bg[a] = np.random.rand(3)
                self.current_col_fg[a] = np.random.rand(3)
                # Set trigger for new stimulus.
                self.dat["variables"]["_trigger_"][a] = 1

            # Update internal data (= variables).
            # -----------------------------------------------------------------
            # Update label.
            if self.mode in [0, 1, 2]:
                if a > self.split:
                    current_label = int(self.dataset['train']["mnist_label"][self.current_sample[a],0])
                else:
                    current_label = int(self.dataset['valid']["mnist_label"][self.current_sample[a],0])
            elif self.mode == 3:
                current_label = int(self.dataset['test']["mnist_label"][self.current_sample[a],0])

            if "mnist_label" in self.p["out"]:
                self.dat["variables"]["mnist_label"][a,current_label,:,:] = 1.0

            # Update image.
            if "mnist_image" in self.p["out"]:
                if self.dat["variables"]["mnist_image"].shape[1] == 1:
                    # Determine fading factor.
                    if self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_elapsed[a]) / float(self.dat["parameter"]["fading"])
                    elif self.current_duration[a] - self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_duration[a] - self.current_elapsed[a]) \
                             / float(self.dat["parameter"]["fading"])
                    else:
                        ff = 1
                    if self.mode in [0, 1, 2]:
                        if a > self.split:
                            self.dat["variables"]["mnist_image"][a,0,:,:] \
                                = ff * self.dataset['train']["mnist_image"][self.current_sample[a],:,:]
                        else:
                            self.dat["variables"]["mnist_image"][a,0,:,:] \
                                = ff * self.dataset['valid']["mnist_image"][self.current_sample[a],:,:]
                    elif self.mode == 3:
                        self.dat["variables"]["mnist_image"][a,0,:,:] \
                            = ff * self.dataset['test']["mnist_image"][self.current_sample[a],:,:]
                elif self.dat["variables"]["mnist_image"].shape[1] == 3:
                    # Determine fading factor.
                    if self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_elapsed[a]) / float(self.dat["parameter"]["fading"])
                    elif self.current_duration[a] - self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_duration[a] - self.current_elapsed[a]) \
                             / float(self.dat["parameter"]["fading"])
                    else:
                        ff = 1
                    if self.mode in [0, 1, 2]:
                        if a > self.split:
                            for c in range(3):
                                self.dat["variables"]["mnist_image"][a,c,:,:] \
                                    = ff * (self.current_col_fg[a][c] * self.dataset['train']["mnist_image"][self.current_sample[a],:,:] \
                                            + self.current_col_bg[a][c] * (1 - self.dataset['train']["mnist_image"][self.current_sample[a],:,:]))
                        else:
                            for c in range(3):
                                self.dat["variables"]["mnist_image"][a,c,:,:] \
                                    = ff * (self.current_col_fg[a][c] * self.dataset['valid']["mnist_image"][self.current_sample[a],:,:] \
                                            + self.current_col_bg[a][c] * (1 - self.dataset['valid']["mnist_image"][self.current_sample[a],:,:]))
                    elif self.mode == 3:
                        for c in range(3):
                            self.dat["variables"]["mnist_image"][a,c,:,:] \
                                = ff * (self.current_col_fg[a][c] * self.dataset['test']["mnist_image"][self.current_sample[a],:,:] \
                                        + self.current_col_bg[a][c] * (1 - self.dataset['test']["mnist_image"][self.current_sample[a],:,:]))

        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "mnist_pred" in self.p["in"] \
                and "mnist_image" in self.p["out"] \
                and "mnist_label" in self.p["out"] \
                and np.sum(self.dat['variables']['_epoch_trigger_']) == 0:
            # Call update function.
            current_label = [0 for a in range(self.net["agents"])]
            for a in range(self.net["agents"]):
                if self.mode in [0, 1, 2]:
                    if a > self.split:
                        current_label[a] = int(self.dataset['train']["mnist_label"][self.current_sample[a],0])
                    else:
                        current_label[a] = int(self.dataset['valid']["mnist_label"][self.current_sample[a],0])
                elif self.mode == 3:
                    current_label[a] = int(self.dataset['test']["mnist_label"][self.current_sample[a],0])
            # Note: For mode == 3, the TCM_train hold the information for the test data.
            mask = np.ones([self.net['agents'],])
            mask[0:self.split] = 0
            self.TCM_train.update_history(self.inputs["mnist_pred"],
                                    int(self.frame_cntr),
                                    current_label,
                                    self.current_elapsed,
                                    self.dat["variables"]["_trigger_"],
                                    mask)
            self.TCM_valid.update_history(self.inputs["mnist_pred"],
                                    int(self.frame_cntr),
                                    current_label,
                                    self.current_elapsed,
                                    self.dat["variables"]["_trigger_"],
                                    1 - mask)

            if self.mode == 2:
                self.cumm_conf_mat += self.TCM_valid.conf_mat
                self.dat["variables"]["conf_mat_train"][:,:,:] \
                    = self.TCM_train.conf_mat[:,:,:]
                self.dat["variables"]["acc_train"][:,:,:] \
                    = self.TCM_train.accuracy[:,:,:]
                self.dat["variables"]["conf_mat_valid"][:,:,:] \
                    = self.cumm_conf_mat
                # Compute accuracy.
                acc = np.zeros([self.cumm_conf_mat.shape[0],], dtype=np.float32)
                for w in range(self.cumm_conf_mat.shape[0]):
                    tmp_sum = np.sum(self.cumm_conf_mat[w,:,:])
                    acc[w] = np.trace(self.cumm_conf_mat[w,:,:]) / max(tmp_sum, 1e-8)
                self.dat["variables"]["acc_valid"][:,0,0] = acc[:]
            elif self.mode == 3:
                self.cumm_conf_mat += self.TCM_train.conf_mat
                self.dat["variables"]["conf_mat_train"][:,:,:] \
                    = self.cumm_conf_mat
                # Compute accuracy.
                acc = np.zeros([self.cumm_conf_mat.shape[0],], dtype=np.float32)
                for w in range(self.cumm_conf_mat.shape[0]):
                    tmp_sum = np.sum(self.cumm_conf_mat[w,:,:])
                    acc[w] = np.trace(self.cumm_conf_mat[w,:,:]) / max(tmp_sum, 1e-8)
                self.dat["variables"]["acc_train"][:,0,0] = acc[:]
                self.dat["variables"]["conf_mat_valid"][:,:,:] = 0
                self.dat["variables"]["acc_valid"][:,:,:] = 0
            else:
                self.dat["variables"]["conf_mat_train"][:,:,:] \
                    = self.TCM_train.conf_mat[:,:,:]
                self.dat["variables"]["acc_train"][:,:,:] \
                    = self.TCM_train.accuracy[:,:,:]
                self.dat["variables"]["conf_mat_valid"][:,:,:] \
                    = self.TCM_valid.conf_mat[:,:,:]
                self.dat["variables"]["acc_valid"][:,:,:] \
                    = self.TCM_valid.accuracy[:,:,:]


