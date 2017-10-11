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



import sys
import numpy as np

try:
    import pickle as pckl
except:
    try:
        import cPickle as pckl
    except:
        pckl = None

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
    return {"out": ["cf10_image", "cf10_label"], 
            "in": ["cf10_pred"]
           }



def if_init(net, name, dat_name, dat_layout, mode=None):
    """Return value for interface parameter / variable.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string identifier for the interface.
    ident : int
        The unique process id for the process of this interface.
    param : dict
        Dictionary of core parameters.
    mn : MetaNetwork
        deprecated
    """
    # Default return is None.
    dat_value = None

    # Return initialized value.
    return dat_value



def if_shm_layout(name, net, param):
    """Return shared memory layout for civar10 interface.

    Parameters
    ----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string identifier for the interface.
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
        = ShmL("np", (), np.int32, p.get("mode", 0), 1, None)
    # Add durations as numpy parameters.
    shm_layout["parameter"]["min_duration"] = ShmL("np", (), np.int32, 12, 1, None)
    shm_layout["parameter"]["max_duration"] = ShmL("np", (), np.int32, 16, 1, None)
    shm_layout["parameter"]["fading"] = ShmL("np", (), np.int32, 4, 1, None)

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
        shm_layout["variables"][o] = ShmL("np", 
                                          np_state_shape(net, tmp_target),
                                          np.float32,
                                          0)
    # We also add a simple trigger variable for new stimulus onsets.
    shm_layout["variables"]["_trigger_"] \
        = ShmL("np", [net["agents"],], np.float32, 0)
    shm_layout["variables"]["_epoch_trigger_"] \
        = ShmL("np", [3,], np.float32, 0)
    # If a prediction is given as input we compute a confusion matrix.
    if "cf10_pred" in p["in"]:

        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "cf10_pred"
        if "remap" in p:
            if "cf10_pred" in p["remap"]:
                tmp_target = p["remap"]["cf10_pred"]
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




class ProcessIf_cifar10(ProcessIf):
    """Interface class providing basic cifar10 data.

        To use this interface, please download the python version of the cifar-10 
        dataset. Extract the cifar-10 dataset and specify the extracted folder 
        in the st_graph file (under 'source_path') that uses the cifar-10 
        dataset.

        Interface parameters:
        ---------------------
        source_path : str
            This has to be the global path to the cifar10 dataset path.
        min_duration : int
            The minimum duration a mnist number will be presented in frames.
            The actual duration will be drawn uniformly between min_duration
            and max_duration.
        max_duration : int
            The maximum duration a mnist number will be presented in frames.
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
        cf10_pred : np.array, shape [agents, 10, 1, 1]
            The cifar10 interface provides a delayed confusion matrix as 
            performance measure. To compute this a classification result is 
            needed to be compared with the ground-truth.

        Outputs:
        --------
        cf10_image : np.array, shape [agents, 3, 32, 32]
            The grey-scale mnist images.
        cf10_label : np.array, shape [agents, 10, 1, 1]
            The one-hot encoded ground-truth label for the current image.
    """
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class.
        ProcessIf.__init__(self, name, ident, net, param)



    def initialize(self):
        """Method to initialize the cifar-10 (load) interface.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        self.max_duration = self.p.get("max_duration", 16)
        self.min_duration = self.p.get("min_duration", 12)
        self.min_duration = self.p.get("fading", 4)
            
        self.mode = None
        self.mode_shuffle = {}
        self.mode_current = {}

        # Load cifar10 dataset.
        # ---------------------------------------------------------------------
        self.image_shape = np.array([3, 32, 32]).astype(np.int32)
        self.no_classes = 10

        self.samples = {}
        self.samples['train'] = 40000
        self.samples['valid'] = 10000
        self.samples['test'] = 10000
        # Structure holding all cifar10 images.
        self.dataset = {'train': {}, 'valid': {}, 'test': {}}

        # Load training data.
        for t in self.dataset:
            self.dataset[t]["cf10_image"] = np.zeros([self.samples[t], 3, 32, 32], dtype=np.float32)
            self.dataset[t]["cf10_label"] = np.zeros([self.samples[t], 1], dtype=np.float32)
        for b in range(4):
            if sys.version[0] == "2":
                data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                      + "/data_batch_" + str(b+1), "rb"))
            elif sys.version[0] == "3":
                data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                      + "/data_batch_" + str(b+1), "rb"), 
                                      encoding="latin1")
            image_data = np.swapaxes(np.reshape(data["data"], [10000, 3, 32, 32]), 2, 3)
            self.dataset['train']["cf10_image"][b*10000:(b+1)*10000,:,:,:] \
                = image_data[:,:,:,:] / 256.0
            # get labels
            self.dataset['train']["cf10_label"][b*10000:(b+1)*10000,0] = np.array(data["labels"])[:]

        # Load validation dataset.
        if sys.version[0] == "2":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/data_batch_5", "rb"))
        elif sys.version[0] == "3":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/data_batch_5", "rb"), 
                                  encoding="latin1")
        image_data = np.swapaxes(np.reshape(data["data"], [self.samples['valid'], 3, 32, 32]), 2, 3)
        self.dataset['valid']["cf10_image"][:,:,:,:] \
            = image_data[:,:,:,:] / 256.0
        self.dataset['valid']["cf10_label"][:,0] = np.array(data["labels"])[:]

        # Load test dataset.
        if sys.version[0] == "2":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/test_batch", "rb"))
        elif sys.version[0] == "3":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/test_batch", "rb"), 
                                  encoding="latin1")
        image_data = np.swapaxes(np.reshape(data["data"], [self.samples['test'], 3, 32, 32]), 2, 3)
        self.dataset['test']["cf10_image"][:,:,:,:] \
            = image_data[:,:,:,:] / 256.0
        self.dataset['test']["cf10_label"][:,0] = np.array(data["labels"])[:]

        for t in ['train', 'valid', 'test']:
            if t + ' samples' in self.p:
                self.samples[t] = min(self.p[t + ' samples'], self.samples[t])

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_duration = []
        self.current_elapsed = []
        self.current_sample = []
        for a in range(self.net["agents"]):
            self.current_duration += [np.random.randint(self.min_duration, 
                                                        self.max_duration + 1)]
            self.current_elapsed += [0]
            if a > self.split:
                self.current_sample += [np.random.randint(0, self.samples['train'])]
            else:
                self.current_sample += [np.random.randint(0, self.samples['valid'])]
        for t in ['train', 'valid', 'test']:
            self.mode_shuffle[t] = np.random.permutation(self.samples[t])
            self.mode_current[t] = 0

        # Instantiate temporal confusion matrix.
        if "cf10_pred" in self.p["in"]:
            self.TCM_train = TemporalConfusionMatrix(self.net, self.name, "cf10_pred")
            self.TCM_valid = TemporalConfusionMatrix(self.net, self.name, "cf10_pred")
            for a in range(self.net["agents"]):
                self.TCM_train.trigger_history[a] = [-1, -1]
                self.TCM_valid.trigger_history[a] = [-1, -1]

            # For cummulative performances (over entire epoch, e.g. valid and test).
            self.cumm_conf_mat = np.zeros(self.TCM_valid.conf_mat.shape, dtype=np.float32)


    def update_frame_writeout(self):
        """Method to update the experimental state of the cifar10 interface.
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
        if "cf10_label" in self.p["out"]:
            self.dat["variables"]["cf10_label"] *= 0.0
        self.dat["variables"]["_trigger_"] *= 0
        self.dat["variables"]["_epoch_trigger_"] *= 0
        # Check for split update.
        if self.split != self.old_split:
            self.old_split = np.copy(self.split)
            for i,I in enumerate(self.current_duration):
                self.current_elapsed[i] = I
        # Update experimental state for all agents.
        for a in range(self.net["agents"]):
            # Update current experimental state.
            # -----------------------------------------------------------------
            self.current_elapsed[a] += 1
            # Check for end of frame.
            if self.current_elapsed[a] > self.current_duration[a]:
                # Set back elapsed.
                self.current_elapsed[a] = 0
                self.current_duration[a] \
                    = np.random.randint(self.dat["parameter"]["min_duration"],
                                        self.dat["parameter"]["max_duration"] + 1)
                # Dependent on presentations of this next frame, 
                # init its parameters.
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

                # Set trigger for new stimulus.
                self.dat["variables"]["_trigger_"][a] = 1

            # Update internal data (= variables).
            # -----------------------------------------------------------------
            # Update label.
            if self.mode in [0, 1, 2]:
                if a > self.split:
                    current_label = int(self.dataset['train']["cf10_label"][self.current_sample[a],0])
                else:
                    current_label = int(self.dataset['valid']["cf10_label"][self.current_sample[a],0])
            elif self.mode == 3:
                current_label = int(self.dataset['test']["cf10_label"][self.current_sample[a],0])
            if "cf10_label" in self.p["out"]:
                self.dat["variables"]["cf10_label"][a,current_label,0,0] = 1.0
            # Update image.
            if "cf10_image" in self.p["out"]:
                # Determine fading factor.
                if self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                    ff = float(self.current_elapsed[a]) / float(self.dat["parameter"]["fading"])
                    if self.mode in [0, 1, 2]:
                        if a > self.split:
                            self.dat["variables"]["cf10_image"][a,:,:,:] \
                                = ff * self.dataset['train']["cf10_image"][self.current_sample[a],:,:,:]
                        else:
                            self.dat["variables"]["cf10_image"][a,:,:,:] \
                                = ff * self.dataset['valid']["cf10_image"][self.current_sample[a],:,:,:]
                    elif self.mode == 3:
                        self.dat["variables"]["cf10_image"][a,:,:,:] \
                            = ff * self.dataset['test']["cf10_image"][self.current_sample[a],:,:,:]
                elif self.current_duration[a] - self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                    ff = float(self.current_duration[a] - self.current_elapsed[a]) \
                         / float(self.dat["parameter"]["fading"])
                    if self.mode in [0, 1, 2]:
                        if a > self.split:
                            self.dat["variables"]["cf10_image"][a,:,:,:] \
                                = ff * self.dataset['train']["cf10_image"][self.current_sample[a],:,:,:]
                        else:
                            self.dat["variables"]["cf10_image"][a,:,:,:] \
                                = ff * self.dataset['valid']["cf10_image"][self.current_sample[a],:,:,:]
                    elif self.mode == 3:
                        self.dat["variables"]["cf10_image"][a,:,:,:] \
                            = ff * self.dataset['test']["cf10_image"][self.current_sample[a],:,:,:]
                else:
                    if self.mode in [0, 1, 2]:
                        if a > self.split:
                            self.dat["variables"]["cf10_image"][a,:,:,:] \
                                = self.dataset['train']["cf10_image"][self.current_sample[a],:,:,:]
                        else:
                            self.dat["variables"]["cf10_image"][a,:,:,:] \
                                = self.dataset['valid']["cf10_image"][self.current_sample[a],:,:,:]
                    elif self.mode == 3:
                        self.dat["variables"]["cf10_image"][a,:,:,:] \
                            = self.dataset['test']["cf10_image"][self.current_sample[a],:,:,:]

        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "cf10_pred" in self.p["in"] \
                and "cf10_image" in self.p["out"] \
                and "cf10_label" in self.p["out"]:
            # Call update function.
            current_label = [0 for a in range(self.net["agents"])]
            for a in range(self.net["agents"]):
                current_label[a] = np.argmax(self.dat["variables"]["cf10_label"][a,:,0,0])
            mask = np.ones([self.net['agents'],])
            mask[0:self.split] = 0
            self.TCM_train.update_history(self.inputs["cf10_pred"],
                                          int(self.frame_cntr),
                                          current_label,
                                          self.current_elapsed,
                                          self.dat["variables"]["_trigger_"],
                                          mask)
            self.TCM_valid.update_history(self.inputs["cf10_pred"],
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
