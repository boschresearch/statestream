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
from skimage.transform import resize

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
    return {"out": ["tcf10_image", "tcf10_trafo"], 
            "in": []
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

    # Return layout.
    return shm_layout




class ProcessIf_transform_cifar10(ProcessIf):
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

        self.mode = self.p.get("mode", 0)
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
            self.dataset[t]["tcf10_image"] = np.zeros([self.samples[t], 3, 32, 32], dtype=np.float32)
        for b in range(4):
            if sys.version[0] == "2":
                data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                      + "/data_batch_" + str(b+1), "rb"))
            elif sys.version[0] == "3":
                data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                      + "/data_batch_" + str(b+1), "rb"), 
                                      encoding="latin1")
            image_data = np.swapaxes(np.reshape(data["data"], [10000, 3, 32, 32]), 2, 3)
            self.dataset['train']["tcf10_image"][b*10000:(b+1)*10000,:,:,:] \
                = image_data[:,:,:,:] / 256.0

        # Load validation dataset.
        if sys.version[0] == "2":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/data_batch_5", "rb"))
        elif sys.version[0] == "3":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/data_batch_5", "rb"), 
                                  encoding="latin1")
        image_data = np.swapaxes(np.reshape(data["data"], [self.samples['valid'], 3, 32, 32]), 2, 3)
        self.dataset['valid']["tcf10_image"][:,:,:,:] \
            = image_data[:,:,:,:] / 256.0

        # Load test dataset.
        if sys.version[0] == "2":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/test_batch", "rb"))
        elif sys.version[0] == "3":
            data = pckl.load(open(self.net["interfaces"][self.name]["source_path"] \
                                  + "/test_batch", "rb"), 
                                  encoding="latin1")
        image_data = np.swapaxes(np.reshape(data["data"], [self.samples['test'], 3, 32, 32]), 2, 3)
        self.dataset['test']["tcf10_image"][:,:,:,:] \
            = image_data[:,:,:,:] / 256.0

        for t in ['train', 'valid', 'test']:
            if t + ' samples' in self.p:
                self.samples[t] = min(self.p[t + ' samples'], self.samples[t])

        self.transforms = ['trans', 'scale', 'rot', 'random']

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_duration = []
        self.current_elapsed = []
        self.current_sample = []
        self.current_image = []
        self.current_transform = []
        for a in range(self.net["agents"]):
            self.current_duration += [0]
            self.current_elapsed += [0]
            self.current_sample += [0]
            self.current_transform += ['trans']
            self.current_image.append(np.zeros(self.image_shape, dtype=np.float32))
        for a in range(self.net["agents"]):
            self.draw_new_sample(a)

        for t in ['train', 'valid', 'test']:
            self.mode_shuffle[t] = np.random.permutation(self.samples[t])
            self.mode_current[t] = 0



    def draw_new_sample(self, a):
        """Draw a new sample.
        """
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
        # Get image from dataset considering mode.
        if self.mode in [0, 1, 2]:
            if a > self.split:
                img = self.dataset['train']["tcf10_image"][self.current_sample[a],:,:,:]
            else:
                img = self.dataset['valid']["tcf10_image"][self.current_sample[a],:,:,:]
        elif self.mode == 3:
            img = self.dataset['test']["tcf10_image"][self.current_sample[a],:,:,:]
        self.current_image[a] = np.copy(img)
        # Draw new transformation.
        self.current_transform[a] = np.random.randint(0, len(self.transforms))



    def update_frame_writeout(self):
        """Method to update the experimental state of the cifar10 interface.
        """
        # Update sampling mode.
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
                self.draw_new_sample(a)

            # Update internal data (= variables).
            # -----------------------------------------------------------------
            # Update image.
            if "tcf10_image" in self.p["out"]:
                # Apply fading factor.
                if self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                    ff = float(self.current_elapsed[a]) / float(self.dat["parameter"]["fading"])
                elif self.current_duration[a] - self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                    ff = float(self.current_duration[a] - self.current_elapsed[a]) \
                         / float(self.dat["parameter"]["fading"])
                else:
                    ff = 1.0
                # Apply augmentation.
                self.dat["variables"]["tcf10_image"][a,:,:,:] = ff * self.current_image[a]
            if "tcf10_transform" in self.p["out"]:
                if self.transforms[self.current_transform[a]] == "trans":
                    self.dat["variables"]["tcf10_transform"][a,0,:,:] = 0.1
                    self.dat["variables"]["tcf10_transform"][a,1,:,:] = 0.05
                elif self.transforms[self.current_transform[a]] == "scale":
                    center = np.array([self.image_shape[1] / 2.0, 
                                       self.image_shape[2] / 2.0], dtype=np.float32)
                    for dx in range(self.image_shape[1]):
                        for dy in range(self.image_shape[2]):
                            v = np.array([center[0] - float(dx), center[1] - float(dy)], dtype=np.float32)
                            v[0] = - v[0] / self.image_shape[1]
                            v[1] = - v[1] / self.image_shape[2]
                            self.dat["variables"]["tcf10_transform"][a,0,dx,dy] \
                                = 0.3 * v[1]
                            self.dat["variables"]["tcf10_transform"][a,1,dx,dy] \
                                = 0.3 * v[0]
                elif self.transforms[self.current_transform[a]] == "rot":
                    self.dat["variables"]["tcf10_transform"][a,0,:,:] = 0.2
                    self.dat["variables"]["tcf10_transform"][a,1,:,:] = 0.2
                elif self.transforms[self.current_transform[a]] == "random":
                    for dx in range(self.image_shape[1]):
                        for dy in range(self.image_shape[2]):
                            v = 0.1 * (np.random.rand(2) - 0.5)
                            self.dat["variables"]["tcf10_transform"][a,0,dx,dy] \
                                = v[0]
                            self.dat["variables"]["tcf10_transform"][a,1,dx,dy] \
                                = v[1]


