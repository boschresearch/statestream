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
    return {"out": ["seg_mnist_image", 
                    "seg_mnist_label",
                    "seg_mnist_mask"],
            "in": ["seg_mnist_pred"]
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
    """Return shared memory layout for mnist segmentation interface.

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
        = ShmL("np", (), np.int32, p.get("mode", 0), 1, None)
    # Add durations as numpy parameters.
    shm_layout["parameter"]["min_duration"] \
        = ShmL("np", (), np.int32, p.get("min_duration", 8), 1, None)
    shm_layout["parameter"]["max_duration"] \
        = ShmL("np", (), np.int32, p.get("max_duration", 16), 1, None)
    shm_layout["parameter"]["fading"] \
        = ShmL("np", (), np.int32, p.get("fading", 4), 1, None)
    shm_layout["parameter"]["hide"] \
        = ShmL("np", (), np.float32, p.get("hide", 0.0), 0.0, 1.0)

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
    # If a prediction is given as input we compute a confusion matrix.
    if "seg_mnist_pred" in p["in"]:
        # Add parameter.

        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "seg_mnist_pred"
        if "remap" in p:
            if "seg_mnist_pred" in p["remap"]:
                tmp_target = p["remap"]["seg_mnist_pred"]
        # Note that for this interface we use an ignore map (last label map).
        # This ignore map will not be part of the confusion matrix.
        no_classes = np_state_shape(net, tmp_target)[1]
        # Create layout for conf-mat.
        shm_layout["variables"]["conf_mat_train"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)
        shm_layout["variables"]["conf_mat_test"] \
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
        shm_layout["variables"]["acc_test"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    1,
                    1],
                   np.float32,
                   0)

    # Return layout.
    return shm_layout

    

class ProcessIf_mnist_segmentation(ProcessIf):
    """Interface class providing basic mnist segmentation.

        To use this interface please download the pickled version of the mnist dataset: mnist.pkl.gz
        The absolute path to this file must be given (under 'source_file') in the st_graph file that want to use mnist.

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

        Interface parameters:
        ---------------------
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
        hide : float
            Probability with which to hide a single mnist number.
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
        seg_mnist_pred : np.array, shape [agents, 10, board_width, board_height]
            The networks current segmentation result. This is needed to comput
            e.g. the temporal confusion matrix.

        Outputs:
        --------
        seg_mnist_image : np.array, shape [agents, 1, board_width, board_height]
            The grey-scale board of mnist images to be segmented.
        seg_mnist_label : np.array, shape [agents, 10, board_width, board_height]
            Ground truth for mnist labels across board.
        seg_mnist_mask : np.array, shape [agents, 1, board_width, board_height]
            Mask defining areas to be used for loss computation.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)



    def draw_sample(self, sample):
        """Draw a new random sample.
        """
        # Reset elapsed duration.
        self.current_elapsed[sample] = 0
        # Draw new random duration.
        self.current_duration[sample] \
            = np.random.randint(self.dat["parameter"]["min_duration"],
                                self.dat["parameter"]["max_duration"] + 1)
        # Draw new mnist samples and determine if hidden.
        if sample > self.split:
            for s in range(self.N_mnist):
                self.current_sample[sample][s] \
                    = np.random.randint(0, self.train_samples - 1)
                self.current_hidden[sample][s] \
                    = np.random.rand() < self.dat["parameter"]["hide"]
        else:
            for s in range(self.N_mnist):
                self.current_sample[sample][s] \
                    = np.random.randint(0, self.test_samples - 1)
                self.current_hidden[sample][s] \
                    = np.random.rand() < self.dat["parameter"]["hide"]

    def update_sample(self, sample):
        """Update sample 'sample'.
        """
        self.current_elapsed[sample] += 1
        if self.current_elapsed[sample] > self.current_duration[sample]:
            self.draw_sample(sample)



    def initialize(self):
        """Method to initialize (load) the mnist interface class.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        self.dat["parameter"]["min_duration"] = self.p.get("min_duration", 16)
        self.dat["parameter"]["max_duration"] = self.p.get("max_duration", 24)
        self.dat["parameter"]["fading"] = self.p.get("fading", 1)
        self.dat["parameter"]["hide"] = self.p.get("hide", 0.0)

        self.border = 5

        # Get shapes of important neuron-pools.
        self.np_shape = {}
        for o in self.p["out"] + self.p["in"]:
            tmp_target = o
            # Consider remapping.
            if "remap" in self.p:
                if o in self.p["remap"]:
                    tmp_target = self.p["remap"][o]
            if o not in self.np_shape:
                self.np_shape[o] = np_state_shape(self.net, tmp_target)

        # Determine number of shown mnist numbers on board.
        self.N_mnist_x = self.np_shape["seg_mnist_image"][2] // 28
        self.N_mnist_y = self.np_shape["seg_mnist_image"][3] // 28
        self.N_mnist = self.N_mnist_x * self.N_mnist_y
        self.no_classes = 10
        self.mnist_image = np.zeros([self.net["agents"],
                                     1,
                                     self.np_shape["seg_mnist_image"][2],
                                     self.np_shape["seg_mnist_image"][3]], dtype=np.float32)
        self.mnist_label = np.zeros([self.net["agents"],
                                     self.no_classes,
                                     self.np_shape["seg_mnist_image"][2],
                                     self.np_shape["seg_mnist_image"][3]], dtype=np.float32)
        self.mnist_mask = np.zeros([self.net["agents"],
                                    1,
                                    self.np_shape["seg_mnist_image"][2],
                                    self.np_shape["seg_mnist_image"][3]], dtype=np.float32)

        # Load mnist dataset.
        # ---------------------------------------------------------------------
        # Structure holding all mnist images.
        self.train_data = {}
        self.test_data = {}
        # Load dataset.
        f = gzip.open(self.net["interfaces"][self.name]["source_file"], "rb")
        if sys.version[0] == "2":
            train_set, valid_set, test_set = pckl.load(f)
        elif sys.version[0] == "3":
            train_set, valid_set, test_set = pckl.load(f, encoding="latin1")
        f.close()

        # Combine train / valid to training dataset.
        train_dataset_x = np.concatenate([train_set[0], valid_set[0]], axis=0)
        train_dataset_y = np.concatenate([train_set[1], valid_set[1]], axis=0)

        self.train_samples = train_dataset_y.shape[0]
        self.test_samples = test_set[1].shape[0]
        self.train_data["mnist_image"] \
            = np.swapaxes(np.reshape(train_dataset_x, 
                                     [self.train_samples,
                                      28, -1]), 
                          1, 2).astype(np.float32)[:,np.newaxis,:,:]
        self.test_data["mnist_image"] \
            = np.swapaxes(np.reshape(test_set[0], 
                                     [self.test_samples,
                                      28, -1]), 
                          1, 2).astype(np.float32)[:,np.newaxis,:,:]
        self.train_data["mnist_label"] \
            = np.asarray(train_dataset_y).astype(np.float32)[:,np.newaxis]
        self.test_data["mnist_label"] \
            = np.asarray(test_set[1]).astype(np.float32)[:,np.newaxis]

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_sample = []
        self.current_hidden = []
        self.current_duration = []
        self.current_elapsed = []
        for a in range(self.net["agents"]):
            # Set to empty.
            self.current_duration.append(0)
            self.current_elapsed.append(0)
            self.current_sample.append([])
            self.current_hidden.append([])
            for s in range(self.N_mnist):
                self.current_sample[-1] += [0]
                self.current_hidden[-1] += [False]
            # Initialize randomly.
            self.draw_sample(a)

        # Instantiate temporal confusion matrix.
        if "seg_mnist_pred" in self.p["in"]:
            self.TCM_train = TemporalConfusionMatrix(self.net, self.name, "seg_mnist_pred")
            self.TCM_test = TemporalConfusionMatrix(self.net, self.name, "seg_mnist_pred")
            for a in range(self.net["agents"]):
                self.TCM_train.trigger_history[a] = [-1, None, None]
                self.TCM_test.trigger_history[a] = [-1, None, None]



    def update_frame_writeout(self):
        """Method to update the experimental state of the mnist interface.
        """
        # Tabula rasa.
        self.mnist_image = 0.0 * self.mnist_image
        self.mnist_label = 0.0 * self.mnist_label
        self.mnist_mask = 0.0 * self.mnist_mask
        self.dat["variables"]["_trigger_"] *= 0.0
        # Check for split update.
        if self.split != self.old_split:
            self.old_split = np.copy(self.split)
            for i,I in enumerate(self.current_duration):
                self.current_elapsed[i] = I
        # Update experimental state for all agents.
        for a in range(self.net["agents"]):
            # Update current experimental state.
            # -----------------------------------------------------------------
            self.update_sample(a)

            # Set trigger for new stimulus.
            if self.current_elapsed[a] == 0:
                self.dat["variables"]["_trigger_"][a] = 1

            # Update internal data (= variables).
            # -----------------------------------------------------------------
            # Update image and label board.
            for s in range(self.N_mnist):
                x = 28 * (s % self.N_mnist_x)
                y = 28 * (s // self.N_mnist_x)
                if a > self.split:
                    c = int(self.train_data["mnist_label"][self.current_sample[a][s],0])
                else:
                    c = int(self.test_data["mnist_label"][self.current_sample[a][s],0])
                if not self.current_hidden[a][s]:
                    if a > self.split:
                        self.mnist_image[a,0,x:x + 28,y:y + 28] \
                            = self.train_data["mnist_image"][self.current_sample[a][s],:,:]
                    else:
                        self.mnist_image[a,0,x:x + 28,y:y + 28] \
                            = self.test_data["mnist_image"][self.current_sample[a][s],:,:]
                # Set label.
                self.mnist_label[a,c,x + self.border:x + 28 - self.border, y + self.border:y + 28 - self.border] = 1
                self.mnist_mask[a,0,x + self.border:x + 28 - self.border, y + self.border:y + 28 - self.border] = 1
        
        # Write board to shared memory.
        if "seg_mnist_image" in self.p["out"]:
            self.dat["variables"]["seg_mnist_image"][:,:,:,:] \
                = self.mnist_image[:,:,:,:]
        if "seg_mnist_label" in self.p["out"]:
            self.dat["variables"]["seg_mnist_label"][:,:,:,:] \
                = self.mnist_label[:,:,:,:]
        if "seg_mnist_mask" in self.p["out"]:
            self.dat["variables"]["seg_mnist_mask"][:,:,:,:] \
                = self.mnist_mask[:,:,:,:]

        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "seg_mnist_pred" in self.p["in"] \
                and "seg_mnist_image" in self.p["out"] \
                and "seg_mnist_label" in self.p["out"]:

            # Call update function.
            current_label = [None for a in range(self.net["agents"])]
            for a in range(self.net["agents"]):
                gt_idx = self.mnist_label[a,:,:,:]
                mask_idx = self.mnist_mask[a,:,:,:]
                current_label[a] = [gt_idx, mask_idx]
            mask = np.ones([self.net['agents'],])
            mask[0:self.split] = 0
            self.TCM_train.update_history(self.inputs["seg_mnist_pred"],
                                          int(self.frame_cntr),
                                          current_label,
                                          self.current_elapsed,
                                          self.dat["variables"]["_trigger_"],
                                          mask)
            self.TCM_test.update_history(self.inputs["seg_mnist_pred"],
                                          int(self.frame_cntr),
                                          current_label,
                                          self.current_elapsed,
                                          self.dat["variables"]["_trigger_"],
                                          1 - mask)
            # Write updated performances to shared memory.
            self.dat["variables"]["conf_mat_train"][:,:,:] \
                = self.TCM_train.conf_mat[:,:,:]
            self.dat["variables"]["acc_train"][:,:,:] \
                = self.TCM_train.accuracy[:,:,:]
            self.dat["variables"]["conf_mat_test"][:,:,:] \
                = self.TCM_test.conf_mat[:,:,:]
            self.dat["variables"]["acc_test"][:,:,:] \
                = self.TCM_test.accuracy[:,:,:]
