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
import PIL
import time
import copy
import os
import multiprocessing as mp
import ctypes

from statestream.interfaces.process_if import ProcessIf
from statestream.interfaces.utils.temporal_confusion_matrix import TemporalConfusionMatrix
from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.meta.neuron_pool import np_state_shape
from statestream.utils.helper import is_scalar_shape
from statestream.utils.helper import LoadSample



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
    return {"out": ["cs_image", 
                    "cs_label", 
                    "cs_class", 
                    "cs_pclass",
                    "cs_mask"], 
            "in": ["cs_pred"]
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
    """Return shared memory layout for cityscapes interface.

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
        = ShmL("np", (), np.int32, p.get("min_duration", 4), 1, None)
    shm_layout["parameter"]["max_duration"] \
        = ShmL("np", (), np.int32, p.get("max_duration", 6), 1, None)

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
    if "cs_pred" in p["in"]:
        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "cs_pred"
        if "remap" in p:
            if "cs_pred" in p["remap"]:
                tmp_target = p["remap"]["cs_pred"]
        no_classes = np_state_shape(net, tmp_target)[1]
        # Create layout for conf-mat.
        shm_layout["variables"]["conf_mat_train"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 16), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)
        shm_layout["variables"]["conf_mat_test"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 16), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)
        shm_layout["variables"]["acc_train"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 16), 
                    1,
                    1],
                   np.float32,
                   0)
        shm_layout["variables"]["acc_test"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 16), 
                    1,
                    1],
                   np.float32,
                   0)

    # Return layout.
    return shm_layout







class ProcessIf_cityscapes(ProcessIf):
    """Interface class providing basic cityscapes data.

        To use this interface please download the cityscapes dataset (fine labeled)
        and specify the global path to the images and labels.

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
        image_path : str
            This has to be the global path to the cityscapes images and should
            contain the subfolders: train/, val/ and test/ 
        label_path : str
            This has to be the global path to the cityscapes labels and should
            contain the subfolders: train/, val/ and test/ 
        processes : int
            Number of parallel processes to load cityscapes data.
        samples : int
            Number of samples to store for efficient data loading.
        min_duration : int
            The minimum duration a cityscapes image will be presented in frames.
            The actual duration will be drawn uniformly between min_duration
            and max_duration.
        max_duration : int
            The maximum duration a cityscapes image will be presented in frames.
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
        cs_pred : np.array, shape [agents, 19, dim_x, dim_y]
            The cityscapes interface provides a delayed confusion matrix as performance
            measure. To compute this a classification result is needed to be compared
            with the ground-truth.

        Outputs:
        --------
        cs_image : np.array, shape [agents, 3, dim_x, dim_y]
            The RGB cityscapes images.
        cs_label : np.array, shape [agents, 19, dim_x, dim_y]
            The one-hot encoded ground-truth label for the current image.
        cs_class : np.array, shape [agents, 3, dim_x, dim_y]
            The RGB encoded ground-truth class labels for the current image.
        cs_pclass : np.array, shape [agents, 3, dim_x, dim_y]
            The RGB encoded converted network prediction cs_pred.
        cs_mask : np.array, shape [agents, 1, dim_x, dim_y]
            The one-hot encoded class label for the ignore / mask class.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)


    def initialize(self):
        """Method to initialize the cityscapes interface class.
        """

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_duration = []
        self.current_elapsed = []
        self.current_isample = []
        for a in range(self.net["agents"]):
            self.current_duration \
                += [np.random.randint(self.dat["parameter"]["min_duration"],
                                      self.dat["parameter"]["max_duration"] + 1)]
            self.current_elapsed += [0]
            self.current_isample += [-1]

        # Initialize data structures for parallel loading.
        # --------------------------------------------------------------------
        # Get label and image paths.
        assert("label_path" in self.p and "image_path" in self.p), \
            "\nError: Cityscapes interface: please specify 'label_path' and " \
            + "'image_path' for cityscapes interface"
        self.label_path = self.p["label_path"]
        self.image_path = self.p["image_path"]

        # Set data items.
        self.data_items = []
        for di in ["cs_image", "cs_class", "cs_label"]:
            if di in self.p['out']:
                self.data_items.append(di)
        self.file_names = {'train': {}, 'test': {}}
        for tt in ['train', 'test']:
            for i in self.data_items:
                self.file_names[tt][i] = []
        self.data_item_shape = {}
        self.data_item_size = {}
        self.data_item_type = {}
        if 'cs_image' in self.data_items:
            self.data_item_type['cs_image'] = 'RGB'
        if 'cs_class' in self.data_items:
            self.data_item_type['cs_class'] = 'RGB-nearest'
        if 'cs_label' in self.data_items:
            self.data_item_type['cs_label'] = 'RGB2one-hot'

        # Get list of all files for which labels exists (= train + val).
        self.train_samples = 0
        self.test_samples = 0
        for t in ["train", "test"]:
            if t == 'test':
                T = 'val'
            else:
                T = 'train'
            cities = os.listdir(self.image_path + os.sep + T)
            # Loop over all cities.
            for c in cities:
                img_files = os.listdir(self.image_path + os.sep + T + os.sep + c)
                cls_allfiles = os.listdir(self.label_path + os.sep + T + os.sep + c)
                if "Thumbs.db" in img_files:
                    img_files.remove("Thumbs.db")
                # Keep only color label images.
                cls_files = []
                for f in cls_allfiles:
                    if f.find("color.png") != -1:
                        cls_files.append(f)
                img_files.sort()
                cls_files.sort()
                assert(len(img_files) == len(cls_files)), \
                    "\nError: Cityscapes interface: Found different number of images / labels for city: " \
                    + str(c) + "  " + str(len(img_files)) + "  " + str(len(lbl_files))
                for s in range(len(img_files)):
                    self.file_names[t]["cs_image"].append(self.image_path + os.sep \
                                            + T + os.sep + c + os.sep + img_files[s])
                    if 'cs_class' in self.data_items:
                        self.file_names[t]["cs_class"].append(self.label_path + os.sep \
                                                + T + os.sep + c + os.sep + cls_files[s])
                    if 'cs_label' in self.data_items:
                        self.file_names[t]["cs_label"].append(self.label_path + os.sep \
                                                + T + os.sep + c + os.sep + cls_files[s])

        # Get number of samples.
        for i in ["cs_image", "cs_class"]:
            if i in self.file_names['train']:
                self.train_samples = len(self.file_names['train'][i])
                self.test_samples = len(self.file_names['test'][i])
                break
            
        # The cityscapes label table.
        # Note: Here the first class is the ignore (aka mask) class.
        self.no_classes = 19
        self.cs_label_info = {
            "aa-unlabeled": {
                "rgb": [  0,  0,  0],
                "class": 0,
                "original class": 0
            },
            "bicycle": {
                "rgb": [119, 11, 32],
                "class": 1,
                "original class": 33
            },
            "building": {
                "rgb": [ 70, 70, 70],
                "class": 2,
                "original class": 11                
            },
            "bus": {
                "rgb": [  0, 60,100],
                "class": 3,
                "original class": 28
            },
            "car": {
                "rgb": [  0,  0,142],
                "class": 4,
                "original class": 26
            },
            "fence": {
                "rgb": [190,153,153],
                "class": 5,
                "original class": 13
            },
            "motorcycle": {
                "rgb": [  0,  0,230],
                "class": 6,
                "original class": 32
            },
            "person": {
                "rgb": [220, 20, 60],
                "class": 7,
                "original class": 24
            },
            "pole": {
                "rgb": [153,153,153],
                "class": 8,
                "original class": 17
            },
            "rider": {
                "rgb": [255,  0,  0],
                "class": 9,
                "original class": 25
            },
            "road": {
                "rgb": [128, 64,128],
                "class": 10,
                "original class": 7
            },
            "sidewalk": {
                "rgb": [244, 35,232],
                "class": 11,
                "original class": 8
            },
            "sky": {
                "rgb": [ 70,130,180],
                "class": 12,
                "original class": 23
            },
            "terrain": {
                "rgb": [152,251,152],
                "class": 13,
                "original class": 22
            },
            "traffic light": {
                "rgb": [250,170, 30],
                "class": 14,
                "original class": 19
            },
            "traffic sign": {
                "rgb": [220,220,  0],
                "class": 15,
                "original class": 20
            },
            "train": {
                "rgb": [  0, 80,100],
                "class": 16,
                "original class": 31
            },
            "truck": {
                "rgb": [  0,  0, 70],
                "class": 17,
                "original class": 27
            },
            "vegetation": {
                "rgb": [107,142, 35],
                "class": 18,
                "original class": 21
            },
            "wall": {
                "rgb": [102,102,156],
                "class": 19,
                "original class": 12
            }
        }

        # Helper for conversion: one-hot -> RGB.
        self.cl_converter = []
        for key in sorted(self.cs_label_info.keys()):
            self.cl_converter.append([self.cs_label_info[key]['rgb'][0], 
                                      self.cs_label_info[key]['rgb'][1], 
                                      self.cs_label_info[key]['rgb'][2]])
        self.cl_converter = np.array(self.cl_converter, np.uint8)

        self.data_item_param = {}
        if 'cs_image' in self.data_items:
            self.data_item_param['cs_image'] = {}
        if 'cs_class' in self.data_items:
            self.data_item_param['cs_class'] = {}
        if 'cs_label' in self.data_items:
            self.data_item_param['cs_label'] = {}
            self.data_item_param['cs_label']['coding'] \
                = copy.copy(self.cs_label_info)

        self.augmentation = self.p.get("augmentation", {})

        # Determine image / label shapes from neuron-pools.
        self.sample_shape = None
        self.sample_size = None
        for o in self.p["out"]:
            tmp_target = o
            # Consider remapping.
            if "remap" in self.p:
                if o in self.p["remap"]:
                    tmp_target = self.p["remap"][o]
            np_shape = np_state_shape(self.net, tmp_target)
            if o == "cs_image":
                self.data_item_shape[o] = np_shape[1:]
            elif o == "cs_class":
                self.data_item_shape[o] = [3, np_shape[2], np_shape[3]]
            elif o == "cs_label":
                # We have to consider the ignore / mask class here.
                self.data_item_shape[o] = [self.no_classes + 1, np_shape[2], np_shape[3]]
            if o in self.data_item_param:
                self.data_item_size[o] = int(np.prod(self.data_item_shape[o]))


        # Get / set number of processes and samples to hold in reserve.
        # Note: The factor two is due to the potential split.
        self.no_procs = max(self.p.get("processes", 2), 2)
        self.no_isamples = max(self.p.get("samples", 2), 
                               self.net["agents"] + 2)

        # Generate necessary shared memory for parallel data loading.
        # --------------------------------------------------------------------
        self.DL_IPC_PROC = {}
        self.DL_IPC_PROC['src_filename'] = {}           # {item} [proc]
        self.DL_IPC_PROC['src_filename_len'] = {}       # {item} [proc]
        self.DL_IPC_PROC['target_sample'] = []          # [proc]
        self.DL_IPC_PROC['proc_state'] = []             # [proc] 0 :: idle, 1 :: loading, 2 :: shutdown
        # [isamples] 0 :: old, 1 :: loading, 2 :: fresh, 3 :: in use
        self.DL_IPC_PROC['sample_state'] = []
        # Generate sample state (one int for each [sample]).
        for s in range(self.no_isamples):
            self.DL_IPC_PROC['sample_state'].append(mp.Array('d', [0]))
        for i in self.data_items:
            self.DL_IPC_PROC['src_filename'][i] = {}
            self.DL_IPC_PROC['src_filename_len'][i] = {}
            for p in range(self.no_procs):
                self.DL_IPC_PROC['src_filename'][i][p] \
                    = mp.Array('d', [0 for j in range(256)])
                self.DL_IPC_PROC['src_filename_len'][i][p] \
                    = mp.Array('d', [0])
        for p in range(self.no_procs):
            self.DL_IPC_PROC['target_sample'].append(mp.Array('d', [0]))
            self.DL_IPC_PROC['proc_state'].append(mp.Array('d', [-1]))
        # Allocate shared memory for data.
        self.DL_IPC_DATA = {}               # {item} [sample]
        for i in self.data_items:
            self.DL_IPC_DATA[i] = {}
            for s in range(self.no_isamples):
                self.DL_IPC_DATA[i][s] \
                    = mp.sharedctypes.RawArray(ctypes.c_float, 
                                               self.data_item_size[i])
        
        # Create numpy respresentation of own shared memory.
        self.dl_net = {}
        for i in self.data_items:
            self.dl_net[i] = {}
            for s in range(self.no_isamples):
                self.dl_net[i][s] = np.frombuffer(self.DL_IPC_DATA[i][s],
                                             dtype=np.float32,
                                             count=self.data_item_size[i])
                self.dl_net[i][s].shape = self.data_item_shape[i]

        # Get class instances for processes.
        self.dl_procs_inst = {}
        for p in range(self.no_procs):
            self.dl_procs_inst[p] = LoadSample(p, 
                                               self.name, 
                                               self.net, 
                                               self.data_item_shape,
                                               self.data_item_type,
                                               self.data_item_param,
                                               augmentation=self.augmentation)
        # Create and launch child processes for data loading.
        self.proc_dl = {}
        for p in range(self.no_procs):
            self.proc_dl[p] = mp.Process(target=self.dl_procs_inst[p].run,
                                         args=(self.DL_IPC_PROC,
                                               self.DL_IPC_DATA))
        # Start processes with initial handshake.
        for p in range(self.no_procs):
            self.proc_dl[p].start()
            while self.DL_IPC_PROC['proc_state'][p][0] != 0:
                time.sleep(0.1)

        # Instantiate temporal confusion matrix.
        if "cs_pred" in self.p["in"]:
            self.TCM_train = TemporalConfusionMatrix(self.net, self.name, "cs_pred")
            self.TCM_test = TemporalConfusionMatrix(self.net, self.name, "cs_pred")

# =============================================================================
# =============================================================================

    def update_always(self):
        """Interface dependent update in every PROCESS cycle.
        """
        # Always try to keep processes as busy as possible.
        for split, tt in enumerate(['train', 'test']):
            P = [int(i + split * (self.no_procs // 2)) for i in range(int(self.no_procs // 2))]
            S = [int(i + split * (self.no_isamples // 2)) for i in range(int(self.no_isamples // 2))]
            for p in P:
                if self.DL_IPC_PROC['proc_state'][p][0] == 0:
                    # Process p is idle,
                    # so look for next to be filled sample slot.
                    for s in S:
                        if self.DL_IPC_PROC['sample_state'][s][0] == 0:
                            # Found a free sample that has to be loaded.
                            # Determine sample to be loaded next.
                            if split == 0:
                                new_sample = np.random.randint(0, self.train_samples - 1)
                            else:
                                new_sample = np.random.randint(0, self.test_samples - 1)
                            # Update file names for all items.
                            for i in self.data_items:
                                file_name = self.file_names[tt][i][new_sample]
                                file_string = np.array([ord(c) for c in file_name])
                                self.DL_IPC_PROC['src_filename'][i][p][0:len(file_string)] \
                                    = file_string[:]
                                self.DL_IPC_PROC['src_filename_len'][i][p][0] \
                                    = len(file_string)
                            # Set sample slot.
                            self.DL_IPC_PROC['target_sample'][p][0] = s
                            # Update target sample slot state.
                            self.DL_IPC_PROC['sample_state'][s][0] = 1
                            # Trigger process.
                            self.DL_IPC_PROC['proc_state'][p][0] = 1
                            # Now process p is busy ... look for new idle process.
                            break

    def update_frame_writeout(self):
        # Reset triggers.
        if "cs_pred" in self.p["in"]:
            self.dat["variables"]["_trigger_"] *= 0.0
        # Check for split update. In case, start fresh.
        if self.split != self.old_split:
            # Wait until all procs are idle, then set all samples to deprecated.
            not_all_procs_idle = True
            while not_all_procs_idle:
                not_all_procs_idle = False
                for p in range(self.no_procs):
                    if self.DL_IPC_PROC['proc_state'][p][0] != 0:
                        not_all_procs_idle = True
                        sleep(0.005)
                        break
            self.old_split = np.copy(self.split)
            for i,I in enumerate(self.current_duration):
                self.current_elapsed[i] = 0
            for i in range(self.no_isamples):
                self.DL_IPC_PROC['sample_state'][i][0] = 0
            self.update_always()
        # Update experimental state for all agents.
        for a in range(self.net["agents"]):
            # Update current experimental state.
            # -----------------------------------------------------------------
            self.current_elapsed[a] += 1
            # Check for end of frame.
            if self.current_elapsed[a] > self.current_duration[a] \
                    or self.current_isample[a] == -1:
                # Set back elapsed.
                self.current_elapsed[a] = 0
                # Set old sample to deprecated.
                if self.current_isample[a] != -1:
                    self.DL_IPC_PROC['sample_state'][self.current_isample[a]][0] = 0
                # Search / wait for new sample.
                new_sample_found = False
                if a > self.split:
                    S = [i for i in range(self.no_isamples // 2)]
                else:
                    S = [int(i + self.no_isamples / 2) for i in range(self.no_isamples // 2)]
                while not new_sample_found:
                    # Loop over all potentially preloaded samples.
                    for s in S:
                        if self.DL_IPC_PROC['sample_state'][s][0] == 2:
                            new_sample_found = True
                            self.current_isample[a] = s
                            self.DL_IPC_PROC['sample_state'][s][0] = 3
                            break
                    if not new_sample_found:
                        # print("\nWarning: Cityscapes interface: no new samples found.")
                        time.sleep(0.01)
                # Draw new duration.
                if self.dat["parameter"]["min_duration"] \
                        == self.dat["parameter"]["max_duration"]:
                    self.current_duration[a] \
                        = self.dat["parameter"]["min_duration"]
                else:
                    self.current_duration[a] \
                        = np.random.randint(self.dat["parameter"]["min_duration"],
                                            self.dat["parameter"]["max_duration"])
                # Set trigger for new stimulus.
                self.dat["variables"]["_trigger_"][a] = 1

        # For each current sample set output image, class and label
        self.first_loaded = True
        for a in range(self.net["agents"]):
            if self.current_isample[a] != -1:
                if "cs_class" in self.p["out"]:
                    self.dat["variables"]['cs_class'][a,:,:,:] \
                        = self.dl_net['cs_class'][self.current_isample[a]][:,:,:] / 255.0
                if "cs_label" in self.p["out"]:
                    # Ignore the first (mask / ignore) class.
                    self.dat["variables"]['cs_label'][a,:,:,:] \
                        = self.dl_net['cs_label'][self.current_isample[a]][1:,:,:]
                if "cs_pclass" in self.p["out"] and "cs_label" in self.p["out"]:
                    max_pred = np.argmax(self.inputs["cs_pred"][a,:,:,:], axis=0) + 1

                    self.dat["variables"]["cs_pclass"][a,:,:,:] \
                        = np.rollaxis(self.cl_converter[max_pred], 2, 0) / 255.0
                if 'cs_image' in self.p["out"]:
                    self.dat["variables"]["cs_image"][a,:,:,:] \
                        = self.dl_net['cs_image'][self.current_isample[a]][:,:,:]
                if 'cs_mask' in self.p["out"]:
                    self.dat["variables"]["cs_mask"][a,0,:,:] \
                        = 1 - self.dl_net['cs_label'][self.current_isample[a]][0,:,:]
            else:
                self.first_loaded = False
        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "cs_pred" in self.p["in"] \
                and "cs_image" in self.p["out"] \
                and "cs_label" in self.p["out"] \
                and self.first_loaded:
            # Call update function.
            current_label = [None for a in range(self.net["agents"])]
            for a in range(self.net["agents"]):
                gt_idx = self.dl_net['cs_label'][self.current_isample[a]][1:,:,:]
                mask_idx = 1 - self.dl_net['cs_label'][self.current_isample[a]][0:1,:,:]
                current_label[a] = [gt_idx, mask_idx]
            mask = np.ones([self.net['agents'],])
            mask[0:self.split] = 0
            self.TCM_train.update_history(self.inputs["cs_pred"],
                                          int(self.frame_cntr),
                                          current_label,
                                          self.current_elapsed,
                                          self.dat["variables"]["_trigger_"],
                                          mask)
            self.TCM_test.update_history(self.inputs["cs_pred"],
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



    def quit(self):
        """Exit method for Cityscapes interface.
        """
        # Sleep to give child-processes the chance to return.
        time.sleep(1.0)
        # Send closing signal to all sub-processes.
        for p in range(self.no_procs):
            self.DL_IPC_PROC['proc_state'][p][0] = 2
