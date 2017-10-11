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
    return {"out": ["css_image", 
                    "css_label", 
                    "css_class",
                    "css_pclass", 
                    "css_mask",
                    "css_speed",
                    "css_yaw"], 
            "in": ["css_pred"]
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
        = ShmL("np", [net["agents"],], np.int32, 0)
    # If a prediction is given as input we compute a confusion matrix.
    if "css_pred" in p["in"]:
        # Add parameter.
        shm_layout["parameter"]["conf-mat window"] \
            = ShmL("np", (), np.int32, int(p.get("conf-mat mean over", 9)), 1, None)
        shm_layout["parameter"]["conf-mat mean over"] \
            = ShmL("np", (), np.int32, int(p.get("conf-mat mean over", 32)), 1, None)

        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "css_pred"
        if "remap" in p:
            if "css_pred" in p["remap"]:
                tmp_target = p["remap"]["css_pred"]
        no_classes = np_state_shape(net, tmp_target)[1]
        # Create layout for conf-mat
        shm_layout["variables"]["_conf_mat_"] \
            = ShmL("np", 
                   [1, 
                    p.get("conf-mat window", 9), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)

    # Return layout.
    return shm_layout







class ProcessIf_cityscapes_sequence(ProcessIf):
    """Interface class providing basic cityscapes sequential data.

        To use this interface please download the pickled version of the mnist 
        dataset: mnist.pkl.gz
        The absolute path to this file must be given (under 'source_file') in 
        the st_graph file that want to use mnist.

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
        vehicle_path : str
            This may be the global path the vehicle_sequence provided
            with the cityscapes dataset.
        processes : int
            Number of parallel processes to load cityscapes data.
        samples : int
            Number of samples to store for efficient data loading.
            This parameter has to be specified and also has to be a multiple
            of net["agents"].
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
        css_pred : np.array, shape [agents, 19, dim_x, dim_y]
            The cityscapes interface provides a delayed confusion matrix as performance
            measure. To compute this a classification result is needed to be compared
            with the ground-truth.

        Outputs:
        --------
        css_image : np.array, shape [agents, 3, dim_x, dim_y]
            The grey-scale mnist images.
        css_label : np.array, shape [agents, 19, dim_x, dim_y]
            The one-hot encoded ground-truth label for the current image.
        css_class : np.array, shape [agents, 3, dim_x, dim_y]
            The RGB encoded ground-truth class labels for the current image.
        css_pclass : np.array, shape [agents, 3, dim_x, dim_y]
            The RGB encoded (converted) network prediction css_pred.
        css_mask : np.array, shape [agents, 1, dim_x, dim_y]
            The one-hot encoded class label for the ignore / mask class.
        css_speed : np.array, shape [agents, 1, 1, 1]
            The ground-truth vehicle speed.
        css_yaw : np.array, shape [agents, 1, 1, 1]
            The ground-truth vehicle yaw angle.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)


    def initialize(self):
        """Method to initialize the cityscapes interface class.
        """
        self.dat["parameter"]["min_duration"] = self.p.get("min_duration", 2)
        self.dat["parameter"]["max_duration"] = self.p.get("max_duration", 4)

        # Initially trigger histories to empty.
        self.trigger_history = []
        for a in range(self.net["agents"]):
            self.trigger_history.append([-1,-1])
        # Initialize delayed confusion matrices as empty.
        if "css_pred" in self.p["in"]:
            self.conf_mat_hist = np.zeros([self.dat["parameter"]["conf-mat mean over"],
                                           self.dat["parameter"]["conf-mat window"],
                                           self.no_classes,
                                           self.no_classes],
                                          dtype=np.float32)

        # Initialize data structures for parallel loading.
        # --------------------------------------------------------------------
        # Get label and image paths
        assert("label_path" in self.p and "image_path" in self.p), \
            "\nError: Cityscapes interface: please specify 'label_path' and " \
            + "'image_path' for cityscapes interface"
        self.label_path = self.p["label_path"]
        self.image_path = self.p["image_path"]

        # Set data items.
        self.data_items = ["css_image", "css_class", "css_label"]
        self.file_names = {}
        for i in self.data_items:
            self.file_names[i] = []
        self.data_item_shape = {}
        self.data_item_size = {}
        self.data_item_type = {
            "css_image": "RGB",
            "css_class": "RGB-nearest",
            "css_label": "RGB2one-hot"
        }

        # Get list of all files for which labels exists (= train + val).
        for t in ["train", "val"]:
            cities = os.listdir(self.image_path + os.sep + t)
            # Loop over all cities.
            for c in cities:
                img_files = os.listdir(self.image_path + os.sep + t + os.sep + c)
                cls_allfiles = os.listdir(self.label_path + os.sep + t + os.sep + c)
                if "Thumbs.db" in img_files:
                    img_files.remove("Thumbs.db")
                # Keep only color label images.
                cls_files = []
                for f in cls_allfiles:
                    if f.find("color.png") != -1:
                        cls_files.append(f)
                img_files.sort()
                cls_files.sort()
                assert(len(img_files) % len(cls_files) == 0), \
                    "\nError: Cityscapes sequence interface: Found different number of images / labels for city: " \
                    + str(c) + "  " + str(len(img_files)) + "  " + str(len(lbl_files))
                for s in range(len(img_files)):
                    self.file_names["css_image"].append(self.image_path + os.sep \
                                                        + t + os.sep + c + os.sep + img_files[s])
                for s in range(len(cls_files)):
                    self.file_names["css_class"].append(self.label_path + os.sep \
                                                        + t + os.sep + c + os.sep + cls_files[s])
                    self.file_names["css_label"].append(self.label_path + os.sep \
                                                        + t + os.sep + c + os.sep + cls_files[s])

        # Get number of samples.
        for i in ["css_class", "css_label"]:
            if i in self.file_names:
                self.no_samples = len(self.file_names[i])
                break

        # Get number of frames in sequence.
        self.no_frames = int(len(self.file_names["css_image"]) / self.no_samples)
        
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

        self.data_item_param = {
            "css_image": {},
            "css_class": {},
            "css_label": {
                "coding": copy.copy(self.cs_label_info)
            }
        }

        # Helper for conversion: one-hot -> RGB.
        self.cl_converter = []
        for key in sorted(self.cs_label_info.keys()):
            self.cl_converter.append([self.cs_label_info[key]['rgb'][0], 
                                      self.cs_label_info[key]['rgb'][1], 
                                      self.cs_label_info[key]['rgb'][2]])
        self.cl_converter = np.array(self.cl_converter, np.uint8)

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
            if o == "css_image":
                self.data_item_shape[o] = np_shape[1:]
            elif o == "css_class":
                self.data_item_shape[o] = [3, np_shape[2], np_shape[3]]
            elif o == "css_label":
                # We have to consider the ignore / mask class here.
                self.data_item_shape[o] = [self.no_classes + 1, np_shape[2], np_shape[3]]
            if o in self.data_item_param:
                self.data_item_size[o] = int(np.prod(self.data_item_shape[o]))


        # Get / set number of processes and samples to hold in reserve.
        self.no_procs = self.p.get("processes", 1)
        self.no_isamples = self.p.get("samples", 2 * self.net["agents"])

        # Number between 0 .. self.no_isamples / self.net["agents"]
        self.current_iframe = 0

        # List of list[4], one for each isample holding information:
        # [sample, duration, elapsed]
        # -1 indicates un-initialized
        self.sample_frame = [[-1,-1,-1] for i in range(self.no_isamples)]

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
                                               self.data_item_param)
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


# =============================================================================
# =============================================================================

    def update_always(self):
        """Interface dependent update in every PROCESS cycle.
        """
        if False:
            print("\n\n")
            for s in range(self.no_isamples):
                print("ID: " + str(s) \
                      + "  SampleFrame: " + str(self.sample_frame[s]) \
                      + "  SampleState: " + str(self.DL_IPC_PROC['sample_state'][s][0]) \
                      + "  iFrame: " + str(self.current_iframe) \
                      + "  iSamps: " + str(self.no_isamples))
            print("\n")
            for p in range(self.no_procs):
                print("Proc: " + str(p) \
                      + " TargetSamp: " + str(self.DL_IPC_PROC['target_sample'][p][0]) \
                      + " ProcState: " + str(self.DL_IPC_PROC['proc_state'][p][0]))

        # Always try to keep processes as busy as possible.
        for p in range(self.no_procs):
            if self.DL_IPC_PROC['proc_state'][p][0] == 0:
                # Process p is idle,
                # so look for next to be filled sample slot.
                idx = [(i + self.current_iframe * self.net["agents"]) % self.no_isamples for i in range(self.no_isamples)]
                for s in idx:
                    if self.DL_IPC_PROC['sample_state'][s][0] == 0:
                        # Determine agent of this isample.
                        this_agent = s % int(self.net["agents"])
                        this_iframe = s // self.net["agents"]
                        this_last_isample = (s - self.net["agents"]) \
                                            % self.no_isamples
                        # Check if last is not initialized.
                        new_init = False
                        if self.sample_frame[this_last_isample][0] == -1:
                            new_init = True
                        else:
                            # Check elapsed < duration.
                            if self.sample_frame[this_last_isample][2] < self.sample_frame[this_last_isample][1]:
                                self.sample_frame[s][0] = self.sample_frame[this_last_isample][0]
                                self.sample_frame[s][1] = self.sample_frame[this_last_isample][1]
                                self.sample_frame[s][2] = self.sample_frame[this_last_isample][2] + 1
                            else:
                                # Elapsed reached duration, hence re-init this agent / slot.
                                new_init = True
                        # Check for re-init / initial-init.
                        if new_init:
                            # Initialize new target isample.
                            self.sample_frame[s][0] = np.random.randint(0, self.no_samples - 1)
                            self.sample_frame[s][1] \
                                = np.random.randint(low=self.dat["parameter"]["min_duration"],
                                                    high=self.dat["parameter"]["max_duration"] + 1)
                            self.sample_frame[s][2] = 0

                        # Update file names for all items.
                        file_name = {}
                        # 19. frame is label so give labels / classes here.
                        if self.sample_frame[s][2] == 19:
                            file_name["css_class"] = self.file_names["css_class"][self.sample_frame[s][0]]
                            file_name["css_label"] = self.file_names["css_label"][self.sample_frame[s][0]]
                        else:
                            file_name["css_class"] = ""
                            file_name["css_label"] = ""
                        # Compute current image index in filename list.
                        current_image_idx = self.sample_frame[s][2] + self.sample_frame[s][0] * self.no_frames
                        file_name["css_image"] = self.file_names["css_image"][current_image_idx]
                        for i in self.data_items:
                            if len(file_name[i]) > 0:
                                file_string = np.array([ord(c) for c in file_name[i]])
                                self.DL_IPC_PROC['src_filename'][i][p][0:len(file_string)] \
                                    = file_string[:]
                                self.DL_IPC_PROC['src_filename_len'][i][p][0] \
                                    = len(file_string)
                            else:
                                self.DL_IPC_PROC['src_filename_len'][i][p][0] = 0
                        # Set sample slot.
                        self.DL_IPC_PROC['target_sample'][p][0] = s
                        # Update target sample slot state.
                        self.DL_IPC_PROC['sample_state'][s][0] = 1
                        # Trigger process.
                        self.DL_IPC_PROC['proc_state'][p][0] = 1
                        # Now process p is busy ...
                        break

    def update_frame_writeout(self):
        # Check if all samples of current iframe are ready.
        isamples_ready = 0 
        while isamples_ready != self.net["agents"]:
            isamples_ready = 0 
            time.sleep(0.002)
            for a in range(self.net["agents"]):
                sample_id = a + self.current_iframe * self.net["agents"]
                if self.DL_IPC_PROC['sample_state'][sample_id][0] == 2:
                    isamples_ready += 1

        # Block (mutex) all samples for reading.
        for a in range(self.net["agents"]):
            sample_id = a + self.current_iframe * self.net["agents"]
            self.DL_IPC_PROC['sample_state'][sample_id][0] = 3

        # Copy from local shm to statestream shm.
        for a in range(self.net["agents"]):
            sample_id = a + self.current_iframe * self.net["agents"]
            if "css_class" in self.p["out"]:
                self.dat["variables"]['css_class'][a,:,:,:] \
                    = self.dl_net['css_class'][sample_id][:,:,:] / 255.0
            if "css_label" in self.p["out"]:
                # Ignore the first (mask / ignore) class.
                self.dat["variables"]['css_label'][a,:,:,:] \
                    = self.dl_net['css_label'][sample_id][1:,:,:]
            if "css_pclass" in self.p["out"] and "css_label" in self.p["out"]:
                max_pred = np.argmax(self.inputs["css_pred"][a,:,:,:], axis=0)
                self.dat["variables"]["css_pclass"][a,:,:,:] \
                    = self.cl_converter[max_pred]
            if 'css_image' in self.p["out"]:
                self.dat["variables"]["css_image"][a,:,:,:] \
                    = self.dl_net['css_image'][sample_id][:,:,:]
            if 'css_mask' in self.p["out"]:
                self.dat["variables"]["css_mask"][a,0,:,:] \
                    = 1 - self.dl_net['css_label'][sample_id][0,:,:]

        # Set old (= current) samples to deprecated (un-block).
        for a in range(self.net["agents"]):
            sample_id = a + self.current_iframe * self.net["agents"]
            self.DL_IPC_PROC['sample_state'][sample_id][0] = 0

        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "css_pred" in self.p["in"] \
                and "css_image" in self.p["out"] \
                and "css_label" in self.p["out"]:
            pass

        # Update current frame.
        self.current_iframe = (self.current_iframe + 1) % (self.no_isamples // self.net["agents"])



    def quit(self):
        """Exit method for Cityscapes interface.
        """
        # Sleep to give child-processes the chance to return.
        time.sleep(1.0)
        # Send closing signal to all sub-processes.
        for p in range(self.no_procs):
            self.DL_IPC_PROC['proc_state'][p][0] = 2
