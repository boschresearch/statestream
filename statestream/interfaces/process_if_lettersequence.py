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
import os

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
    return {"in": ["ls_current", "ls_next"], 
            "out": ["ls_pred"]
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
    """Return shared memory layout for text corpus interface.

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
        = ShmL("np", (), np.int32, p.get("mode", 0), 1, None)
    # Add durations as numpy parameters.
    shm_layout["parameter"]["min_duration"] \
        = ShmL("np", (), np.int32, p.get("min_duration", 16), 1, None)
    shm_layout["parameter"]["max_duration"] \
        = ShmL("np", (), np.int32, p.get("max_duration", 24), 1, None)

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
    if "ls_pred" in p["in"]:
        # Add parameter.
        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "ls_pred"
        if "remap" in p:
            if "ls_pred" in p["remap"]:
                tmp_target = p["remap"]["ls_pred"]
        no_classes = np_state_shape(net, tmp_target)[1]
        # Create layout for conf-mat.
        shm_layout["variables"]["_conf_mat_"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    no_classes, 
                    no_classes],
                   np.float32,
                   0)
        shm_layout["variables"]["_acc_"] \
            = ShmL("np", 
                   [p.get("conf-mat window", 9), 
                    1,
                    1],
                   np.float32,
                   0)

    # Return layout.
    return shm_layout

    

class ProcessIf_lettersequence(ProcessIf):
    """Interface class providing basic text corpus data.

        To use this interface please provide a folder with
        text files.

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
        source_path : str
            Path to text corpus.
        offset : int
            The offset in letters between the two letter
            sequences (may also be negative).
        min_duration : int
            The minimum duration for a sequence of letters.
        max_duration : int
            The maximum duration for a sequence of letters.
        conf-mat window : int
        conf-mat mean over : int

        Inputs:
        -------
        ls_pred : np.array, shape [agents, 128, length, 1]
            The received network prediction.
        Outputs:
        --------
        ls_seq_0 : np.array, shape [agents, 128, length, 1]
            The first letter sequence. The length is defined
            through the associated neuron-pool.
        ls_seq_1 : np.array, shape [agents, 128, length, 1]
            The second letter sequence. Again the its length
            is defined through the associated neuron-pool.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)

    def initialize(self):
        """Method to initialize (load) the mnist interface class.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        self.dat["parameter"]["min_duration"] = self.p.get("min_duration", 16)
        self.dat["parameter"]["max_duration"] = self.p.get("max_duration", 24)
        self.dat["parameter"]["offset"] = self.p.get("offset", 1)

        # Determine number of files in corpus and load them.
        files = os.listdir(self.p["source_path"])
        self.no_files = len(files)

        # Read all files.
        self.text_data = []
        for i in range(self.no_files):
            with open(self.p["source_path"] + os.sep + files[i]) as f:
                L = f.readlines()
                # Generate one string out of the file.
                text_str = ""
                for l in L:
                    text_str = text_str + l
                self.text_data.append(text_str)
            f.close()

        # Get shapes of associated neuron-pools.
        self.np_shapes = {}
        for o in self.p["out"] + self.p["in"]:
            tmp_target = o
            # Consider remapping.
            if "remap" in self.p:
                if o in self.p["remap"]:
                    tmp_target = self.p["remap"][o]
            self.np_shapes[o] = np_state_shape(self.net, tmp_target)

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_duration = []
        self.current_elapsed = []
        self.current_file = []
        # This offset is not to be confused with the offset parameter.
        # This offset specifies the offset of the letter sequenceS in the text.
        # The offset parameter specifies the offset between the sequences
        # relative to one another.
        self.current_offset = []
        for a in range(self.net["agents"]):
            self.current_duration \
                += [np.random.randint(self.dat["parameter"]["min_duration"],
                                      self.dat["parameter"]["max_duration"] + 1)]
            self.current_elapsed += [0]
            if self.no_files == 1:
                self.current_file += [0]
            else:
                self.current_file += [np.random.randint(0, self.no_files - 1)]
            offset_max = len(self.text_data[self.current_file[a]]) \
                         - 1 - self.current_duration[a] \
                         - self.dat["parameter"]["offset"]
            self.current_offset += [np.random.randint(0, offset_max)]

        # Instantiate temporal confusion matrix.
        if "ls_pred" in self.p["in"]:
            self.TCM = TemporalConfusionMatrix(self.net, self.name, "ls_pred")
            for a in range(self.net["agents"]):
                self.TCM.trigger_history[a] = [0, 0]



    def update_frame_writeout(self):
        # Update states.
        # Reset trigger.
        if "ls_pred" in self.p["in"]:
            self.dat["variables"]["_trigger_"] *= 0.0
        # Check for duration end (new sequence).
        for a in range(self.net["agents"]):
            self.current_elapsed[a] += 1
            if self.current_elapsed[a] >= self.current_duration[a]:
                self.current_elapsed[a] = 0
                self.current_duration[a] \
                    = np.random.randint(self.dat["parameter"]["min_duration"],
                                          self.dat["parameter"]["max_duration"] + 1)
                if self.no_files == 1:
                    self.current_file[a] = 0
                else:
                    self.current_file[a] = np.random.randint(0, self.no_files - 1)
                offset_max = len(self.text_data[self.current_file[a]]) \
                             - 2 - self.dat["parameter"]["max_duration"] \
                             - abs(self.dat["parameter"]["offset"])
                self.current_offset[a] = np.random.randint(0, offset_max)
                # Set trigger for new stimulus.
                self.dat["variables"]["_trigger_"][a] = 1

        for s in ["ls_seq_0", "ls_seq_1"]:
            if s in self.p["out"]:
                # Tabula rasa for outputs.
                self.dat["variables"][s] *= 0
                seq_len = self.np_shapes[s][2]
                # Write outputs.
                for a in range(self.net["agents"]):
                    if s == "ls_seq_0":
                        char_pos = int(self.current_offset[a]) + int(self.current_elapsed[a])
                    else:
                        char_pos = int(self.current_offset[a]) + int(self.current_elapsed[a]) \
                                   + int(self.dat["parameter"]["offset"])
                        char_pos = max(char_pos, 0)
                    substring = self.text_data[self.current_file[a]][char_pos:char_pos + seq_len]
                    ascii_list = [np.clip(ord(c), 0, 127) for c in substring]
                    idx_list = [a, ascii_list, range(seq_len), 0]
                    try:
                        self.dat["variables"][s][idx_list] = 1
                    except:
                        print("\nFAILURE while setting to 1:")
                        print("\seqlen: " + str(seq_len) + "  dat: **" + str(self.text_data[self.current_file[a]][char_pos:char_pos + seq_len]) + "**  ascii: " + str(ascii_list) + "  str: **" + str(substring) + "**  elapsed: " + str(self.current_elapsed[a]) + " paramoff: " + str(self.dat["parameter"]["offset"]))

        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "ls_pred" in self.p["in"] \
                and "ls_seq_0" in self.p["out"] \
                and "ls_seq_1" in self.p["out"]:
            # Call update function.
            current_label = [0 for a in range(self.net["agents"])]
            for a in range(self.net["agents"]):
                current_label[a] = np.argmax(self.dat["variables"]["ls_seq_1"][a, :, 0, 0])
            self.TCM.update_history(self.inputs["ls_pred"],
                                    int(self.frame_cntr),
                                    current_label,
                                    self.current_elapsed,
                                    self.dat["variables"]["_trigger_"])

            # Write updated performances to shared memory.
            self.dat["variables"]["_conf_mat_"][:,:,:] \
                = self.TCM.conf_mat[:,:,:]
            self.dat["variables"]["_acc_"][:,:,:] \
                = self.TCM.accuracy[:,:,:]
