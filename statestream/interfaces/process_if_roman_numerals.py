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

from statestream.utils.pygame_import import pg

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
    return {"in": ["rn_image", "rn_label"], 
            "out": ["rn_pred"]
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
    """Return shared memory layout for roman numeral interface.

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
    shm_layout["parameter"]["fading"] \
        = ShmL("np", (), np.int32, p.get("fading", 4), 1, None)


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
    if "rn_pred" in p["in"]:
        # Add parameter.

        # Determine number of classes, for this we need the real prediction np.
        tmp_target = "rn_pred"
        if "remap" in p:
            if "rn_pred" in p["remap"]:
                tmp_target = p["remap"]["rn_pred"]
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

    

class ProcessIf_roman_numerals(ProcessIf):
    """Interface class providing basic roman numeral data.

        First ten number are provided:
            I, II, III, IV, V, VI, VII, VIII, IX, X

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
        min_duration : int
            The minimum duration a roman numeral will be presented in frames.
            The actual duration will be drawn uniformly between min_duration
            and max_duration.
        max_duration : int
            The maximum duration a roman numeral will be presented in frames.
        fading : int
            The number of neuronal frames to fade in/out a roman numeral.
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
        rn_pred : np.array, shape [agents, 10, 1, 1]
            The roman numeral interface provides a delayed confusion matrix as performance
            measure. To compute this a classification result is needed to be compared
            with the ground-truth.

        Outputs:
        --------
        rn_image : np.array, shape [agents, 1, np_x, np_y]
            The grey-scale roman numeral images.
        rn_label : np.array, shape [agents, 10, *, *]
            The one-hot encoded ground-truth label for the current image. Spatial
            dimensions will be taken from the associated neuron-pool.

        Notes
        -----
        The session wide split parameter does not influence the behavior of this
        interface, because data is generated randomly anyway, hence a split into
        train and test data makes no sense.
    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)



    def draw_sample(self, sample):
        """Draw a new random sample.
        """
        # Set back elapsed.
        self.current_elapsed[sample] = 0
        # Draw new duration.
        self.current_duration[sample] \
            = np.random.randint(self.dat["parameter"]["min_duration"],
                                self.dat["parameter"]["max_duration"] + 1)
        # Draw new numeral.
        self.current_numeral[sample] = np.random.randint(0, self.no_classes)
        # Draw new colors.
        self.current_col_bg[sample] = np.random.rand(3)
        self.current_col_fg[sample] = np.random.rand(3)

        # Draw new globel numeral offset.
        if self.minmax_numeral_offset == 0:
            self.current_n_offset[sample] = [0, 0]
        else:
            self.current_n_offset[sample] \
                = [np.random.uniform(-self.minmax_numeral_offset,
                                     self.minmax_numeral_offset),
                   np.random.uniform(-self.minmax_numeral_offset,
                                     self.minmax_numeral_offset)
                   ]
        # Draw new global numeral scale.
        if self.min_numeral_scale == 1:
            self.current_n_scale[sample] = 1
        else:
            self.current_n_scale[sample] \
                = np.random.uniform(low=self.min_numeral_scale,
                                    high=1.0)
        # Draw new globel numeral blur.
        if self.min_numeral_blur == self.max_numeral_blur:
            self.current_n_blur[sample] = self.min_numeral_blur
        else:
            self.current_n_blur[sample] \
                = np.random.uniform(low=self.min_numeral_blur,
                                    high=self.max_numeral_blur)
        # Determine offset for atoms and their corners and thickness.
        self.current_a_offset[sample] = []
        self.current_c_offset[sample] = []
        self.current_a_thick[sample] = []
        for a in range(self.atoms[self.current_numeral[sample]]):
            if self.minmax_atom_offset == 0:
                offset = [0, 0]
            else:
                offset \
                    = [np.random.uniform(-self.minmax_atom_offset,
                                        self.minmax_atom_offset),
                       np.random.uniform(-self.minmax_atom_offset,
                                        self.minmax_atom_offset)
                       ]
            self.current_a_offset[sample].append(offset)

            if self.minmax_corner_offset == 0:
                c_offset = [[0, 0], [0, 0]]
            else:
                c_offset \
                    = [[np.random.uniform(-self.minmax_corner_offset,
                                        self.minmax_corner_offset),
                       np.random.uniform(-self.minmax_corner_offset,
                                        self.minmax_corner_offset)
                       ],
                       [np.random.uniform(-self.minmax_corner_offset,
                                        self.minmax_corner_offset),
                       np.random.uniform(-self.minmax_corner_offset,
                                        self.minmax_corner_offset)
                       ]]
            self.current_c_offset[sample].append(c_offset)

            if self.min_atom_thick == self.max_atom_thick:
                self.current_a_thick[sample].append(self.max_atom_thick)
            else:
                self.current_a_thick[sample].append(
                    np.random.randint(self.min_atom_thick,
                                        self.max_atom_thick))

        # Set trigger for new stimulus.
        self.dat["variables"]["_trigger_"][sample] = 1



    def update_sample(self, sample):
        """Update a sample.
        """
        self.current_elapsed[sample] += 1
        # Check duration.
        if self.current_elapsed[sample] > self.current_duration[sample]:
            self.draw_sample(sample)
            # Redraw numeral.
            self.draw_numeral(sample)



    def draw_numeral(self, sample):
        """Draw the numeral with its current parametrization.
        """
        # Tabula rasa for this numeral.
        if self.np_shape["rn_image"][0] == 3:
            self.current_image[sample].fill((self.current_col_bg[sample][0],
                                             self.current_col_bg[sample][1],
                                             self.current_col_bg[sample][2]))
        else:
            self.current_image[sample].fill((0,0,0))

        # Draw atoms.
        for a in range(self.atoms[self.current_numeral[sample]]):
            # Determine corner coordinates.
            c0_x = self.corners[self.current_numeral[sample]][a][0][0]
            c0_y = self.corners[self.current_numeral[sample]][a][0][1]
            c1_x = self.corners[self.current_numeral[sample]][a][1][0]
            c1_y = self.corners[self.current_numeral[sample]][a][1][1]
            # Update coordinates with numeral + atom + corner offset.
            c0_x += self.current_n_offset[sample][0] \
                    + self.current_a_offset[sample][a][0] \
                    + self.current_c_offset[sample][a][0][0]
            c0_y += self.current_n_offset[sample][1] \
                    + self.current_a_offset[sample][a][1] \
                    + self.current_c_offset[sample][a][0][1]
            c1_x += self.current_n_offset[sample][0] \
                    + self.current_a_offset[sample][a][0] \
                    + self.current_c_offset[sample][a][1][0]
            c1_y += self.current_n_offset[sample][1] \
                    + self.current_a_offset[sample][a][1] \
                    + self.current_c_offset[sample][a][1][1]
            # Scale coordinates to image resolution.
            c0_x *= self.current_n_scale[sample] * self.np_shape["rn_image"][1]
            c0_y *= self.current_n_scale[sample] * self.np_shape["rn_image"][2]
            c1_x *= self.current_n_scale[sample] * self.np_shape["rn_image"][1]
            c1_y *= self.current_n_scale[sample] * self.np_shape["rn_image"][2]
            # Draw line on surface.
            if self.np_shape["rn_image"][0] == 3:
                col = (int(255 * self.current_col_fg[sample][0]),
                       int(255 * self.current_col_fg[sample][1]),
                       int(255 * self.current_col_fg[sample][2]))
            else:
                col = (255, 255, 255)
            pg.draw.line(self.current_image[sample], 
                         col,
                         (c0_x, c0_y),
                         (c1_x, c1_y),
                         self.current_a_thick[sample][a])
        # TODO: Finally blur all channels.

        # Get image as array.
        img_arr = np.zeros([self.np_shape["rn_image"][1],
                            self.np_shape["rn_image"][2],
                            3], dtype=np.int)
        pg.pixelcopy.surface_to_array(img_arr, self.current_image[sample], kind='P')
        self.current_array[sample] = img_arr[:,:,0] / 256.0



    def initialize(self):
        """Method to initialize (load) the roman numeral interface class.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------
        self.dat["parameter"]["min_duration"] = self.p.get("min_duration", 16)
        self.dat["parameter"]["max_duration"] = self.p.get("max_duration", 24)
        self.dat["parameter"]["fading"] = self.p.get("fading", 1)

        # Some boundaries for randomized numeral images.
        self.minmax_numeral_offset = 0.02
        self.min_numeral_scale = 0.8
        self.min_numeral_blur = 1.0
        self.max_numeral_blur = 2.0

        self.minmax_atom_offset = 0.02
        self.minmax_corner_offset = 0.02
        self.min_atom_thick = 1
        self.max_atom_thick = 3

        # Determine some dimensions.
        self.no_classes = 10
        # Add all outputs as variables.
        self.np_shape = {}
        for o in self.p["out"] + self.p["in"]:
            tmp_target = o
            # Consider remapping.
            if "remap" in self.p:
                if o in self.p["remap"]:
                    tmp_target = self.p["remap"][o]
            # Set shape.
            self.np_shape[o] = np_state_shape(self.net, tmp_target)[1:]

        # Define number of atoms (lines) for each numeral.
        self.atoms = [1, 2, 3, 3, 2, 3, 4, 5, 3, 2]

        # Define corners of atoms for each numeral.
        # self.corners[class][atom][corner=0,1][xy=0,1]
        self.corners = [[] for c in range(self.no_classes)]
        # Define (I).
        self.corners[0].append([[0.5,0], [0.5,1]])
        # Define (II).
        self.corners[1].append([[0.333,0], [0.333,1]])
        self.corners[1].append([[0.667,0], [0.667,1]])
        # Define (III).
        self.corners[2].append([[0.25,0], [0.25,1]])
        self.corners[2].append([[0.5,0], [0.5,1]])
        self.corners[2].append([[0.75,0], [0.75,1]])
        # Define (IV).
        self.corners[3].append([[0.25,0], [0.5,1]])
        self.corners[3].append([[0.75,0], [0.5,1]])
        self.corners[3].append([[0.25,0], [0.25,1]])
        # Define (V).
        self.corners[4].append([[0.25,0], [0.5,1]])
        self.corners[4].append([[0.75,0], [0.5,1]])
        # Define (VI).
        self.corners[5].append([[0.25,0], [0.5,1]])
        self.corners[5].append([[0.75,0], [0.5,1]])
        self.corners[5].append([[0.75,0], [0.75,1]])
        # Define (VII).
        self.corners[6].append([[0,0], [0.25,1]])
        self.corners[6].append([[0.5,0], [0.25,1]])
        self.corners[6].append([[0.5,0], [0.5,1]])
        self.corners[6].append([[0.75,0], [0.75,1]])
        # Define (VIII).
        self.corners[7].append([[0,0], [0.25,1]])
        self.corners[7].append([[0.5,0], [0.25,1]])
        self.corners[7].append([[0.5,0], [0.5,1]])
        self.corners[7].append([[0.75,0], [0.75,1]])
        self.corners[7].append([[1,0], [1,1]])
        # Define (IX).
        self.corners[8].append([[0.25,0], [0.75,1]])
        self.corners[8].append([[0.75,0], [0.25,1]])
        self.corners[8].append([[0.25,0], [0.25,1]])
        # Define (X).
        self.corners[9].append([[0.25,0], [0.75,1]])
        self.corners[9].append([[0.75,0], [0.25,1]])

        # Initialize experimental state for all agents.
        # ---------------------------------------------------------------------
        self.current_duration = [0 for s in range(self.net["agents"])]
        self.current_elapsed = [0 for s in range(self.net["agents"])]
        self.current_numeral = [0 for s in range(self.net["agents"])]
        self.current_col_bg = [0 for s in range(self.net["agents"])]
        self.current_col_fg = [0 for s in range(self.net["agents"])]

        self.current_n_offset = [[] for s in range(self.net["agents"])]
        self.current_n_scale = [0 for s in range(self.net["agents"])]
        self.current_n_blur = [0 for s in range(self.net["agents"])]
        self.current_a_offset = [[] for s in range(self.net["agents"])]
        self.current_c_offset = [[] for s in range(self.net["agents"])]
        self.current_a_thick = [0 for s in range(self.net["agents"])]

        self.current_image = []
        self.current_array = []
        for a in range(self.net["agents"]):
            self.draw_sample(a)
            self.current_image.append(pg.Surface(self.np_shape["rn_image"][1:]))
            self.current_array.append(np.zeros(self.np_shape["rn_image"], 
                                               dtype=np.float32))

        # Instantiate temporal confusion matrix.
        if "rn_pred" in self.p["in"]:
            self.TCM = TemporalConfusionMatrix(self.net, self.name, "rn_pred")
            for a in range(self.net["agents"]):
                self.TCM.trigger_history[a] = [0, int(self.current_numeral[a]),0]



    def update_frame_writeout(self):
        """Method to update the experimental state of the roman numeral interface.
        """
        # Clean labels and trigger.
        if "rn_label" in self.p["out"]:
            self.dat["variables"]["rn_label"] *= 0.0
        if "rn_pred" in self.p["in"]:
            self.dat["variables"]["_trigger_"] *= 0.0
        # Update experimental state for all agents.
        for a in range(self.net["agents"]):
            # Update current experimental state.
            # -----------------------------------------------------------------
            self.update_sample(a)

            # Update internal data (= variables).
            # -----------------------------------------------------------------
            # Update label.
            if "rn_label" in self.p["out"]:
                self.dat["variables"]["rn_label"][a,self.current_numeral[a],:,:] = 1.0
            # Update image.
            if "rn_image" in self.p["out"]:
                if self.dat["variables"]["rn_image"].shape[1] == 1:
                    # Determine fading factor.
                    if self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_elapsed[a]) / float(self.dat["parameter"]["fading"])
                    elif self.current_duration[a] - self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_duration[a] - self.current_elapsed[a]) \
                             / float(self.dat["parameter"]["fading"])
                    else:
                        ff = 1
                    self.dat["variables"]["rn_image"][a,0,:,:] \
                        = ff * self.current_array[a][:,:]
                elif self.dat["variables"]["rn_image"].shape[1] == 3:
                    # Determine fading factor.
                    if self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_elapsed[a]) / float(self.dat["parameter"]["fading"])
                    elif self.current_duration[a] - self.current_elapsed[a] < self.dat["parameter"]["fading"]:
                        ff = float(self.current_duration[a] - self.current_elapsed[a]) \
                             / float(self.dat["parameter"]["fading"])
                    else:
                        ff = 1
                    for c in range(3):
                        self.dat["variables"]["rn_image"][a,c,:,:] \
                            = ff * (self.current_col_fg[a][c] * self.current_array[a][:,:] \
                                    + self.current_col_bg[a][c] * (1 - self.current_array[a][:,:]))

        # Update (if needed) the confusion matrix variable.
        # -----------------------------------------------------------------
        if "rn_pred" in self.p["in"] \
                and "rn_image" in self.p["out"] \
                and "rn_label" in self.p["out"]:

            # Call update function.
            self.TCM.update_history(self.inputs["rn_pred"],
                                    int(self.frame_cntr),
                                    self.current_numeral,
                                    self.current_elapsed,
                                    self.dat["variables"]["_trigger_"])

            # Write updated performances to shared memory.
            self.dat["variables"]["_conf_mat_"][:,:,:] \
                = self.TCM.conf_mat[:,:,:]
            self.dat["variables"]["_acc_"][:,:,:] \
                = self.TCM.accuracy[:,:,:]
