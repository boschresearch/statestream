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
import os
from time import gmtime, strftime
import copy
import matplotlib.image as Image

from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.utils.core_client import STCClient



class CClient_save_images(STCClient):
    """This is a client to save some NP states as images.
    """
    def __init__(self, name, net, param, session_id, IPC_PROC):
        # Initialize parent ProcessIf class.
        STCClient.__init__(self, name, net, param, session_id, IPC_PROC)



    def initialize(self):
        """Method to initialize this core client type.
        """
        # Get some core client parameters.
        # ---------------------------------------------------------------------
        self.items = copy.deepcopy(self.p['items'])
        self.save_path = self.p['save_path'] + os.sep \
                         + strftime("%a-%d-%b-%Y-%H-%M-%S", gmtime())

        # Make directory for every item.
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        for i in self.items:
            if not os.path.isdir(self.save_path + os.sep + i):
                os.makedirs(self.save_path + os.sep + i)

        # Generate internal structure to store data
        self.store_data = {}
        for i,I in self.items.items():
            item_shape = self.net['neuron_pools'][i]['shape']
            self.store_data[i] = [np.zeros(item_shape, dtype=np.float32) for i in range(I['offset'] + 1)]

        self.current_frame = 0

    def readin(self):
        """Method to store current state of items internally.
        """
        self.current_frame += 1
        # Update data storage.
        for i,I in self.items.items():
            current_frame = self.current_frame % (I['offset'] + 1)
            self.store_data[i][current_frame][:] \
                = self.shm.dat[i]["state"][0,:]

    def writeout(self):
        """Method to save stored data to images.
        """
        for i,I in self.items.items():
            current_frame = (self.current_frame - I['offset']) % (I['offset'] + 1)
            savefile = self.save_path + os.sep + i + os.sep + str(self.current_frame).rjust(8, '0') + '.png'
            item_shape = self.net['neuron_pools'][i]['shape']
            if item_shape[0] == 1:
                dat = self.store_data[i][current_frame][0,:,:]
            elif item_shape[0] == 2:
                dat = None
            elif item_shape[0] == 3:
                dat = np.swapaxes(self.store_data[i][current_frame], 0, 2)
            Image.imsave(savefile, dat)
