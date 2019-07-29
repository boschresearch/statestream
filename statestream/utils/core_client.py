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



import numpy as np
import os
import copy
import time

from statestream.utils.shared_memory import SharedMemory



class STCClient(object):
    def __init__(self, name, net, param, session_id, IPC_PROC):
        """Process wrapper, managing shared memory and syncronization.
        """
        # Initialize process.
        self.name = name
        self.param = copy.deepcopy(param)
        self.net = copy.deepcopy(net)
        self.shm = SharedMemory(self.net, self.param, session_id=session_id)
        self.IPC_PROC = IPC_PROC

        # Get parameter dictionary.
        self.p = self.net["core_clients"][self.name]

        # Get / set activation flag and start frame.
        self.active = self.p.get("state", False)
        self.start_frame = self.p.get("start frame", 2)

        # Message printed in terminal.
        self.mesg = []
            


    def initialize(self):
        """These instructions will be executed once at the beginning.
        """
        pass



    def interrupt(self):
        """In case the core shuts down this will be executed.
        """
        pass
        

    
    def before_readin(self):
        """These instructions will be executed before read phase.
        """
        pass



    def readin(self):
        """These instructions will be executed during read phase.
        """
        pass



    def writeout(self):
        """These instructions will be executed during write phase.
        """
        pass


