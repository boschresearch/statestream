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

from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.utils.core_client import STCClient



class CClient_lr_scheduler(STCClient):
    """This is a client to schedule the learning rate.
    """
    def __init__(self, name, net, param, session_id, IPC_PROC):
        # Initialize parent ProcessIf class.
        STCClient.__init__(self, name, net, param, session_id, IPC_PROC)



    def initialize(self):
        """Method to initialize this core client type.
        """
        # Get some core client parameters.
        # ---------------------------------------------------------------------
        self.lr_schedule = self.p.get("schedule", "constant")
        self.plasticity = self.p["plast"]
        if self.lr_schedule == "exponential_decay":
            self.exp_decay = self.p.get("decay", 0.99)



    def before_readin(self):
        """Method is executed before readin phase.
        """
        if self.lr_schedule == "exponential_decay":
            self.shm.dat[self.plasticity]["parameter"]["lr"] *= self.exp_decay
