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

from statestream.utils.yaml_wrapper import load_yaml

from statestream.meta.process import process_state, process_trigger
from statestream.meta.network import MetaNetwork
from statestream.meta.synapse_pool import sp_needs_rebuild
from statestream.meta.neuron_pool import np_needs_rebuild
from statestream.meta.plasticity import plast_needs_rebuild

from statestream.utils.shared_memory import SharedMemory
from statestream.meta.network import S2L



class STProcess(object):
    def __init__(self, name, ident, metanet, param):
        """Process wrapper, managing shared memory and syncronization.
        """
        # Initialize process.
        self.name = name
        self.id = ident
        self.param = param
        self.mn = copy.deepcopy(metanet)
        self.net = copy.deepcopy(self.mn.net)
        # Get and set process type.
        self.item_type = None
        if self.name in self.net["neuron_pools"]:
            self.item_type = "np"
        elif self.name in self.net["synapse_pools"]:
            self.item_type = "sp"
        elif self.name in self.net["plasticities"]:
            self.item_type = "plast"
        # Get parameter dictionary.
        if self.item_type is not None:
            self.p = self.net[S2L(self.item_type)][self.name]
            # Get / set device for process.
            self.device = self.p.get("device", "cpu")
            
        # Initialize profiler.
        self.profiler = {
            "read": np.zeros([self.param["core"]["profiler_window"]]),
            "write": np.zeros([self.param["core"]["profiler_window"]])
        }
        
        # Begin with empty dictionary of nps / sps.
        self.nps = {}
        self.sps = {}



    def initialize(self):
        """Process dependent initialization.
        """
        pass
    
    def update_frame_readin(self):
        """Process dependent read update.
        """
        pass

    def update_frame_writeout(self):
        """Process dependent write update.
        """
        pass

    def run(self, IPC_PROC, dummy):
        """Central method of STProcess holding forever loop.
        """
        # Get and set process pid.
        IPC_PROC["pid"][self.id] = os.getpid()
        # Initially set state.
        # This also releases the core from its block after calling run.
        self.state = process_state["C"]
        IPC_PROC["state"][self.id] = self.state
        # Create local references to IPC.
        self.IPC_PROC = IPC_PROC

        # Begin with zero frame.
        self.frame_cntr = 0

        # Get shared memory.
        self.shm = SharedMemory(self.net, 
                                self.param,
                                session_id=IPC_PROC['session_id'].value)

        # Do process specific initialization / compilation.
        self.initialize()

        # Set eigen state to wait after write, and hence always begin with
        # reading.
        self.state = process_state["WaW"]
        IPC_PROC["state"][self.id] = self.state

        # Flag for very first readin.
        red_once = False

        # Enter forever loop.
        while self.state != process_state["E"]:
            # Update state dependent on trigger.
            trigger = IPC_PROC["trigger"].value
            if trigger == process_trigger["-E"] \
                    or IPC_PROC["pid"][self.id] == -1:
                self.state = process_state["E"]
                IPC_PROC["state"][self.id] = self.state
                break
            elif trigger == process_trigger["WaW-R"] \
                    and self.state == process_state["WaW"]:
                self.state = process_state["R"]
            elif trigger == process_trigger["WaR-W"] \
                    and self.state == process_state["WaR"]:
                self.state = process_state["W"]
            elif trigger == process_trigger["-B"]:
                # Load new network file.
                tmp_filename = os.path.expanduser('~') \
                               + '/.statestream/edit_net-' \
                               + str(IPC_PROC['session_id'].value) \
                               + '.st_graph'
                with open(tmp_filename) as f:
                    edited_net = load_yaml(f)
                # Check if this process has to be re-build.
                rebuild = False
                if self.item_type == "sp":
                    if sp_needs_rebuild(self.net, edited_net, self.name):
                        rebuild = True
                elif self.item_type == "np":
                    if np_needs_rebuild(self.net, edited_net, self.name):
                        rebuild = True
                elif self.item_type == "plast":
                    if plast_needs_rebuild(self.net, edited_net, self.name):
                        rebuild = True
                # End or update process as needed.
                if rebuild:
                    self.state = process_state["E"]
                else:
                    # Update local net, even this process is not affected.
                    self.mn = MetaNetwork(edited_net)
                    self.net = copy.deepcopy(self.mn.net)
                    # Get shared memory.
                    self.shm = SharedMemory(self.net, 
                                            self.param,
                                            session_id=int(IPC_PROC["session_id"].value))
                    time.sleep(0.5)

            # Dependent on state, do something.
            if self.state == process_state["R"]:
                # Start timer for reading phase.
                timer_start = time.time()
                # Get frame_cntr.
                self.frame_cntr = IPC_PROC["now"].value

                # Process dependent read actions.
                self.update_frame_readin()

                self.state = process_state["WaR"]
                # End timing of read phase.
                idx = int(self.frame_cntr) \
                      % int(self.param["core"]["profiler_window"])
                self.profiler["read"][idx] \
                    = time.time() - timer_start
                red_once = True
            elif self.state == process_state["W"]:
                # Start timer of write phase.
                timer_start = time.time()

                # Process dependent write actions.
                if red_once:
                    self.update_frame_writeout()
                
                self.state = process_state["WaW"]
                # End timing of write phase.
                idx = int(self.frame_cntr) \
                      % int(self.param["core"]["profiler_window"])
                self.profiler["write"][idx] \
                    = time.time() - timer_start
                # Write profiler information.
                IPC_PROC["profiler"][self.id][0] \
                    = np.mean(self.profiler["read"])
                IPC_PROC["profiler"][self.id][1] \
                    = np.mean(self.profiler["write"])
            elif self.state == process_state["WaR"]:
                time.sleep(0.001)
            elif self.state == process_state["WaW"]:
                # Check for np reset and reset state / pars 
                # in IPC and vars in theano.
                if IPC_PROC["reset"][self.id] == 1:
                    # reset state
                    self.shm.dat[self.name]["state"].fill(0)
                    # TODO: re-init all parameter
                    # TODO: write nice function to init all parameter or specified
                time.sleep(0.001)
            elif self.state == process_state["E"]:
                pass

            # Update ipc state.
            IPC_PROC["state"][self.id] = self.state
