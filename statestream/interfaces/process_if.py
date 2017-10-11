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
import time
import copy
import os
import sys
import importlib

from statestream.utils.pygame_import import pg, pggfx
from statestream.utils.yaml_wrapper import load_yaml
from statestream.meta.process import process_state, process_trigger
from statestream.meta.network import MetaNetwork
from statestream.utils.shared_memory import SharedMemory
from statestream.utils.helper import is_scalar_shape



class ProcessIf(object):
    """General interface class.

    Parameters
    ----------
    name : str
        The unique string identifier for the interface.
    ident : int
        The unique process id for the process of this interface.
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.
    """
    def __init__(self, name, ident, net, param):
        # Initialize process.
        self.name = name
        self.id = ident
        self.param = param
        self.mn = MetaNetwork(net)
        self.net = copy.deepcopy(self.mn.net)
        self.p = self.net["interfaces"][name]
        self.state = process_state["I"]

        # Get and set device.
        self.device = self.net["interfaces"][name].get("device", "cpu")

        # Get / Set interface screen resolution.
        self.screen_width = self.p.get("screen_width", 800)
        self.screen_height = self.p.get("screen_height", 600)

        # Get local representation of shared memory for this type of interface.
        self.type = self.p["type"]
        if_shm_layout \
            = getattr(importlib.import_module("statestream.interfaces.process_if_" \
                                              + self.type), "if_shm_layout")

        # Allocate variables / parameters for local storage.
        self.dat_layout = if_shm_layout(self.name, self.net, self.param)
        self.dat = {}
        for t in ["parameter", "variables"]:
            self.dat[t] = {}
            for i,i_l in self.dat_layout[t].items():
                if i_l.type == "np":
                    if is_scalar_shape(i_l.shape):
                        self.dat[t][i] = np.array([1,], dtype=i_l.dtype)
                    else:
                        self.dat[t][i] = np.zeros(i_l.shape, dtype=i_l.dtype)
        # Set internal state.
        self.state = process_state["I"]
        # Flag for color correction (blue).
        self.ccol = False
        # Initialize profiler.
        self.profiler = {
            "read": np.zeros([self.param["core"]["profiler_window"]]),
            "write": np.zeros([self.param["core"]["profiler_window"]])
        }
        # Some mouse related variables.
        self.LMB_click = False
        self.LMB_clicks = 0
        self.LMB_drag = None
        self.LMB_hold = False
        self.LMB_origin = np.zeros([2,])
        self.LMB_last_click = 0

        # Flag for color correction (blue).
        self.ccol = True
        
        # Begin at zeroth frame.
        self.frame_cntr = 0



    # Color correction for blue-channel.
    def cc(self, col):
        """Deprecated.
        """
        if self.ccol:
            return (col[0]/2, col[1], col[2])
        else:
            return (col[0], col[1], col[2])



    def viz(self):
        # Init if not done.
        if not self.viz_init:
            # Import and initialize pygame here.
            pg.init()
            self.font = pg.font.SysFont("Courier", 20)
            self.font_small = pg.font.SysFont("Courier", 16)
            self.clock = pg.time.Clock()
            pg.mouse.set_visible(1)
            pg.key.set_repeat(1, 100)
            self.background = pg.Surface((self.screen_width,
                                          self.screen_height)).convert()
            self.background.fill(self.cc((0,0,0)))
            self.viz_init = True
        # Get current mouse position.
        self.POS = pg.mouse.get_pos()
        # Catch events.
        self.current_events = pg.event.get()

        for event in self.current_events:
            if event.type == pg.QUIT:
                self.viz_close = True
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.viz_close = True

        # Blit background - tabula rasa
        self.screen.blit(self.background, (0, 0))
            
        # Print interface name.
        self.screen.blit(self.font.render("Visualization of interface " \
            + self.name, 1, self.cc((255,255,255))), (10, 10))

        # Update screen.
        self.update_screen()

        # Flip display.
        pg.display.flip()

    def initialize(self):
        """Interface dependent initialization (e.g. load dataset).
        """
        pass
    
    def update_frame_readin(self):
        """Interface dependent update during read phase.
        """
        pass

    def update_frame_writeout(self):
        """Interface dependent frame update (e.g. show new image).
        """
        pass

    def update_always(self):
        """Interface dependent update in every PROCESS cycle.
        """
        pass

    def update_screen(self):
        """Interface dependent update of interface visualization.
        """
        pass

    def quit(self):
        """Interface dependent quit function called once at end.
        """
        pass

    def run(self, IPC_PROC, dummy):
        """Main method to synchronize interface with core.

        Parameters:
        -----------
        IPC_PROC : dict
            Dictionary of process save shared memory for synchronization.
        """
        # Get set if pid.
        IPC_PROC["pid"][self.id].value = os.getpid()
        # Initially set state.
        IPC_PROC["state"][self.id].value = self.state

        # Get shared memory.
        self.shm = SharedMemory(self.net, 
                                self.param,
                                session_id=IPC_PROC["session_id"].value)

        # Flag for closing viz from if gui.
        self.viz_close = False
        self.viz_init = False
        self.local_viz_init = False
        self.screen = None

        # Initialize interface inputs (necessary for clean read/write).
        self.inputs = {}
        for i in self.p["in"]:
            tmp_target = i
            if "remap" in self.p:
                if i in self.p["remap"]:
                    tmp_target = self.p["remap"][i]
            self.inputs[i] \
                = np.zeros(self.shm.layout[tmp_target]["state"].shape, 
                           dtype=self.shm.layout[tmp_target]["state"].dtype)

        # Initialize split.
        self.split = 0
        self.old_split = 0

        # Do interface specific initialization.
        self.initialize()

        # Set eigen state.
        self.state = process_state["WaW"]
        IPC_PROC["state"][self.id].value = self.state

        # Start forever.
        while self.state != process_state["E"]:
            # Update state dependent on process trigger.
            trigger = IPC_PROC["trigger"].value
            if trigger == process_trigger["-E"]:
                self.state = process_state["E"]
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
                # Update local net, even this process is not affected.
                self.mn = MetaNetwork(edited_net)
                self.net = copy.deepcopy(self.mn.net)
                self.shm = SharedMemory(self.net, 
                                        self.param,
                                        session_id=int(IPC_PROC["session_id"].value))

            # Dependent on state, do something.
            if self.state == process_state["R"]:
                # Start read timer.
                timer_start = time.time()
                # Things to be updated always.
                self.update_always()
                # Get current frame_cntr.
                self.frame_cntr = copy.copy(IPC_PROC["now"].value)
                # Read necessary np states.
                for i in self.p["in"]:
                    tmp_target = i
                    if "remap" in self.p:
                        if i in self.p["remap"]:
                            tmp_target = self.p["remap"][i]
                    self.inputs[i] = np.copy(self.shm.dat[tmp_target]["state"])

                # Read necessary if parameters.
                for par in self.shm.dat[self.name]["parameter"]:
                    if is_scalar_shape(self.shm.layout[self.name]["parameter"][par].shape):
                        self.dat["parameter"][par] \
                            = self.shm.dat[self.name]["parameter"][par][0]
                    else:
                        self.dat["parameter"][par] \
                            = np.copy(self.shm.dat[self.name]["parameter"][par])

                # Read split parameter.
                self.split = int(IPC_PROC["plast split"].value)

                # Handle interface visualization.
                if self.screen is None:
                    if IPC_PROC["if viz"][self.name].value == 1:
                        # Import and initialize pygame here.
                        from statestream.utils.pygame_import import pg, pggfx
                        self.screen = pg.display.set_mode((self.screen_width,
                                                           self.screen_height), 
                                                           pg.SRCALPHA,
                                                           32)
                        # Set initialization flag.
                        self.viz_init = False
                        self.local_viz_init = False
                if self.screen is not None:
                    if IPC_PROC["if viz"][self.name].value == 0:
                        # Quit viz.
                        pg.display.quit()
                        # Reset screen.
                        self.screen = None
                        # reset viz close flag
                        self.viz_close = False
                    else:
                        self.viz()
                if self.viz_close and self.screen is not None:
                    # Quit interface visualization.
                    pg.display.quit()
                    # Reset screen flag.
                    self.screen = None
                    # Reset viz close flag.
                    self.viz_close = False
                    # Reset IPC flag.
                    IPC_PROC["if viz"][self.name].value = 0

                # Update the current frame.
                self.update_frame_readin()
                # Update state.
                self.state = process_state["WaR"]
                # End timing of read phase.
                idx = int(self.frame_cntr) % int(self.param["core"]["profiler_window"])
                self.profiler["read"][idx] = time.time() - timer_start
            elif self.state == process_state["W"]:
                # Start timer for write phase.
                timer_start = time.time()
                # Things to be updated always.
                self.update_always()
                # Update the current frame.
                self.update_frame_writeout()

                # Write update for all variables to shared memory (inl. np states).
                for var in self.shm.dat[self.name]["variables"]:
                    # Check for batch or single mode.
                    if self.dat["parameter"]["mode"] == 1 \
                            and var in self.p["out"]:
                        for a in range(self.net["agents"] - 1):
                            self.dat["variables"][var][a + 1,:,:,:] \
                                = self.dat["variables"][var][0,:,:,:]
                    # Write to shared memory.
                    self.shm.set_shm([self.name, "variables", var],
                                     self.dat["variables"][var])

                self.state = process_state["WaW"]
                # End timing of write phase
                idx = int(self.frame_cntr) % int(self.param["core"]["profiler_window"])
                self.profiler["write"][idx] = time.time() - timer_start
                # Write profiler information to IPC.
                IPC_PROC["profiler"][self.id][0] = np.mean(self.profiler["read"])
                IPC_PROC["profiler"][self.id][1] = np.mean(self.profiler["write"])
            elif self.state in [process_state["WaR"], process_state["WaW"]]:
                # Things to be updated always.
                self.update_always()
                time.sleep(0.001)
            elif self.state == process_state["E"]:
                # Cleanup.
                pass

            # Update ipc state.
            IPC_PROC["state"][self.id].value = self.state

        # Call interface dependent quit function.
        self.quit()

