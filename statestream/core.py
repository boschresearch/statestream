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



from __future__ import print_function

import os
import sys

import statestream as sstream
SSTREAMPATH=os.path.dirname(sstream.__file__)
sys.argv = [os.path.abspath(a) for a in sys.argv]
os.chdir(SSTREAMPATH)

from time import sleep, gmtime, strftime, time
import copy
import multiprocessing as mp
import numpy as np
import ctypes
import importlib
import h5py
import matplotlib
matplotlib.use('agg')


try:
    import pickle as pckl
except:
    try:
        import cPickle as pckl
    except:
        pckl = None


import statestream.meta.network as mn
from statestream.meta.process import process_state, process_trigger
from statestream.meta.neuron_pool import np_needs_rebuild
from statestream.meta.synapse_pool import sp_needs_rebuild
from statestream.meta.plasticity import plast_needs_rebuild

from statestream.processes.process_np import ProcessNp
from statestream.processes.process_sp import ProcessSp
from statestream.processes.process_plast import ProcessPlast
from statestream.processes.process_sys_client import ProcessSysClient

from statestream.utils.core_client import STCClient
from statestream.utils.defaults import DEFAULT_CORE_PARAMETER
from statestream.utils.helper import is_float_dtype
from statestream.utils.helper import is_int_dtype
from statestream.utils.helper import is_scalar_shape
from statestream.utils.keyboard import Keyboard
from statestream.utils.save_load_network import save_network
from statestream.utils.shared_memory import SharedMemory
from statestream.utils.tmem import update_tmem
from statestream.utils.tikz_graph_generator import generate_rollout_graph
from statestream.utils.yaml_wrapper import load_yaml, dump_yaml



class StateStream(object):
    def __init__(self):
        # Create local statestream path in home if not yet existing.
        self.home_path = os.path.expanduser('~')
        if not os.path.isdir(self.home_path + '/.statestream'):
            os.makedirs(self.home_path + '/.statestream')
        # Create theano compile meta directory.
        if not os.path.isdir(self.home_path + '/.statestream/compiledirs'):
            os.makedirs(self.home_path + '/.statestream/compiledirs')

        # Begin with empty parameter dictionary.
        self.param = {}
        
        # Load core parameters.
        # ---------------------------------------------------------------------
        # Read local system settings if given, else use defaults.
        tmp_filename = self.home_path + '/.statestream/stcore.yml'
        if not os.path.isfile(tmp_filename):
            # Write default core parameters to file.
            with open(tmp_filename, 'w+') as f:
                dump_yaml(DEFAULT_CORE_PARAMETER, f)
                # Use default parameter dictionary
                self.param["core"] = DEFAULT_CORE_PARAMETER
        else:
            # Load core parameters from file.
            with open(tmp_filename) as f:
                tmp_dictionary = load_yaml(f)
                # Check if core parameter file is empty.
                if tmp_dictionary is None:
                    print("Warning: Found empty core parameter file ~/.statestream/stcore.yml")
                    tmp_dictionary = {}
                # Create core parameters from default and loaded.
                self.param["core"] = {}
                for p,P in DEFAULT_CORE_PARAMETER.items():
                    self.param["core"][p] = tmp_dictionary.get(p, P)

        # Load network parameters.
        # ---------------------------------------------------------------------
        # Read graph file.
        self.flag_load_at_start = False
        if len(sys.argv) not in [1, 2]:
            print("Error: Expected call: python core.py [st_net or st_graph file] or python core.py for initialization.")
            sys.exit()
        else:
            if len(sys.argv) == 2:
                # Check if st_net or st_graph file.
                if len(sys.argv[1]) > 10:
                    if sys.argv[1].endswith(".st_graph"):
                        with open(sys.argv[1]) as f:
                            tmp_net = load_yaml(f)
                            tmp_net["__source_root__"] = [sys.argv[1][:]]
                            self.mn = mn.MetaNetwork(tmp_net)
                    elif sys.argv[1].endswith(".st_net"):
                        # Set flag to initialize with loaded for later.
                        self.flag_load_at_start = True
                        self.load_at_start_filename = sys.argv[1]
                        # Load st_net file and get graph dictionary.
                        with open(sys.argv[1], "rb") as f:
                            # Load it here only for graph initialization.
                            loadList = pckl.load(f)
                            loadList[0][1]["__source_root__"] = [sys.argv[1][:]]
                            # Get network graph.
                            self.mn = mn.MetaNetwork(loadList[0][1])
                    else:
                        print("Error: Invalid filename ending. Expected .st_net or .st_graph.")
                        sys.exit()
                else:
                    print("Error: Source filename is too short.")
                    sys.exit()
            else:
                sys.exit()

        # Check module specification for sanity.
        if not mn.is_sane_module_spec(self.mn.net):
            sys.exit()

        # Check sanity of meta.
        if not self.mn.is_sane():
            sys.exit()

        # Generate meta network with ids.
        self.net = self.mn.net

        # Generate compileable tikz file.
        if not os.path.isdir(self.param["core"]["save_path"] + os.sep + "graph_tikz"):
            os.makedirs(self.param["core"]["save_path"] + os.sep + "graph_tikz")
        tikz_file = self.param["core"]["save_path"] + os.sep \
                    + "graph_tikz" + os.sep + self.net["name"] + ".tex"
        generate_rollout_graph(given_net=self.net, savefile=tikz_file)

        # Set float counter to zero.
        self.floats = 0
        # Generate ipc for process handling.
        self.IPC_PROC = {}
        # Add trigger ipc:
        #   trigger == 0 :: switch "WaW" -> "R"
        #   trigger == 1 :: switch "WaR" -> "W"
        #   trigger == 2 :: switch       -> "E"
        self.IPC_PROC["trigger"] = mp.Value("d", 1)
        self.IPC_PROC["now"] = mp.Value("d", 0)
        self.IPC_PROC["break"] = mp.Value("d", 1)
        self.IPC_PROC["one-step"] = mp.Value("d", 0)
        # Save / load message (considering initial load from file).
        if self.flag_load_at_start:
            self.IPC_PROC["save/load"] = mp.Value("d", 2)
        else:
            self.IPC_PROC["save/load"] = mp.Value("d", 0)
        # General string message (esp. for viz -> core).
        self.IPC_PROC["string"] = mp.Array("d", [0 for i in range(256)])
        self.string = np.zeros([256], dtype=np.uint8)
        # Instruction len (len of IPC_PROC["string"]) and instruction flag.
        self.IPC_PROC["instruction len"] = mp.Value("d", 0)
        self.IPC_PROC["instruction"] = mp.Value("d", 0)
        # Index of Agent Of Interest.
        self.IPC_PROC["AOI"] = mp.Value("d", 0)
        # Index of Agent Of Interest.
        self.IPC_PROC["plast split"] = mp.Value("d", 0)
        # Flag to reset entire network to initial state (1 for initial state).
        # TODO: reset to tmem_state
        self.IPC_PROC["net reset"] = mp.Value("d", 0)
        # Add ipc for process management.
        #   "I", "W", "R", "WaW", "WaR", "E"
#        self.IPC_PROC["state"] = mp.Value("d", process_state["I"]) for i in range(self.param["core"]["max_processes"])]
        self.IPC_PROC["state"] = mp.Array("d", [process_state["I"] for i in range(self.param["core"]["max_processes"])])
        # The system process identifier for each process.
#        self.IPC_PROC["pid"] = [mp.Value("d", 0) for i in range(self.param["core"]["max_processes"])]
        self.IPC_PROC["pid"] = mp.Array("d", [0 for i in range(self.param["core"]["max_processes"])])
        # Temporal period of process.
#        self.IPC_PROC["period"] = [mp.Value("d", 0) for i in range(self.param["core"]["max_processes"])]
        self.IPC_PROC["period"] = mp.Array("d", [0 for i in range(self.param["core"]["max_processes"])])
        # Temporal offset to temporal period of process.
        self.IPC_PROC["period offset"] \
            = mp.Array("d", [0 for i in range(self.param["core"]["max_processes"])])
        # Pause flag for each process.
        # 0 :: run, 1 :: pause, 2 :: off
        self.IPC_PROC["pause"] = mp.Array("d", [0 for i in range(self.param["core"]["max_processes"])])
        self.IPC_PROC["reset"] = mp.Array("d", [0 for i in range(self.param["core"]["max_processes"])])
        self.IPC_PROC["profiler"] \
            = [mp.Array("d", [0, 0]) for i in range(self.param["core"]["max_processes"])]
        self.IPC_PROC["if viz"] = {}
        for i in self.net["interfaces"]:
            self.IPC_PROC["if viz"][i] = mp.Value("d", 0)
            
        # GUI and CORE processes are handled separately
        # GR 1*** :: shutdown
        self.IPC_PROC["gui request"] = mp.Array("d", [0, 0, 0, 0])
        self.IPC_PROC["gui pid"] = mp.Value("d", 0)
        self.IPC_PROC["gui flag"] = mp.Value("d", 0)
        self.IPC_PROC["rvgui pid"] = mp.Value("d", 0)
        self.IPC_PROC["rvgui flag"] = mp.Value("d", 0)
        self.IPC_PROC["core pid"] = mp.Value("d", 0)
        self.IPC_PROC["delay"] = mp.Value("d", 0)
        self.IPC_PROC["tmem time steps"] \
            = mp.sharedctypes.RawArray(ctypes.c_float, len(self.param["core"]["temporal_memory"]))

        # Create shared memory for statestream and attach instance for core.
        self.shm = SharedMemory(self.net, self.param)
        # Get / set session id.
        self.IPC_PROC["session_id"] = mp.Value("i", self.shm.session_id)
        print("Opened session id: " + str(self.shm.session_id))

        # Write shared memory log.
        self.shm_logfile = self.home_path + '/.statestream/shm-' \
            + str(int(self.IPC_PROC["session_id"].value)) + '.log'
        with open(self.shm_logfile, "w+") as f:
            idx = np.argsort(self.shm.log_bytes)
            for l in range(len(self.shm.log_bytes)):
                i = len(self.shm.log_bytes) - l - 1
                if self.shm.log_bytes[idx[i]] < 2**10:
                    b = str(int(self.shm.log_bytes[idx[i]])) + " B"
                elif self.shm.log_bytes[idx[i]] >= 2**10 and self.shm.log_bytes[idx[i]] < 2**20:
                    b = str(int(self.shm.log_bytes[idx[i]] / 2**10)) + " kB"
                elif self.shm.log_bytes[idx[i]] >= 2**20 and self.shm.log_bytes[idx[i]] < 2**30:
                    b = str(int(self.shm.log_bytes[idx[i]] / 2**20)) + " MB"
                f.write(format(b.ljust(12) + self.shm.log_lines[idx[i]]) + '\n')

        # Create identifier for every item that needs a process.
        print("Create identifiers for processes ...")
        self.proc_id = {}
        self.proc_name = {}
        self.proc_type = {}
        self.device = {}
        self.bottleneck = {}
        current_id = 0
        for t in ["np", "sp", "plast", "if"]:
            if mn.S2L(t) in self.net:
                for i,I in self.net[mn.S2L(t)].items():
                    # In case of SP, determine if this SP really needs its own process.
                    # I.e., iff SP parameters receive updates.
                    needs_process = False
                    if t == "sp":
                        for p, P in self.net["plasticities"].items():
                            # par[0]: np or sp
                            # par[1]: np/sp name
                            # par[2]: parameter name
                            for par in P["parameter"]:
                                if par[0] == "sp":
                                    if "share params" in self.net["synapse_pools"][par[1]]:
                                        for spar, SPAR in self.net["synapse_pools"][par[1]]["share params"].items():
                                            if SPAR[0] == i:
                                                needs_process = True
                                    # or if self is this sp.
                                    elif par[1] == i:
                                        # Add param only if not shared.
                                        if "share params" in self.net["synapse_pools"][i]:
                                            if par[2] not in self.net["synapse_pools"][i]["share params"]:
                                                needs_process = True
                                        else:
                                            needs_process = True
                    else:
                        needs_process = True

                    # If needed, prepare process for item.
                    if needs_process:
                        self.shm.proc_id[i][0] = current_id
                        self.proc_id[i] = int(current_id)
                        self.proc_name[int(current_id)] = copy.copy(i)
                        self.proc_type[i] = t
                        self.IPC_PROC["period"][current_id] = I.get("period", 1)
                        self.IPC_PROC["period offset"][current_id] = I.get("period offset", 0)
                        self.IPC_PROC["pause"][current_id] = I.get("at start", 0)
                        current_id += 1
                    else:
                        self.shm.proc_id[i][0] = -1
                        self.proc_id[i] = -1

                    # Not needed here, but remember also devices for processes.
                    self.device[i] = I.get("device", "cpu")
                    # Get / set bottleneck factor.
                    self.bottleneck[i] = I.get("bottleneck", -1)
        print("    ... done creating identifiers for processes.")

        # Initialize temporal memory for everything.
        self.tmem_updates = [0 for i in range(len(self.param["core"]["temporal_memory"]))]
        self.tmem_update = [False for i in range(len(self.param["core"]["temporal_memory"]))]
        self.tmem_time_steps = [0 for i in range(len(self.param["core"]["temporal_memory"]))]

        # Create instances for processes.
        print("Create instances for " + str(current_id) + " processes ...")
        self.inst = {}
        for p,p_id in self.proc_id.items():
            if p_id != -1:
                if self.proc_type[p] == "np":
                    self.inst[p] = ProcessNp(p, p_id, self.mn, self.param)
                elif self.proc_type[p] == "sp":
                    self.inst[p] = ProcessSp(p, p_id, self.mn, self.param)
                elif self.proc_type[p] == "plast":
                    self.inst[p] = ProcessPlast(p, p_id, self.mn, self.param)
                elif self.proc_type[p] == "if":
                    I = self.net["interfaces"][p]
                    class_name = "ProcessIf_" + I["type"]
                    ProcIf = getattr(importlib.import_module("statestream.interfaces.process_if_" \
                                                             + I["type"]), 
                                     class_name)
                    self.inst[p] = ProcIf(p, p_id, self.mn, self.param)
        print("    ... done creating instances for processes.")

        # Instantiate core clients.
        self.cclients = {}
        if "core_clients" in self.net:
            for cc,CC in self.net["core_clients"].items():
                CClientConst = getattr(importlib.import_module("examples.core_clients." + CC["type"]), 
                                       "CClient_" + CC["type"])
                self.cclients[cc] = CClientConst(cc, self.net, self.param, self.shm.session_id, self.IPC_PROC)

        # Set initial frame counter to zero.
        self.frame_cntr = 0

        self.state = process_state["I"]
        self.shutdown = False
        self.gui_shutdown = False
        self.rvgui_shutdown = False

        # Get keyboard control.
        self.terminal = Keyboard()
        self.terminal_input = ""
        self.terminal_command = ""
        self.terminal_current_show = ""
        self.terminal_current_shown = False
        self.terminal_clean = False

        # Try to generate logging director for manipulations.
        if not os.path.isdir(self.home_path + '/.statestream/log_manipulation'):
            os.makedirs(self.home_path + '/.statestream/log_manipulation')
        # Set brainview save file.
        self.manip_logfile = self.home_path + '/.statestream/log_manipulation/' \
                             + self.net['name'] + '.manip_log'
        
        print("Done initialising core. Shared memory allocated: ~" \
              + str(int(np.sum(self.shm.log_bytes) / 2**20)) + " [MB]")


    def get_free_proc_id(self):
        """Return the next free process id.
        """
        proc_ids = []
        proc_id = None
        # Collect all proc ids.
        for p,p_id in self.proc_id.items():
            proc_ids.append(p_id)
        for i in range(self.param["core"]["max_processes"]):
            if i not in proc_ids:
                proc_id = copy.copy(i)
                break
        return int(proc_id)


    def get_proc_name_by_id(self, proc_id):
        """Return process with the given PID.
        """
        proc_name = None
        for p,p_id in self.proc_id.items():
            if proc_id == p_id:
                proc_name = copy.copy(p)
                break
        return proc_name


    def register_sys_client(self, client_param):
        """Register a new system client.
        """
        # Get internal PID for new process.
        proc_id = self.get_free_proc_id()
        # Append to some housekeeping structures.
        self.proc_id[client_param['name']] = proc_id
        self.proc_name[proc_id] = client_param['name']
        self.proc_type[client_param['name']] = 'client'
        self.device[client_param['name']] = client_param.get('device', 'cpu')
        self.bottleneck[client_param['name']] = client_param.get('bottleneck', -1)
        # Update shared memory.
        self.shm.add_sys_client(client_param)
        self.shm.update_sys_client()
        # Create process, start it and wait for initial handshake.
        self.inst[client_param['name']] = ProcessSysClient(client_param['name'], 
                                                           proc_id, 
                                                           self.mn, 
                                                           self.param,
                                                           client_param)
        self.proc[client_param['name']] = mp.Process(target=self.inst[client_param['name']].run,
                                                     args=(self.IPC_PROC, []))
        self.IPC_PROC["period"][proc_id] = 1
        self.IPC_PROC["period offset"][proc_id] = 0
        self.proc[client_param['name']].start()
        while self.IPC_PROC["state"][proc_id] == process_state["I"]:
            sleep(0.001)
        # Store pid of currently started process.
        self.PIDS[client_param['name']] \
            = int(self.IPC_PROC["pid"][proc_id])
        # Update PIDs logfile.
        with open(self.home_path + "/.statestream/pid.log", "w+") as f:
            dump_yaml(self.PIDS, f)


    def remove_sys_client(self, client_name):
        """Remove a single system client process.
        """
        if client_name in self.proc_id:
            # Send specific terminasion signal to client process.
            self.IPC_PROC["pid"][self.proc_id[client_name]] = -1
            self.proc[client_name].join()
            for p_id, p_name in self.proc_name.items():
                if p_name == client_name:
                    self.proc_name.pop(p_id)
                    break
            self.proc_id.pop(client_name)
            self.proc_type.pop(client_name)
            self.device.pop(client_name)
            self.bottleneck.pop(client_name)
            self.proc.pop(client_name)
            self.inst.pop(client_name)
            self.shm.remove_sys_client(client_name)
            self.shm.update_sys_client()


    def write_manip_log(self, entry):
        """Write the current manipulation log to file.
        """
        with open(self.manip_logfile, "a+") as f:
            f.write(entry + "\n")


    def init_from_extern(self, what=None):
        """Initialize model parts from external sources.
        """
        # If no target provided, initialize everything due to specification.
        if what is None:
            # Load models from specification.
            if "external_models" in self.net:
                models = {}
                config = {}
                for m, M in self.net["external_models"].items():
                    if M["type"] == "keras":
                        # Load model file.
                        models[m] = h5py.File(M["model_file"], 
                                              mode='r')
                        # Instantiate model configuration.
                        config[m] = models[m].attrs.get('model_config')
                        #config[m] = json.loads(self.model_config.decode('utf-8'))
                # Load external parameter.
                for t in ["neuron_pools", "synapse_pools"]:
                    for i, I in self.net[t].items():
                        for P in I:
                            # Check that we have an init entry and it is a list.
                            if not P.startswith("init ") \
                                    or not isinstance(I[P], list):
                                continue
                            # Check that first list entry for init is a loaded model.
                            if I[P][0] not in models:
                                continue
                            # Get name of to be initialized parameter.
                            p = P.split()[1]
                            # Get model id.
                            m = I[P][0]
                            # Get source layer and parameter ids of external source.
                            layer = I[P][1]
                            param = I[P][2]
                            # Dependent on model type get parameter value.
                            if self.net["external_models"][m]["type"] == "keras":
                                val = np.array(models[m]["model_weights"][layer][param], 
                                               dtype=np.float32)
                            # Add dimensions for W.
                            if p[0] == "W":
                                if len(val.shape) == 2:
                                    val = val[:,:,np.newaxis,np.newaxis]
                                    val = np.swapaxes(val, 0, 1)
                            print("\n KP: " + str(param) + "  " + str(val.shape))
                            # Set shared memory to value.
                            self.shm.set_shm([i, "parameter", p], val)

        # TODO: Enable also init of WHAT from external.




    def run(self):
        """This is the main core routine.
        """

        # Init from extern is False in the beginning.
        init_from_extern_once = False
        # Get and set core pid.
        self.IPC_PROC["core pid"].value = os.getpid()
        # Get instances of processes.
        print("Start " + str(len(self.proc_id)) + " processes ...")
        self.proc = {}
        for p in self.proc_id:
            if self.proc_id[p] != -1:
                self.proc[p] = mp.Process(target=self.inst[p].run,
                                          args=(self.IPC_PROC, []))
        print("   ... done starting processes.")

        # initialize core profiler with zeros
        self.profiler_core_read = np.zeros([self.param["core"]["profiler_window"],])
        self.profiler_core_write = np.zeros([self.param["core"]["profiler_window"],])
        self.profiler_core_overall = np.zeros([self.param["core"]["profiler_window"],])

        # Dictionary storing all system process ids.
        self.PIDS = {}
        
        # Get pids of CORE process.
        self.PIDS["_core_"] = int(self.IPC_PROC["core pid"].value)
        # Write all pids to log file.
        with open(self.home_path + "/.statestream/pid.log", "w+") as f:
            dump_yaml(self.PIDS, f)
            f.close()

        # At the very beginning initialize all processes, starting with zeroth.
        pending_procs = len(self.proc_id)

        # Start with no delay.
        self.delay = 0.0
        self.delayed = False

        # Start with empty command line.
        self.current_line = ""

        # Execute initialization method for all core clients.
        for cc,CC in self.cclients.items():
            CC.initialize()

        print("Core enters forever loop.")
        
        # Enter forever loop.
        while self.state != process_state["E"]:
            # Check for pending process startups.
            if pending_procs != 0:
                pending_procs = 0
                for p,p_id in self.proc_id.items():
                    if p_id != -1:
                        if self.IPC_PROC["state"][p_id] == process_state["I"]:
                            self.proc[p].start()
                            while self.IPC_PROC["state"][p_id] == process_state["I"]:
                                sleep(0.001)
                            # Store pid of currently started process.
                            self.PIDS[p] \
                                = (int(self.IPC_PROC["pid"][p_id]))
                            # Update pending process counter.
                            pending_procs += 1
                        elif self.IPC_PROC["state"][p_id] \
                                != process_state["WaW"]:
                            pending_procs += 1
                        # Write all pids to log file.
                        with open(self.home_path + "/.statestream/pid.log", "w+") as f:
                            dump_yaml(self.PIDS, f)

            # Start timer for loop cycle.
            timer_start_overall = time()
            # Check if all processes are in the same state.
            WaR_shared = True
            WaW_shared = True
            for p,p_id in self.proc_id.items():
                if p_id != -1:
                    if WaR_shared:
                        if self.frame_cntr % self.IPC_PROC["period"][p_id] \
                                == self.IPC_PROC["period offset"][p_id]:
                            if self.IPC_PROC["state"][p_id] \
                                    != process_state["WaR"]:
                                WaR_shared = False
                    if WaW_shared:
                        if self.frame_cntr % int(self.IPC_PROC["period"][p_id]) \
                                == self.IPC_PROC["period offset"][p_id]:
                            if self.IPC_PROC["state"][p_id] \
                                    != process_state["WaW"]:
                                WaW_shared = False

            # Check if WaR_shared.
            if WaR_shared and self.IPC_PROC["trigger"].value \
                    != process_trigger["WaR-W"] and not self.delayed:
                # Start timer for core write.
                timer_start_write = time()
                # Update overall frame counter.
                self.frame_cntr += 1
                self.IPC_PROC["now"].value = self.frame_cntr
                # Trigger write for all processes.
                self.IPC_PROC["trigger"].value = process_trigger["WaR-W"]
                # Execute writeout method for all clients.
                for cc,CC in self.cclients.items():
                    if self.frame_cntr == CC.start_frame:
                        CC.active = True
                    if CC.active:
                        CC.writeout()
                # End timer for write phase.
                self.profiler_core_write[int(self.frame_cntr) % int(self.param["core"]["profiler_window"])] \
                    = time() - timer_start_write

            # Check if WaW_shared.
            if WaW_shared and self.IPC_PROC["trigger"].value \
                    != process_trigger["WaW-R"] and not self.delayed:
                # Start timer for core read.
                timer_start_read = time()
                # For very first time initialize everthing incl. external sources.
                if not init_from_extern_once:
                    # Very first initialization.
                    self.shm.init()
                    # Init from external models.
                    self.init_from_extern()
                    # Do not initialize a second time again.
                    init_from_extern_once = True
                    
                # Evaluate instructions.
                if self.IPC_PROC["instruction"].value == 1: 
                    # Get instruction.
                    instruction_len = int(self.IPC_PROC["instruction len"].value)
                    self.string[0:instruction_len] = self.IPC_PROC["string"][0:instruction_len]
                    # Set back instruction flag.
                    self.IPC_PROC["instruction"].value = 0
                    # Convert ascii to str.
                    instruction = ""
                    for c in range(instruction_len):
                        instruction += chr(self.string[c])
                    # Evaluate instruction.
                    cmd_line = instruction.split()
                    if len(cmd_line) > 0:
                        # General setter for parameter.
                        if cmd_line[0].startswith("set") and len(cmd_line) == 6:
                            # Add logging entry.
                            self.write_manip_log(str(self.shm.session_id) + " " + str(self.frame_cntr) \
                                                     + strftime(" %a-%d-%b-%Y-%H-%M-%S ", gmtime())
                                                     + instruction)
                            shml = self.shm.layout[cmd_line[1]][cmd_line[2]][cmd_line[3]]
                            if is_scalar_shape(shml.shape):
                                if is_int_dtype(shml.dtype):
                                    if cmd_line[4] == "int":
                                        self.shm.set_shm([cmd_line[1], cmd_line[2], cmd_line[3]],
                                                         int(cmd_line[5]))
                                if is_float_dtype(shml.dtype):
                                    if cmd_line[4] == "float":
                                        self.shm.set_shm([cmd_line[1], cmd_line[2], cmd_line[3]],
                                                         float(cmd_line[5]))
                        elif cmd_line[0] == "register_sys_client":
                            # Load client parameters and register system client.
                            tmp_filename = os.path.expanduser('~') \
                                           + '/.statestream/system_client_parameter_' \
                                           + str(int(self.IPC_PROC['session_id'].value)) \
                                           + '.yml'
                            with open(tmp_filename) as f:
                                self.register_sys_client(load_yaml(f))
                        elif cmd_line[0] == "remove_sys_client":
                            self.remove_sys_client(cmd_line[1])
                        elif cmd_line[0] == "init":
                            # General re-initialization of parts of the network.
                            if len(cmd_line) == 1:
                                # Re-init entire network.
                                init_from_extern_once = False
                            else:
                                # Re-init all items given by their process ids.
                                for i in range(len(cmd_line) - 1):
                                    proc_id_tmp = int(cmd_line[i + 1])
                                    # Search for proces with this proc_id.

                                    # Re-init shared memory for this process.
                        elif cmd_line[0] == "edit":
                            # Wait for all processes to get in same state.
                            sleep(1.0)
                            # General editing with necessary rebuilds of parts of the network.
                            # Generate st_graph filename and load it.
                            tmp_filename = os.path.expanduser('~') \
                                           + '/.statestream/edit_net-' \
                                           + str(self.IPC_PROC['session_id'].value) \
                                           + '.st_graph'
                            with open(tmp_filename) as f:
                                self.edited_net = load_yaml(f)
                            # (For now) create new shm entirely (hence destroy all trained
                            # parameters).
                            self.shm.delete()
                            self.shm = SharedMemory(self.edited_net, 
                                                    self.param, 
                                                    force_id=self.IPC_PROC['session_id'].value)
                            init_from_extern_once = False
                            # Send trigger to all processes to reload network file.
                            trigger = copy.copy(self.IPC_PROC["trigger"].value)
                            self.IPC_PROC["trigger"].value = process_trigger["-B"]
                            # Re-join with to be re-build processes.
                            rebuild_proc_ids = []
                            for s, S in self.net["synapse_pools"].items():
                                if sp_needs_rebuild(self.net, self.edited_net, s):
                                    proc_idx = self.proc_id[s]
                                    rebuild_proc_ids.append(proc_idx)
                                    self.proc[s].join()
                                    # Re-init process instance.
                                    self.inst[s] = ProcessSp(s, 
                                                             proc_idx, 
                                                             self.mn, 
                                                             self.param)
                                    self.proc[s] = mp.Process(target=self.inst[s].run,
                                                              args=(self.IPC_PROC, []))
                                    self.IPC_PROC["state"][proc_idx] = 0
                            for n, N in self.net["neuron_pools"].items():
                                if np_needs_rebuild(self.net, self.edited_net, n):
                                    proc_idx = self.proc_id[n]
                                    rebuild_proc_ids.append(proc_idx)
                                    self.proc[n].join()
                                    # Re-init process instance.
                                    self.inst[n] = ProcessNp(n, 
                                                             proc_idx, 
                                                             self.mn, 
                                                             self.param)
                                    self.proc[n] = mp.Process(target=self.inst[n].run,
                                                              args=(self.IPC_PROC, []))
                                    self.IPC_PROC["state"][proc_idx] = 0
                            for p, P in self.net["plasticities"].items():
                                if plast_needs_rebuild(self.net, self.edited_net, p):
                                    proc_idx = self.proc_id[p]
                                    rebuild_proc_ids.append(proc_idx)
                                    self.proc[p].join()
                                    # Re-init process instance.
                                    self.inst[p] = ProcessPlast(p, 
                                                                proc_idx, 
                                                                self.mn, 
                                                                self.param)
                                    self.proc[p] = mp.Process(target=self.inst[p].run,
                                                              args=(self.IPC_PROC, []))
                                    self.IPC_PROC["state"][proc_idx] = 0
                            # Reset trigger.
                            self.IPC_PROC["trigger"].value = trigger
                            # Finally overwrite own net and meta-net.
                            self.mn = mn.MetaNetwork(self.edited_net)
                            self.net = self.mn.net
                            # Begin to startup rebuild processes.
                            pending_procs = len(rebuild_proc_ids)

                        elif len(cmd_line[-1]) >= 2:
                            # Check if last string is a number (parameter).
                            if not cmd_line[-1][-1].isdigit():
                                print("\nCORE RECEIVED INST: " + str(cmd_line))
                                if cmd_line[0].startswith("reset") and len(cmd_line) >= 5:
                                    # Get target of reset.
                                    reset_target_idx = 2
                                    while cmd_line[reset_target_idx][0] in "#!":
                                        reset_target_idx += 1
                                    reset_target = cmd_line[reset_target_idx][0:-1]
                                    # Get type of update (state, par).
                                    if cmd_line[-2].startswith("state"):
                                        self.shm.init([reset_target, "state"], mode=None)
                                    elif cmd_line[-3].startswith("par"):
                                        par = cmd_line[-2][0:-1]
                                        self.shm.init([reset_target, "parameter", par], mode=None)
                                    if cmd_line[-2].startswith("period"):
                                        # Get / set new period and offset for process.
                                        proc_id = self.proc_id[reset_target]
                                        self.IPC_PROC["period"][proc_id] \
                                            = int(cmd_line[-1][:-1].split("#")[0])
                                        self.IPC_PROC["period offset"][proc_id] \
                                            = int(cmd_line[-1][:-1].split("#")[1])

                # Proceed only if not in break or if in one-step.
                if self.IPC_PROC["break"].value == 0 \
                        or (self.IPC_PROC["break"].value == 1 \
                        and self.IPC_PROC["one-step"].value == 1):


                    # Adapt "no! bottleneck" item's period to bottleneck duration.
                    # new_period = old_period * ("no bn" items_dur) / (bottleneck_dur * bn_factor)
                    if self.frame_cntr % (2 * self.param["core"]["profiler_window"]) == 0 \
                            and self.IPC_PROC["one-step"].value != 1:
                        # Compute bottleneck (slowest node) duration (read + write).
                        self.bottleneck_dur = 0
                        for p,p_id in self.proc_id.items():
                            if p_id != -1:
                                # Only consider items which could be the bottleneck.
                                if self.bottleneck[p] == -1:
                                    dur = self.IPC_PROC['profiler'][p_id][0] \
                                          + self.IPC_PROC['profiler'][p_id][1]
                                    if dur > self.bottleneck_dur:
                                        self.bottleneck_dur = dur
                        # Update periods.
                        for p,p_id in self.proc_id.items():
                            if p_id != -1:
                                if self.bottleneck[p] > 0 and self.bottleneck_dur > 0:
                                    # Compute item duration.
                                    dur = self.IPC_PROC['profiler'][p_id][0] \
                                          + self.IPC_PROC['profiler'][p_id][1]
                                    new_period \
                                        = int(self.IPC_PROC["period"][p_id] * float(dur) \
                                            / (self.bottleneck_dur * self.bottleneck[p]))
                                    if new_period > 0:
                                        self.IPC_PROC["period"][p_id] = new_period


                    # Reset one-step.
                    self.IPC_PROC["one-step"].value = 0

                    # Before READING, check for load / save.
                    # Save network (everything) to file.
                    if self.IPC_PROC["save/load"].value == 1:
                        # Get save file name and save model.
                        ascii_str = copy.copy(self.IPC_PROC["string"][0:255])
                        str_len = int(self.IPC_PROC["instruction len"].value)
                        saveFile = "".join([chr(int(ascii_str[i])) for i in range(str_len)])
                        # Set back IPC.
                        self.IPC_PROC["save/load"].value = 0
                        save_network(save_file=saveFile, 
                                     net=copy.deepcopy(self.net), 
                                     shm=self.shm)
                    elif self.IPC_PROC["save/load"].value == 2:
                        # Dependent on load source, get loadList.
                        loadList = None
                        if self.flag_load_at_start:
                            self.flag_load_at_start = False
                            with open(self.load_at_start_filename, "rb") as f:
                                # Load it here only for graph initialization.
                                loadList = pckl.load(f)
                        else:
                            # Get load file name.
                            ascii_str = copy.copy(self.IPC_PROC["string"][0:255])
                            loadFile = "".join([chr(int(ascii_str[i])) for i in range(int(self.IPC_PROC["instruction len"].value))])
                            with open(loadFile, "rb") as f:
                                loadList = pckl.load(f)
                        # Set back IPC.
                        self.IPC_PROC["save/load"].value = 0
                        # Now, if loadList is available copy everything.
                        if loadList is not None:
                            # Load states only if same number of agents.
                            if loadList[0][0]["agents"] == self.net["agents"]:
                                # Load states where possible.
                                for n in loadList[2]:
                                    n_idx = loadList[2].index(n)
                                    # Check for existance.
                                    if n in self.net["neuron_pools"]:
                                        # Check for correct shape.
                                        if self.net["neuron_pools"][n]["shape"] \
                                                == loadList[0][0]["neuron_pools"][n]["shape"]:
                                            self.shm.dat[n]["state"][:,:,:,:] \
                                                = loadList[1][n_idx][:,:,:,:]
                            else:
                                print("\nError: incompatible number of agents")
                            # Load parameter & variables where possible.
                            par_var = ["parameter", "variables"]
                            for mod in par_var:
                                mod_idx = par_var.index(mod)
                                for p in loadList[2 * mod_idx + 4]:
                                    p_idx = loadList[2 * mod_idx + 4].index(p)
                                    # Check for existence of item dictionary.
                                    if p[2] in self.shm.dat[p[0]][mod]:
                                        # Check shape and set val.
                                        val = np.array(loadList[2 * mod_idx + 3][p_idx])
                                        if self.shm.dat[p[0]][mod][p[2]].shape == val.shape:
                                            self.shm.set_shm([p[0], mod, p[2]], val)
                    
                    # Execute before_readin method for all core clients.
                    for cc,CC in self.cclients.items():
                        if CC.active:
                            CC.before_readin()
                    
                    # Set all processes to reading phase.
                    self.IPC_PROC["trigger"].value = process_trigger["WaW-R"]
    
                    # Execute readin method for all core clients.
                    for cc,CC in self.cclients.items():
                        if CC.active:
                            CC.readin()

                    # Update temporal memory.
                    update_tmem(self.param["core"]["temporal_memory"], 
                                self.tmem_updates, 
                                self.tmem_update, 
                                0, 
                                self.frame_cntr, 
                                self.tmem_time_steps)
                    # Save state / parameters according to tmem_update in reverse temporal order.
                    for t in range(len(self.param["core"]["temporal_memory"])):
                        t_index = len(self.param["core"]["temporal_memory"]) - 1 - t
                        if self.tmem_update[t_index]:
                            if t_index == 0:
                                # Copy current frame from shared memory.
                                for n in self.net["neuron_pools"]:
                                    self.shm.dat[n]["tmem"][0]["state"][:,:,:,:] \
                                        = self.shm.dat[n]["state"][:,:,:,:]
                                for I in ["np", "sp", "plast", "if"]:
                                    for i in self.net[mn.S2L(I)]:
                                        for mod in ["parameter", "variables"]:
                                            for par in self.shm.dat[i][mod]:
                                                self.shm.set_shm([i, "tmem", 0, mod, par],
                                                                  self.shm.dat[i][mod][par])
                            else:
                                # Copy from t_index - 1 -> t_index.
                                for n in self.net["neuron_pools"]:
                                    self.shm.dat[n]["tmem"][t_index]["state"][:,:,:,:] \
                                        = self.shm.dat[n]["tmem"][t_index - 1]["state"][:,:,:,:]
                                for I in ["np", "sp", "plast", "if"]:
                                    for i in self.net[mn.S2L(I)]:
                                        for mod  in ["parameter", "variables"]:
                                            for par in self.shm.dat[i][mod]:
                                                self.shm.set_shm([i, "tmem", t_index, mod, par],
                                                                  self.shm.dat[i]["tmem"][t_index - 1][mod][par])
                        # Update tmem time steps.
                        self.IPC_PROC["tmem time steps"][t_index] \
                            = self.tmem_time_steps[t_index]

                # End timer.
                self.profiler_core_read[int(self.frame_cntr) % int(self.param["core"]["profiler_window"])] \
                    = time() - timer_start_read



            # Check keyboard.
            if self.terminal.event():
                c = self.terminal.getch()
                if ord(c) == 10: # return
                    self.terminal_command = self.terminal_input
                    self.terminal_input = ""
                elif ord(c) == 127: # backspace
                    if len(self.terminal_input) > 0:
                        self.terminal_input = self.terminal_input[0:-1]
                elif ord(c) < 127:
                    self.terminal_input += c
            # Print command line.
            if pending_procs != 0:
                new_line = strftime("%a, %d %b %Y %H:%M:%S", gmtime()) \
                    + " @ initializing: (" + str(pending_procs) \
                    + " remaining) << " + self.terminal_input \
                    +" <<                     "
            else:
                new_line = strftime("%a, %d %b %Y %H:%M:%S", gmtime()) \
                    + " @ {:08d}".format(self.frame_cntr) + " << " \
                    + self.terminal_input + " <<                    "

            # Evaluate command.
            self.terminal_command = self.terminal_command.rstrip()
            if self.terminal_command != "":
                if self.terminal_command == "exit":
                    if self.IPC_PROC["gui flag"].value == 1:
                        # Send quit to gui.
                        self.IPC_PROC["gui flag"].value = 2
                        self.gui_shutdown = True
                    if self.IPC_PROC["rvgui flag"].value == 1:
                        # Sent quit to rollout view.
                        self.IPC_PROC["rvgui flag"].value = 2
                        self.rvgui_shutdown = True
                    self.shutdown = True
                elif self.terminal_command in ["profile core", "state", "nps", "sps", "ccs", "?", "help"]:
                    self.terminal_current_show = copy.copy(self.terminal_command)
                elif self.terminal_command.startswith('cstat'):
                    self.terminal_current_show = copy.copy(self.terminal_command)
                elif self.terminal_command == "viz off":
                    if self.IPC_PROC["gui flag"].value == 1:
                        # Send quit to gui.
                        self.IPC_PROC["gui flag"].value = 2
                        self.gui_shutdown = True
                elif self.terminal_command == "viz on":
                    import statestream.visualization.visualization
                    if self.IPC_PROC["gui flag"].value != 1:
                        self.IPC_PROC["gui flag"].value = 1
                        # Reload visualization python module.
                        if sys.version[0] == "2":
                            reload(statestream.visualization.visualization)
                        elif sys.version[0] == "3":
                            importlib.reload(statestream.visualization.visualization)
                        from statestream.visualization.visualization import Visualization
                        # Create and start visualization process.
                        self.inst_viz = Visualization(self.net, self.param)
                        self.proc_viz = mp.Process(target=self.inst_viz.run,
                                                   args=(self.IPC_PROC, None))
                        self.proc_viz.start()
                elif self.terminal_command == "rv off":
                    if self.IPC_PROC["rvgui flag"].value == 1:
                        # Send quit to gui.
                        self.IPC_PROC["rvgui flag"].value = 2
                        self.rvgui_shutdown = True
                elif self.terminal_command == "rv on":
                    import statestream.visualization.rollout_view
                    if self.IPC_PROC["rvgui flag"].value != 1:
                        self.IPC_PROC["rvgui flag"].value = 1
                        # Reload visualization python module.
                        if sys.version[0] == "2":
                            reload(statestream.visualization.rollout_view)
                        elif sys.version[0] == "3":
                            importlib.reload(statestream.visualization.rollout_view)
                        from statestream.visualization.rollout_view import rollout_view
                        # Create and start visualization process.
                        self.inst_rv = rollout_view(self.net, self.param)
                        self.proc_rv = mp.Process(target=self.inst_rv.run,
                                                   args=(self.IPC_PROC, None))
                        self.proc_rv.start()
                elif self.terminal_command == "clean on":
                    self.terminal_clean = True
                elif self.terminal_command == "clean off":
                    self.terminal_clean = False
                elif self.terminal_command == "stream":
                    self.IPC_PROC["break"].value = 0
                elif self.terminal_command == "pause":
                    self.IPC_PROC["break"].value = 1
                elif self.terminal_command.startswith("ccmsg"):
                    if self.terminal_command.split()[1] in self.net.get("core_clients", {}):
                        self.terminal_current_show = copy.copy(self.terminal_command)
                elif self.terminal_command.startswith("savegraph"):
                    # Update from shared memory.
                    raw_net = copy.deepcopy(self.net)
                    self.shm.update_net(raw_net)
                    # Write original graph parameters to file.
                    if self.terminal_command.split()[1].endswith(".st_graph"):
                        save_file = self.param["core"]["save_path"] + os.sep \
                                    + self.terminal_command.split()[1]
                    else:
                        save_file = self.param["core"]["save_path"] + os.sep \
                                    + self.terminal_command.split()[1] \
                                    + ".st_graph"
                    with open(save_file, "w+") as f:
                        print("Graph written to " + str(save_file) + "\n")
                        dump_yaml(raw_net,
                                  f)
                elif self.terminal_command.startswith("savenet"):
                    # Send command line to core.
                    if self.terminal_command.split()[1].endswith(".st_net"):
                        save_file = self.param["core"]["save_path"] + os.sep \
                                    + self.terminal_command.split()[1]
                    else:
                        save_file = self.param["core"]["save_path"] + os.sep \
                                    + self.terminal_command.split()[1] \
                                    + ".st_net"
                    save_file_c = np.array([ord(tmp_c) for tmp_c in save_file])
                    self.IPC_PROC["instruction len"].value = len(save_file_c)
                    self.IPC_PROC["string"][0:len(save_file_c)] = save_file_c[:]
                    self.IPC_PROC["save/load"].value = 1
                elif self.terminal_command.startswith("period "):
                    # Change period setting for a process.
                    # Assume first option is the process ID and second the new
                    # period.
                    if len(self.terminal_command.split()) == 3:
                        loc_pid = int(self.terminal_command.split()[1])
                        period = int(self.terminal_command.split()[2])
                        if period > 0:
                            self.IPC_PROC["period"][loc_pid] = period
                elif self.terminal_command.startswith("offset "):
                    # Change offset setting for a process.
                    # Assume first option is the process ID and second the new
                    # offset.
                    if len(self.terminal_command.split()) == 3:
                        loc_pid = int(self.terminal_command.split()[1])
                        offset = int(self.terminal_command.split()[2])
                        if period > 0:
                            self.IPC_PROC["period offset"][loc_pid] = offset
                elif self.terminal_command.startswith("bottleneck "):
                    # Change bottleneck setting for a process.
                    # Assume first option is the process ID and second the new
                    # bottleneck factor.
                    if len(self.terminal_command.split()) == 3:
                        loc_pid = int(self.terminal_command.split()[1])
                        proc_name = self.get_proc_name_by_id(loc_pid)
                        bn_factor = float(self.terminal_command.split()[2])
                        if bn_factor <= 0:
                            # Reset process to be a potential bottleneck.
                            self.bottleneck[proc_name] = -1
                            self.IPC_PROC["period"][loc_pid] = 1
                        else:
                            self.bottleneck[proc_name] = bn_factor
                elif self.terminal_command.startswith("split "):
                    # Change split parameter.
                    if len(self.terminal_command.split()) == 2:
                        split = max(int(self.terminal_command.split()[1]), 0)
                        split = min(split, self.net['agents'])
                        self.IPC_PROC['plast split'].value = split
                elif self.terminal_command.startswith('ccstop'):
                    cc_name = self.terminal_command.split()[1]
                    if cc_name in self.cclients:
                        self.cclients[cc_name].active = False
                elif self.terminal_command.startswith('ccstart'):
                    cc_name = self.terminal_command.split()[1]
                    if cc_name in self.cclients:
                        self.cclients[cc_name].active = True
                else:
                    self.terminal_current_show \
                        = "unknown " + self.terminal_command
                    print("\n    Unknown command: " \
                          + self.terminal_current_show[7:] + "\n")
                self.terminal_current_shown = False
                self.terminal_command = ""

            # Terminal input "shm ..." overwrites everything.
            if self.terminal_input.startswith("shm"):
                # List some shared memory entries dependent on "shm ...".
                shm_lines = self.shm.pprint_list(self.terminal_input)
                self.terminal_current_show = "shm"
                self.terminal_current_shown = False
                self.terminal_command = ""
            # Set back.
            if self.terminal_current_show == "shm" \
                    and not self.terminal_input.startswith("shm"):
                self.terminal_current_show = ""
                self.terminal_current_shown = False

            if self.current_line != new_line:
                # Clear screen.
                if self.terminal_clean:
                    os.system("clear")
                    print("statestream terminal - " + self.net["name"])

                # Print new line.
                self.current_line = new_line
                print(self.current_line, end="\r")

                if self.terminal_clean:
                    self.terminal_current_shown = False

                if not self.terminal_current_shown:
                    self.terminal_current_shown = True
                    if self.terminal_current_show == "profile core":
                        print("\n    Profile of core process:")
                        print("        read phase : " \
                              + str(np.mean(self.profiler_core_read)) + " +- " \
                              + str(np.max(np.abs(self.profiler_core_read \
                                                - np.mean(self.profiler_core_read)))))
                        print("        write phase: " \
                              + str(np.mean(self.profiler_core_write)) + " +- " \
                              + str(np.max(np.abs(self.profiler_core_write \
                                                - np.mean(self.profiler_core_write)))))
                        print("        overall    : " \
                              + str(np.mean(self.profiler_core_overall)) + " +- " \
                              + str(np.max(np.abs(self.profiler_core_overall \
                                                - np.mean(self.profiler_core_overall)))))
                    elif self.terminal_current_show.startswith("cstat"):
                        if len(self.terminal_current_show.split()) == 2:
                            lines = int(self.terminal_current_show.split()[1])
                        else:
                            lines = 256
                        print("\n")
                        print("    " + "id".ljust(4) \
                              + "name".ljust(18) \
                              + "type".ljust(8) \
                              + "state".ljust(8) \
                              + "device".ljust(8) \
                              + "read".ljust(8) \
                              + "write".ljust(8) \
                              + "period".ljust(8) \
                              + "offset".ljust(8) \
                              + "\n")
                        # Sort by duration.
                        durs = []
                        for p,p_id in self.proc_id.items():
                            if p_id != -1:
                                durs.append([-self.IPC_PROC['profiler'][p_id][0] \
                                             - self.IPC_PROC['profiler'][p_id][1], p])
                                durs[-1][0] *= self.IPC_PROC["period"][p_id]
                        idx = sorted((e[0], e[1]) for i,e in enumerate(durs))
                        for i,p_dur_name in enumerate(idx):
                            if i >= lines:
                                print("    ... " + str(len(durs) - lines) \
                                      + " more processes")
                                break
                            p = p_dur_name[1]
                            p_id = self.proc_id[p]
                            prof = [copy.copy(self.IPC_PROC['profiler'][p_id][0]),
                                    copy.copy(self.IPC_PROC['profiler'][p_id][1])]
                            print("    " + str(p_id).ljust(4) \
                                  + str(p).ljust(18) \
                                  + self.proc_type[p].ljust(8) \
                                  + str(int(self.IPC_PROC["state"][p_id])).ljust(8) \
                                  + self.device[p].ljust(8) \
                                  + str(int(1000 * prof[0])).ljust(8) \
                                  + str(int(1000 * prof[1])).ljust(8) \
                                  + str(int(self.IPC_PROC["period"][p_id])).ljust(8) \
                                  + str(int(self.IPC_PROC["period offset"][p_id])).ljust(8))
                        print("")
                    elif self.terminal_current_show == "state":
                        print("\n")
                        print("session id:".rjust(16) + "   " \
                              + str(self.shm.session_id))
                        print("agents / split:".rjust(16) + "   " \
                              + str(self.net["agents"]) + " / " + str(int(self.IPC_PROC['plast split'].value)))
                        print("NPs / SPs:".rjust(16) + "   " \
                              + str(len(self.net["neuron_pools"])) + " / " + str(len(self.net["synapse_pools"])))
                        if self.IPC_PROC["break"].value == 0:
                            print("state:".rjust(16) + "   streaming")
                        else:
                            print("state:".rjust(16) + "   paused")
                        if self.IPC_PROC["gui flag"].value == 0:
                            print("viz:".rjust(16) + "   off")
                        else:
                            print("viz:".rjust(16) + "   on")
                        if self.terminal_clean:
                            print("clean:".rjust(16) + "   on")
                        else:
                            print("clean:".rjust(16) + "   off")
                        print("")
                    elif self.terminal_current_show == "nps":
                        print("\n    neuron_pools: " \
                              + str(len(self.net["neuron_pools"])))
                        for n,N in self.net["neuron_pools"].items():
                            print("        " + n.ljust(16) + " of shape " \
                                  + str(N["shape"]))
                    elif self.terminal_current_show == "sps":
                        print("\n    synapse_pools: " \
                              + str(len(self.net["synapse_pools"])))
                        for s,S in self.net["synapse_pools"].items():
                            print("        " + s.ljust(16) + " from " \
                                  + str(S["source"][0][0]).ljust(16) + " to " \
                                  + str(S["target"]).ljust(16))
                    elif self.terminal_current_show == "ccs":
                        print("\n\n    core-clients: " \
                              + str(len(self.net.get("core_clients", {}))))
                        for cc,CC in self.cclients.items():
                            print("        " + cc.ljust(16) + " of type " \
                                  + str(CC.p["type"]) + "  (active: " + str(CC.active) + ")\n")
                    elif self.terminal_current_show.startswith('ccmsg'):
                        cc_name = self.terminal_current_show.split()[1]
                        print("\n    messages of core-client " + cc_name)
                        max_lines = 8
                        line = 0
                        for l in reversed(self.cclients[cc_name].mesg):
                            line += 1
                            if line < max_lines:
                                print("        " + str(l))
                            else:
                                print("        " \
                                      + str(len(self.cclients[cc_name].mesg) - max_lines) \
                                      + " past messages")
                                break
                    elif self.terminal_current_show == "shm":
                        print("\n")
                        for l in range(len(shm_lines)):
                            print("    " + shm_lines[l])
                    elif self.terminal_current_show in ["?", "help"]:
                        print("\n    Commands:")
                        print("        bottleneck [PID] [factor]  Reset the bottleneck factor for process PID.")
                        print("        ccmsg [cc name]            Show messages of core-client with given name.")
                        print("        ccs                        Show list of core-clients.")
                        print("        ccstart/ccstop [cc_name]   Start / stop core-client with given name.")
                        print("        clean on/off               Enable/disable clear screen for terminal.")
                        print("        exit                       Exit statestream.")
                        print("        help, ?                    Show this info.")
                        print("        nps                        Show list of neuron_pools.")
                        print("        pause                      Pause network.")
                        print("        period [PID] [new period]  Reset new period for process PID.")
                        print("        offset [PID] [new period]  Reset new offset for process PID.")
                        print("        profile core               Print some profiling info for core process.")
                        print("        cstat (lines)             Print state of the first 'lines' item processes.")
                        print("        savegraph [filename]       Save graph to yaml file.")
                        print("        savenet [filename]         Save network to yaml file.")
                        print("        shm. ...                   Introspect shared memory.")
                        print("        split [new split]          Reset split parameter.")
                        print("        sps                        Show list of synapse_pools.")
                        print("        state                      Show current system status.")
                        print("        stream                     Run network after pause.")
                        print("        viz on/off                 Open/close visualization.")
                        print("\n")
                    elif self.terminal_current_show.startswith("unknown ") \
                            and self.terminal_clean:
                        print("\n    Unknown command: " \
                              + self.terminal_current_show[7:] + "\n")

                sys.stdout.flush()

            if self.gui_shutdown:
                # Wait for gui to return.
                if self.IPC_PROC['gui flag'].value == 0: # 1
                    self.proc_viz.join()
                    self.proc_viz = None
                    self.inst_viz = None
                    self.gui_shutdown = False
            if self.rvgui_shutdown:
                # Wait for gui to return.
                if self.IPC_PROC['rvgui flag'].value == 0: # 1
                    self.proc_rv.join()
                    self.proc_rv = None
                    self.inst_rv = None
                    self.rvgui_shutdown = False

            # Check for shutdown.
            if (self.IPC_PROC["gui request"][0] == 1 or self.shutdown) and not self.gui_shutdown:
                # Send trigger for all processes to end.
                self.IPC_PROC["trigger"].value = 2
                print("\n")
                # Wait for processes to return.
                for p in self.proc_id:
                    print("Shutdown process for: " + str(p) + " ...")
                    try:
                        self.proc[p].join()
                    except:
                        print("    ... process for: " + str(p) \
                              + " seems to be down already.")
                self.state = process_state["E"]
            
            # End timer.
            self.profiler_core_overall[int(self.frame_cntr) % int(self.param["core"]["profiler_window"])] \
                = time() - timer_start_overall

            if not WaR_shared and not WaW_shared:
                sleep(0.001)

            # Sleep cummulated for delay.
            if self.IPC_PROC["delay"].value != 0 and not self.delayed:
                self.delayed = True
                self.delay = float(self.IPC_PROC["delay"].value) / 1000.0

            if self.delayed:
                self.delay -= float(time() - timer_start_overall)
                if self.delay < 0.0:
                    self.delayed = False

        # Tell core clients that core shutsdown.
        for cc,CC in self.cclients.items():
            CC.interrupt()

        # Free shared memory.
        self.shm.delete()

        # End keyboard control.
        self.terminal.reset_term()
        
        # Comment for the last line.
        print("Core: Last line of statestream code. Backend shutting down ...")
            
            
            
            
            
            
if __name__ == "__main__":
    inst_statestream = StateStream()
    inst_statestream.run()
                    