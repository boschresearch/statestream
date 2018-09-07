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
import importlib
import SharedArray

from statestream.utils.helper import is_scalar_shape
from statestream.utils.shared_memory_layout import SharedMemoryLayout

from statestream.meta.network import get_item_type
from statestream.meta.network import S2L
from statestream.meta.neuron_pool import np_shm_layout, np_init
from statestream.meta.synapse_pool import sp_shm_layout, sp_init



def shared_layout(net, param):
    """Generates shared-memory layout from net and param.
    """
    layout = {}

    for t in ["np", "sp", "plast", "if"]:
        for i,I in net[S2L(t)].items():
            # Begin with empty layout.
            layout[i] = {}
            # Empty tmem layout structure. Rest will be filled during shm creation.
            layout[i]["tmem"] = []
            for tmem in range(len(param["core"]["temporal_memory"])):
                layout[i]["tmem"].append({"parameter": {}, "variables": {}})
                if t == "plast":
                    layout[i]["tmem"][tmem]["updates"] = {}
            # Get item shm data layout.
            if t == "np":
                layout[i].update(np_shm_layout(i, net, param))
            elif t == "sp":
                layout[i].update(sp_shm_layout(i, net, param))
            elif t == "plast":
                plast_shm_layout \
                    = getattr(importlib.import_module("statestream.meta.plasticities." + I["type"]), 
                                                      "plast_shm_layout")
                layout[i].update(plast_shm_layout(i, net, param))
                layout[i]["updates"] = {}
                for par in I["parameter"]:
                    if par[1] not in layout[i]["updates"]:
                        layout[i]["updates"][par[1]] = {}
                    shml = layout[par[1]]["parameter"][par[2]]
                    layout[i]["updates"][par[1]][par[2]] = shml
            elif t == "if":
                if_shm_layout \
                    = getattr(importlib.import_module("statestream.interfaces.process_if_" + I["type"]),
                                                      "if_shm_layout")
                layout[i].update(if_shm_layout(i, net, param))

    return layout    




class SharedMemory(object):
    def __init__(self, net, param, session_id=None, force_id=None):
        self.net = net
        self.param = param
        # Get list of all existing shared memory arrays.
        shm_list = SharedArray.list()
        shm_list_name = []
        for i in range(len(shm_list)):
            if sys.version[0] == "2":
                shm_list_name.append(shm_list[i].name)
            elif sys.version[0] == "3":
                shm_list_name.append(shm_list[i].name.decode("utf-8"))
        # Initially start with invalid session id.
        self.session_id = None
        # Begin with empty structure holding the entire layout.
        self.dat = {}

        # Estimate of bytes reserved in shared memory.
        self.log_lines = []
        self.log_bytes = []

        # Build layout.
        # ---------------------------------------------------------------------
        self.layout = shared_layout(self.net, self.param)

        # Dependent on given session_id initialize shared memory.
        # ---------------------------------------------------------------------
        if session_id is None:
            if force_id is None:
                # Determine next free session id.
                for tmp_session_id in range(2**10):
                    id_taken = False
                    for i in range(len(shm_list)):
                        if shm_list_name[i].find("statestream." + str(tmp_session_id) + ".") != -1:
                            id_taken = True
                            break
                    # Take the first free session id and break.
                    if not id_taken:
                        self.session_id = tmp_session_id
                        session_name = "statestream." + str(self.session_id) + "."
                        break
            else:
                self.session_id = force_id
                session_name = "statestream." + str(self.session_id) + "."

            # Allocate all shared memory.
            # ---------------------------------------------------------
            for t in ["np", "sp", "plast", "if"]:
                for i, I in self.net[S2L(t)].items():
                    # Allocate process identifiers.
                    shm_name = session_name + "core.proc_id." + i
                    SharedArray.create(shm_name, 1, dtype=np.int32)
                        
                    # Allocate shm for neuron pool states.
                    if t == "np":
                        shm_name = session_name + "net." + i + ".state"
                        SharedArray.create(shm_name, 
                                           self.layout[i]["state"].shape, 
                                           dtype=self.layout[i]["state"].dtype)
                        # Allocate also tmem for np states.
                        for tmem in range(len(param["core"]["temporal_memory"])):
                            self.layout[i]["tmem"][tmem]["state"] = self.layout[i]["state"]
                            tmem_shm_name = session_name + "net.tmem." + str(tmem) + "." + i + ".state"
                            SharedArray.create(tmem_shm_name, 
                                               self.layout[i]["state"].shape, 
                                               dtype=self.layout[i]["state"].dtype)
                    # Allocate parameters and variables (incl. tmem).
                    for T in ["parameter", "variables"]:
                        shm_name = session_name + "net." + i + "." + T
                        for d,d_l in self.layout[i][T].items():
                            dat_name = shm_name + "." + d
                            if is_scalar_shape(d_l.shape):
                                SharedArray.create(dat_name, 1, dtype=d_l.dtype)
                            else:
                                SharedArray.create(dat_name, d_l.shape, dtype=d_l.dtype)
                            # Allocate also tmem for parameters / variables.
                            for tmem in range(len(param["core"]["temporal_memory"])):
                                self.layout[i]["tmem"][tmem][T][d] = d_l
                                tmem_shm_name = session_name + "net.tmem." + str(tmem) \
                                                + "." + i + "." + T + "." + d
                                if is_scalar_shape(d_l.shape):
                                    SharedArray.create(tmem_shm_name, 1, dtype=d_l.dtype)
                                else:
                                    SharedArray.create(tmem_shm_name, d_l.shape, dtype=d_l.dtype)
            # Allocate shm for plasticity updates (incl. tmem).
            for i,I in self.net["plasticities"].items():
                shm_name = session_name + "net." + i + ".updates."
                for par in I["parameter"]:
                    shml = self.layout[par[1]]["parameter"][par[2]]
                    dat_name = shm_name + par[0] + "." + par[1] + "." + par[2]
                    if is_scalar_shape(shml.shape):
                        SharedArray.create(dat_name, 1, dtype=shml.dtype)
                    else:
                        SharedArray.create(dat_name, shml.shape, dtype=shml.dtype)
                    # Allocate also tmem for updates.
                    for tmem in range(len(param["core"]["temporal_memory"])):
                        if par[1] not in self.layout[i]["tmem"][tmem]["updates"]:
                            self.layout[i]["tmem"][tmem]["updates"][par[1]] = {}
                        self.layout[i]["tmem"][tmem]["updates"][par[1]][par[2]] = shml
                        tmem_shm_name = session_name + "net.tmem." + str(tmem) \
                                        + "." + i + ".updates." \
                                        + par[0] + "." + par[1] + "." + par[2]
                        if is_scalar_shape(shml.shape):
                            SharedArray.create(tmem_shm_name, 1, dtype=shml.dtype)
                        else:
                            SharedArray.create(tmem_shm_name, shml.shape, dtype=shml.dtype)
        else:
            # Set session name for shm.
            session_name = "statestream." + str(session_id) + "."
            # Check if shared memory for this session_id was already created.
            for i in range(len(shm_list)):
                if shm_list_name[i].find(session_name) != -1:
                    self.session_id = session_id
                    break
            assert (self.session_id != None), \
                "Error: SharedMemory() Given session_id was not found: " \
                + str(session_id) + "   " + session_name

        # Attach all shared memory.
        # ---------------------------------------------------------------------
        self.proc_id = {}
        for t in ["np", "sp", "plast", "if"]:
            for i,I in self.net[S2L(t)].items():
                self.dat[i] = {}
                # Begin with empty list of dicts for temporal memory.
                self.dat[i]["tmem"] = [{} for tmem in range(len(param["core"]["temporal_memory"]))]
                # Process relevant memory.
                shm_name = session_name + "core.proc_id." + i
                self.proc_id[i] = SharedArray.attach(shm_name)
                # Network data shared memory for neuron pool states.
                if t == "np":
                    shm_name = session_name + "net." + i + ".state"
                    self.dat[i]["state"] = SharedArray.attach(shm_name)
                    self.log_lines += [str(i) + ".state"]
                    self.log_bytes += [self.dat[i]["state"].nbytes]
                    # Attach also tmem for np states.
                    for tmem in range(len(param["core"]["temporal_memory"])):
                        tmem_dat_name = session_name + "net.tmem." + str(tmem) + "." + i + ".state"
                        self.dat[i]["tmem"][tmem]["state"] = SharedArray.attach(tmem_dat_name)
                        self.log_lines += [str(i) + ".tmem." + str(tmem) + ".state"]
                        self.log_bytes += [self.dat[i]["tmem"][tmem]["state"].nbytes]
                # Network data shared memory for plasticity updates.
                if t == "plast":
                    # Begin with empty dict also for temporal memory.
                    self.dat[i]["updates"] = {}
                    for tmem in range(len(param["core"]["temporal_memory"])):
                        self.dat[i]["tmem"][tmem]["updates"] = {}
                    shm_name = session_name + "net." + i + ".updates."
                    for par in I["parameter"]:
                        # First time add parameter for specific item (incl. tmem).
                        if par[1] not in self.dat[i]["updates"]:
                            self.dat[i]["updates"][par[1]] = {}
                            for tmem in range(len(param["core"]["temporal_memory"])):
                                self.dat[i]["tmem"][tmem]["updates"][par[1]] = {}
                        # Specify shm update id.
                        dat_name = shm_name + par[0] + "." + par[1] + "." + par[2]
                        # Attach shm.
                        self.dat[i]["updates"][par[1]][par[2]] = SharedArray.attach(dat_name)
                        self.log_lines += [str(i) + ".updates." + str(par[1]) + "." + str(par[2])]
                        self.log_bytes += [self.dat[i]["updates"][par[1]][par[2]].nbytes]
                        # Attach also tmem for updates.
                        for tmem in range(len(param["core"]["temporal_memory"])):
                            tmem_shm_name = session_name + "net.tmem." + str(tmem) \
                                            + "." + i + ".updates." \
                                            + par[0] + "." + par[1] + "." + par[2]
                            self.dat[i]["tmem"][tmem]["updates"][par[1]][par[2]] \
                                = SharedArray.attach(tmem_shm_name)
                            self.log_lines += [str(i) + ".tmem." + str(tmem) \
                                               + ".updates." + str(par[1]) + "." + str(par[2])]
                            self.log_bytes += [self.dat[i]["tmem"][tmem]["updates"][par[1]][par[2]].nbytes]
                # Network data shared memory for variables and parameter.
                for T in ["parameter", "variables"]:
                    # Begin with empty dict also for temporal memory.
                    self.dat[i][T] = {}
                    for tmem in range(len(param["core"]["temporal_memory"])):
                        self.dat[i]["tmem"][tmem][T] = {}
                    # Determine shm id item "prefix".
                    shm_name = session_name + "net." + i + "." + T
                    # Loop over all vars/pars of this item.
                    for d,d_l in self.layout[i][T].items():
                        dat_name = shm_name + "." + d
                        self.dat[i][T][d] = SharedArray.attach(dat_name)
                        self.log_lines += [str(i) + "." + str(T) + "." + str(d)]
                        self.log_bytes += [self.dat[i][T][d].nbytes]
                        # Attach also tmem for parameter / variables.
                        for tmem in range(len(param["core"]["temporal_memory"])):
                            tmem_shm_name = session_name + "net.tmem." + str(tmem) \
                                            + "." + i + "." + T + "." + d
                            self.dat[i]["tmem"][tmem][T][d] = SharedArray.attach(tmem_shm_name)
                            self.log_lines += [str(i) + ".tmem." + str(tmem) + "." \
                                + str(T) + "." + str(d)]
                            self.log_bytes += [self.dat[i]["tmem"][tmem][T][d].nbytes]
                        
                        
    
    def delete(self):
        """Method to free statestream shared memory of the particular session.
        """
        if self.session_id != None:
            shm_list = SharedArray.list()
            shm_list_name = []
            for i in range(len(shm_list)):
                if sys.version[0] == "2":
                    shm_list_name.append(shm_list[i].name)
                elif sys.version[0] == "3":
                    shm_list_name.append(shm_list[i].name.decode("utf-8"))

            for i in range(len(shm_list)):
                if shm_list_name[i].find("statestream." + str(self.session_id) + ".") != -1:
                    SharedArray.delete(shm_list_name[i])


    def add_sys_client(self, client_param):
        """Create shared memory for a single system client.
        """
        client_shm_name = 'statestream.' \
                          + str(self.session_id) + '.' \
                          + 'sys_clients.' \
                          + str(client_param['name']) + '.'

        # Create and attach client specific shared memory.
        for T in ['parameter', 'variables']:
            if T in client_param:
                for pv,PV in client_param[T].items():
                    shm_name = client_shm_name + T + '.' + pv
                    try:
                        SharedArray.create(shm_name, PV['shape'], dtype=np.float32)
                    except:
                        dat = SharedArray.attach(shm_name)
                        if dat.shape != PV['shape']:
                            print('\nError: Shared memory: Tried to create already existing memory: ' + shm_name)



    def update_sys_client(self):
        """Update this instance of shared memory to existing clients.
        """
        # Determine all clients, currently in shared memory.
        clients = {}
        shm_list = SharedArray.list()
        client_shm_name = 'statestream.' \
                          + str(self.session_id) + '.' \
                          + 'sys_clients.'
        for shm_name_raw in shm_list:
            if sys.version[0] == "2":
                shm_name = shm_name_raw.name
            elif sys.version[0] == "3":
                shm_name = shm_name_raw.name.decode("utf-8")
            if shm_name.startswith(client_shm_name):
                shm_name_split = shm_name.split('.')
                client_name = shm_name_split[3]
                if client_name not in clients:
                    clients[client_name] = {
                        'parameter': {},
                        'variables': {}
                    }
                clients[client_name][shm_name_split[4]][shm_name_split[5]] \
                    = shm_name
        # Update client shared memroy dat and layout.
        for c,C in clients.items():
            if c not in self.dat:
                self.dat[c] = {
                    'parameter': {},
                    'variables': {}
                }
                self.layout[c] = {
                    'parameter': {},
                    'variables': {}
                }
            for t,T in C.items():
                for d,D in T.items():
                    self.dat[c][t][d] = SharedArray.attach(D)
                    self.layout[c][t][d] = SharedMemoryLayout('np',
                                                              self.dat[c][t][d].shape,
                                                              self.dat[c][t][d].dtype,
                                                              0.0)

        # Determine all items in dat / layout which are not in shared memory.
        # Remove deprecated shared memory from layout and dat.
        remove_items = []
        for i,I in self.layout.items():
            if i not in clients and i not in self.net['neuron_pools'] \
                    and i not in self.net['synapse_pools'] \
                    and i not in self.net['plasticities'] \
                    and i not in self.net['interfaces']:
                remove_items.append(i)
        for i in remove_items:
            self.dat.pop(i)
            self.layout.pop(i)



    def remove_sys_client(self, client_name):
        """Remove shared memory for system client.
        """
        client_shm_name = 'statestream.' \
                          + str(self.session_id) + '.' \
                          + 'sys_clients.' \
                          + str(client_name) + '.'
        # Delete shared memory.
        for T in ['parameter', 'variables']:
            for d,d_l in self.layout[client_name][T].items():
                shm_name = client_shm_name + T + '.' + str(d)
                try:
                    SharedArray.delete(shm_name)
                except:
                    print("\nERROR: Unable to delete non-existing shared memory: " + str(shm_name) + "\n")


    def pprint_list(self, what=""):
        """Return a list of lines containing shm info about what.
        """
        lines = []
        w = what.split(".")
        if len(w) > 1:
            if len(w[1]) == 1:
                if w[1] == "n":
                    i_type = "neuron_pools"
                elif w[1] == "s":
                    i_type = "synapse_pools"
                elif w[1] == "p":
                    i_type = "plasticities"
                elif w[1] == "i":
                    i_type = "interfaces"
            else:
                return []
        if what in ["shm", "shm."]:
            lines.append("[n]euron pools")
            lines.append("[s]ynapse pools")
            lines.append("[p]lasticities")
            lines.append("[i]nterfaces")
        if len(w) == 2:
            # shm.i_type
            if w[1] != "":
                cntr = 0
                for i in self.net[i_type]:
                    if cntr == 0:
                        # Append new line.
                        lines.append("    " + i.ljust(18))
                    else:
                        # Append to existing line.
                        lines[-1] = lines[-1] + i.ljust(18)
                    if cntr < 3:
                        cntr += 1
                    else:
                        cntr = 0
        elif len(w) == 3:
            # shm.i_type.item_name
            if w[1] != "":
                cntr = 0
                for i in self.net[i_type]:
                    if i.startswith(w[2]):
                        if cntr == 0:
                            # Append new line.
                            lines.append("    " + i.ljust(18))
                        else:
                            # Append to existing line.
                            lines[-1] = lines[-1] + i.ljust(18)
                        if cntr < 3:
                            cntr += 1
                        else:
                            cntr = 0
        elif len(w) == 4:
            # shm.i_type.item_name.data_type
            if w[1] != "":
                if w[2] in self.net[i_type]:
                    # Assuming all classes of data begin 
                    # with a different letter.
                    if w[3] == "":
                        for e in self.dat[w[2]]:
                            lines.append("    [" + e[0] + "]" + e[1:])
                    else:
                        dat_type = "x"
                        if w[3][0] in ["v", "p"]:
                            if w[3] == "v":
                                dat_type = "variables"
                            else:
                                dat_type = "parameter"
                            for vp in self.dat[w[2]][dat_type]:
                                lines.append("    " + vp)
                        elif w[3].startswith("s"):
                            if w[3] == "s":
                                lines.append("    shape: " + str(self.layout[w[2]]["state"].shape))
                                lines.append("     type: " + str(self.layout[w[2]]["state"].dtype))
                                nbytes = self.dat[w[2]]["state"].nbytes
                                lines.append("   memory: " + str(nbytes) + " B")
                            if w[3].startswith("s[") and w[3][-1] == "]":
                                # Get data.
                                s = eval("self.dat[w[2]]['state']" + w[3][1:])
                                if len(s.shape) == 0:
                                    lines.append("    value: " + str(s))
                                if len(s.shape) == 1:
                                    for i in range(min(s.shape[0], 16)):
                                        lines.append(str(i).ljust(4) + "   " + str(s[i]))
                                    if s.shape[0] >= 16:
                                        lines.append("...")

        elif len(w) == 5:
            if w[1] != "":
                if w[2] in self.net[i_type]:
                    dat_type = "x"
                    if w[3][0] in ["v", "p"]:
                        if w[3] == "v":
                            dat_type = "variables"
                        else:
                            dat_type = "parameter"
                        for vp in self.dat[w[2]][dat_type]:
                            if vp.startswith(w[4]) and len(w[4]) < len(vp):
                                lines.append("    " + vp)
                            if vp == w[4]:
                                lines.append("    shape: " + str(self.layout[w[2]][dat_type][vp].shape))
                                lines.append("     type: " + str(self.layout[w[2]][dat_type][vp].dtype))
                                nbytes = self.dat[w[2]][dat_type][vp].nbytes
                                lines.append("   memory: " + str(nbytes) + " B")
                            if w[4].startswith(vp) and w[4][-1] == "]" and "[" in w[4]:
                                # Get data.
                                s = eval("self.dat[w[2]][dat_type][vp]" + w[4][len(vp):])
                                if len(s.shape) == 0:
                                    lines.append("    value: " + str(s))
                                if len(s.shape) == 1:
                                    for i in range(min(s.shape[0], 16)):
                                        lines.append(str(i).ljust(4) + "   " + str(s[i]))
                                    if s.shape[0] >= 16:
                                        lines.append("...")



        return lines



    def init(self, what=[], mode=None):
        """Method to recusively initialize a subset of the network.
        
            what:
                []                      Initialize everything.
                ["state"]               Initialize all states.
                ["parameter"]           Initialize all parameter.
                ["variables"]           Initialize all variables.
                ["updates"]             Initialize all updates.
                
                [np_id, "state"]        Initialize state of neuron pool np_id.
                [item_id,               Initialize parameter par_id of item item_id.
                    "parameter",
                    par_id]        
                [item_id,               Initialize variable var_id of item item_id.
                    "variables",
                    var_id]
                [plast_id,              Initialize updates [tar_id, par_id] of plasticity plast_id.
                    "updates",
                    tar_id,
                    par_id]
        """
        # Do not initialize meta-variables.
        if len(what) >= 1:
            if what[0] in self.dat \
                    and what[0] not in self.net['neuron_pools'] \
                    and what[0] not in self.net['synapse_pools'] \
                    and what[0] not in self.net['plasticities'] \
                    and what[0] not in self.net['interfaces']:
                return
        # Adjust mode in some cases.
        if isinstance(mode, list):
            if "external_models" in self.net:
                # In case of external model init, set mode here to none.
                if mode[0] in self.net["external_models"]:
                    mode = None
        # Determine item to be set and its type.
        item_id = None
        item_type = None
        if len(what) >= 1:
            item_id = what[0]
            if item_id == "state":
                # Initialize all states. 
                for n in self.net["neuron_pools"]:
                    self.init([n, "state"], mode=mode)
                # Done with initialization.
                return None
            elif item_id in ["parameter", "variables"]:
                # Initialize all parameters or variables.
                for i in self.dat:
                    for d, d_l in self.layout[i][item_id].items():
                        self.init([i, item_id, d], mode=mode)
                # Done with initialization.
                return None
            elif item_id == "updates":
                # Initialize all updates.
                for i in self.net["plasticities"]:
                    for target_i in self.dat[i]["updates"]:
                        for target_p in self.dat[i]["updates"][target_i]:
                            self.init([i, "updates", target_i, target_p], mode=mode)
                # Done with initialization.
                return None
            else:
                # Assume what[0] is an item.
                # Determine item type.
                item_type = get_item_type(self.net, item_id)
        else:
            # len ought to be zero, so everthing should be set.
            self.init(["state"], mode=0.0)
            self.init(["parameter"], mode=mode)
            self.init(["variables"], mode=0.0)
            self.init(["updates"], mode=0.0)
            # Done with initialization.
            return None
                        
        # Dependent on len of what, determine what is to be set.
        set_flag = False
        if len(what) == 1:
            # Re-init a single item.
            if item_type == "np":
                pass
            elif item_type == "sp":
                pass
            elif item_type == "plast":
                pass
            elif item_type == "if":
                pass
            # TODO
        elif len(what) == 2:
            if what[1] in ["state"]:
                dat_name = "__state__"
                dat_layout = self.layout[item_id]["state"]
                set_flag = True
            else:
                raise NameError("SharedMemory.init() inconsistent what parameter for what of length " \
                     + str(len(what)) + ".")
        elif len(what) == 3:
            if what[1] in ["parameter", "variables"]:
                dat_name = what[2]
                dat_layout = self.layout[item_id][what[1]][what[2]]
                set_flag = True
            else:
                raise NameError("SharedMemory.init() inconsistent what parameter for what of length " \
                    + str(len(what)) + ".")
        elif len(what) == 4:
            if what[1] == "updates":
                dat_name = [what[2], [what[3]]]
                dat_layout = self.layout[item_id]["updates"][what[2]][what[3]]
                set_flag = True
            else:
                raise NameError("SharedMemory.init() inconsistent what parameter for what of length " \
                    + str(len(what)) + ".")
        else:
            raise NameError("SharedMemory.init() Unexpected what of length " + str(len(what)) + ".")

        # Set if something is to be set.
        if set_flag:
            if item_type == "np":
                value = np_init(self.net, item_id, dat_name, dat_layout, mode=mode)
            elif item_type == "sp":
                value = sp_init(self.net, item_id, dat_name, dat_layout, mode=mode)
            elif item_type == "plast":
                # Determine plasticity type.
                plast_type = self.net["plasticities"][item_id]["type"]
                # Get correct plasticity initializer.
                try:
                    plast_init \
                        = getattr(importlib.import_module("statestream.meta.plasticities." + plast_type), 
                                                          "plast_init")
                    value = plast_init(self.net, item_id, dat_name, dat_layout, mode=mode)
                except:
                    value = None
            elif item_type == "if":
                # Determine interface type.
                if_type = self.net["interfaces"][item_id]["type"]
                # Get correct plasticity initializer.
                try:
                    if_init = getattr(importlib.import_module("statestream.interfaces." + if_type), 
                                                              "if_init")
                    value = if_init(self.net, item_id, dat_name, dat_layout, mode=mode)
                except:
                    value = None
            # Fallback if invalid value.
            if value is None:
                value = self.init_fallback(item_id, dat_name, dat_layout, mode=mode)
            # Finally set value.
            self.set_shm(what, value)



    def init_fallback(self, item_id, dat_name, dat_layout, mode=None):
        """Fallback to default initialization.
        """
        # Get local dictionary.
        if item_id in self.net["neuron_pools"]:
            p = self.net["neuron_pools"][item_id]
        elif item_id in self.net["synapse_pools"]:
            p = self.net["synapse_pools"][item_id]
        elif item_id in self.net["plasticities"]:
            p = self.net["plasticities"][item_id]
        elif item_id in self.net["interfaces"]:
            p = self.net["interfaces"][item_id]
        # Dependent on scalar or not, try to initialize.            
        if is_scalar_shape(dat_layout.shape):
            # Scalar values.
            if mode is None:
                dat_value = np.array(p.get(dat_name, dat_layout.default), 
                                     dtype=dat_layout.dtype)
            else:
                dat_value = np.array(mode, dtype=dat_layout.dtype)
                if mode in ["one", 1.0]:
                    dat_value = np.array(1.0, dtype=dat_layout.dtype)
                else:
                    dat_value = np.array(0.0, dtype=dat_layout.dtype)
        else:
            # If mode is None, set to default.
            if mode is None:
                dat_value = np.ones(dat_layout.shape, dtype=dat_layout.dtype)
                try:
                    dat_value *= dat_layout.default
                except:
                    dat_value *= 0
                    print("Warning: No valid initialization for " + str(dat_name) \
                        + " of item " + str(item_id) + ". Set to zero.")
            else:
                # Dependent on specified mode set value.
                if mode in ["zero", 0.0]:
                    dat_value = np.zeros(dat_layout.shape, dtype=dat_layout.dtype)
                elif mode in ["one", 1.0]:
                    dat_value = np.ones(dat_layout.shape, dtype=dat_layout.dtype)
        # Return initialized value.
        return dat_value



    def set_shm(self, which, value):
        """Method to set a specific array in shared memory to value.
        """
        if len(which) == 2:
            if self.layout[which[0]][which[1]].min is not None:
                value = np.maximum(value, self.layout[which[0]][which[1]].min)
            if self.layout[which[0]][which[1]].max is not None:
                value = np.minimum(value, self.layout[which[0]][which[1]].max)
            shape = self.layout[which[0]][which[1]].shape
            if is_scalar_shape(shape):
                self.dat[which[0]][which[1]][0] = value
            elif value.shape == self.dat[which[0]][which[1]].shape:
                self.dat[which[0]][which[1]][:] = value
            else:
                print("\nError set_shm: incompatible shapes: " \
                      + str(value.shape) + "   " \
                      + str(self.dat[which[0]][which[1]].shape) \
                      + " for " + str(which))
        elif len(which) == 3:
            if self.layout[which[0]][which[1]][which[2]].min is not None:
                value = np.maximum(value, self.layout[which[0]][which[1]][which[2]].min)
            if self.layout[which[0]][which[1]][which[2]].max is not None:
                value = np.minimum(value, self.layout[which[0]][which[1]][which[2]].max)
            shape = self.layout[which[0]][which[1]][which[2]].shape
            if is_scalar_shape(shape):
                self.dat[which[0]][which[1]][which[2]][0] = value
            elif value.shape == self.dat[which[0]][which[1]][which[2]].shape:
                self.dat[which[0]][which[1]][which[2]][:] = value
            else:
                print("\nError set_shm: incompatible shapes: " \
                      + str(value.shape) + "   " \
                      + str(self.dat[which[0]][which[1]][which[2]].shape) \
                      + " for " + str(which))
        elif len(which) == 4:
            if self.layout[which[0]][which[1]][which[2]][which[3]].min is not None:
                value = np.maximum(value, self.layout[which[0]][which[1]][which[2]][which[3]].min)
            if self.layout[which[0]][which[1]][which[2]][which[3]].max is not None:
                value = np.minimum(value, self.layout[which[0]][which[1]][which[2]][which[3]].max)
            shape = self.layout[which[0]][which[1]][which[2]][which[3]].shape
            if is_scalar_shape(shape):
                self.dat[which[0]][which[1]][which[2]][which[3]][0] = value
            elif value.shape == self.dat[which[0]][which[1]][which[2]][which[3]].shape:
                self.dat[which[0]][which[1]][which[2]][which[3]][:] = value
            else:
                print("\nError set_shm: incompatible shapes: " \
                      + str(value.shape) + "   " \
                      + str(self.dat[which[0]][which[1]][which[2]][which[3]].shape) \
                      + " for " + str(which))
        elif len(which) == 5:
            if self.layout[which[0]][which[1]][which[2]][which[3]][which[4]].min is not None:
                value = np.maximum(value, self.layout[which[0]][which[1]][which[2]][which[3]][which[4]].min)
            if self.layout[which[0]][which[1]][which[2]][which[3]][which[4]].max is not None:
                value = np.minimum(value, self.layout[which[0]][which[1]][which[2]][which[3]][which[4]].max)
            shape = self.layout[which[0]][which[1]][which[2]][which[3]][which[4]].shape
            if is_scalar_shape(shape):
                self.dat[which[0]][which[1]][which[2]][which[3]][which[4]][0] = value
            elif value.shape == self.dat[which[0]][which[1]][which[2]][which[3]][which[4]].shape:
                self.dat[which[0]][which[1]][which[2]][which[3]][which[4]][:] = value
            else:
                print("\nError set_shm: incompatible shapes: " \
                      + str(value.shape) + "   " \
                      + str(self.dat[which[0]][which[1]][which[2]][which[3]][which[4]].shape) \
                      + " for " + str(which))
        else:
            raise NameError("SharedMemory.set_shm() expected item \
                specification of length 2-5, got " + str(len(which)))



    def get_shm(self, which):
        """Method to get a specific array in shared memory.
        """
        if len(which) == 2:
            shape = self.layout[which[0]][which[1]].shape
            if is_scalar_shape(shape):
                return self.dat[which[0]][which[1]][0]
            else:
                return self.dat[which[0]][which[1]][:]
        elif len(which) == 3:
            shape = self.layout[which[0]][which[1]][which[2]].shape
            if is_scalar_shape(shape):
                return self.dat[which[0]][which[1]][which[2]][0]
            else:
                return self.dat[which[0]][which[1]][which[2]][:]
        elif len(which) == 4:
            shape = self.layout[which[0]][which[1]][which[2]][which[3]].shape
            if is_scalar_shape(shape):
                return self.dat[which[0]][which[1]][which[2]][which[3]][0]
            else:
                return self.dat[which[0]][which[1]][which[2]][which[3]][:]
        elif len(which) == 5:
            shape = self.layout[which[0]][which[1]][which[2]][which[3]][which[4]].shape
            if is_scalar_shape(shape):
                return self.dat[which[0]][which[1]][which[2]][which[3]][which[4]][0]
            else:
                return self.dat[which[0]][which[1]][which[2]][which[3]][which[4]][:]
        else:
            raise NameError("SharedMemory.get_shm() expected item \
                specification of length 2-5, got " + str(len(which)))



    def update_net(self, net):
        """Update the given net (its parameters, etc.) from shared memory.
        """
        # Search all parameters of the network in shared memory.
        for i in self.dat:
            # Determine item type.
            i_type = get_item_type(self.net, i)
            if i_type is not None:
                for p in self.dat[i]['parameter']:
                    # TODO: For now update of metas is not done.
                    try:
                        if p in net[S2L(i_type)][i] and is_scalar_shape(self.layout[i]['parameter'][p].shape):
                            net[S2L(i_type)][i][p] = float(self.dat[i]['parameter'][p][0])
                    except:
                        pass


