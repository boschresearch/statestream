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
import copy
import os
import sys

from statestream.meta.neuron_pool import np_state_shape
from statestream.meta.losses import all_losses, has_target
from statestream.utils.yaml_wrapper import load_yaml



# String lookup for short to long conversion.
def S2L(t):
    if t.startswith("np"):
        return "neuron_pools"
    elif t.startswith("sp"):
        return "synapse_pools"
    elif t.startswith("plast"):
        return "plasticities"
    elif t.startswith("if"):
        return "interfaces"
    return None

# =============================================================================
# =============================================================================
# =============================================================================

def cm2fd(meta):
    """Function to convert ruamel CommentedMap to dictionary respecting tuples.
    """
    if isinstance(meta, dict):
        fd = {}
        for k, v in meta.items():
            if k[0] == "(" and k[-1] == ")":
                fd[eval(k)] = cm2fd(v)
            else:
                fd[k] = cm2fd(v)
        return fd
    else:
        if isinstance(meta, str):
            if meta[0] == "(" and meta[-1] == ")":
                return eval(meta)
            else:
                return meta
        else:
            return meta

# =============================================================================
# =============================================================================
# =============================================================================

def get_item_type(net, item_id):
    """Function to determine the item type for an item.
    
    Note: For meta-variables, which in shared-memory are stored
    besides items, this will return None.

    Parameter
    ---------
    net : dictionary
        The network dictionary.
    item_id : string
        The item name / identifier.
    
    Return
    ------
    item_type : str
        The type of the item (np, sp, plast or if).
    """
    item_type = None
    if item_id in net["neuron_pools"]:
        item_type = "np"
    elif item_id in net["synapse_pools"]:
        item_type = "sp"
    elif item_id in net["plasticities"]:
        item_type = "plast"
    elif item_id in net["interfaces"]:
        item_type = "if"
    return item_type

# =============================================================================
# =============================================================================
# =============================================================================

def network_rollback(net, depth, nps, sps):
    """Function to rollback an initialized network rollout.
    """
    # This for loop: build sps[d] and nps[d-1].
    for d in range(depth):
        D = depth - d
        # Loop over all sps.
        for s,S in net["synapse_pools"].items():
            # If sp targets np in current depth D ...
            if S["target"] in nps[D]:
                # ... add sp.
                sps[D].append(s)
                # Add all source nps.
                sources = [item for sub_list in S["source"] for item in sub_list]
                for src in sources:
                    if src not in nps[D - 1]:
                        nps[D - 1].append(src)
    # Now prune nps (and target sps) which have no valid inputs.
    for d in range(depth - 1):
        del_nps = []
        del_sps = []
        for n in nps[d + 1]:
            # Count input sps for this neuron_pool.
            np_ins = False
            for s in sps[d + 1]:
                # Check if np is target of sp.
                if net["synapse_pools"][s]["target"] == n:
                    np_ins = True
                    break
            # If np has no inputs, then memorize it and all its target sps for deletion.
            if not np_ins:
# TODO: This is actually a relevant warning in some cases, but is distracting in general.
#       Best introduce a general verbosity level: debug, warning, error
#                print("\n    Warning: Pruned np " + str(n) + " during rollout.")
                del_nps.append(n)
                for s in sps[d + 2]:
                    S = net["synapse_pools"][s]
                    # Get all sp sources.
                    srcs = [src for sub_list in S["source"] for src in sub_list]
                    # If n in sources, memorize sp for deletion.
                    if n in srcs:
                        del_sps.append(s)
        # Remove memorized nps / sps for this depth.
        for n in del_nps:
            if n in nps[d + 1]:
                # Remove all n.
                nps[d + 1] \
                    = [new_n for new_n in nps[d + 1] if new_n != n]
        for s in del_sps:
            if s in sps[d + 2]:
                # Remove all s.
                sps[d + 2] \
                    = [new_s for new_s in sps[d + 2] if new_s != s]

# =============================================================================
# =============================================================================
# =============================================================================

def is_sane_module_spec(net):
    """Check if specifications of modules in net is sane.
    """
    is_sane = True
    if "modules" in net:
        if not isinstance(net["modules"], dict):
            print("\nError: Modules specification in st_graph must be a dict.")
            is_sane = False
        else:
            for m,M in net["modules"].items():
                if not isinstance(M, dict):
                    print("\nError: Specification of a module must be a dict: " \
                          + str(m))
                    is_sane = False
                else:
                    if "neuron_pools" in M:
                        pass
                    if "synapse_pools" in M:
                        pass
                    if "plasticities" in M:
                        pass
                    if "interfaces" in M:
                        pass
                    if "core_clients" in M:
                        pass

    return is_sane

# =============================================================================
# =============================================================================
# =============================================================================

class MetaNetwork(object):
    def __init__(self, net):
        self.raw_net = copy.deepcopy(net)
        self.net = copy.deepcopy(net)

        # Decorate network graph.
        # =====================================================================
        # Import from other st_graph files.
        if "import" in self.net:
            example_path = os.path.dirname(os.path.abspath(__file__)) \
                           + "/../../examples/"
            for imp in self.net["import"]:
                # Create file name.
                st_graph_file = example_path + str(imp[0]) + ".st_graph"
                # Look if external st_graph file exists and load it as meta network.
                if not os.path.isfile(st_graph_file):
                    print("\nError: Unable to import " + str(imp[0]) \
                          + ". File not found: " + st_graph_file)
                with open(st_graph_file) as f:
                    loc_mn = MetaNetwork(load_yaml(f))
                    loc_net = loc_mn.net
                    E = imp[1]
                    if E in ["neuron_pools", "synapse_pools", "plasticities", "interfaces"]:
                        # Import all items of a type.
                        if E in loc_net:
                            # Add type if not yet present.
                            if E not in self.net:
                                self.net[E] = {}
                            # Add all entries.
                            for i,I in loc_net[E].items():
                                # Add empty item if not exist already.
                                if i not in self.net[E]:
                                    self.net[E][i] = {}
                                # Add item parameters if not exist already.
                                for p,P in I.items():
                                    if p not in self.net[E][i]:
                                        self.net[E][i][p] = copy.copy(P)
                    elif E == "modules":
                        # Import all modules from source.
                        if E in loc_net:
                            # Add type if not yet present.
                            if E not in self.net:
                                self.net[E] = {}
                            # Add all entries.
                            for m,M in loc_net[E].items():
                                # Add empty item if not exist already.
                                if m not in self.net[E]:
                                    self.net[E][m] = copy.copy(M)
                    elif E.startswith("modules."):
                        # Import a single module definition.
                        mods = E.split(".")
                        if "modules" in loc_net:
                            if mods[1] in loc_net["modules"]:
                                if not "modules" in self.net:
                                    self.net["modules"] = {}
                                if mods[1] not in self.net["modules"]:
                                    self.net["modules"][mods[1]] \
                                        = copy.copy(loc_net["modules"][mods[1]])
                    elif E == "tag_specs":
                        # Import all modules from source.
                        if E in loc_net:
                            # Add type if not yet present.
                            if E not in self.net:
                                self.net[E] = {}
                            # Add all entries.
                            for m,M in loc_net[E].items():
                                # Add empty item if not exist already.
                                if m not in self.net[E]:
                                    self.net[E][m] = copy.copy(M)
                    elif E.startswith("tag_specs."):
                        # Import a single module definition.
                        mods = E.split(".")
                        if "tag_specs" in loc_net:
                            if mods[1] in loc_net["tag_specs"]:
                                if not "tag_specs" in self.net:
                                    self.net["tag_specs"] = {}
                                if mods[1] not in self.net["tag_specs"]:
                                    self.net["tag_specs"][mods[1]] \
                                        = copy.copy(loc_net["tag_specs"][mods[1]])
                    elif E.startswith("neuron_pools.") \
                            or E.startswith("synapse_pools.") \
                            or E.startswith("plasticities.") \
                            or E.startswith("interfaces."):
                        # Import all items of type.tag
                        mods = E.split(".")
                        for i,I in loc_net[mods[0]].items():
                            if "tags" in I:
                                if mods[1] in I["tags"]:
                                    if not i in self.net[mods[0]]:
                                        self.net[mods[0]][i] = {}
                                    # Add item parameters if not exist already.
                                    for p,P in I.items():
                                        if p not in self.net[mods[0]][i]:
                                            self.net[mods[0]][i][p] = copy.copy(P)
                    elif E.startswith("globals"):
                        # Import all or specified global variable(s).
                        if "globals" in loc_net:
                            if E == "globals":
                                # Import all global variables.
                                if not "globals" in self.net:
                                    self.net["globals"] = {}
                                for i,I in loc_net["globals"].items():
                                    self.net["globals"][i] = copy.copy(I)
                            elif E.startswith("globals."):
                                # Import specific global variable.
                                if not "globals" in self.net:
                                    self.net["globals"] = {}
                                mods = E.split(".")
                                if mods[1] in loc_net["globals"]:
                                    self.net["globals"][mods[1]] = copy.copy(loc_net["globals"][mods[1]])
                    else:
                        # Assume that a tag is given and import all items with this tag.
                        for t in ["neuron_pools", "synapse_pools", "plasticities", "interfaces"]:
                            for i,I in loc_net[t].items():
                                if "tags" in I:
                                    if E in I["tags"]:
                                        if not i in self.net[t]:
                                            self.net[t][i] = copy.copy(I)



        # Convert modules to subgraphs.
        if "modules" in self.net:
            # Loop over all module types.
            for mt,MT in self.net["modules"].items():
                # Loop over all instances of module type.
                if mt in self.net:
                    for mi,MI in self.net[mt].items():
                        # Add all core-clients for this module.
                        if "core_clients" in MT:
                            for cc,CC in MT["core_clients"].items():
                                cc_name = mi + "_" + cc
                                if "core_clients" not in self.net:
                                    self.net["core_clients"] = {}
                                if cc_name not in self.net["core_clients"]:
                                    self.net["core_clients"][cc_name] = {}
                                    for i,I in CC.items():
                                        if isinstance(I, str):
                                            if I.startswith("_"):
                                                self.net["core_clients"][cc_name][i] \
                                                    = MI[I[1:]]
                                            else:
                                                if I in MT.get('neuron_pools', {}) \
                                                        or I in MT.get('synapse_pools', {}) \
                                                        or I in MT.get('interfaces', {}) \
                                                        or I in MT.get('plasticities', {}):
                                                    self.net["core_clients"][cc_name][i] \
                                                        = mi + "_" + I
                                                else:
                                                    self.net["core_clients"][cc_name][i] \
                                                        = I
                                        elif isinstance(I, list):
                                            self.net["core_clients"][cc_name][i] = []
                                            for li in range(len(I)):
                                                if isinstance(I[li], str):
                                                    if I[li].startswith("_"):
                                                        self.net["core_clients"][cc_name][i].append(MI[I[li][1:]])
                                                    else:
                                                        self.net["core_clients"][cc_name][i].append(I[li])
                                                elif isinstance(I[li], list):
                                                    pass
                                                else:
                                                    self.net["core_clients"][cc_name][i].append(I[li])
                                        else:
                                            self.net["core_clients"][cc_name][i] = I
                        # Add all neuron-pools for this module.
                        if "neuron_pools" in MT:
                            for n,N in MT["neuron_pools"].items():
                                np_name = mi + "_" + n
                                if np_name not in self.net["neuron_pools"]:
                                    self.net["neuron_pools"][np_name] = {}
                                    for i,I in N.items():
                                        if isinstance(I, str):
                                            if I.startswith("_"):
                                                self.net["neuron_pools"][np_name][i] \
                                                    = MI[I[1:]]
                                            else:
                                                self.net["neuron_pools"][np_name][i] \
                                                    = I
                                        elif isinstance(I, list):
                                            self.net["neuron_pools"][np_name][i] = []
                                            for li in range(len(I)):
                                                if isinstance(I[li], str):
                                                    if I[li].startswith("_"):
                                                        self.net["neuron_pools"][np_name][i].append(MI[I[li][1:]])
                                                    else:
                                                        self.net["neuron_pools"][np_name][i].append(I[li])
                                                elif isinstance(I[li], list):
                                                    pass
                                                else:
                                                    self.net["neuron_pools"][np_name][i].append(I[li])
                                        else:
                                            self.net["neuron_pools"][np_name][i] = I

                        # Add all synapse-pools for this module.
                        if "synapse_pools" in MT:
                            for s,S in MT["synapse_pools"].items():
                                sp_name = mi + "_" + s
                                if sp_name not in self.net["synapse_pools"]:
                                    self.net["synapse_pools"][sp_name] = {}
                                    for i,I in S.items():
                                        if isinstance(I, str):
                                            if I.startswith("_"):
                                                self.net["synapse_pools"][sp_name][i] \
                                                    = MI[I[1:]]
                                            else:
                                                if i == "target" and I in MT["neuron_pools"]:
                                                    self.net["synapse_pools"][sp_name][i] \
                                                        = mi + "_" + I
                                                else:
                                                    self.net["synapse_pools"][sp_name][i] \
                                                        = I
                                        elif isinstance(I, list):
                                            self.net["synapse_pools"][sp_name][i] = []
                                            for li in range(len(I)):
                                                if isinstance(I[li], str):
                                                    if I[li].startswith("_"):
                                                        self.net["synapse_pools"][sp_name][i].append(MI[I[li][1:]])
                                                    else:
                                                        self.net["synapse_pools"][sp_name][i].append(I[li])
                                                elif isinstance(I[li], list):
                                                    self.net["synapse_pools"][sp_name][i].append([])
                                                    for lii in range(len(I[li])):
                                                        if isinstance(I[li][lii], str):
                                                            if I[li][lii].startswith("_"):
                                                                self.net["synapse_pools"][sp_name][i][-1].append(MI[I[li][lii][1:]])
                                                            else:
                                                                if i == "source" and I[li][lii] in MT["neuron_pools"]:
                                                                    src_np_name = mi + "_" + I[li][lii]
                                                                    self.net["synapse_pools"][sp_name][i][-1].append(src_np_name)
                                                                else:
                                                                    self.net["synapse_pools"][sp_name][i][-1].append(I[li][lii])
                                                        else:
                                                            self.net["synapse_pools"][sp_name][i][-1].append(I[li][lii])
                                                else:
                                                    self.net["synapse_pools"][sp_name][i].append(I[li])
                                        elif isinstance(I, dict) and i == "share params":
                                            self.net["synapse_pools"][sp_name][i] = {}
                                            for li in I:
                                                if isinstance(I[li], list):
                                                    self.net["synapse_pools"][sp_name][i][li] = []
                                                    for lii in range(len(I[li])):
                                                        if isinstance(I[li][lii], str):
                                                            if I[li][lii].startswith("_"):
                                                                self.net["synapse_pools"][sp_name][i][li].append(MI[I[li][lii][1:]])
                                                            else:
                                                                if I[li][lii] in MT["synapse_pools"]:
                                                                    src_sp_name = mi + "_" + I[li][lii]
                                                                    self.net["synapse_pools"][sp_name][i][li].append(src_sp_name)
                                                                else:
                                                                    self.net["synapse_pools"][sp_name][i][li].append(I[li][lii])
                                        else:
                                            self.net["synapse_pools"][sp_name][i] = I

                        # Add all plasticities for this module.
                        if "plasticities" in MT:
                            for p,P in MT["plasticities"].items():
                                plast_name = mi + "_" + p
                                if plast_name not in self.net["plasticities"]:
                                    self.net["plasticities"][plast_name] = {}
                                    for i,I in P.items():
                                        if isinstance(I, str):
                                            if I.startswith("_"):
                                                self.net["plasticities"][plast_name][i] \
                                                    = MI[I[1:]]
                                            else:
                                                new_I = copy.copy(I)
                                                if i in ["source", "target"]:
                                                    if I in MT["neuron_pools"]:
                                                        new_I = mi + "_" + I
                                                self.net["plasticities"][plast_name][i] \
                                                    = new_I
                                        elif isinstance(I, list):
                                            self.net["plasticities"][plast_name][i] = []
                                            if i == "parameter":
                                                for par in I:
                                                    if (par[0] == "sp" and par[1] in MT["synapse_pools"]) \
                                                            or (par[0] == "np" and par[1] in MT["neuron_pools"]):
                                                        val = copy.copy(par)
                                                        val[1] = mi + "_" + par[1]
                                                        self.net["plasticities"][plast_name][i].append(copy.copy(val))
                                                    elif par[1].startswith('_'):
                                                        val = copy.copy(par)
                                                        val[1] = MI[par[1][1:]]
                                                        self.net["plasticities"][plast_name][i].append(copy.copy(val))
                                                    else:
                                                        self.net["plasticities"][plast_name][i].append(copy.copy(par))
                                            else:
                                                for li in range(len(I)):
                                                    if isinstance(I[li], str):
                                                        if I[li].startswith("_"):
                                                            self.net["plasticities"][plast_name][i].append(MI[I[li][1:]])
                                                        else:
                                                            self.net["plasticities"][plast_name][i].append(I[li])
                                                    else:
                                                        self.net["plasticities"][plast_name][i].append(I[li])
                                        else:
                                            self.net["plasticities"][plast_name][i] = I

                        # Add all interfaces for this module.
                        if "interfaces" in MT:
                            for p,P in MT["interfaces"].items():
                                if_name = mi + "_" + p
                                if if_name not in self.net["interfaces"]:
                                    self.net["interfaces"][if_name] = {}
                                    for i,I in P.items():
                                        if isinstance(I, str):
                                            if I.startswith("_"):
                                                self.net["interfaces"][if_name][i] \
                                                    = MI[I[1:]]
                                            else:
                                                self.net["interfaces"][if_name][i] \
                                                    = I
                                        elif i == "remap":
                                            self.net["interfaces"][if_name]["remap"] = {}
                                            for rm,RM in I.items():
                                                if RM in MT["neuron_pools"]:
                                                    np_name = mi + "_" + RM
                                                elif RM.startswith('_'):
                                                    np_name = MI[RM[1:]] 
                                                else:
                                                    np_name = RM
                                                self.net["interfaces"][if_name]["remap"][rm] = np_name
                                        elif isinstance(I, list):
                                            self.net["interfaces"][if_name][i] = []
                                            for li in range(len(I)):
                                                if isinstance(I[li], str):
                                                    if I[li].startswith("_"):
                                                        self.net["interfaces"][if_name][i].append(MI[I[li][1:]])
                                                    else:
                                                        self.net["interfaces"][if_name][i].append(I[li])
                                                else:
                                                    self.net["interfaces"][if_name][i].append(I[li])
                                        else:
                                            self.net["interfaces"][if_name][i] = copy.copy(I)

        # Check for tag specifications.
        if "tag_specs" in self.net:
            for t,T in self.net["tag_specs"].items():
                # Search for items with this tag.
                for mod in ["neuron_pools", "synapse_pools", "plasticities", "interfaces"]:
                    for i,I in self.net[mod].items():
                        if "tags" in I:
                            if t in I["tags"]:
                                # Apply all tag specifications.
                                for ts,TS in T.items():
                                    # Do not overwrite existing parameters.
                                    if ts not in I:
                                        I[ts] = TS

        # Check for global integers and floats.
        if "globals" in self.net:
            for mod in ["neuron_pools", "synapse_pools", "plasticities", "interfaces"]:
                for s,S in self.net[mod].items():
                    for i,I in S.items():
                        if isinstance(I, str):
                            contains_global = False
                            for gi,GI in self.net["globals"].items():
                                if gi in I:
                                    I = I.replace(gi, str(GI))
                                    contains_global = True
                            if contains_global:
                                S[i] = eval(I)
                        elif isinstance(I, list):
                            for e,E in enumerate(I):
                                if isinstance(E, str):
                                    contains_global = False
                                    for gi,GI in self.net["globals"].items():
                                        if gi in E:
                                            E = E.replace(gi, str(GI))
                                            contains_global = True
                                    if contains_global:
                                        S[i][e] = eval(E)
                                elif isinstance(E, list):
                                    for ee,EE in enumerate(E):
                                        if isinstance(EE, str):
                                            contains_global = False
                                            for gi,GI in self.net["globals"].items():
                                                if gi in EE:
                                                    EE = EE.replace(gi, str(GI))
                                                    contains_global = True
                                            if contains_global:
                                                S[i][e][ee] = eval(EE)

        # Time to check sanity of build network.
        if not self.is_sane():
            sys.exit()

        # Add devices for all nps and plasts to tags.
        for i in ["np", "sp", "plast"]:
            for n,N in self.net[S2L(i)].items():
                if "device" in N:
                    dev = N["device"]
                else:
                    dev = "cpu"
                if "tags" in N:
                    if dev not in N["tags"]:
                        N["tags"].append(dev)
                else:
                    N["tags"] = [dev]

        self.no_nps = len(self.net.get('neuron_pools', {}))
        self.no_sps = len(self.net.get('synapse_pools', {}))
        self.no_ifs = len(self.net.get('interfaces', {}))
        self.no_plasts = len(self.net.get('plasticities', {}))
        self.no_neurons = 0
        self.no_synapses = 0
        self.no_processes = 2 + self.no_plasts + self.no_ifs    # viz + main


        # Neuron-pool shapes.
        self.np_shape = {}
        for n,N in self.net["neuron_pools"].items():
            # update number of neurons
            self.no_neurons += np.prod(N["shape"])
            # update number of processes
            np_is_produced = False
            for p,P in self.net["interfaces"].items():
                if n in P["out"]:
                    np_is_produced = True
                    break
            if not np_is_produced:
                self.no_processes += 1
            # dependent on np shape set its 4D explicit shape
            if len(N["shape"]) == 2:
                self.np_shape[n] = [self.net["agents"],
                                    N["shape"][0],
                                    N["shape"][1],
                                    1]
            elif len(N["shape"]) == 1:
                self.np_shape[n] = [self.net["agents"],
                                    N["shape"][0],
                                    1,
                                    1]
            elif len(N["shape"]) == 3:
                self.np_shape[n] = [self.net["agents"],
                                    N["shape"][0],
                                    N["shape"][1],
                                    N["shape"][2]]
            else:
                print("Error: np " + n + " is of shape: " + str(N["shape"]))

        # Compute number of weights / synapses.
        for s,S in self.net["synapse_pools"].items():
            tgt_shape = self.np_shape[S["target"]]
            for f in range(len(S["source"])):
                for src in range(len(S["source"][f])):
                    src_shape = self.np_shape[S["source"][f][src]]
                    if "rf" in S:
                        if isinstance(S["rf"], list):
                            rf = S["rf"][f][src]
                        else:
                            rf = S["rf"]
                        self.no_synapses += tgt_shape[0] * src_shape[0] * rf * rf
                    else:
                        # Assume fully connected.
                        self.no_synapses += np.prod(src_shape) * np.prod(tgt_shape)


        # Get list of all tags and dict of tag -> items associations.
        self.tags = []
        self.tags_t2i = {}
        for mod in ["np", "sp", "if", "plast"]:
            for i in self.net[S2L(mod)]:
                if "tags" in self.net[S2L(mod)][i]:
                    for t in self.net[S2L(mod)][i]["tags"]:
                        if t not in self.tags_t2i:
                            self.tags_t2i[t] = []
                        if i not in self.tags_t2i[t]:
                            self.tags_t2i[t].append(i)
                        if t not in self.tags:
                            self.tags.append(t)

        # Compute nets for nps.
        self.net_np_nps = {}
        self.net_np_sps = {}
        for n in self.net["neuron_pools"]:
            self.net_np_nps[n] = [n]
            self.net_np_sps[n] = []
            for s,S in self.net["synapse_pools"].items():
                assert("target" in S), "ERROR: st_meta_network: " \
                    + "no target parameter found for sp " + s
                if S["target"] == n:
                    # further on with graph for np
                    self.net_np_sps[n].append(s)
                    sources = S["source"]
                    for source_np in [item for sub_list in sources for item in sub_list]:
                        if source_np not in self.net_np_nps[n]:
                            self.net_np_nps[n].append(source_np)

        # Compute nets for plasts.
        self.net_plast_nps = {}
        self.net_plast_sps = {}
        self.net_plast_depth = {}
        for p,P in self.net["plasticities"].items():
            if P["type"] in ["hebbian"]:
                # For hebb we actually need no deep computation.
                self.net_plast_depth[p] = 0
                # Generate empty lists.
                self.net_plast_nps[p] = [[] for i in range(self.net_plast_depth[p] + 1)]
                self.net_plast_sps[p] = [[] for i in range(self.net_plast_depth[p] + 1)]
                # Add target to np list at time 0.
                self.net_plast_nps[p][-1].append(P["target"])
                # Check number of parameter.
                if len(P["parameter"]) != 1:
                    print("WARNING: For plasticity " + p + " of " + P["type"] \
                          + " type more than one parameter was found.")
                # Add all sources of sps.
                src_sp = P["parameter"][0][1]
                for sublist in range(len(self.net["synapse_pools"][src_sp]["source"])):
                    for src in range(len(self.net["synapse_pools"][src_sp]["source"][sublist])):
                        self.net_plast_nps[p][-1].append(self.net["synapse_pools"][src_sp]["source"][sublist][src])
                # Add sp.
                if P["parameter"][0][0] == "sp":
                    self.net_plast_sps[p][-1].append(P["parameter"][0][1])
                else:
                    print("Warning: For " + P["type"] + " plasticity " + p \
                          + " found parameter other then from sp: " \
                          + str(P["parameter"][0]))
            elif P["type"] == "loss":
                # Set depth of plasticity graph to max of source and target.
                if has_target(P.get("loss_function", None)):
                    self.net_plast_depth[p] = max(P["source_t"], P["target_t"])
                else:
                    self.net_plast_depth[p] = P["source_t"]
                # lists of needed nps / sps for each level of depth
                # note: nps[0] is all inputs, and these are all inputs
                #       sps[d] are the input sps to nps[d], hence:
                #       sps[0] is always []
                #       the minimal distance from target to overall inputs must be at least depth
                # In principle this builds a reverse pyramid from target to all sources of distance depth.
                self.net_plast_nps[p] = [[] for i in range(self.net_plast_depth[p] + 1)]
                self.net_plast_sps[p] = [[] for i in range(self.net_plast_depth[p] + 1)]
                # Initialize with source and target layer.
                if has_target(P.get("loss_function", None)):
                    self.net_plast_nps[p][P["target_t"]].append(P["target"])
                self.net_plast_nps[p][P["source_t"]].append(P["source"])
                # Consider masking neuron-pool.
                if "mask" in P:
                    if has_target(P.get("loss_function", None)):
                        self.net_plast_nps[p][P["target_t"]].append(P["mask"])
                    else:
                        self.net_plast_nps[p][P["source_t"]].append(P["mask"])
                # Consider uncertainty.
                if P["loss_function"] == "reg_uncertainty":
                    self.net_plast_nps[p][P["source_t"]].append(P["uncertainty"])

                network_rollback(self.net, self.net_plast_depth[p], self.net_plast_nps[p], self.net_plast_sps[p])

                # Nice print of rolled-out network for specific loss.
                # if p == "loss_scalar":
                #     print("\nNeuron-pools:")
                #     for np_d in range(len(self.net_plast_nps[p])):
                #         print("\nDepth: " + str(np_d) + "   " + str(self.net_plast_nps[p][np_d]))
                #     print("\nSynapse-pools:")
                #     for sp_d in range(len(self.net_plast_sps[p])):
                #         print("\nDepth: " + str(sp_d) + "   " + str(self.net_plast_sps[p][sp_d]))

            elif P["type"] == "L_regularizer":
                # set depth of plasticity graph to 0 (no rollout necessary)
                self.net_plast_depth[p] = 0
                self.net_plast_nps[p] = [[]]
                self.net_plast_sps[p] = [[]]
                # initialize with all np / sp of which parameters are regularized
                for par in P["parameter"]:
                    if par[0] == "np":
                        if par[1] not in self.net_plast_nps[p][0]:
                            self.net_plast_nps[p][0].append(par[1])
                    elif par[0] == "sp":
                        # add sps
                        if par[1] not in self.net_plast_sps[p][0]:
                            self.net_plast_sps[p][0].append(par[1])
                        # and also add all sp"s source nps (needed to init sp) and targets
                        self.net_plast_nps[p][0].append(self.net["synapse_pools"][par[1]]["target"])
                        sources = [item for sub_list in self.net["synapse_pools"][par[1]]["source"] for item in sub_list]
                        for src in sources:
                            if src not in self.net_plast_nps[p][0]:
                                self.net_plast_nps[p][0].append(src)


    def is_sane(self):
        """Sanity check for entire network structure.
        """
        is_sane = True

        if not isinstance(self.net, dict):
            print("    Error: Got no dictionary from st_graph file.")
            return False

        # Check for name and agents.
        if "name" not in self.net:
            print("    Error: Please specify a network name (e.g. name: test).")
            is_sane = False
        if "agents" not in self.net:
            print("    Error: Please specify number of agents (aka. batchsize) " \
                  + "(e.g. agents: 16).")
            is_sane = False
        # Check that np, sp, plast, if is in net.
        T = []
        for t in ["np", "sp", "plast", "if"]:
            if S2L(t) not in self.net:
                print("    Error: Unable to find " + S2L(t) + ".")
                is_sane = False
            else:
                if not isinstance(self.net[S2L(t)], dict):
                    print("    Error: Network definition of " + str(S2L(t)) \
                          + " must be a dictionary.")
                    is_sane = False
                else:
                    T.append(t)
        # Check for valid item names.
        for t in T:
            for n in self.net[S2L(t)]:
                if self.net[S2L(t)][n] is None:
                    print("    Error: Found empty item: '" + str(n) + "'")
                    is_sane = False
                for c in n:
                    if not c.isalpha() and not c.isdigit() and not c == "_":
                        print("    Error: Item names may only contain " \
                              + "[a-zA-Z0-9_]. Invalid name: '" + str(n) + "'")
                        is_sane = False
                        break
        # Check for unique item names (across item types).
        for t0 in T:
            for t1 in T:
                if t0 != t1:
                    for n0 in self.net[S2L(t0)]:
                        for n1 in self.net[S2L(t1)]:
                            if n0 == n1:
                                print("    Error: Item names must be unique. " \
                                      + "Found '" + n0 + "' as " + t0 + " and " + t1)
                                is_sane = False
        # Do some general np checks.
        if "np" in T:
            for n, N in self.net["neuron_pools"].items():
                # Check for valid 'bias_shape' np parameter values.
                if "bias_shape" in N:
                    if N["bias_shape"] not in ["full", "feature", "spatial", "scalar", False]:
                        print("    Error: Invalid value of 'bias_shape' parameter found for np " \
                              + n)
                        is_sane = False
                # Check for valid 'gain_shape' np parameter values.
                if "gain_shape" in N:
                    if N["gain_shape"] not in ["full", "feature", "spatial", "scalar", False]:
                        print("    Error: Invalid value of 'gain_shape' parameter found for np " \
                              + n)
                        is_sane = False
                # Check for valid batch normalization.
                if "batchnorm_mean" in N:
                    if N["batchnorm_mean"] not in ["full", "feature", "spatial", "scalar", False]:
                        print("    Error: Invalid value of 'batchnorm_mean' parameter found for np " \
                              + n)
                        is_sane = False
                if "batchnorm_std" in N:
                    if N["batchnorm_std"] not in ["full", "feature", "spatial", "scalar", False]:
                        print("    Error: Invalid value of 'batchnorm_std' parameter found for np " \
                              + n)
                        is_sane = False
                # Check for valid layer normalization.
                if "layernorm_mean" in N:
                    if N["layernorm_mean"] not in ["full", "feature", "spatial", False]:
                        print("    Error: Invalid value of 'layernorm_mean' parameter found for np " \
                              + n)
                        is_sane = False
                if "layernorm_std" in N:
                    if N["layernorm_std"] not in ["full", "feature", "spatial", False]:
                        print("    Error: Invalid value of 'layernorm_std' parameter found for np " \
                              + n)
                        is_sane = False
        # Some sp checks.
        if "sp" in T:
            for s, S in self.net["synapse_pools"].items():
                if S is not None:
                    # Check that all synapse pools have source and target.
                    if "source" not in S:
                        print("    Error: No source(s) specified for synapse \
                              pool '" + s + "'.")
                        is_sane = False
                    else:
                        # Check for correct factor_shapes length and values.
                        if "factor_shapes" in S:
                            if not isinstance(S["factor_shapes"], list):
                                print("    Error: For synapse \
                                      pool '" + s + "'. factor_shapes must be a list\
                                      with length equal to number of factors.")
                                is_sane = False
                            else:
                                if len(S["factor_shapes"]) != len(S["source"]):
                                    print("    Error: For synapse \
                                          pool '" + s + "'. If factor_shapes defined, \
                                          they must be defined for all factors.")
                                    is_sane = False
                                else:
                                    for f in S["factor_shapes"]:
                                        if f not in ["full", "feature", "spatial", "scalar"]:
                                            print("    Error: For synapse \
                                                  pool '" + s + "'. Unexpected factor shape: " \
                                                  + str(f) + ". Expected: full, feature, spatial or scalar.")
                                            is_sane = False
                        # Some checks for bias shapes.
                        if "bias_shapes" in S:
                            # Check if list.
                            if not isinstance(S["bias_shapes"], list):
                                print("    Error: For synapse \
                                      pool '" + s + "'. bias_shapes must be a list\
                                      with length equal to number of factors.")
                                is_sane = False
                            else:
                                # Check for correct length of list.
                                if len(S["bias_shapes"]) != len(S["source"]):
                                    print("    Error: For synapse \
                                          pool '" + s + "'. If bias_shapes defined, \
                                          they must be defined for all factors.")
                                    is_sane = False
                                else:
                                    # Check consistency with factor_shapes.
                                    if "factor_shapes" in S:
                                        if isinstance(S["factor_shapes"], list):
                                            if len(S["factor_shapes"]) == len(S["source"]):
                                                for f in range(len(S["factor_shapes"])):
                                                    if S["factor_shapes"][f] == "feature":
                                                        if S["bias_shapes"][f] not in ["feature", "scalar"]:
                                                            print("    Error: For synapse \
                                                                  pool '" + s + "'. Factor of shape feature " \
                                                                  + " allows only bias shapes feature and scalar."\
                                                                  + " Got: " + S["bias_shapes"][f])
                                                            is_sane = False
                                                    elif S["factor_shapes"][f] == "spatial":
                                                        if S["bias_shapes"][f] not in ["spatial", "scalar"]:
                                                            print("    Error: For synapse \
                                                                  pool '" + s + "'. Factor of shape spatial " \
                                                                  + " allows only bias shapes spatial and scalar."\
                                                                  + " Got: " + S["bias_shapes"][f])
                                                            is_sane = False
                                                    if S["factor_shapes"][f] == "scalar":
                                                        if S["bias_shapes"][f] not in ["scalar"]:
                                                            print("    Error: For synapse \
                                                                  pool '" + s + "'. Factor of shape scalar " \
                                                                  + " allows only bias shape scalar."\
                                                                  + " Got: " + S["bias_shapes"][f])
                                                            is_sane = False


                    if "target" not in S:
                        print("    Error: No target specified for synapse pool '" \
                              + s + "'.")
                        is_sane = False

        # Check sp source / target existence.
        if "sp" in T and "np" in T:
            for s,S in self.net["synapse_pools"].items():
                if S is not None:
                    if "target" in S:
                        if S["target"] not in self.net["neuron_pools"]:
                            print("    Error: For sp " + s + " target np '" \
                                  + S["target"] + "' not found.")
                            is_sane = False
                    if "source" in S:
                        sources = [src for sublist in S["source"] for src in sublist]
                        for src in sources:
                            if src not in self.net["neuron_pools"]:
                                print("    Error: For sp " + s + " source np '" \
                                      + str(src) + "' not found.")
                                is_sane = False
                                break
        # Check correct sizes of sp's list(list) parameters which should all be the
        # same size as source.
        for s, S in self.net.get("synapse_pools", {}).items():
            if "source" in S:
                for t in ["target_shapes", "weight_shapes", "dilation", "weight_fnc", "ppp"]:
                    if t in S:
                        if not isinstance(S[t], list):
                            print("    Error: For sp " + s + ". Parameter " + t + " must be list "\
                                  + "of lists of same size as sp's sources. Found no list: " + str(S[t]))
                            is_sane = False
                        else:
                            if len(S[t]) != len(S["source"]):
                                print("    Error: For sp " + s + ". Parameter " + t + " must be list "\
                                      + "of lists of same size as sp's sources. Found: " + str(S[t]))
                                is_sane = False
                            else:
                                for srcs, t_l in zip(S["source"], S[t]):
                                    if not isinstance(t_l, list):
                                        print("    Error: For sp " + s + ". Parameter " + t + " must be list "\
                                              + "of lists of same size as sp's sources. Found no list: " + str(S[t]))
                                        is_sane = False
                                        break
                                    else:
                                        if len(t_l) != len(srcs):
                                            print("    Error: For sp " + s + ". Parameter " + t + " must be list "\
                                                  + "of lists of same size as sp's sources. Found: " + str(S[t]))
                                            is_sane = False
                                            break
        # Test for sp that 'act' parameter is a list of length factors.
        for s, S in self.net.get("synapse_pools", {}).items():
            if "act" in S:
                if not isinstance(S["act"], list):
                    print("    Error: For sp " + s + ". Parameter act must be list "\
                          + "of activations, one for each factor. Found: " + str(S["act"]))
                    is_sane = False
                else:
                    if len(S["act"]) != len(S["source"]):
                        print("    Error: For sp " + s + ". Parameter act must be list "\
                              + "of activations, one for each factor. Found: " + str(S["act"]))
                        is_sane = False
        # For sps with target / source without space check for None rf.
        for s, S in self.net.get("synapse_pools", {}).items():
            # Determine number of factors.
            if "source" in S and "target" in S:
                target_shape = np_state_shape(self.net, S["target"])
                no_factors = len(S["source"])
                # Check spatiality of target.
                if target_shape[2] == 1 and target_shape[3] == 1:
                    if "rf" in S:
                        if isinstance(S["rf"], list):
                            for srcs in S["rf"]:
                                for src in srcs:
                                    if src != 1:
                                        print("    Error: For sp " + s \
                                              + " no rf parameter allowed for sp with target " \
                                              + "without spatial dimensions.")
                                        is_sane = False
                        else:
                            if S["rf"] != 1:
                                print("    Error: For sp " + s \
                                      + " no rf parameter allowed for sp with target " \
                                      + "without spatial dimensions.")
                                is_sane = False
                # Get list of all sources.
                sources = [src for sublist in S["source"] for src in sublist]
                rf = S.get("rf", [[]])
                # rf may not be a list of list only iff one source.
                if not isinstance(rf, list) and len(sources) != 1:
                    print("    Error: For sp " + s \
                          + " Found non-list rf parameter for sp with more than one source.")
                    is_sane = False
                # Check spatiality of sources.
                for src in sources:
                    if src in self.net.get('neuron_pools', {}):
                        source_shape = np_state_shape(self.net, src)
                        if source_shape[2] == 1 and source_shape[3] == 1:
                            if "rf" in S:
                                if isinstance(S["rf"], list):
                                    for srcs in S["rf"]:
                                        for rf in srcs:
                                            if rf != 1:
                                                print("    Error: For sp " + s \
                                                      + " source np " + str(src) + " has no space, but rf defined.")
                                                is_sane = False
                                else:
                                    if S["rf"] not in [0, 1, None]:
                                        print("    Error: For sp " + s \
                                              + " source np " + str(src) + " has no space, but rf defined.")
                                        is_sane = False
        # Check consistancy of spatial dims for connected nps.
        for s, S in self.net.get("synapse_pools", {}).items():
            if "source" in S and "target" in S:
                target_shape = np_state_shape(self.net, S["target"])
                # If target has no spatial dimension everything is fully connected.
                if target_shape[2] == 1 and target_shape[3] == 1:
                    continue
                # Check consistency for all sources.
                sources = [src for sublist in S["source"] for src in sublist]
                for src in sources:
                    source_shape = np_state_shape(self.net, src)
                    # Nothing to do if source has no spatial dimensions.
                    if source_shape[2] == 1 and source_shape[3] == 1:
                        break
                    # Determine pooling.
                    if source_shape[2] == target_shape[2]:
                        pooling = 1
                    elif source_shape[2] > target_shape[2]:
                        pooling = int(source_shape[2] \
                                      / target_shape[2])
                    elif target_shape[2] > source_shape[2]:
                        pooling = -int(target_shape[2] \
                                       / source_shape[2])
                    # Now check for consistancy.
                    if pooling > 0:
                        if pooling * target_shape[2] != source_shape[2] \
                                or pooling * target_shape[3] != source_shape[3]:
                            print("    Error: For sp " + s \
                                  + " inconsistant spatial dims for source: " + src)
                            is_sane = False
                    else:
                        if -pooling * source_shape[2] != target_shape[2] \
                                or -pooling * source_shape[3] != target_shape[3]:
                            print("    Error: For sp " + s \
                                  + " inconsistant spatial dims for source: " + src)
                            is_sane = False
        # Check plasticity and their parameters.
        for p, P in self.net.get("plasticities", {}).items():
            # If samples in plasticity check its larger than agents.
            if "samples" in P:
                if P["samples"] <= self.net["agents"]:
                    print("    Error: For plast " + p + ", samples must be larger than agents.")
                    is_sane = False
            # Check for type.
            if "type" not in P:
                print("    Error: For plast " + p + ", no type found.")
                is_sane = False
            # Check target / source consistancy.
            if "source" in P:
                if P["source"] not in self.net["neuron_pools"]:
                    print("    Error: For plast " + p \
                          + " source defined as " + str(P["source"] \
                          + " but no neuron-pool found."))
                    is_sane = False
            if "target" in P:
                if P["target"] not in self.net["neuron_pools"]:
                    print("    Error: For plast " + p \
                          + " target defined as " + str(P["target"] \
                          + " but no neuron-pool found."))
                    is_sane = False
            if "source" in P and "target" in P:
                if P["source"] in self.net["neuron_pools"] \
                        and P["target"] in self.net["neuron_pools"]:
                    source_shape = np_state_shape(self.net, P["source"])
                    target_shape = np_state_shape(self.net, P["target"])
                    # For 'loss' plasticity both those have to be equal.
                    if "type" in P:
                        if P["type"] == "loss":
                            if source_shape != target_shape:
                                print("    Error: For plast " + p \
                                      + " of type loss, target and source np " \
                                      + " must be of same shape.")
                                is_sane = False
            # Check parameter structure.
            if not "parameter" in P:
                print("    Error: For plast " + p \
                      + " no parameters defined.")
                is_sane = False
            else:
                if len(P["parameter"]) == 0:
                    print("    Error: For plast " + p \
                          + " no parameters defined.")
                    is_sane = False
                else:
                    for par in range(len(P["parameter"])):
                        if len(P["parameter"][par]) != 3:
                            print("    Error: For plast " + p \
                                  + " wrong parameter specification. Expected: " \
                                  + "[item_type, item_name, par_name]. Got: " \
                                  + str(P["parameter"][par]))
                            is_sane = False
                        else:
                            if P["parameter"][par][0] not in ["np", "sp", "plast", "if"]:
                                print("    Error: For plast " + p \
                                      + " wrong parameter specification. Expected: " \
                                      + "[item_type, item_name, par_name]. Got: " \
                                      + str(P["parameter"][par]))
                                is_sane = False
                            else:
                                # Check that item exists.
                                if P["parameter"][par][1] not in self.net[S2L(P["parameter"][par][0])]:
                                    print("    Error: For plast " + p \
                                          + " unable to find item " + str(P["parameter"][par][1]) \
                                          + " for parameter " + str(P["parameter"][par]))
                                    is_sane = False
                                else:
                                    # Check for doubles.
                                    for p0 in range(len(P["parameter"])):
                                        if p0 != par:
                                            found_double = True
                                            for spec in range(len(P["parameter"])):
                                                if P["parameter"][par][spec]!= P["parameter"][p0][spec]:
                                                    found_double = False
                                                    break
                                            if found_double:
                                                print("    Error: For plast " + p \
                                                      + " double specification for parameter: " \
                                                      + str(P["parameter"][par]))
                                                is_sane = False

        # Check interfaces.
        if "interfaces" in self.net:
            if isinstance(self.net["interfaces"], dict):
                for i,I in self.net["interfaces"].items():
                    if not isinstance(I, dict):
                        print("    Error: Expected dictionary for interface" \
                              + " definition of interface: " + str(i) + ".")
                        is_sane = False
                    else:
                        # Inputs must be specified as list for each interface.
                        if not "in" in I or not "out" in I:
                            print("    Error: Expected in/out in interface" \
                                  + " specification of interface: " + str(i) + ".")
                            is_sane = False
                        else:
                            if not isinstance(I["in"], list) or \
                                    not isinstance(I["out"], list):
                                print("    Error: Expected in/out in interface" \
                                      + " specification to be of type list for" \
                                      + " interface: " + str(i) + ".")
                                is_sane = False
                            else:
                                # Check that in/out of interface are actual nps.
                                for n in I["in"] + I["out"]:
                                    tgt_np = n
                                    if "remap" in I:
                                        if n in I["remap"]:
                                            tgt_np = I["remap"][n]
                                    if not tgt_np in self.net["neuron_pools"]:
                                        print("    Error: No associated neuron-" \
                                              + "pool found for in/out " + str(tgt_np) \
                                              + " of interface " + str(i) + ".")
                                        is_sane = False

        return is_sane

# =============================================================================
# =============================================================================
# =============================================================================

def tag_filter(net, mod, tags):
    """Method returns all items of type mod which match the tag filter.
    """
    # List of all items.
    filtered_items = [i for i in net[mod]]
    for t in tags:
        # Remove leading tag "#!" and ending "!".
        if t[-1] == "!":
            tag = t[1:-1]
        else:
            tag = t[1:]
        # Dependent of inclusive or exclusive tag.
        if t[0] == "!":
            for i,I in net[mod].items():
                if "tags" in I:
                    if tag in I["tags"] and i in filtered_items:
                        filtered_items.remove(i)
        elif t[0] == "#":
            for i,I in net[mod].items():
                if "tags" in I:
                    if tag not in I["tags"] and i in filtered_items:
                        filtered_items.remove(i)
                else:
                    if i in filtered_items:
                        filtered_items.remove(i)
    return filtered_items

# =============================================================================
# =============================================================================
# =============================================================================

def suggest_data(net, mn, inp):
    """Method gives a list of suggestions for current input command list inp.
    """
    # Marks "!" indicate finished selection.
    sugs = []
    if len(inp) == 1:
        if inp[0] == "":
            # Empty input.
            sugs += ["reset", "mv", "show", "exit"]
        else:
            for c in ["reset", "mv", "show", "exit"]:
                if c.find(inp[0]) != -1:
                    sugs.append(c)
    elif len(inp) == 2:
        if inp[1] == "":
            if inp[0] in ["reset!"]:
                # item type
                sugs += ["np", "sp", "if", "plast"]
            elif inp[0] == "mv!":
                # for meta variable name suggest mv_name
                sugs += ["mv_name"]
        else:
            if inp[0] in ["reset!"]:
                # only matching types
                for c in ["np", "sp", "if", "plast"]:
                    if c.find(inp[1]) != -1:
                        sugs.append(c)
            elif inp[0] == "mv!":
                # for meta variables all names (= current input) are allowed
                sugs.append(inp[1])
    elif len(inp) >= 3:
        # dependent on main command parse string command
        if inp[0] == "reset!":
            if inp[-2] in ["np!", "sp!", "plast!", "if!"]:
                if len(inp[-1]) == 0:
                    sugs += ["#tag", "!tag (not that tag)", "item name"]
                elif inp[-1][0] == "#" or inp[-1][0] == "!":
                    for t in mn.tags[inp[1][0:-1]]:
                        if t.find(inp[-1][1:]) != -1:
                            sugs.append(inp[-1][0] + t)
                else:
                    for c in net[S2L(inp[1])]:
                        if c.find(inp[-1]) != -1:
                            sugs.append(c)
            elif inp[-2][0] == "#" or inp[-2][0] == "!":
                if len(inp[-1]) == 0:
                    filtered_items = tag_filter(net, S2L(inp[1]), inp[2:-1])
                    sugs += ["#tag", "!tag (not that tag)"] + filtered_items
                elif inp[-1][0] == "#" or inp[-1][0] == "!":
                    for t in mn.tags[inp[1][0:-1]]:
                        if t.find(inp[-1][1:]) != -1:
                            sugs.append(inp[-1][0] + t)
                else:
                    filtered_items = tag_filter(net, S2L(inp[1]), inp[2:-1])
                    for i in filtered_items:
                        if i.find(inp[-1]) != -1:
                            sugs.append(i)
            elif inp[-2][0:-1] in net[S2L(inp[1])]:
                if inp[-3] in ["np!", "sp!", "plast!", "if!"] \
                        or inp[-3][0] == "#" or inp[-3][0] == "!":
                    if inp[1] == "np!":
                        items = ["state", "par", "period"]
                    elif inp[1] == "plast!":
                        items = ["par", "period"]
                    else:
                        items = ["par"]
                    if inp[0] == "show":
                        items.append("var")
                    for i in items:
                        if i.find(inp[-1]) != -1:
                            sugs.append(i)
            elif inp[-2] == "par!":
                if inp[-3][0:-1] in net[S2L(inp[1])]:
                    ps = eval("st_" + inp[1][0:-1] + "_parameter_shape")(inp[-3][0:-1], net, mn)
                    for p in ps:
                        if p.find(inp[-1]) != -1:
                            sugs.append(p)
            elif inp[-2] == "state!":
                if inp[-3][0:-1] in net[S2L(inp[1])]:
                    items = ["zero"]
                    for p in items:
                        if p.find(inp[-1]) != -1:
                            sugs.append(p)
            elif inp[-2] == "period!":
                if inp[-3][0:-1] in net[S2L(inp[1])]:
                    sugs += ["uint_period#uint_offset"]
            else:
                # no explicit information in inp[-2], use inp[-3]
                if len(inp) >= 4:
                    if inp[-3] == "par!":
                        if inp[-4][0:-1] in net[S2L(inp[1])]:
                            # get shapes of all parameters
                            ps = eval("st_" + inp[1][0:-1] \
                                      + "_parameter_shape")(inp[-4][0:-1], net, mn)
                            items = []
                            if inp[-2][0] == "W":
                                items += ["bilin", "loc_inh", "xavier", "one", "zero"]
                                if ps[inp[-2][0:-1]][0] == ps[inp[-2][0:-1]][1]:
                                    items.append("id")
                            if inp[-2][0] == "b":
                                items = ["zero"]
                        for i in items:
                            if i.find(inp[-1]) != -1:
                                sugs.append(i)
    
    return sugs

# =============================================================================
# =============================================================================
# =============================================================================

def expand_path(net, p, self_conn_flag=True):
    """Expand a given path p by one.

     expand a given path p (list of np ids) by one given the network and return
     list of all paths (= list of paths = list of list of np ids)
    self_conn_flag :: flag specifying if np self connections are expanded
    """
    p_list = []
    for s, S in net["synapse_pools"].items():
        if p[-1] == S["target"]:
            for n,N in net["neuron_pools"].items():
                for i in range(len(S["source"])):
                    for j in range(len(S["source"][i])):
                        if n == S["source"][i][j]:
                            if self_conn_flag:
                                p_list.append(p + [n])
                            else:
                                if n != p[-1]:
                                    p_list.append(p + [n])
    return p_list



def shortest_path(net, source_np, target_np):
    """Returns shortest path from source_np to target_np.
    """
    # initialize all paths to target np as only target_np
    path_list = [[target_np]]
    # set flag
    source_found = False
    # cntr for sanity check
    exps = 0
    # default empty
    shortest_path = []
    # while not expanded to source_np
    while not source_found and exps < len(net["neuron_pools"]):
        # initial next path expansion with empty
        new_path_list = []
        # expand all paths to target_np by one
        for p in path_list:
            new_path_list += expand_path(net, p)
        # update path list through expanded paths
        path_list = new_path_list
        # check if source found
        for p in path_list:
            if p[-1] == source_np:
                source_found = True
                shortest_path = p
                break
        exps += 1
    return shortest_path



def find_sps_between(net, source_np, target_np):
    """Return list of all direct sps from source_np to target_np.
    """
    sp_list = []
    for s,S in net["synapse_pools"].items():
        if target_np == S["target"]:
            for i in range(len(S["source"])):
                for j in range(len(S["source"][i])):
                    if source_np == S["source"][i][j] and s not in sp_list:
                        sp_list.append(s)
    return sp_list



# =============================================================================
# =============================================================================
# =============================================================================


def compute_distance_matrix(net):
    """Computes and returns the quadratic graph distance matrix over all NPs.

    Distances (between NPs) are stored in a dict of 2-tuples of NPs.
    Distance here means always shortest distance.
    Non-existing paths between nodes result in a None distance, and there
    are no distances <= 0.
    """
    dm = {}
    for n0 in net['neuron_pools']:
        for n1 in net['neuron_pools']:
            dm[(n0, n1)] = None
    # Add all "one" distances.
    for s,S in net['synapse_pools'].items():
        sp_srcs = [src for srcs in S['source'] for src in srcs]
        for n in sp_srcs:
            dm[(n, S['target'])] = 1
    # Iteratively update distance matrix for a conservative #NPs times.
    for i in range(len(net['neuron_pools']) - 1):
        changed = False
        for path,dist in dm.items():
            # Try to extend currently longest distances / paths.
            if dist is not None:
                if dist == i + 1:
                    for s,S in net['synapse_pools'].items():
                        sp_srcs = [src for srcs in S['source'] for src in srcs]
                        if path[1] in sp_srcs:
                            if dm[(path[0], S['target'])] is None:
                                dm[(path[0], S['target'])] = dist + 1
                                changed = True
        if not changed:
            break
    return dm

# =============================================================================
# =============================================================================
# =============================================================================

def get_the_input(net, dist_mat=None):
    """Returns a meaningfull network input.
        
    Meaningfull here means an input of the longest w/o rec. path in the graph.

    We here assume, that the path of maximal length starts with an input.
    """
    # Get distance matrix.
    if dist_mat is None:
        dm = compute_distance_matrix(net)
    else:
        dm = dist_mat

    # Get maximal path length.
    max_dist = 0
    max_path = None
    for path,dist in dm.items():
        if dist is not None:
            if dist > max_dist:
                max_dist = copy.copy(dist)
                max_path = copy.copy(path)
    the_input = None
    if max_path is not None:
        the_input = max_path[0]
    return the_input



def clever_sort(net, source, the_input=None, dist_mat=None):
    """Two-step sorting of neuron-pools relative to a source.

    First, sort all NPs relative to their distance to the source NP.

    Second, sort all NPs with same distance to sort by their distance
    to the network input.

    Returns
    =======
    clever_sorted : list
        Sorted list of all neuron-pools.
    """
    # Get distance matrix.
    if dist_mat is None:
        dm = compute_distance_matrix(net)
    else:
        dm = dist_mat
    # Get the input.
    if the_input is None:
        ti = get_the_input(net, dist_mat)
    else:
        ti = the_input

    # Get distances from (+) / to (-) source.
    dists = {}
    dists[0] = [source]
    min_dist = 0
    max_dist = 0
    for n in net['neuron_pools']:
        if n != source:
            if dm[(n, source)] is not None:
                if -dm[(n, source)] in dists:
                    if n not in dists[-dm[(n, source)]]:
                        dists[-dm[(n, source)]].append(n)
                        if -dm[(n, source)] < min_dist:
                            min_dist = -dm[(n, source)]
                else:
                    dists[-dm[(n, source)]] = [n]
                    if -dm[(n, source)] < min_dist:
                        min_dist = -dm[(n, source)]
            if dm[(source, n)] is not None:
                if dm[(source, n)] in dists:
                    if n not in dists[dm[(source, n)]]:
                        dists[dm[(source, n)]].append(n)
                        if dm[(source, n)] > max_dist:
                            max_dist = dm[(source, n)]
                else:
                    dists[dm[(source, n)]] = [n]
                    if dm[(source, n)] > max_dist:
                        max_dist = dm[(source, n)]

    # Remove doubles from dist lists (+- dists may both occur for a NP).

    # Sort dists intern using distances to the_input.
    for d,nps in dists.items():
        if len(nps) != 1:
            pass

    clever_sorted = []
    for i in np.arange(min_dist, max_dist):
        if i in dists:
            for n in dists[i]:
                if n not in clever_sorted:
                    clever_sorted.append(n)

    # Add all non-connected NPs at end.
    for n in net['neuron_pools']:
        if n not in clever_sorted:
            clever_sorted.append(n)

    return clever_sorted

