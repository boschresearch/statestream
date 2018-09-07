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

import sys
import numpy as np

import statestream.meta.network as mn
from statestream.utils.yaml_wrapper import load_yaml, dump_yaml



def my_str(in_str):
    """Convert a given string to tex-able output string.
    """
    out_str = in_str.replace("_", "\\textunderscore ")
    return out_str



def generate_rollout_graph(loadfile=None, given_net=None, savefile=None):
    """Generates a tex-able tikz graph file and saves it to filename.

    Parameter:
    ----------
    loadfile : str
        The st_graph network specification file.
    savefile : str
        The filename of the tikz save file.
    """
    # Check if st_graph file and load it.
    if loadfile is not None:
        if len(loadfile) > 10:
            if loadfile[-9:] == ".st_graph":
                with open(loadfile) as f:
                    raw_net = load_yaml(f)
            else:
                print("Error: Invalid filename ending. Expected .st_graph.")
                sys.exit()
        else:
            print("Error: Source filename is too short.")
            sys.exit()
        
        # Generate meta network.
        MN = mn.MetaNetwork(raw_net)

        # Check sanity of meta.
        if not MN.is_sane():
            sys.exit()

        net = MN.net
    else:
        MN = mn.MetaNetwork(given_net)
        net = MN.net

    # =====================================================================

    # Some sanity checks, otherwise no tiks is generated.
    is_sane = True
    if len(net["neuron_pools"]) == 0:
        is_sane = False

    if not is_sane:
        return

    # =====================================================================

    # Determine order of neuron-pools for plotting.
    np_lists = {}
    np_types = ["in", "hidden", "out"]
    for t in np_types:
        np_lists[t] = []
    for np in net["neuron_pools"]:
        np_set = False
        # Check if input or output.
        for i,I in net["interfaces"].items():
            if np in I["in"]:
                np_lists["out"].append(np)
                np_set = True
            elif np in I["out"]:
                np_lists["in"].append(np)
                np_set = True
            if "remap" in I:
                for r,R in I["remap"].items():
                    if np == R:
                        if r in I["in"]:
                            np_lists["out"].append(np)
                            np_set = True
                        if r in I["out"]:
                            np_lists["in"].append(np)
                            np_set = True
            if np_set:
                break
        if not np_set:
            np_lists["hidden"].append(np)

    # Sort all stand-alone input nps to the very beginning.
    for np in np_lists["in"]:
        # Determine if stand-alone (no source / target of any sp).
        stand_alone = True
        for s,S in net["synapse_pools"].items():
            if S["target"] == np:
                stand_alone = False
            for srcs in S["source"]:
                if np in srcs:
                    stand_alone = False
                    break
        # Swap to end of hidden.
        if stand_alone:
            np_i = np_lists["in"].index(np)
            np_lists["in"][np_i], np_lists["in"][0] \
                = np_lists["in"][0], np_lists["in"][np_i]

    np_lists["hidden"].sort()
    # =====================================================================



    # =====================================================================
    # =====================================================================
    # Begin with empty.
    tikz_lines = []

    # Import some dependencies.
    tikz_lines.append("")
    tikz_lines.append("\\documentclass[tikz, border=1cm]{standalone}")
    tikz_lines.append("\\usepackage{tikz}")
    tikz_lines.append("\\usetikzlibrary{shapes}")
    tikz_lines.append("\\usetikzlibrary{arrows}")
    tikz_lines.append("\\usetikzlibrary{calc}")
    tikz_lines.append("\\usetikzlibrary{positioning}")
    tikz_lines.append("\\usetikzlibrary{intersections}")
    tikz_lines.append("\\usetikzlibrary{backgrounds}")

    # Define block styles.
    tikz_lines.append("")
    tikz_lines.append("\\definecolor{RBgreen}{rgb} {0.,0.6,0.}")
    tikz_lines.append("\\tikzstyle{sty_np} = [rectangle, " \
                                           + "draw=black, " \
                                           + "fill=green!60, " \
                                           + "text centered, " \
                                           + "align=center, " \
                                           + "minimum width=8em, " \
                                           + "minimum height=8em, " \
                                           + "font=\\bf]")
    tikz_lines.append("\\tikzstyle{sty_np_dormant} = [rectangle, " \
                                           + "draw=black, " \
                                           + "fill=black!30, " \
                                           + "text centered, " \
                                           + "align=center, " \
                                           + "minimum width=8em, " \
                                           + "minimum height=8em, " \
                                           + "font=\\bf]")
    tikz_lines.append("\\tikzstyle{sty_sp} = [draw, line width=1mm, blue, -latex]")
    tikz_lines.append("\\tikzstyle{sty_sp_plast} = [draw, line width=0.5mm, red, -latex]")
    tikz_lines.append("\\tikzstyle{sty_conv} = [rectangle, draw=black, fill=red, minimum width=1em, text centered, minimum height=4em]")
    tikz_lines.append("\\tikzstyle{sty_sp_dormant} = [draw, dotted, line width=0.3mm, draw=black!40, -latex]")

    tikz_lines.append("\\tikzstyle{sty_concat} = [circle, draw=black, fill=yellow]")
    tikz_lines.append("\\tikzstyle{sty_line_label} = [draw, dotted, line width=0.3mm, RBgreen, -latex]")

    # Begin picture.
    tikz_lines.append("")
    tikz_lines.append("\\begin{document}")
    tikz_lines.append("\\begin{tikzpicture}[node distance = 0.45cm, auto]")

    # Set coordinate for positioning.
    tikz_lines.append("")
    tikz_lines.append("    \\coordinate (input);")

    # Rollout one-step function
    tikz_lines.append("")
    for t in [0, 1]:
        for l in np_types:
            for np in range(len(np_lists[l])):
                if t == 0:
                    if np == 0:
                        if l == "in":
                            tikz_lines.append("    \\node [sty_np, right = 6em of input]"\
                                              + " (" + np_lists[l][np] + "_" + str(t) + ")" \
                                              + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                              + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                        else:
                            # Incase no hidden or even input layers, take the last input.
                            if len(np_lists[np_types[np_types.index(l) - 1]]) == 0:
                                if np_types[np_types.index(l) - 1] == "in":
                                    tikz_lines.append("    \\node [sty_np, right = 6em of input]"\
                                                      + " (" + np_lists[l][np] + "_" + str(t) + ")" \
                                                      + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                      + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                                else:
                                    tikz_lines.append("    \\node [sty_np, right = 6em of " + np_lists[np_types[np_types.index(l) - 2]][-1] + "_0]"\
                                                      + " (" + np_lists[l][np] + "_" + str(t) + ")" \
                                                      + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                      + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                            else:
                                tikz_lines.append("    \\node [sty_np, right = 6em of " + np_lists[np_types[np_types.index(l) - 1]][-1] + "_0]"\
                                                  + " (" + np_lists[l][np] + "_" + str(t) + ")" \
                                                  + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                  + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                    else:
                        tikz_lines.append("    \\node [sty_np, right = 6em of " + np_lists[l][np - 1] + "_0]" \
                                          + " (" + np_lists[l][np] + "_" + str(t) + ")" \
                                          + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                          + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                else:
                    tikz_lines.append("    \\node [sty_np, below = 6em of " + np_lists[l][np] + "_" + str(t - 1) + "]" \
                                      + " (" + np_lists[l][np] + "_" + str(t) + ")" \
                                      + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                      + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
        if t > 0:
            for s,S in net["synapse_pools"].items():
                src = S["source"][0][0]
                tgt = S["target"]
                tikz_lines.append("    \\path [sty_sp]" \
                                  + " (" + src + "_" + str(t - 1) + ".south east) -- (" + tgt + "_" + str(t) + ".north west);")
    if len(np_lists["in"]) == 0:
        last_base = np_lists["hidden"][0] + "_1"
    else:
        last_base = np_lists["in"][0] + "_1"



    # Rollout all plasticities.
    for p,P in net["plasticities"].items():
        if P["type"] == "loss":
            rollout = P["source_t"] + 1
            for t in range(rollout):
                for l in np_types:
                    for np in range(len(np_lists[l])):
                        node_name = np_lists[l][np] + "_" + str(p) + "_" + str(t)
                        if np_lists[l][np] in MN.net_plast_nps[p][t]:
                            tex_style_np = "sty_np"
                        else:
                            tex_style_np = "sty_np_dormant"
                        if t == 0:
                            if np == 0:
                                if l == "in":
                                    tikz_lines.append("    \\node [" + tex_style_np + ", below = 8em of " + last_base + "]"\
                                                      + " (" + node_name + ")" \
                                                      + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                      + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                                else:
                                    # Incase no hidden layers, take the last input.
                                    if len(np_lists[np_types[np_types.index(l) - 1]]) == 0:
                                        parent_node = np_lists[np_types[np_types.index(l) - 2]][-1] + "_" + str(p)
                                        tikz_lines.append("    \\node [" + tex_style_np + ", right = 6em of " + parent_node + "_0]"\
                                                          + " (" + node_name + ")" \
                                                          + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                          + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                                    else:
                                        parent_node = np_lists[np_types[np_types.index(l) - 1]][-1] + "_" + str(p)
                                        tikz_lines.append("    \\node [" + tex_style_np + ", right = 6em of " + parent_node + "_0]"\
                                                          + " (" + node_name + ")" \
                                                          + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                          + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                            else:
                                parent_node = np_lists[l][np - 1] + "_" + str(p)
                                tikz_lines.append("    \\node [" + tex_style_np + ", right = 6em of " + parent_node + "_0]" \
                                                  + " (" + node_name + ")" \
                                                  + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                                  + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                        else:
                            parent_node = np_lists[l][np] + "_" + str(p) + "_" + str(t - 1)
                            tikz_lines.append("    \\node [" + tex_style_np + ", below = 6em of " + parent_node + "]" \
                                              + " (" + node_name + ")" \
                                              + " {\\textbf{" + my_str(np_lists[l][np]) + "}"\
                                              + " \\\\ $\\mathbf{" + str(net["neuron_pools"][np_lists[l][np]]["shape"]) + "}$};")
                        # Update base, esp. for next plasticity plot.
                        if t == rollout - 1 and np == 0 and l == "in":
                            last_base = node_name
                if t > 0:
                    for s,S in net["synapse_pools"].items():
                        if s in MN.net_plast_sps[p][t]:
                            tex_style_np = "sty_sp"
                        else:
                            tex_style_np = "sty_sp_dormant"
                        tgt = S["target"]
                        tgt_node = tgt + "_" + str(p) + "_" + str(t)
                        for f in range(len(S["source"])):
                            for src in S["source"][f]:
                                src_node = src + "_" + str(p) + "_" + str(t - 1)
                                tikz_lines.append("    \\path [" + tex_style_np + "]" \
                                                  + " (" + src_node + ".south east) -- (" + tgt_node + ".north west);")


    # End picture.
    tikz_lines.append("")
    tikz_lines.append("\\end{tikzpicture}")
    tikz_lines.append("\\end{document}")
    tikz_lines.append("")
    # =====================================================================
    # =====================================================================



    # Create output file.
    try:
        with open(savefile, "w+") as f:
            for l in tikz_lines:
                f.write(l + "\n")
    except:
        print("\nWARNING: Unable to access " + str(savefile) + ". No rollout graph tex file generated.\n")
