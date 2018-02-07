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
import importlib
import shutil
from ctypes import c_int
import matplotlib



from statestream.utils.pygame_import import pg, pggfx
from statestream.utils.yaml_wrapper import load_yaml, dump_yaml

from statestream.meta.network import suggest_data, \
                                     shortest_path, \
                                     MetaNetwork, \
                                     S2L
from statestream.meta.synapse_pool import sp_get_dict

from statestream.ccpp.cgraphics import cgraphics_colorcode, \
                                       cgraphics_np_force, \
                                       cgraphics_vec_to_RGBangle

from statestream.utils.rearrange import rearrange_3D_to_2D, \
                                        rearrange_4D_to_2D

from statestream.utils.helper import is_scalar_shape
from statestream.utils.helper import is_int_dtype
from statestream.utils.helper import is_float_dtype
from statestream.utils.helper import is_list_of_lists
from statestream.utils.shared_memory import SharedMemory

from statestream.visualization.graphics import num2str, brighter, darker
from statestream.visualization.graphics import blit_plot, \
                                               blit_hist, \
                                               blit_pcp, \
                                               plot_circle, \
                                               empty_subwin

from statestream.visualization.base_widgets import Collector
from statestream.visualization.widgets.list_selection_window import ListSelectionWindow
from statestream.visualization.widgets.list_mselection_window import ListMSelWindow
from statestream.visualization.widgets.parameter_specification_window import ParamSpecWindow

from statestream.utils.defaults import DEFAULT_VIZ_PARAMETER, DEFAULT_COLORS



def draw_dashed_line(surf, col, start_pt, end_pt, width=1, dash=10, offset=0.0, dash_type=0):
    """Draw a dashed line.

    Parameter
    ---------
    surf : pygame Surface
        The graphical surface to draw the line on.
    col : (int, int, int)
        The RGB color to draw the dashed line with.
    start_pt : (int, int)
        Position on screen to start the line.
    end_pt : (int, int)
        Position on screen to end the line.
    width : int
        The line width in pixel.
    dash : int
        Length of dash for dashed line.
    offset : int
        An offset to draw moving dashed lines.
    dash_type: int
        Specifies the type of dash used:
        0: normal line
        1: arrow
    """
    diff = end_pt - start_pt
    length = np.linalg.norm(diff)
    slope = diff / length
    if dash_type == 0:
        for i in range(0, int(length / dash) - 1, 2):
            start_i = start_pt + ((i + offset) * dash) * slope
            end_i = start_pt + ((i + 1 + offset) * dash) * slope
            pg.draw.line(surf, col, start_i, end_i, width)
    elif dash_type == 1:
        orth = 0.5 * dash * slope
        orth[0], orth[1] = orth[1], -orth[0]
        for i in range(0, int(length / dash) - 1, 2):
            start_i = start_pt + ((i + offset) * dash) * slope
            end_i = start_pt + ((i + 1 + offset) * dash) * slope
            pg.draw.line(surf, col, start_i + orth, end_i, width)
            pg.draw.line(surf, col, start_i - orth, end_i, width)



class Visualization(object):
    def __init__(self, net, param):
        """Initialize main visualization class.

        Main visualization class with viz process having access to the entire
        shared memory.

        Parameters
        ----------
        net : dict
            The network dictionary.
        param : dict
            The core parameters.    
        """
        # --------------------------------------------------------------------
        # Read local system settings if given, else use defaults.
        self.vparam = {}
        tmp_filename = os.path.expanduser('~') + '/.statestream/stviz.yml'
        if not os.path.isfile(tmp_filename):
            # Write default core parameters to file.
            with open(tmp_filename, 'w+') as f:
                dump_yaml(DEFAULT_VIZ_PARAMETER,
                          f)
                # Use default parameter dictionary
                self.vparam = DEFAULT_VIZ_PARAMETER
        else:
            # Load core parameters from file.
            with open(tmp_filename) as f:
                tmp_dictionary = load_yaml(f)
                # Create core parameters from default and loaded.
                for p,P in DEFAULT_VIZ_PARAMETER.items():
                    self.vparam[p] = tmp_dictionary.get(p, P)

        # Get some variables.
        self.mn = MetaNetwork(net)
        self.net = copy.deepcopy(self.mn.net)
        self.param = copy.deepcopy(param)

        # Generate networkx graph.
        #self.nx_graph = generate_graph(self.net, lod='np-np')
        
        # Set internal state.
        self.state_running = True
        self.all_up = False
        self.all_up_current = 0

        # Some edit mode variables, handling online computation graph changes.
        self.edit_mode = False
        self.edited_items = []
        self.edit_net = None
        
        # Initially show name only if mouse over.
        self.show_name = 0
        # Initialy show no profiler information.
        self.show_profiler = 0

        # Detailed visualization profiler.
        self.viz_prof_frames = 12
        self.viz_prof = ["item force comput.",
                         "draw items",
                         "draw connections",
                         "draw subwindows",
                         "monitor comput.",
                         "LMB",
                         "mini-map"]
        self.viz_prof_start = {}
        self.viz_prof_dur = {}
        for vp in self.viz_prof:
            self.viz_prof_start[vp] = 0.0
            self.viz_prof_dur[vp] \
                = np.zeros([self.viz_prof_frames,], dtype=np.float32)
        
        # Define settings on the right and their sub-settings.
        # DEPRECATED, still active but will be replaced.
        self.active_setting = None
        self.settings = {}
        self.settings['monitor'] = {}
        for s in self.settings:
            self.settings[s]['rect'] = pg.Rect(0, 0, 2, 2)
            self.settings[s]['sets'] = {}
        self.settings['monitor']['sets']['mean_state'] = {}
        self.settings['monitor']['sets']['mean_state']['type'] = 'float'
        self.settings['monitor']['sets']['mean_state']['y'] = 160
        self.settings['monitor']['sets']['mean_state']['rect'] \
            = pg.Rect(0, 0, 2, 2)
        self.settings['monitor']['sets']['mean_state']['value'] = 1e+6
        self.settings['monitor']['sets']['mean_state']['manip px'] = 0
        self.settings['monitor']['sets']['mean_state']['manip dv'] = 0
        self.settings['monitor']['sets']['mean_state']['manip orig'] \
            = copy.copy(self.settings['monitor']['sets']['mean_state']['value'])
        self.settings['monitor']['sets']['m state 0/1'] = {}
        self.settings['monitor']['sets']['m state 0/1']['type'] = 'iter'
        self.settings['monitor']['sets']['m state 0/1']['y'] = 230
        self.settings['monitor']['sets']['m state 0/1']['rect'] \
            = pg.Rect(0, 0, 2, 2)
        self.settings['monitor']['sets']['m state 0/1']['value'] = 'on'
        self.settings['monitor']['sets']['m state 0/1']['values'] \
            = ['off', 'soft off', 'soft on', 'on']
        self.settings['monitor']['sets']['var_state'] = {}
        self.settings['monitor']['sets']['var_state']['type'] = 'float'
        self.settings['monitor']['sets']['var_state']['y'] = 310
        self.settings['monitor']['sets']['var_state']['rect'] \
            = pg.Rect(0, 0, 2, 2)
        self.settings['monitor']['sets']['var_state']['value'] = 1e+6
        self.settings['monitor']['sets']['var_state']['manip px'] = 0
        self.settings['monitor']['sets']['var_state']['manip dv'] = 0
        self.settings['monitor']['sets']['var_state']['manip orig'] \
            = copy.copy(self.settings['monitor']['sets']['var_state']['value'])
        self.settings['monitor']['sets']['v state 0/1'] = {}
        self.settings['monitor']['sets']['v state 0/1']['type'] = 'iter'
        self.settings['monitor']['sets']['v state 0/1']['y'] = 380
        self.settings['monitor']['sets']['v state 0/1']['rect'] \
            = pg.Rect(0, 0, 2, 2)
        self.settings['monitor']['sets']['v state 0/1']['value'] = 'on'
        self.settings['monitor']['sets']['v state 0/1']['values'] \
            = ['off', 'soft off', 'soft on', 'on']
        self.settings['monitor']['sets']['n state 0/1'] = {}
        self.settings['monitor']['sets']['n state 0/1']['type'] = 'iter'
        self.settings['monitor']['sets']['n state 0/1']['y'] = 450
        self.settings['monitor']['sets']['n state 0/1']['rect'] \
            = pg.Rect(0, 0, 2, 2)
        self.settings['monitor']['sets']['n state 0/1']['value'] = 'on'
        self.settings['monitor']['sets']['n state 0/1']['values'] \
            = ['off', 'on']

        # Options for right clicks.
        self.options = {
            'np': ['state'],
            'sp': ['on/off'],
            'plast': ['on/off'],
            'if': ['if viz']
        }

        # Create graph structure.i
        self.graph = {}
        self.graph_i = ['sp', 'np', 'plast', 'if']
        self.graph_I = ['synapse_pools', 
                        'neuron_pools', 
                        'plasticities', 
                        'interfaces']
        # Visibility flag for graph item types.
        self.graph_type_viz_flag = {}
        for i in self.graph_i:
            self.graph_type_viz_flag[i] = True

        self.conn_i = ['plast', 'if', 'sw', 'sp']
        self.conn_type_viz_flag = {}
        self.conn_type_viz_flag['sw'] = True
        self.conn_type_viz_flag['sp'] = True
        self.conn_type_viz_flag['plast'] = False
        self.conn_type_viz_flag['if'] = False

        # Some graph parameters.
        self.graph_friction = 0.5
        self.np_dist = 160.0
        self.force_np_repulsion = True
        self.force_sp_repulsion = True
        self.dash_size = 10

        # Color for items.
        self.graph_col = {
            'empty': (0,0,0),
            'sp': self.vparam['sp_color'],
            'np': self.vparam['np_color'],
            'plast': self.vparam['plast_color'],
            'if': self.vparam['if_color']
        }
        self.graph_col_idle = {
            'empty': (0,0,0),
            'sp': (128,0,0),
            'np': (0,128,0),
            'plast': (0,0,128),
            'if': (128,128,0)
        }



        # Create empty graph structure.
        for i in range(len(self.graph_i)):
            # Create empty structure.
            if self.graph_I[i] in self.net:
                for x in self.net[self.graph_I[i]]:
                    self.graph[x] = {}
                    self.graph[x]['type'] = self.graph_i[i]
                    self.graph[x]['TYPE'] = self.graph_I[i]
                    self.graph[x]['pos'] = np.array([400 + 720 * np.random.rand(), 
                                                     100 + 500 * np.random.rand()])
                    self.graph[x]['vel'] = np.zeros([2])
                    self.graph[x]['force'] = np.zeros([2])
                    self.graph[x]['col'] = (0,0,0)
                    self.graph[x]['rad'] = 10
                    self.graph[x]['COL'] = (0,0,0)
                    self.graph[x]['WIDTH'] = 2
                    self.graph[x]['RAD'] = 10
                    self.graph[x]['state'] = 0
                    self.graph[x]['pause'] = 0
                    self.graph[x]['rect'] = pg.Rect(0, 0, 2, 2)
                    self.graph[x]['sw'] = []
                    self.graph[x]['in'] = []
                    self.graph[x]['out'] = []
        # Fill with default structure.
        for i in range(len(self.graph_i)):
            # Fill with default.
            if self.graph_I[i] in self.net:
                for x in self.net[self.graph_I[i]]:
                    X = self.net[self.graph_I[i]][x]
                    # Add sps to sps and nps.
                    if self.graph_i[i] == 'sp':
                        sources = [item for sub_list in X['source'] for item in sub_list]
                        for s in sources:
                            self.graph[x]['in'].append(s)
                            self.graph[s]['out'].append(x)
                        self.graph[x]['out'].append(X['target'])
                        self.graph[X['target']]['in'].append(x)
                    # Add all in/outs for plasts (also to sps and nps).
                    elif self.graph_i[i] == 'plast':
                        if X['type'] == 'loss':
                            self.graph[x]['in'].append(X['source'])
                            self.graph[X['source']]['out'].append(x)
                            if 'target' in X:
                                self.graph[x]['in'].append(X['target'])
                                self.graph[X['target']]['out'].append(x)
                            for p in X['parameter']:
                                self.graph[x]['out'].append(p[1])
                                self.graph[p[1]]['in'].append(x)
                        if X['type'] == 'L_regularizer':
                            for p in X['parameter']:
                                self.graph[x]['out'].append(p[1])
                        if X['type'] in ['hebbian']:
                            # Add source and target as inputs.
                            self.graph[x]['in'] += [X['source'], X['target']]
                            # Add target sp as output.
                            self.graph[x]['out'] += [X['parameter'][0][1]]
                    elif self.graph_i[i] == 'if':
                        for Is in X['in']:
                            # Check for remapping.
                            tmp_target = Is
                            if 'remap' in X:
                                if Is in X['remap']:
                                    tmp_target = X['remap'][Is]
                            self.graph[x]['in'].append(tmp_target)
                            self.graph[tmp_target]['out'].append(x)
                        for Os in X['out']:
                            # Check for remapping.
                            tmp_target = Os
                            if 'remap' in X:
                                if Os in X['remap']:
                                    tmp_target = X['remap'][Os]
                            self.graph[x]['out'].append(tmp_target)
                            self.graph[tmp_target]['in'].append(x)

        # Blitting queue for subwindows.
        self.subwin_blit_queue = []
                            
        # Compute neuron and synapse index.
        self.np_n2i = {}
        self.np_i2n = {}
        self.sp_n2i = {}
        self.sp_i2n = {}
        cntr = 0
        for x in self.net["neuron_pools"]:
            self.np_n2i[x] = cntr
            cntr += 1
        cntr = 0
        for x in self.net["synapse_pools"]:
            self.sp_n2i[x] = cntr
            cntr += 1

        # Compute distance matrix for the moment.
        # conn_mat[src + tgt * no_nps] = min_path_len(src, tgt)
        self.conn_mat = np.zeros([len(self.np_n2i)**2,], 
                                 dtype=c_int)
        self.conn_mat_sp = np.zeros([len(self.sp_n2i)**2,], 
                                 dtype=c_int)
        for s,S in self.net["synapse_pools"].items():
            if not "NOSPRING" in S.get('tags', []):
                tgt_np = S["target"]
                tgt_np_id = self.np_n2i[tgt_np]
                for srcs in S["source"]:
                    for src_np in srcs:
                        src_np_id = self.np_n2i[src_np]
                        conn_id = src_np_id + tgt_np_id * len(self.np_n2i)
                        self.conn_mat[conn_id] = 1

        self.distmat = {}
        for n, N in self.net['neuron_pools'].items():
            self.distmat[n] = {
                -1: [],
                1: [],
            }
            for s, S in self.net['synapse_pools'].items():
                # Get all sources.
                sources = [src for src_list in S['source'] for src in src_list]
                if n == S['target']:
                    self.distmat[n][1] += sources
                elif n in sources:
                    self.distmat[n][-1] += [S['target']]

        # Size of subwindow controls.
        self.sw_size = 16

        # Set current preview to None.
        self.current_preview_active = False
        self.current_preview = None
        self.current_preview_src = ''

        # Origin of items on screen. These are the only variables
        # in the real global coordinate frame.
        self.origin = np.array([0,0], dtype=np.float32)
        
        # Remove doubles from in/out.
        for x in self.graph:
            self.graph[x]['in'] = list(set(self.graph[x]['in']))
            self.graph[x]['out'] = list(set(self.graph[x]['out']))

        # This is the list of elected graph items.
        self.items_selected = []
        
        # Online monitor structures.
        # monitor state: 0 off / 1 soft viz but off / 2 soft viz 
        # and on / 3 only hard on
        self.monitor_item = {}
        self.monitor_bad_items = {}
        self.monitor_item['mean_state'] = {}
        self.monitor_item['var_state'] = {}
        self.monitor_item['nan_state'] = {}
        self.monitor_bad_items['mean_state'] = []
        self.monitor_bad_items['var_state'] = []
        self.monitor_bad_items['nan_state'] = []
        for n,n_shape in self.mn.np_shape.items():
            self.monitor_item['mean_state'][n] \
                = np.zeros([n_shape[0],]).astype(np.float32)
            self.monitor_item['var_state'][n] \
                = np.zeros([n_shape[0],]).astype(np.float32)
            self.monitor_item['nan_state'][n] = 0
        
        # Begin with empty list of meta variables.
        self.meta_variables = []
        self.new_mv_list = []
        self.new_mv_subwins = []

        # Begin with empty core communication queue.
        self.core_comm_queue = []

        # Determine all available types for meta variables.
        local_path = os.path.dirname(os.path.abspath(__file__))
        local_files = os.listdir(local_path \
                      + os.sep + '..' + os.sep \
                      + 'meta/system_clients/')
        self.meta_variable_types = []
        for t in local_files:
            # Ignore all .pyc files.
            if t[-1] != 'c':
                if t[0:2] != '__':
                    self.meta_variable_types.append(t[0:-3])
        
        # Initially set debug flag to false.
        self.debug_items = ['LMB_hold', 
                            'LMB_drag_type', 
                            'LMB_drag_inst', 
                            'is_shift_pressed', 
                            'is_ctrl_pressed', 
                            'dbg_dummy']
        self.debug_flag = False
        self.dbg_dummy = None
        
        # Command line variables.
        self.command_line = []
        self.is_typing_command = False
        self.is_shift_pressed = False
        self.is_ctrl_pressed = False
        self.perform_command = False
        
        # Some hot-keys.
        self.hotkey = {
            'c': False,
            'f': False,
            'g': False,
            's': False,
        }

        # Flag for color correction (blue).
        self.ccol = self.vparam['color_correction']
        
        # General delay.
        self.delay_rect = pg.Rect(0, 0, 2, 2)
        self.delay = 0
        
        # Get home directory.
        self.home_path = os.path.expanduser('~')
        self.this_path = os.path.abspath(__file__)[0:-16]
        if not os.path.isdir(self.home_path + '/.statestream/viz'):
            # Create dictionary.
            os.makedirs(self.home_path + '/.statestream/viz')
            # Initially copy graphview for running example.
            shutil.copyfile(self.this_path + "../../docs/resources/demo-graphview-00", 
                            self.home_path + "/.statestream/viz/demo-graphview-00")
        # Set graphview save file.
        self.graphview_file = self.home_path + '/.statestream/viz/' \
                              + self.net['name'] + '-graphview'
        
# =============================================================================

    def cc(self, col):
        """Color correction for blue-channel.
        """
        if self.ccol:
            return (col[0]/2, col[1], col[2])
        else:
            return (col[2], col[1], col[0])
        
# =============================================================================

    def update_core_comm_queue(self):
        """Method to update communication to core.
        """
        if self.core_comm_queue:
            if self.IPC_PROC["save/load"].value == 0 \
                    and self.IPC_PROC["instruction"].value == 0:
                msg = self.core_comm_queue[0]
                core_string = np.array([ord(c) for c in msg['instruction string']])
                self.IPC_PROC['instruction len'].value = len(core_string)
                self.IPC_PROC['string'][0:len(core_string)] = core_string[:]
                if 'save/load' in msg:
                    self.IPC_PROC["save/load"].value = copy.copy(msg['save/load'])
                elif 'instruction' in msg:
                    self.IPC_PROC["instruction"].value = copy.copy(msg['instruction'])
                self.core_comm_queue.pop(0)

# =============================================================================

    def add_meta_variable(self, mv_type=None, mv_param=[], selected_values=[], blitted=[], itemized=False):
        '''Add specified meta variable.
        '''
        # Add a new meta variable.
        # Define meta variables parameters.
        client_param = {}
        for p,P in enumerate(mv_param):
            if P["name"] == "name":
                client_param['name'] = P["value"]
                break
        client_param['type'] = mv_type
        client_param['params'] = copy.deepcopy(mv_param)
        client_param['param'] = copy.deepcopy(self.param)
        client_param['selected_values'] = copy.deepcopy(selected_values)
        # Add mv-type dependent parameters / variables.
        shm_layout = getattr(importlib.import_module('statestream.meta.system_clients.' \
                                                     + mv_type), 'client_shm_layout')
        pv = shm_layout(self.param, self.net, mv_param, selected_values)
        client_param['parameter'] = pv['parameter']
        client_param['variables'] = pv['variables']
        # Save parameters to tmp file.
        filename = self.home_path + '/.statestream/system_client_parameter_' \
                   + str(int(self.IPC_PROC['session_id'].value)) + '.yml'
        with open(filename, 'w') as outfile:
            dump_yaml(client_param, outfile)

        # Send instruction to core to instantiate new system client.
        core_comm = {}
        core_comm['instruction string'] = 'register_sys_client'
        core_comm['instruction'] = 1
        self.core_comm_queue.append(copy.deepcopy(core_comm))
        mv_construct \
            = getattr(importlib.import_module('statestream.meta.system_clients.' \
                                              + mv_type), mv_type)
        self.meta_variables.append(mv_construct(self.net, 
                                                self.param, 
                                                mv_param=mv_param, 
                                                client_param=client_param))
        self.meta_variables[-1].itemized = itemized

# =============================================================================

    def get_visible_pos(self, default_pos=[0,0], font_size='small', strlist1=None, strlist2=None):
        # Determine a 'good' position to place frames.
        max_str = 0
        if strlist2 is None:
            for i in strlist1:
                max_str = max(max_str, len(i) + 1)
        else:
            for i1,i2 in zip(strlist1, strlist2):
                max_str = max(max_str, len(i1) + len(i2) + 1)
        # Default position to place selection window on screen.
        pos_x = int(min(default_pos[0], 
                        self.screen.get_size()[0] - 0.7 * max_str * self.font_size[font_size]))
        pos_y = int(min(default_pos[1],
                        self.screen.get_size()[1] - (len(strlist1) + 4) * self.font_size[font_size]))
        return [pos_x, pos_y]

# =============================================================================
    
    def subwin_mode_exists(self, shape):
        """Determines which subwinmodes exist for a given state shape.
        """
        modes = []
        l_shape = shape
        if isinstance(shape, tuple):
            l_shape = list(shape)
        if isinstance(l_shape, list):
            if len(l_shape) > 0:
                modes.append('hist')
                modes.append('maps')
                if len(l_shape) == 3:
                    if l_shape[0] in [2,3]:
                        modes.append('scatter')
                if len(l_shape) == 3 and l_shape[0] == 2:
                    modes.append('angle2D')
                if len(l_shape) == 4 and l_shape[1] == 2:
                    modes.append('angle2D')
                if l_shape[0] == 3:
                    modes.append('RGB 0')
                if len(l_shape) > 1:
                    if l_shape[1] == 3:
                        modes.append('RGB 1')
                shape_0 = copy.copy(l_shape)
                del shape_0[0]
                if np.prod(shape_0) > 1:
                    modes.append('plot 0')
                if len(l_shape) > 1:
                    shape_1 = copy.copy(l_shape)
                    del shape_1[1]
                    if np.prod(shape_1) > 1:
                        modes.append('plot 1')
        modes.sort()
        return modes

# =============================================================================

    def magic_shape(self, shapes, magic, patterns):
        """Compute shape of magic(shapes).
        
        Parameters:
        -----------
        shapes: list of lists of ints
            List of shapes of all data sources entering the magic function.
        magic: string
            List containing the magic function. All data sources are
            referenced via #0, #1, ..., #9.
        patterns: list of strings
            Access patterns for the shapes.

        Returns:
        dat_shape: list of ints
            The shape of magic(shapes).
        """
        dat_shape = None
        if magic in [None, '']:
            if patterns is not None:
                dat = np.zeros(shapes[0], dtype=np.float32)
                m = 'dat' + patterns[0]
                dat_shape = eval(m).shape
            else:
                dat_shape = shapes[0]
        else:
            if len(shapes) <= 10:
                m = copy.copy(magic)
                dat = []
                for s,shape in enumerate(shapes):
                    dat.append(np.zeros(shape, dtype=np.float32))
                    var = 'dat[' + str(s) + ']'
                    if patterns is not None:
                         var = var + str(patterns[s])
                    m = m.replace('#' + str(s), var)
                dat_shape = eval(m).shape
        return dat_shape

# =============================================================================

    def create_subwin(self, source, parent_type, mode=None, magic='', patterns=None, glob_id=-1):
        """Function creates a subwindow.

        Parameters:
        -----------
        source: list of strings
            The source of the sub-window data in shared memory:
                dat = statestream. \
                      session_id. \
                      x[0]. \
                      x[1]. \
                      ...
        parent_type: string
            The parent type of the sub-window: 'meta_var', 'item'
        mode: string
            The current blitting mode of the source data. Available modes depend
            on the shape of the source data:
                'plot *'          (= 'curves', 'plot')
                'maps'          (= 'image')
                'RGB *'
                'scatter'
                'hist'
        magic: string
            A magic function applied on the source before visualization.
        patterns: list of strings
            Patterns used to retrieve data from sources (e.g. '[0,:,:,:]').
            The default is ''.
        """
        SW = empty_subwin()
        SW['type'] = copy.copy(parent_type)
        SW['source'] = copy.deepcopy(source)
        # Determine shape of source and final data.
        source_shape = None
        try:
            if len(source) == 2:
                source_shape = self.shm.layout[source[0]][source[1]].shape
            elif len(source) == 3:
                source_shape = self.shm.layout[source[0]][source[1]][source[2]].shape
            elif len(source) == 4:
                source_shape = self.shm.layout[source[0]][source[1]][source[2]][source[3]].shape
            elif len(source) == 5:
                source_shape = self.shm.layout[source[0]][source[1]][source[2]][source[3]][source[4]].shape
            elif len(source) == 6:
                source_shape = self.shm.layout[source[0]][source[1]][source[2]][source[3]][source[4]][source[5]].shape
        except:
            return None

        dat_shape = self.magic_shape([source_shape], magic, patterns)
        #print("\nSRCSHAPE: " + str(source_shape) + "  magic: " + str(magic) + "  pattern: " + str(patterns) + "  dat_shape: " + str(dat_shape) + "  src: " + str(source))

        SW['src_shape'] = source_shape
        SW['dat_shape'] = dat_shape

        # Determine visualization mode
        available_modes = self.subwin_mode_exists(dat_shape)
        SW['mode'] = available_modes[0]
        if mode is not None:
            if mode in available_modes:
                SW['mode'] = copy.copy(mode)

        # Set colormap flag.
        if SW['mode'] in ['maps', 'angle2D'] \
                or SW['mode'].startswith('plot'):
            SW['cm flag'] = True

        SW['magic'] = copy.copy(magic)
        if patterns is not None:
            SW['patterns'] = copy.copy(patterns)
        else:
            SW['patterns'] = ['']

        # Determine visualization shape.
        SW['shape'] = [1,1]
        if SW['mode'] == 'maps':
            if len(dat_shape) == 1:
                SW['shape'] = [1, dat_shape[0]]
            elif len(dat_shape) == 2:
                SW['shape'] = dat_shape
            elif len(dat_shape) == 3:
                if dat_shape[0] == 1 and dat_shape[1] * dat_shape[2] > 1:
                    SW['shape'] = dat_shape[1:]
                elif dat_shape[1] * dat_shape[2] == 1:
                    SW['shape'] = [1, dat_shape[0]]
                else:
                    SW['tileable'] = True
                    SW['sub shape'][0] = int(np.ceil(np.sqrt(dat_shape[0])))
                    SW['sub shape'][1] \
                        = int(np.ceil(float(dat_shape[0]) / SW['sub shape'][0]))
                    SW['shape'] = [SW['sub shape'][0] * dat_shape[1],
                                   SW['sub shape'][1] * dat_shape[2]]
            elif len(dat_shape) == 4:
                SW['tileable'] = True
                if dat_shape[0] == 1 and dat_shape[1] == 1:
                    SW['sub shape'] = [1, 1]
                elif dat_shape[0] == 1 and dat_shape[1] != 1:
                    SW['sub shape'][0] = int(np.ceil(np.sqrt(dat_shape[1])))
                    SW['sub shape'][1] \
                        = int(np.ceil(float(dat_shape[1]) / SW['sub shape'][0]))
                elif dat_shape[0] != 1 and dat_shape[1] == 1:
                    SW['sub shape'][0] = int(np.ceil(np.sqrt(dat_shape[0])))
                    SW['sub shape'][1] \
                        = int(np.ceil(float(dat_shape[0]) / SW['sub shape'][0]))
                else:
                    SW['sub shape'] = dat_shape[0:2]
                SW['shape'] = [SW['sub shape'][0] * dat_shape[2],
                               SW['sub shape'][1] * dat_shape[3]]
        elif SW['mode'] == 'angle2D':
            if len(dat_shape) == 3:
                SW['shape'] = dat_shape[1:]
            elif len(dat_shape) == 4:
                SW['tileable'] = True
                SW['sub shape'][0] = int(np.ceil(np.sqrt(dat_shape[0])))
                SW['sub shape'][1] \
                    = int(np.ceil(float(dat_shape[0]) / SW['sub shape'][0]))
                SW['shape'] = [SW['sub shape'][0] * dat_shape[2],
                               SW['sub shape'][1] * dat_shape[3]]
        elif SW['mode'].startswith('RGB'):
            if int(SW['mode'].split()[1]) == 0:
                if len(dat_shape) == 2:
                    SW['shape'] = [dat_shape[1], 1]
                elif len(dat_shape) == 3:
                    SW['shape'] = dat_shape[1:]
                elif len(dat_shape) == 4:
                    SW['tileable'] = True
                    SW['sub shape'][0] = int(np.ceil(np.sqrt(dat_shape[1])))
                    SW['sub shape'][1] \
                        = int(np.ceil(float(dat_shape[1]) / SW['sub shape'][0]))
                    SW['shape'] = [SW['sub shape'][0] * dat_shape[2],
                                   SW['sub shape'][1] * dat_shape[3]]
            elif int(SW['mode'].split()[1]) == 1:
                if len(dat_shape) == 2:
                    SW['shape'] = [dat_shape[0], 1]
                elif len(dat_shape) == 3:
                    SW['shape'] = [dat_shape[0], dat_shape[2]]
                elif len(dat_shape) == 4:
                    SW['tileable'] = True
                    SW['sub shape'][0] = int(np.ceil(np.sqrt(dat_shape[0])))
                    SW['sub shape'][1] \
                        = int(np.ceil(float(dat_shape[0]) / SW['sub shape'][0]))
                    SW['shape'] = [SW['sub shape'][0] * dat_shape[2],
                                   SW['sub shape'][1] * dat_shape[3]]

        # Create some sub-window internal visualization structures.
        if SW['mode'] in ['maps', 'angle2D'] or SW['mode'].startswith('RGB'):
            SW['buffer'] = pg.Surface(SW['shape'])
            SW['buffer'].convert()
            SW['array'] = np.zeros(SW['shape'], dtype=np.int32)
            SW['array double'] = np.zeros(SW['shape'], dtype=np.double)

        # By default set sub-window at screen center.
        if SW['type'] == 'meta_var':
            SW['pos'] = np.array([self.vparam['screen_width'] // 2, 
                                  self.vparam['screen_height'] // 2],
                                 dtype=np.float32)

        # Finally integrate into visualization structures.
        if parent_type == 'meta_var':
            for mv,MV in enumerate(self.meta_variables):
                if source[0] == MV.name:
                    MV.SW.append(SW)
                    MV.adjust_subwin()
                    break
        else:
            self.graph[source[0]]['sw'].append(SW)

        if glob_id != -1:
            self.subwin_blit_queue.insert(glob_id, SW)
        else:
            self.subwin_blit_queue.append(SW)

        self.update_subwindow_idx()

        return SW

# =============================================================================

    def delete_subwindow(self, SW):
        """Delete a given sub-window.
        """
        self.update_subwindow_idx()
        glob_idx = copy.copy(SW['glob_idx'])
        loc_idx = copy.copy(SW['loc_idx'])
        # Delete subwindow from blitting queue.
        del self.subwin_blit_queue[glob_idx]
        # Delete subindow from graph / mv list.
        if SW['type'] == 'meta_var':
            for mv,MV in enumerate(self.meta_variables):
                if MV.name == SW['source'][0]:
                    del MV.SW[loc_idx]
        else:
            del self.graph[SW['source'][0]]['sw'][loc_idx]
        self.update_subwindow_idx()

# =============================================================================

    def update_subwindow_idx(self):
        """Update local and global index of each subwind.
        """
        for sw,SW in enumerate(self.subwin_blit_queue):
            SW['glob_idx'] = sw
        for x,X in self.graph.items():
            for sw,SW in enumerate(X['sw']):
                SW['loc_idx'] = sw
        for mv,MV in enumerate(self.meta_variables):
            for sw,SW in enumerate(MV.SW):
                SW['loc_idx'] = sw

# =============================================================================

    def cb_button_LMB_click_hide_items(self, item_type):
        """Change visibility for item type.
        """
        self.graph_type_viz_flag[item_type] \
            = not self.graph_type_viz_flag[item_type]

# =============================================================================

    def cb_button_LMB_click_hide_conns(self, conn_type):
        '''Change visibility for connections.
        '''
        self.conn_type_viz_flag[conn_type] \
            = not self.conn_type_viz_flag[conn_type]
        
# =============================================================================

    def cb_button_LMB_click_break(self):
        '''System break button clicked, pause system.
        '''
        if self.IPC_PROC['break'].value == 0:
            self.IPC_PROC['break'].value = 1
            for m in self.monitor_bad_items:
                self.monitor_bad_items[m] = []
            # Change sprites for buttons.
            self.buttons['one-step'].sprite = 'one-step'
        else:
            self.IPC_PROC['break'].value = 0
            # Change sprites for buttons.
            self.buttons['one-step'].sprite = 'empty'

# =============================================================================

    def cb_button_LMB_click_one_step(self):
        '''System one-step button clicked (proceed for one step).
        '''
        if self.buttons['break'].value == 1:
            self.IPC_PROC['one-step'].value \
                = (self.IPC_PROC['one-step'].value + 1) % 2

# =============================================================================

    def cb_button_LMB_click_debug(self):
        '''System debug button was clicked (show some debug information).
        '''
        self.debug_flag = not self.debug_flag

# =============================================================================

    def cb_button_LMB_click_profiler(self):
        '''Button for profiler mode was clicked.
        '''
        self.show_profiler = (self.show_profiler + 1) % 3

# =============================================================================

    def cb_button_LMB_click_name(self):
        '''Button to change name presentation mode was clicked.
        '''
        self.show_name = (self.show_name + 1) % 3

# =============================================================================

    def cb_button_LMB_click_np_force(self):
        '''Button to switch on/off repulsive forces between nps.
        '''
        self.force_np_repulsion = not self.force_np_repulsion

    def cb_button_LMB_click_sp_force(self):
        '''Button to switch on/off repulsive forces between sps.
        '''
        self.force_sp_repulsion = not self.force_sp_repulsion

# =============================================================================

    def cb_button_LMB_click_close_subwins(self):
        '''Close all sub-windows.
        '''
        # Empty blitting queue.
        self.subwin_blit_queue = []
        for x,X in self.graph.items():
            X['sw'] = []
        for mv,MV in enumerate(self.meta_variables):
            MV.SW = []

    def cb_test(self, info):
        print('VIZ OUT CB TEST: ' + str(info))
        
# =============================================================================

    def cb_parameter_specification_done(self, modus=None, ptype=None, param=None):
        '''Callback for parameter specification completed.

        Parameter
        =========
        modus : string
            Defines for which task a parameter was entered.
        ptype : string
            Specifies parameters for the task.
        param : list of dicts
            Returned parameters from user.
        '''
        # Dependent on modus do something with the received parameters
        if modus == 'meta_variable':
            # Determine all selected values.
            # Create a metavariable from parameters.
            mv = {}
            mv['mv_type'] = ptype
            mv['mv_param'] = param
            mv['selected_values'] = self.mv_all_selected_values
            self.new_mv_list.append(mv)
        elif modus == "single param":
            # Update a single scalar integer parameter.
            # Generate single string.
            command_line = "set " + ptype + " parameter " + param[0]["name"] \
                           + " " + param[0]["type"] + " " \
                           + str(param[0]["value"])
            core_comm = {}
            core_comm['instruction string'] = copy.copy(command_line)
            core_comm['instruction'] = 1
            self.core_comm_queue.append(copy.deepcopy(core_comm))
        elif modus == "save model comment":
            # Save entire current model.
            # Define save path.
            save_path = self.param["core"]["save_path"] + os.sep + self.net["name"]
            # Check if save path exists, otherwise create.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Instruct core to save current model using the given comment.
            model_name = str(param[0]["value"]).replace(" ", "_")
            # Check if model_name ends with st_net.
            if not model_name.endswith(".st_net"):
                model_name += ".st_net"
            # Generate global save file name.
            save_filename = save_path + os.sep + model_name
            core_comm = {}
            core_comm['instruction string'] = copy.copy(save_filename)
            core_comm['save/load'] = 1
            self.core_comm_queue.append(copy.deepcopy(core_comm))
        elif modus == "edit param np":
            # Update edited net.
            self.edit_net["neuron_pools"][ptype]["act"] = str(param[0]["value"])
            if ptype not in self.edited_items:
                self.edited_items.append(ptype)
        elif modus == "edit param sp":
            # Update edited net.
            self.edit_net["synapse_pools"][ptype]["rf"] = eval(param[0]["value"])
            self.edit_net["synapse_pools"][ptype]["dilation"] = eval(param[0]["value"])
            # Add item to list of edited items.
            if ptype not in self.edited_items:
                self.edited_items.append(ptype)
        elif modus == "edit param if":
            # Update edited net.
            pass
        elif modus in ["sw magic", "sw pattern"]:
            # Apply a new magic function or pattern.
            old_SW = self.subwin_blit_queue[int(ptype)]
            sw_source = copy.copy(old_SW['source'])
            sw_type = copy.copy(old_SW['type'])
            sw_mode = copy.copy(old_SW['mode'])
            if modus == "sw magic":
                sw_magic = param[0]['value']
                sw_patterns = copy.copy(old_SW['patterns'])
            elif modus == "sw pattern":
                sw_magic = copy.copy(old_SW['magic'])
                sw_patterns = [param[0]['value']]
            sw_pos = np.copy(old_SW['pos'])
            sw_size = np.copy(old_SW['size'])
            # Delete current sub-window and replace it with new one in new mode.
            self.delete_subwindow(old_SW)
            new_SW = self.create_subwin(sw_source,
                                        sw_type,
                                        mode=sw_mode,
                                        magic=sw_magic,
                                        patterns=sw_patterns,
                                        glob_id=int(ptype))
            new_SW['pos'] = np.copy(sw_pos)
            new_SW['size'] = np.copy(sw_size)

# =============================================================================

    def cb_selection_window_over(self, source, modus, selections, button_id):
        """Callback for mouse over selection window.
        """
        if modus in ['save graphview', 'load graphview']:
            # Load and update current graphview version of slot.
            # Determine selection length.
            if modus == 'save graphview':
                sels = len(selections) - 1
            elif modus == 'load graphview':
                sels = len(selections)
            # Determine filename and load image.
            if button_id < sels:
                self.current_preview_active = True
                filename = self.graphview_file + "-{:02d}.jpg".format(button_id)
                if self.current_preview_src != filename:
                    self.current_preview = self.load_image(filename)
                    self.current_preview_src = filename

# =============================================================================

    def cb_list_mselection_done(self, modus, ptype, param, value):
        """A list multiple-selection is done.
        """
        if modus == 'meta_variable':
            # Get meta-variable type dependent list of parameters.
            mv_get_param \
                = getattr(importlib.import_module('statestream.meta.system_clients.' + ptype), 
                          'get_parameter')
            mv_param = mv_get_param(self.net, self.items_selected)
            # Set unique meta variable name.
            for p in range(len(mv_param)):
                if mv_param[p]['name'] == 'name':
                    # Search for unique name (index).
                    for i in range(2**10):
                        index_free = True
                        for mv,MV in enumerate(self.meta_variables):
                            if MV.name == "meta_var_" + str(i):
                                index_free = False
                                break
                        if index_free:
                            mv_param[p]['default'] = "meta_var_" + str(i)
                            break
            # Determine which of the values over all selected items were selected.
            self.mv_all_selected_values = []
            for c in self.items_selected:
                for i in range(len(param)):
                    if value[i]:
                        self.mv_all_selected_values.append((c,param[i]))
            # Determine visible position for drawing.
            max_len = 0
            for p in mv_param:
                max_len = max(max_len, len(p['name']) + len(str(p['default'])) + 4)
            strlist1 = [''.rjust(max_len) for i in mv_param]
            specwin_pos = self.get_visible_pos(default_pos=self.POS,
                                              font_size='small',
                                              strlist1=strlist1,
                                              strlist2=None)
            # Open new parameter specification window for meta variable parameters.
            ParamSpecWindow(parent=None,
                            wcollector=self.wcollector,
                            pos=np.array(specwin_pos, dtype=np.float32),
                            modus='meta_variable',
                            ptype=ptype,
                            parameter=mv_param,
                            cb_LMB_clicked=self.cb_parameter_specification_done)

# =============================================================================

    def cb_selection_window_click(self, source, modus, selections, button_id):
        '''Callback for button click in selection window.
        '''
        # Dependent on modus evaluate selection choise.
        if modus == 'item':
            if selections[button_id] == 'if viz':
                self.IPC_PROC['if viz'][source].value = 1
            elif selections[button_id] == 'on/off':
                if self.graph[source]['type'] in ['np', 'sp', 'plast']:
                    if self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value in [0, 1]:
                        self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value = 2
                        self.graph[source]['pause'] = 2
                    elif self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value == 2:
                        self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value = 0
                        self.graph[source]['pause'] = 0
            elif selections[button_id] == 'interfaces':
                if self.graph[source]['type'] in ['if']:
                    # Get interface type.
                    if_type = self.net["interfaces"][source]["type"]
                    # Get sub-interfaces of interface.
                    if_if \
                        = getattr(importlib.import_module("statestream.interfaces.process_if_" + if_type), 
                                  'if_interfaces')
                    in_out = if_if()
                    selections = []
                    sel_info = []
                    for i in in_out["in"] + in_out["out"]:
                        if i not in self.net["interfaces"][source]["in"] + self.net["interfaces"][source]["out"]:
                            tgt_np = "__none__"
                        else:
                            tgt_np = i
                            if 'remap' in self.net['interfaces'][source]:
                                if i in self.net['interfaces'][source]['remap']:
                                    tgt_np = self.net['interfaces'][source]['remap'][i]
                            selections.append(i.ljust(16) + tgt_np.rjust(16))
                            sel_info.append('')
                    selwin_pos = self.get_visible_pos(default_pos=self.POS,
                                                      font_size='small',
                                                      strlist1=selections,
                                                      strlist2=sel_info)
                    # Create selection window with current interface settings.
                    ListSelectionWindow(parent=None,
                                        wcollector=self.wcollector,
                                        pos=np.array(selwin_pos, 
                                                     dtype=np.float32),
                                        selections=selections,
                                        selections_info=sel_info,
                                        source=source,
                                        modus="online_remap",
                                        cb_LMB_clicked=self.cb_selection_window_click,
                                        cb_over=self.cb_selection_window_over)
            elif selections[button_id] == 'pause':
                if self.graph[source]['type'] == 'np':
                    if self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value == 0:
                        self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value = 1
                        self.graph[source]['pause'] = 1
                    elif self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value == 1:
                        self.IPC_PROC['pause'][self.shm.proc_id[source][0]].value = 0
                        self.graph[source]['pause'] = 0
            elif selections[button_id] == 'edit':
                # Edit hardwired parameters.
                # If not alread change into edit mode.
                if not self.edit_mode:
                    self.edit_mode = True
                    self.edit_net = copy.deepcopy(self.net)
                # At first create item type dependent list of editable parameters.
                if self.graph[source]['type'] == 'np':
                    # Get default values for parameters.
                    def_act = self.edit_net["neuron_pools"][source].get('act', 'Id')
                    # Create ParamSpecWindow to enter new parameters.
                    ParamSpecWindow(parent=None,
                                    wcollector=self.wcollector,
                                    pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                    modus='edit param np',
                                    ptype=source,
                                    parameter=[
                                    {
                                        "name": "activation function: ",
                                        "type": "string",
                                        "min": None,
                                        "max": None,
                                        "default": def_act
                                    }],
                                    cb_LMB_clicked=self.cb_parameter_specification_done)
                elif self.graph[source]['type'] == 'sp':
                    # Get default values for receptive fields.
                    def_rf = sp_get_dict(self.edit_net["synapse_pools"][source], "rf", 0)
                    def_dil = sp_get_dict(self.edit_net["synapse_pools"][source], "dilation", 1)
                    # Create ParamSpecWindow to enter new parameters.
                    ParamSpecWindow(parent=None,
                                    wcollector=self.wcollector,
                                    pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                    modus='edit param sp',
                                    ptype=source,
                                    parameter=[
                                    {
                                        "name": "receptive field(s): ",
                                        "type": "string",
                                        "min": None,
                                        "max": None,
                                        "default": def_rf
                                    },
                                    {
                                        "name": "dilations(s): ",
                                        "type": "string",
                                        "min": None,
                                        "max": None,
                                        "default": def_dil
                                    }],
                                    cb_LMB_clicked=self.cb_parameter_specification_done)
            else:
                # If scalar parameter was selected open ParamSpecWindow, else 
                # open a sub-window.
                selection = selections[button_id]
                scalar = False
                # Enable online change of scalar parameters.
                if selection.split()[0] == 'par:':
                    if is_scalar_shape(self.shm.layout[source]["parameter"][selection.split()[1]].shape):
                        if is_int_dtype(self.shm.layout[source]["parameter"][selection.split()[1]].dtype):
                            # Open new parameter specification window to change int-type parameter.
                            ParamSpecWindow(parent=None,
                                            wcollector=self.wcollector,
                                            pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                            modus='single param',
                                            ptype=source,
                                            parameter=[
                                            {
                                                "name": selection.split()[1],
                                                "type": "int",
                                                "min": self.shm.layout[source]["parameter"][selection.split()[1]].min,
                                                "max": self.shm.layout[source]["parameter"][selection.split()[1]].max,
                                                "default": self.shm.dat[source]["parameter"][selection.split()[1]][0]
                                            }],
                                            cb_LMB_clicked=self.cb_parameter_specification_done)
                            scalar = True
                        elif is_float_dtype(self.shm.layout[source]["parameter"][selection.split()[1]].dtype):
                            # Open new parameter specification window to change floating type parameter.
                            ParamSpecWindow(parent=None,
                                            wcollector=self.wcollector,
                                            pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                            modus='single param',
                                            ptype=source,
                                            parameter=[
                                            {
                                                "name": selection.split()[1],
                                                "type": "float",
                                                "min": self.shm.layout[source]["parameter"][selection.split()[1]].min,
                                                "max": self.shm.layout[source]["parameter"][selection.split()[1]].max,
                                                "default": self.shm.dat[source]["parameter"][selection.split()[1]][0]
                                            }],
                                            cb_LMB_clicked=self.cb_parameter_specification_done)
                            scalar = True
                # If selected is not scalar parameter, visualize it.
                if not scalar:
                    patterns = None
                    shm_source = [source]
                    if selection.split()[0] == "state":
                        shm_source.append('state')
                        patterns = ['[0,:,:,:]']
                    elif selection.split()[0] == "par:":
                        shm_source.append('parameter')
                        shm_source.append(selection[5:])
                    elif selection.split()[0] == "var:":
                        shm_source.append('variables')
                        shm_source.append(selection[5:])
                    elif selection.split()[0] == "upd:":
                        shm_source.append('updates')
                        shm_source.append(selection.split()[1])
                        shm_source.append(selection.split()[2])
                    self.create_subwin(shm_source, 'item', mode='maps', patterns=patterns)
        elif modus == 'save graphview':
            # Dump graphview in selected slot.
            self.dump_graphview(id=button_id)
        elif modus == 'load graphview':
            # Load graphview from selected slot.
            self.load_graphview(id=button_id)
        elif modus == 'sw mode':
            # Temporarily save some data of this sub-window.
            old_SW = self.subwin_blit_queue[int(source)]
            sw_source = copy.copy(old_SW['source'])
            sw_type = copy.copy(old_SW['type'])
            sw_magic = copy.copy(old_SW['magic'])
            sw_patterns = copy.copy(old_SW['patterns'])
            sw_pos = np.copy(old_SW['pos'])
            sw_size = np.copy(old_SW['size'])
            # Delete current subwindow and replace it with new one in new mode.
            self.delete_subwindow(old_SW)
            new_SW = self.create_subwin(sw_source,
                                        sw_type,
                                        mode=selections[button_id],
                                        magic=sw_magic,
                                        patterns=sw_patterns,
                                        glob_id=int(source))
            new_SW['pos'] = np.copy(sw_pos)
            new_SW['size'] = np.copy(sw_size)
        elif modus == 'tag':
            # Add all items with this tag to selection.
            for x, X in self.graph.items():
                if selections[button_id] in self.net[S2L(X['type'])][x].get('tags', []):
                    if x not in self.items_selected:
                        self.items_selected.append(x)
        elif modus == 'meta_var viz_sel var':
            # Select 'variable' and 'viz mode' for a meta-variable.
            # User has now selected the 'varialbe' and on the basis of this 'vars' shape
            # provide a selection window of available viz options.
            # Get meta-variable.
            for mv,MV in enumerate(self.meta_variables):
                if MV.name == source:
                    mv_idx = mv
                    break
            var = selections[button_id].lstrip().rstrip()
            shm_source = [self.meta_variables[mv_idx].name,
                          'variables',
                          var]
            self.create_subwin(shm_source,
                               'meta_var')
        elif modus == 'meta_variable_type':
            if selections[button_id] == 'on/off':
                for i in self.items_selected:
                    if self.IPC_PROC['pause'][self.shm.proc_id[i][0]].value in [0, 1]:
                        self.IPC_PROC['pause'][self.shm.proc_id[i][0]].value = 2
                        self.graph[i]['pause'] = 2
                    elif self.IPC_PROC['pause'][self.shm.proc_id[i][0]].value == 2:
                        self.IPC_PROC['pause'][self.shm.proc_id[i][0]].value = 0
                        self.graph[i]['pause'] = 0
            else:
                # Open new selection window for all possibible instances of a meta-var type.
                # Collect all possible
                sub_types = []
                mv_type = selections[button_id]
                mv_exists \
                    = getattr(importlib.import_module('statestream.meta.system_clients.' + mv_type), 
                              'exists_for')
                sub_mv = mv_exists(self.net, self.items_selected, self.shm)
                for smv in sub_mv:
                    sub_types.append(smv)
                modus = 'meta_variable'
                value = [True, False]
                dvalue = False
                max_len = 0
                for i in value:
                    max_len = max(max_len, len(str(i)))
                strlist2 = [''.rjust(max_len) for i in sub_types]
                selwin_pos = self.get_visible_pos(default_pos=self.POS,
                                                  font_size='small',
                                                  strlist1=sub_types,
                                                  strlist2=strlist2)
                ListMSelWindow(parent=None,
                               wcollector=self.wcollector,
                               pos=np.array(selwin_pos, dtype=np.float32),
                               modus=modus,
                               ptype=mv_type,
                               parameter=sub_types,
                               value=value,
                               dvalue=dvalue,
                               cb_LMB_clicked=self.cb_list_mselection_done)

        elif modus == 'load model comment':
            # Build up model load name.
            model_name = self.param["core"]["save_path"] + os.sep \
                         + self.net["name"] + os.sep \
                         + selections[button_id].replace(" ", "_") + ".st_net"
            core_comm = {}
            core_comm['instruction string'] = copy.copy(model_name)
            core_comm['save/load'] = 2
            self.core_comm_queue.append(copy.deepcopy(core_comm))

# =============================================================================

    def cb_button_over_info(self):
        '''Show some network information if mouse over settings info button.
        '''
        Y0 = 160
        dy = 16
        cntr = 0
        self.screen.blit(self.fonts['small'].render('name            :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.net['name']), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('agents          :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.net['agents']), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('neuron-pools    :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.mn.no_nps), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('synapse-pools   :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.mn.no_sps), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('plasticities    :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.mn.no_plasts), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('interfaces      :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.mn.no_ifs), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('processes       :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(self.mn.no_processes), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('neurons         :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(num2str(self.mn.no_neurons), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('synapses        :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(num2str(self.mn.no_synapses), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('shm nbytes      :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(num2str(np.sum(self.shm.log_bytes)), 
                                                    1,
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('tmem capacity   :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(len(self.param['core']['temporal_memory'])), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render('tmem distance   :', 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        self.screen.blit(self.fonts['small'].render(str(np.prod(self.param['core']['temporal_memory'])), 
                                                    1, 
                                                    self.cc(self.vparam['number_color'])), 
                         (240, Y0 + cntr * dy))

        # Blit overview over shortcuts.
        cntr += 2
        self.screen.blit(self.fonts['small'].render("s/S".rjust(10) + " :  zoom in/out", 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render("r/w".rjust(10) + " :  read/write viz settings", 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render("R/W".rjust(10) + " :  read/write model", 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render("cntr+r".rjust(10) + " :  re-init model", 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))
        cntr += 1
        self.screen.blit(self.fonts['small'].render("t".rjust(10) + " :  tag selection", 
                                                    1, 
                                                    self.cc(self.vparam['text_color'])), 
                         (64, Y0 + cntr * dy))

# =============================================================================

    # Dump graphview parametrization to local directory.
    def dump_graphview(self, id=0):
        '''Method to dump some visualization settings.
        '''
        # Generate graphview dictionary.
        graphview = {}
        graphview['items'] = {}
        graphview['settings'] = {}
        graphview['meta_vars'] = []
        graphview['subwindows'] = []
        # Add subwindows.
        for x, X in self.graph.items():
            graphview['items'][x] = {}
            graphview['items'][x]['pos'] = [float(X['pos'][0]), float(X['pos'][1])]
            graphview['items'][x]['pause'] = self.graph[x]['pause']
        # Add meta variables.
        for mv,MV in enumerate(self.meta_variables):
            mv_dict = {}
            mv_dict['name'] = MV.name
            mv_dict['type'] = MV.type
            mv_dict['values_child'] = []
            mv_dict['values_value'] = []
            for v in MV.sv:
                mv_dict['values_child'].append(v[0])
                mv_dict['values_value'].append(v[1])
            mv_dict['mv_params'] = MV.mv_param
            mv_dict['itemized'] = MV.itemized
            graphview['meta_vars'].append(mv_dict)
        # Add all subwindows.
        for sw,SW in enumerate(self.subwin_blit_queue):
            sw_dict = {}
            sw_dict['type'] = copy.copy(SW['type'])
            sw_dict['source'] = copy.copy(SW['source'])
            sw_dict['mode'] = copy.copy(SW['mode'])
            sw_dict['magic'] = copy.copy(SW['magic'])
            sw_dict['patterns'] = copy.deepcopy(SW['patterns'])
            sw_dict['pos'] = [float(SW['pos'][0]), 
                              float(SW['pos'][1])]
            sw_dict['size'] = [int(SW['size'][0]),
                               int(SW['size'][1])]
            sw_dict['tiles'] = copy.copy(SW['tiles'])
            sw_dict['tileable'] = copy.copy(SW['tileable'])
            sw_dict['glob_idx'] = copy.copy(SW['glob_idx'])
            graphview['subwindows'].append(copy.deepcopy(sw_dict))
        # Add item type / connection visibility settings.
        graphview['settings']['item type viz flag'] = {}
        for i in self.graph_i:
            graphview['settings']['item type viz flag'][i] = self.graph_type_viz_flag[i]
        graphview['settings']['conn type viz flag'] = {}
        for i in self.conn_i:
            graphview['settings']['conn type viz flag'][i] = self.conn_type_viz_flag[i]
        # Show name / profiler settings.
        graphview['settings']['show name'] = self.show_name
        graphview['settings']['show profiler'] = self.show_profiler
        graphview['np_dist'] = self.np_dist
        filename = self.graphview_file + "-{:02d}".format(id)
        # Save graphview.
        with open(filename, 'w') as outfile:
            dump_yaml(graphview, outfile)
        # Save scaled screen for preview.
        self.save_image(filename + ".jpg",
                        pg.transform.scale(self.screen, 
                                         (self.vparam['screen_width'] // 2, 
                                          self.vparam['screen_height'] // 2)))

# =============================================================================

    def load_graphview(self, id=0):
        # Try to load and apply graphview save file.
        filename = self.graphview_file + "-{:02d}".format(id)
        if os.path.isfile(filename):
            with open(filename, 'r') as infile:
                graphview = load_yaml(infile)
                # Add item positions.
                for x,X in self.graph.items():
                    if x in graphview['items']:
                        X['pos'][0] = graphview['items'][x]['pos'][0]
                        X['pos'][1] = graphview['items'][x]['pos'][1]
                # Remove all meta variables.
                while len(self.meta_variables) > 0:
                    del self.meta_variables[-1]
                # Add meta variables.
                for MV in graphview['meta_vars']:
                    sel_val = []
                    mv_valid = True
                    for i in range(len(MV['values_child'])):
                        if MV['values_child'][i] not in self.graph:
                            mv_valid = False
                            break
                        else:
                            sel_val.append((MV['values_child'][i], MV['values_value'][i]))
                    if mv_valid:
                        mv = {}
                        mv['mv_type'] = MV['type']
                        mv['mv_param'] = MV['mv_params']
                        mv['selected_values'] = sel_val
                        if 'itemized' in MV:
                            mv['itemized'] = MV['itemized']
                        # Check for subwindow.
                        self.new_mv_list.append(copy.deepcopy(mv))

                # Add item / connection type visibility settings.
                for i in self.graph_i:
                    self.graph_type_viz_flag[i] = graphview['settings']['item type viz flag'][i]
                    self.buttons['hide ' + i].value = int(not self.graph_type_viz_flag[i])
                for i in self.conn_i:
                    self.conn_type_viz_flag[i] = graphview['settings']['conn type viz flag'][i]
                    self.buttons['hide conn ' + i].value = int(not self.conn_type_viz_flag[i])
                # Remove all existing subwindows.
                self.cb_button_LMB_click_close_subwins()
                # Initialize subwindows ONLY for items.
                for sw,SW in enumerate(graphview.get('subwindows', [])):
                    if SW['type'] == 'item' and SW['source'][0] in self.graph:
                        new_SW = self.create_subwin(SW['source'],
                                                    SW['type'],
                                                    mode=SW['mode'],
                                                    magic=SW['magic'],
                                                    patterns=SW['patterns'])
                        new_SW['pos'] = np.copy(SW['pos'])
                        new_SW['size'] = np.copy(SW['size'])
                        new_SW['tiles'] = SW['tiles']
                        new_SW['tileable'] = SW['tileable']
                    elif SW['type'] == 'meta_var':
                        self.new_mv_subwins.append(copy.deepcopy(SW))
                # Show name / profiler settings.
                self.show_name = graphview['settings']['show name']
                self.buttons['name'].value = self.show_name
                self.show_profiler = graphview['settings']['show profiler']
                self.buttons['profiler'].value = self.show_profiler
                self.np_dist = graphview.get('np_dist', self.np_dist)
                for i in self.graph:
                    self.graph[i]['pause'] = int(self.IPC_PROC['pause'][self.shm.proc_id[i][0]].value)

# =============================================================================

    def load_image(self, filename):
        """Load an image while considering color corrections.
        """
        pixel_array = pg.PixelArray(pg.image.load(filename).convert())
        width = pixel_array.shape[0]
        height = pixel_array.shape[1]
        # Colorcorrect image.
        if self.ccol:
            for x in range(width):
                for y in range(height):
                    col = pixel_array[x,y]
                    b = col % 256
                    g = (col // 256**1) % 256
                    r = (col // 256**2) % 256
                    a = (col // 256**3) % 256
                    pixel_array[x,y] = r + g * 256 \
                                       + int(b // 2) * 256**2 + a * 256**3
        else:
            for x in range(width):
                for y in range(height):
                    col = pixel_array[x,y]
                    r = col % 256
                    g = (col // 256) % 256
                    b = (col // (256 * 256)) % 256
                    pixel_array[x,y] = r + g * 256 + b * 256 * 256
        return pixel_array.make_surface()

    def save_image(self, filename, surf):
        """Save an image while considering color corrections.
        """
        # Save surface to file.
        pg.image.save(surf, filename)



# =============================================================================



    def run(self, IPC_PROC, dummy):
        '''Main running method for visualization.
        '''
        # Reference to self.IPC_PROC.
        self.IPC_PROC = IPC_PROC
        # Get and set viz pid.
        self.IPC_PROC['gui pid'].value = os.getpid()
        # Init pygame.
        pg.display.init()
        pg.font.init()
        self.screen = pg.display.set_mode((self.vparam['screen_width'],
                                      self.vparam['screen_height']), pg.SRCALPHA, 32)
        clock = pg.time.Clock()
        background = pg.Surface(self.screen.get_size()).convert()
        background.fill(self.cc(self.vparam['background_color']))

        pg.mouse.set_visible(1)
        pg.key.set_repeat(1, 100)

        # Colorcodes.
        self.cm = ['magma', 'viridis', 'Spectral', 'hsv', 'Greys']
        self.CM = {}
        self.CM_RAW = {}
        self.CM_CC = {}
        for cms in self.cm:
            colormap = np.array(matplotlib.pyplot.get_cmap(cms)(np.arange(256) / 256))
            colormap = np.clip(colormap, 0.0, 0.999)
            self.CM_CC[cms] = np.copy(colormap)
            self.CM_CC[cms][:,0] = colormap[:,2] * 255
            self.CM_CC[cms][:,1] = colormap[:,1] * 255
            self.CM_CC[cms][:,2] = colormap[:,0] * 255
            self.CM_CC[cms] = self.CM_CC[cms].astype(np.int32)
            self.CM[cms] = colormap[:,:]
            if self.ccol:
                self.CM_RAW[cms] = np.fliplr(((colormap[:,0] * 255).astype(np.int32) + \
                               (colormap[:,1] * 255).astype(np.int32) * 256 + \
                               (colormap[:,2] * 127).astype(np.int32) * 256 * 256)[np.newaxis, :])
            else:
                self.CM_RAW[cms] = np.fliplr(((colormap[:,2] * 255).astype(np.int32) + \
                               (colormap[:,1] * 255).astype(np.int32) * 256 + \
                               (colormap[:,0] * 255).astype(np.int32) * 256 * 256)[np.newaxis, :])
        # Get shared memory.
        self.shm = SharedMemory(self.net, 
                                self.param,
                                session_id=\
                                    int(self.IPC_PROC['session_id'].value))

        # current visualization frame
        current_viz_frame = 0
        # current neural frame (from ipc)
        current_frame = 0
        # previous neural frame
        last_frame = 0
        # new neural frame flag
        self.new_frame = False
        
        # Set all fonts.
        self.fonts = {}
        self.font_size = {}
        for f in ['tiny', 'small', 'large', 'huge']:
            self.fonts[f] = pg.font.SysFont('Courier', 
                                            self.vparam['font ' + f])
            self.font_size[f] = self.vparam['font ' + f]

        # Load button sprites.
        button_surf = self.load_image('resources/buttons.png')
        self.button_source_rect = {
            'small grey empty': [0 * 16, 0, 16, 16],
            'small red exit': [1 * 16, 0, 16, 16],
            'small grey scale': [2 * 16, 0, 16, 16],
            'small grey drag': [3 * 16, 0, 16, 16],
            'small grey anker': [4 * 16, 0, 16, 16],
            'small left': [5 * 16, 0, 16, 16],
            'small right': [6 * 16, 0, 16, 16],
            'small color': [7 * 16, 0, 16, 16],
            'small item enable': [8 * 16, 0, 16, 16],
            'small item disable': [9 * 16, 0, 16, 16],
            'small item on': [10 * 16, 0, 16, 16],
            'small meta enable': [11 * 16, 0, 16, 16],
            'small meta disable': [12 * 16, 0, 16, 16],
            'small meta on': [13 * 16, 0, 16, 16],
            'small hist': [14 * 16, 0, 16, 16],
            'small star': [15 * 16, 0, 16, 16],
            'small red empty': [1 * 16, 16, 16, 16],
            'small value': [6 * 16, 32, 16, 16],
            'small curve': [7 * 16, 32, 16, 16],
            'small hash': [8 * 16, 32, 16, 16],
            'small tiles on': [9 * 16, 32, 16, 16],
            'small tiles off': [10 * 16, 32, 16, 16],
            'small bracket': [11 * 16, 32, 16, 16],
            'empty': [4 * 32, 96, 32, 32],
            'pause': [2 * 32, 64, 32, 32],
            'play': [3 * 32, 64, 32, 32],
            'close sw': [4 * 32, 64, 32, 32],
            'clock': [6 * 32, 64, 32, 32],
            'sel clock': [7 * 32, 64, 32, 32],
            'conn sw off': [5 * 32, 2 * 32, 32, 32],
            'conn sw on': [5 * 32, 3 * 32, 32, 32],
            'conn off': [6 * 32, 2 * 32, 32, 32],
            'conn on': [6 * 32, 3 * 32, 32, 32],
            'one-step': [1 * 32, 2 * 32, 32, 32],
            'color': [1 * 48, 4 * 32, 48, 48],
            'info': [2 * 48, 4 * 32, 48, 48],
            'monitor': [3 * 48, 4 * 32, 48, 48],
            'magnet': [4 * 48, 4 * 32, 48, 48],
            'cubeScalar': [0 * 24, 48 + 4 * 32, 24, 24],
            'cubeMaps': [1 * 24, 48 + 4 * 32, 24, 24],
            'cubePlane': [2 * 24, 48 + 4 * 32, 24, 24],
            'cubeVector': [3 * 24, 48 + 4 * 32, 24, 24],
            'alphaSphere': [4 * 24, 48 + 4 * 32, 24, 24]
        }
        self.button_sprite = {}
        for b,B in self.button_source_rect.items():
            self.button_sprite[b] = pg.Surface([B[2], B[3]], pg.SRCALPHA, 32)
            self.button_sprite[b].fill((0,0,0,0))
            self.button_sprite[b].blit(button_surf, (0,0), area=B)
        # Add sprites for item types.
        for y in ['on', 'off']:
            for t in self.graph_i:
                sprite_name = t + ' ' + y
                self.button_sprite[sprite_name] \
                    = pg.Surface([32, 32], pg.SRCALPHA, 32)
                self.button_sprite[sprite_name].fill((0,0,0,0))
                col = self.graph_col[t]
                if y == 'on':
                    y_col = DEFAULT_COLORS['light']
                else:
                    y_col = DEFAULT_COLORS['red']
                plot_circle(self.button_sprite[sprite_name], 
                            15, 15, 12, 
                            self.cc(y_col), 
                            self.cc(col), 
                            2)
        # Add sprites for connection types.
        for y in ['on', 'off']:
            for t in self.conn_i:
                if t != 'sw':
                    sprite_name = 'conn ' + t + ' ' + y
                    self.button_sprite[sprite_name] \
                        = pg.Surface([32, 32], pg.SRCALPHA, 32)
                    col = self.graph_col[t]
                    if y == 'on':
                        y_col = DEFAULT_COLORS['light']
                    else:
                        y_col = DEFAULT_COLORS['red']
                    pg.draw.line(self.button_sprite[sprite_name], 
                                 self.cc(y_col), 
                                 (4, 10), 
                                 (22, 10), 2)
                    pg.draw.line(self.button_sprite[sprite_name], 
                                 self.cc(y_col), 
                                 (4, 28), 
                                 (22, 10), 2)
                    pg.draw.line(self.button_sprite[sprite_name], 
                                 self.cc(y_col), 
                                 (22, 28), 
                                 (22, 10), 2)
                    plot_circle(self.button_sprite[sprite_name], 
                                22, 10, 6, 
                                self.cc(DEFAULT_COLORS['light']), 
                                self.cc(col), 2)
        # Some modified clocks.
        self.button_sprite['no clock'] = pg.Surface([32, 32], pg.SRCALPHA, 32)
        self.button_sprite['no clock'].fill((0,0,0,0))
        self.button_sprite['no clock'].blit(self.button_sprite['clock'], (0,0))
        pg.draw.line(self.button_sprite['no clock'], 
                     self.cc(DEFAULT_COLORS['red']), 
                     (4, 32 - 4), 
                     (32 - 4, 4), 8)
        # Some name buttons.
        self.button_sprite['name'] = pg.Surface([32, 32], pg.SRCALPHA, 32)
        self.button_sprite['name'].fill((0,0,0,0))
        self.button_sprite['name'].blit(self.button_sprite['empty'], (0,0))
        self.button_sprite['name'].blit(self.fonts['small'].render('na', 
                                                                   1, 
                                                                   self.cc(self.vparam['background_color'])), 
                                        (6, 0))
        self.button_sprite['name'].blit(self.fonts['small'].render('me', 
                                                                   1, 
                                                                   self.cc(self.vparam['background_color'])), 
                                        (6, 16))
        self.button_sprite['sel name'] = pg.Surface([32, 32], pg.SRCALPHA, 32)
        self.button_sprite['sel name'].fill((0,0,0,0))
        self.button_sprite['sel name'].blit(self.button_sprite['empty'], (0,0))
        self.button_sprite['sel name'].blit(self.fonts['small'].render('na', 
                                                                       1, 
                                                                       self.cc(self.vparam['text_color'])), 
                                            (6, 0))
        self.button_sprite['sel name'].blit(self.fonts['small'].render('me', 
                                                                       1, 
                                                                       self.cc(self.vparam['text_color'])), 
                                            (6, 16))
        self.button_sprite['no name'] = pg.Surface([32, 32], pg.SRCALPHA, 32)
        self.button_sprite['no name'].fill((0,0,0,0))
        self.button_sprite['no name'].blit(self.button_sprite['empty'], (0,0))
        self.button_sprite['no name'].blit(self.fonts['small'].render('na', 
                                                                      1, 
                                                                      self.cc(self.vparam['background_color'])), 
                                           (6, 0))
        self.button_sprite['no name'].blit(self.fonts['small'].render('me', 
                                                                      1, 
                                                                      self.cc(self.vparam['background_color'])), 
                                           (6, 16))
        pg.draw.line(self.button_sprite['no name'], 
                     self.cc(DEFAULT_COLORS['red']), 
                     (4, 32 - 4), 
                     (32 - 4, 4), 8)
        # Sprite for np force on/off.
        for y in ['on', 'off']:
            sprite_name = 'np force ' + y
            self.button_sprite[sprite_name] = pg.Surface([32, 32], pg.SRCALPHA, 32)
            col = self.graph_col['np']
            if y == 'on':
                y_col = DEFAULT_COLORS['light']
            else:
                y_col = DEFAULT_COLORS['red']
            pg.draw.line(self.button_sprite[sprite_name], self.cc(y_col), (8, 8), (23, 23), 2)
            pg.draw.line(self.button_sprite[sprite_name], self.cc(y_col), (8, 12), (19, 23), 2)
            pg.draw.line(self.button_sprite[sprite_name], self.cc(y_col), (12, 8), (23, 19), 2)
            plot_circle(self.button_sprite[sprite_name], 
                        24, 8, 6, 
                        self.cc(DEFAULT_COLORS['light']), 
                        self.cc(col), 2)
            plot_circle(self.button_sprite[sprite_name], 
                        8, 24, 6, 
                        self.cc(DEFAULT_COLORS['light']), 
                        self.cc(col), 2)
        # Sprite for sp force on/off.
        for y in ['on', 'off']:
            sprite_name = 'sp force ' + y
            self.button_sprite[sprite_name] = pg.Surface([32, 32], pg.SRCALPHA, 32)
            col = self.graph_col['sp']
            if y == 'on':
                y_col = DEFAULT_COLORS['light']
            else:
                y_col = DEFAULT_COLORS['red']
            pg.draw.line(self.button_sprite[sprite_name], self.cc(y_col), (8, 8), (23, 23), 2)
            pg.draw.line(self.button_sprite[sprite_name], self.cc(y_col), (8, 12), (19, 23), 2)
            pg.draw.line(self.button_sprite[sprite_name], self.cc(y_col), (12, 8), (23, 19), 2)
            plot_circle(self.button_sprite[sprite_name], 
                        24, 8, 6, 
                        self.cc(DEFAULT_COLORS['light']), 
                        self.cc(col), 2)
            plot_circle(self.button_sprite[sprite_name], 
                        8, 24, 6, 
                        self.cc(DEFAULT_COLORS['light']), 
                        self.cc(col), 2)

        # Create widget collector.
        self.wcollector = Collector(self.screen, 
                                    self.vparam,
                                    self.button_sprite, 
                                    self.fonts,
                                    self.ccol)

        # Dictionary of buttons.
        self.buttons = {}
        # Add info button.
        self.buttons['info'] = self.wcollector.add_button(None, sprite='info', 
                                       pos=np.asarray([4, 170], dtype=np.float32), 
                                       cb_over=lambda: self.cb_button_over_info())

        # Add break and one-step buttons.
        X0 = 200
        Y0 = 8
        self.buttons['break'] = self.wcollector.add_button(None, sprite=['pause', 'play'], 
                               pos=np.asarray([X0, Y0], dtype=np.float32), 
                               cb_LMB_clicked=lambda: self.cb_button_LMB_click_break())
        self.buttons['break'].value = self.IPC_PROC['break'].value
        if self.IPC_PROC['break'].value == 1:
            self.buttons['one-step'] = self.wcollector.add_button(None, sprite='one-step', 
                                   pos=np.asarray([X0 + 32, Y0], dtype=np.float32), 
                                   cb_LMB_clicked=lambda: self.cb_button_LMB_click_one_step())
        else:
            self.buttons['one-step'] = self.wcollector.add_button(None, sprite='empty', 
                                   pos=np.asarray([X0 + 32, Y0], dtype=np.float32), 
                                   cb_LMB_clicked=lambda: self.cb_button_LMB_click_one_step())
        # Add debug button.
        self.buttons['debug'] = self.wcollector.add_button(None, sprite='empty', 
                pos=np.asarray([X0 + 3 * 32, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_debug())
        # Hide buttons for items.
        for t in range(len(self.graph_i)):
            self.buttons['hide ' + self.graph_i[t]] \
                = self.wcollector.add_button(None, sprite=[self.graph_i[t] + ' on', self.graph_i[t] + ' off'], 
                                             pos=np.asarray([X0 + 4 * 32 + t * 32, Y0], dtype=np.float32), 
                                             cb_LMB_clicked=lambda x=self.graph_i[t]: self.cb_button_LMB_click_hide_items(x))
        # Hide buttons for connections.
        for t in range(len(self.conn_i)):
            self.buttons['hide conn ' + self.conn_i[t]] \
                = self.wcollector.add_button(None, sprite=['conn ' + self.conn_i[t] + ' on', 'conn ' + self.conn_i[t] + ' off'], 
                                             pos=np.asarray([X0 + 8 * 32 + t * 32, Y0], dtype=np.float32), 
                                             cb_LMB_clicked=lambda x=self.conn_i[t]: self.cb_button_LMB_click_hide_conns(x))
        # Add button to close all sub-windows.
        self.buttons['close all sw'] = self.wcollector.add_button(None,
                sprite='close sw', 
                pos=np.asarray([X0 + 12 * 32, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_close_subwins())
        # Add buttons for profiler.
        self.buttons['profiler'] = self.wcollector.add_button(None, 
                                    sprite=['no clock', 'clock', 'sel clock'],
                pos=np.asarray([X0 + 13 * 32, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_profiler())
        self.buttons['name'] \
            = self.wcollector.add_button(None, 
                                         sprite=['no name', 'name', 'sel name'],
                                         pos=np.asarray([X0 + 14 * 32, Y0], 
                                                        dtype=np.float32), 
                                         cb_LMB_clicked=lambda: self.cb_button_LMB_click_name())
        # Add np force on/off button.
        self.buttons['np force'] = self.wcollector.add_button(None, 
                                    sprite=['np force on', 'np force off'],
                pos=np.asarray([X0 + 15 * 32, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_np_force())
        # Add sp force on/off button.
        self.buttons['sp force'] = self.wcollector.add_button(None, 
                                    sprite=['sp force on', 'sp force off'],
                pos=np.asarray([X0 + 16 * 32, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_sp_force())

        # Load potential graphview settings.
        self.load_graphview()

        # Some timers
        timer_overall = np.ones([12])

        # Initially set min/max item rectangle.
        min_item_pos = None
        max_item_pos = None
        self.minimap_size = np.array([200, 200], dtype=np.float32)

        self.over_info = False


        # Needed to compute NN FPS.
        NNFPS_old_timer = time.time()
        NNFPS_old_frame = 0
        NNFPS = 0

        # State of mouse buttons.
        RMB_click = False
        RMB_hold = False
        RMB_hold_origin = np.zeros([2])
        RMB_drag_type = None
        RMB_drag_inst = None
        
        LMB_click = False
        LMB_hold = False
        LMB_hold_origin = np.zeros([2])
        LMB_drag_type = None
        LMB_drag_inst = None
        LMB_last_click = time.time()
        LMB_clicks = 0
        
        # Some nice arrays for numerical computation.
        self.items_double_array_0 = np.zeros([len(self.graph),], 
                                             dtype=np.double)
        self.items_double_array_1 = np.zeros([len(self.graph),], 
                                             dtype=np.double)
        self.items_double_array_2 = np.zeros([len(self.graph),], 
                                             dtype=np.double)
        self.items_double_array_3 = np.zeros([len(self.graph),], 
                                             dtype=np.double)
        self.items_double_array_4 = np.zeros([len(self.graph),], 
                                             dtype=np.double)
        self.items_double_array_5 = np.zeros([len(self.graph),], 
                                             dtype=np.double)

        # Get compressed time steps.
        self.tmem_time_steps = np.frombuffer(self.IPC_PROC['tmem time steps'],
                                        dtype=np.float32,
                                        count=len(self.param['core']['temporal_memory']))
        self.tmem_time_steps.shape \
            = [len(self.param['core']['temporal_memory'])]

        # Temporary draw board.
        self.tmp_surf = pg.Surface((self.vparam['screen_width'],
                                    self.vparam['screen_height']))

        # Enter forever loop.
        while self.state_running:
            # Set back all (items) up flag.
            self.all_up = True

            # Current viz frame.
            current_viz_frame += 1

            # Compute current (relative) profiler frame.
            viz_prof_frame = current_viz_frame % self.viz_prof_frames

            # Start viz timer.
            timer_overall[current_viz_frame % 12] = time.time()

            # Get current frame.
            current_frame = int(copy.copy(self.IPC_PROC['now'].value))
            # Get agent of interest.
            aoi = int(copy.copy(self.IPC_PROC['AOI'].value))
            # Determine beginning of new neuronal frame.
            if current_frame > last_frame:
                # Update last frame.
                last_frame = current_frame
                # Set new_frame flag.
                self.new_frame = True
            else:
                self.new_frame = False


            # =================================================================
            # Update all meta variables here at an early point.
            # =================================================================
            # Check for pending shm update due to new meta variable.
            for mv,MV in enumerate(self.meta_variables):
                if not MV.name in self.shm.dat:
                    self.shm.update_sys_client()
            # Add system-clients for pending meta-variables.
            if len(self.new_mv_list) > 0 \
                    and self.IPC_PROC['instruction'].value == 0 \
                    and self.all_up:
                # Get name.
                for p,P in enumerate(self.new_mv_list[0]['mv_param']):
                    if P["name"] == "name":
                        if P["value"] in self.graph:
                            P["value"] = P["value"] + "_MV"
                        mv_name = P["value"]
                        break
                self.add_meta_variable(mv_type=self.new_mv_list[0]['mv_type'], 
                                       mv_param=self.new_mv_list[0]['mv_param'], 
                                       selected_values=self.new_mv_list[0]['selected_values'],
                                       blitted=self.new_mv_list[0].get('blitted', False),
                                       itemized=self.new_mv_list[0].get('itemized', False))
                del self.new_mv_list[0]
            if len(self.new_mv_subwins) > 0:
                SW = self.new_mv_subwins[0]
                SW_inst = self.create_subwin(SW['source'], 
                                             'meta_var',
                                             mode=SW['mode'],
                                             magic=SW['magic'],
                                             patterns=SW['patterns'])
                if SW_inst is not None:
                    SW_inst['pos'] = np.copy(SW['pos'])
                    SW_inst['size'] = np.copy(SW['size'])
                    del self.new_mv_subwins[0]
            # =================================================================


            # =================================================================
            # Get all states.
            # =================================================================
            for x,X in self.graph.items():
                if X['type'] in ['np', 'sp', 'if', 'plast']:
                    self.graph[x]['state'] \
                        = self.IPC_PROC['state'][self.shm.proc_id[x][0]].value
            # =================================================================


            
            # Get current mouse position.
            POS = pg.mouse.get_pos()
            self.POS = POS

            self.subwin_info = []


            # Determine mouse over.
            self.mouse_over = ['bg', None]
            # Determine if over item.
            for x,X in self.graph.items():
                if X['rect'].collidepoint(POS):
                    if x in self.net['synapse_pools'] \
                            and not self.graph_type_viz_flag['sp']:
                        break
                    else:
                        self.mouse_over = ['item', x]
                        self.over_info = True
                        break

            # Determine mouse over subwindow
            for sw,SW in enumerate(self.subwin_blit_queue):
                if SW['rect'].collidepoint(self.POS):
                    self.mouse_over = ['sw', sw]
                    self.over_info = None
                    break

            # Update break / one-step buttons.
            self.buttons['break'].value = self.IPC_PROC['break'].value
            if self.buttons['break'].value == 1:
                self.buttons['one-step'].sprite = 'one-step'
            else:
                self.buttons['one-step'].sprite = 'empty'


            # =================================================================
            # Handle event queue
            # =================================================================
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.shutdown()
                    
                    # TODO: GUI REQUEST TO QUIT EVERYTHING:
                    #self.IPC_PROC['gui request'][0] = 1

                elif event.type == pg.MOUSEBUTTONUP and event.button == 3:
                    RMB_hold = False
                    RMB_drag_type = None
                    RMB_drag_inst = None
                    if RMB_hold_origin == POS:
                        RMB_click = True
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
                    RMB_hold = True
                    RMB_hold_origin = POS
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                    # In case we selected some items with LMB, add these items
                    # to selection.
                    if LMB_drag_type == '__seldrag__':
                        sel_rect = pg.Rect(int(min(self.POS[0], LMB_hold_origin[0])),
                                           int(min(self.POS[1], LMB_hold_origin[1])),
                                           int(abs(self.POS[0] - LMB_hold_origin[0])),
                                           int(abs(self.POS[1] - LMB_hold_origin[1])))
                        for x, X in self.graph.items():
                            if sel_rect.collidepoint(X['pos']):
                                if x not in self.items_selected \
                                        and self.graph_type_viz_flag[X['type']]:
                                    self.items_selected.append(x)
                    # Reset LMB variables.
                    LMB_hold = False
                    LMB_drag_type = None
                    LMB_drag_inst = None
                    if LMB_hold_origin == POS:
                        LMB_click = True
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    LMB_hold = True
                    LMB_hold_origin = POS
                    # Check for double click (2 x clicked in 300 ms).
                    if 1000 * (time.time() - LMB_last_click) < 300:
                        LMB_clicks = 2
                    else:
                        LMB_clicks = 1
                    LMB_last_click = time.time()
                        
                elif event.type == pg.KEYUP:
                    # Check for controll keys.
                    if event.key == pg.K_RSHIFT or event.key == pg.K_LSHIFT:
                        self.is_shift_pressed = False
                    elif event.key == pg.K_RCTRL or event.key == pg.K_LCTRL:
                        self.is_ctrl_pressed = False
                    if not self.is_typing_command \
                            and self.wcollector.active_textentry is None:
                        if event.key == pg.K_g:
                            self.hotkey['g'] = False
                        elif event.key == pg.K_c:
                            self.hotkey['c'] = False
                        elif event.key == pg.K_s:
                            self.hotkey['s'] = False
                elif event.type == pg.KEYDOWN:
                    # Check set controll keys.
                    if event.key == pg.K_RSHIFT or event.key == pg.K_LSHIFT:
                        self.is_shift_pressed = True
                    elif event.key == pg.K_RCTRL or event.key == pg.K_LCTRL:
                        self.is_ctrl_pressed = True
                    # Check if there is an active textentry field in the widget collector.
                    if self.wcollector.active_textentry is None \
                            and self.wcollector.top_widget is None:
                        if event.key == pg.K_ESCAPE:
                            self.shutdown()
                        elif event.key == pg.K_RETURN:
                            if self.is_typing_command:
                                self.is_typing_command = False
                                self.perform_command = True
                            else:
                                self.is_typing_command = True
                                self.command_line = ['']
                        # Check keyboard for inactive viz konsole.
                        if not self.is_typing_command:
                            if event.key in [pg.K_w, pg.K_r] \
                                    and not self.is_shift_pressed \
                                    and not self.is_ctrl_pressed:
                                # Open selection to save / recall current graphview.
                                # Determine existing graphviews for this model.
                                graphviews = os.listdir(self.home_path + '/.statestream/viz/')
                                this_bvs = 0
                                for bv in range(len(graphviews)):
                                    if graphviews[bv].find('-graphview-') != -1 \
                                            and graphviews[bv].find(self.net['name']) != -1 \
                                            and graphviews[bv].find(".jpg") == -1:
                                        this_bvs += 1
                                selections = [" slot " + str(s) + " " for s in range(this_bvs)]
                                sel_info = ['' for s in range(this_bvs)]
                                # For saving also add 'new'.
                                if event.key == pg.K_w:
                                    selections.append("new slot")
                                    sel_info.append('')
                                    modus = "save graphview"
                                else:
                                    modus = "load graphview"
                                # Add selection window to widget collector.
                                ListSelectionWindow(parent=None,
                                                    wcollector=self.wcollector,
                                                    pos=np.array([POS[0], POS[1]], 
                                                                 dtype=np.float32),
                                                    selections=selections,
                                                    selections_info=sel_info,
                                                    source='',
                                                    modus=modus,
                                                    cb_LMB_clicked=self.cb_selection_window_click,
                                                    cb_over=self.cb_selection_window_over)
                            elif event.key == pg.K_w \
                                    and self.is_shift_pressed \
                                    and not self.is_ctrl_pressed:
                                # Write entire model.
                                # Create a text entry window for the model comment.
                                ParamSpecWindow(parent=None,
                                                wcollector=self.wcollector,
                                                pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                                modus='save model comment',
                                                ptype=None,
                                                parameter=[
                                                {
                                                    "name": "comment",
                                                    "type": "string",
                                                    "min": None,
                                                    "max": None,
                                                    "default": "some comment describing the current model"
                                                }],
                                                cb_LMB_clicked=self.cb_parameter_specification_done)
                            elif event.key == pg.K_r \
                                    and self.is_shift_pressed \
                                    and not self.is_ctrl_pressed \
                                    and self.IPC_PROC['instruction'].value == 0 \
                                    and len(self.new_mv_list) == 0:
                                # Read entire model via core.
                                # Define save path.
                                save_path = self.param["core"]["save_path"] \
                                            + os.sep + self.net["name"]
                                # Only proceed if save path exists, otherwise 
                                # there are no saved models.
                                if os.path.exists(save_path):
                                    # Get a list of all available models.
                                    model_files = os.listdir(save_path)
                                    # Generate a selection list with avaiable models.
                                    selwin_sels = []
                                    selwin_info = []
                                    for m in model_files:
                                        if m.endswith(".st_net"):
                                            selwin_sels.append(m[0:-7].replace("_", " "))
                                            selwin_info.append("")
                                    # Create list selection window with all
                                    # available models.
                                    ListSelectionWindow(parent=None,
                                                        wcollector=self.wcollector,
                                                        pos=np.array([POS[0], POS[1]], 
                                                                     dtype=np.float32),
                                                        selections=selwin_sels,
                                                        selections_info=selwin_info,
                                                        source=None,
                                                        modus='load model comment',
                                                        cb_LMB_clicked=self.cb_selection_window_click)

                            elif event.key == pg.K_r \
                                    and not self.is_shift_pressed \
                                    and self.is_ctrl_pressed:
                                # Re-init selection by sending instruction to core.
                                if len(self.items_selected) == 0:
                                    # Re-init entire network.
                                    core_comm = {}
                                    core_comm['instruction string'] = "init"
                                    core_comm['instruction'] = 1
                                    self.core_comm_queue.append(copy.deepcopy(core_comm))
                                else:
                                    # Determine proc ids of all selected items.
                                    instruction_str = "init"
                                    for i in self.items_selected:
                                        instruction_str += " " + str(self.shm.proc_id[i][0])
                                    core_comm = {}
                                    core_comm['instruction string'] = instruction_str
                                    core_comm['instruction'] = 1
                                    self.core_comm_queue.append(copy.deepcopy(core_comm))
                            elif event.key == pg.K_t:
                                # Check if mouse over item.
                                if self.mouse_over[0] == "item":
                                    x = self.mouse_over[1]
                                    X = self.graph[x]
                                    selections = self.net[S2L(X['type'])][x].get('tags', 
                                                                                 ['empty'])
                                    sel_info = ['' for s in selections]
                                    # Add selection window to widget collector.
                                    ListSelectionWindow(parent=None,
                                                        wcollector=self.wcollector,
                                                        pos=np.array([POS[0], POS[1]], 
                                                                     dtype=np.float32),
                                                        selections=selections,
                                                        selections_info=sel_info,
                                                        source=x,
                                                        modus='tag',
                                                        cb_LMB_clicked=self.cb_selection_window_click)
                                else:
                                    # Show all tags for selection.
                                    selections = copy.copy(self.mn.tags)
                                    sel_info = ['' for s in selections]
                                    # Add selection window to widget collector.
                                    ListSelectionWindow(parent=None,
                                                        wcollector=self.wcollector,
                                                        pos=np.array([POS[0], POS[1]], 
                                                                     dtype=np.float32),
                                                        selections=selections,
                                                        selections_info=sel_info,
                                                        source=None,
                                                        modus='tag',
                                                        cb_LMB_clicked=self.cb_selection_window_click)
                            elif event.key == pg.K_e:
                                # Start / end network editing.
                                if not self.is_shift_pressed:
                                    # End editing and remove all edit changes.
                                    self.edited_items = []
                                    self.edit_net = None
                                    self.edit_mode = False
                                else:
                                    # Save edited network to .st_graph file.
                                    tmp_filename = os.path.expanduser('~') \
                                                   + '/.statestream/edit_net-' \
                                                   + str(self.IPC_PROC['session_id'].value) \
                                                   + '.st_graph'
                                    with open(tmp_filename, "w+") as f:
                                        dump_yaml(self.edit_net, f)
                                    # Instruct core to re-build edited network.
                                    core_comm = {}
                                    core_comm['instruction string'] = "edit"
                                    core_comm['instruction'] = 1
                                    self.core_comm_queue.append(copy.deepcopy(core_comm))
                                    # Set back internal edit mode.
                                    self.edited_items = []
                                    self.edit_net = None
                                    self.edit_mode = False
                            elif event.key == pg.K_SPACE:
                                # Loop through subwindows with mouse over.
                                sw_over_list = []
                                for sw, SW in enumerate(self.subwin_blit_queue):
                                    # Check for first / last mouse over.
                                    if SW['rect'].collidepoint(POS):
                                        sw_over_list.append(sw)
                                if len(sw_over_list) > 1:
                                    # Roll all mouse over subwins by one.
                                    for sw in range(len(sw_over_list) - 1):
                                        self.subwin_blit_queue[sw_over_list[sw]], \
                                        self.subwin_blit_queue[sw_over_list[sw + 1]] = \
                                        self.subwin_blit_queue[sw_over_list[sw + 1]], \
                                        self.subwin_blit_queue[sw_over_list[sw]]
                                elif len(sw_over_list) == 1:
                                    # Put window on top.
                                    self.subwin_blit_queue[sw_over_list[0]], \
                                    self.subwin_blit_queue[-1] = \
                                    self.subwin_blit_queue[-1], \
                                    self.subwin_blit_queue[sw_over_list[0]]
                                else:
                                    # Switch between pausing and streaming.
                                    self.cb_button_LMB_click_break()
                            elif event.key == pg.K_s:
                                # Zoom in and out.
                                self.hotkey['s'] = True
                            elif event.key == pg.K_g:
                                self.hotkey['g'] = True
                                self.hotkey_origin = np.array([POS[0], POS[1]])
                            elif event.key == pg.K_c:
                                self.hotkey['c'] = True
                                self.hotkey_origin = np.array([POS[0], POS[1]])
                            elif event.key == pg.K_a:
                                # Complete selection.
                                if len(self.items_selected) == 0:
                                    # Add all items.
                                    for x in self.graph:
                                        self.items_selected.append(x)
                                else:
                                    # Check if only sps or only nps selected.
                                    only_nps = True
                                    only_sps = True
                                    for x in self.items_selected:
                                        if self.graph[x]["type"] != "np":
                                            only_nps = False
                                        if self.graph[x]["type"] != "sp":
                                            only_sps = False
                                        if not only_nps and not only_sps:
                                            break
                                    # Update selection.
                                    if only_nps:
                                        for x, X in self.graph.items():
                                            if X["type"] == "np" \
                                                    and x not in self.items_selected:
                                                self.items_selected.append(x)
                                    if only_sps:
                                        for x, X in self.graph.items():
                                            if X["type"] == "sp" \
                                                    and x not in self.items_selected:
                                                self.items_selected.append(x)
                                    if not only_nps and not only_sps:
                                        # Check if all selected
                                        all_selected = True
                                        for x, X in self.graph.items():
                                            if X["type"] in ["np", "sp", "if", "plast"]:
                                                if x not in self.items_selected:
                                                    all_selected = False
                                                    break
                                        if all_selected:
                                            self.items_selected = []
                                        else:
                                            for x, X in self.graph.items():
                                                if X["type"] in ["np", "sp", "if", "plast"] \
                                                        and x not in self.items_selected:
                                                    self.items_selected.append(x)

                        # Check for characters [a..z, 0..9] if command line typing.
                        if self.is_typing_command:
                            key_name = pg.key.name(event.key)
                            if len(key_name) == 1:
                                if self.is_shift_pressed:
                                    if key_name[0].isalpha():
                                        self.command_line[-1] += key_name.upper()
                                    elif key_name[0].isdigit():
                                        if key_name in ['3', '#']:
                                            self.command_line[-1] += '#'
                                        elif key_name in ['1', '!']:
                                            self.command_line[-1] += '!'
                                    elif key_name in ['-', '_']:
                                        self.command_line[-1] += '_'
                                else:
                                    if key_name[0].isalpha() or key_name[0].isdigit() or key_name == '=':
                                        self.command_line[-1] += key_name
                            else:
                                if event.key == pg.K_SPACE:
                                    # Get suggestions.
                                    sugs = suggest_data(self.net, self.mn, self.command_line)
                                    # If only one suggestion, take it.
                                    if len(sugs) == 1:
                                        self.command_line[-1] = sugs[0] + '!'
                                        self.command_line.append('')
                                    else:
                                        # take whatever written
                                        self.command_line[-1] += '!'
                                        self.command_line.append('')
                                elif event.key == pg.K_BACKSPACE:
                                    if len(self.command_line[-1]) > 0:
                                        if self.command_line[-1][-1] == '!':
                                            self.command_line = self.command_line[0:-1]
                                            self.command_line.append('')
                                        else:
                                            self.command_line[-1] = self.command_line[-1][0:-1]
                                    else:
                                        if len(self.command_line) > 1:
                                            self.command_line = self.command_line[0:-2]
                                            self.command_line.append('')
                                elif event.key == pg.K_MINUS:
                                    self.command_line[-1] += '_'
                                elif event.key == pg.K_UNDERSCORE:
                                    self.command_line[-1] += '_'
                    else:
                        self.wcollector.key(event.key, self.is_shift_pressed, event.unicode)
            # =================================================================
                

            # Update zoom factor.
            if self.hotkey['s']:
                if self.is_shift_pressed:
                    factor = 1.0 / self.vparam["FPS"]
                else:
                    factor = -1.0 / self.vparam["FPS"]
                # Determine what should be zoomed.
                if self.mouse_over[0] == 'sw':
                    SW = self.subwin_blit_queue[self.mouse_over[1]]
                    if SW['mode'] == 'scatter':
                        new_area = (1.0 + factor) * SW['area']
                        new_minimum = (SW['min'] + SW['max']) / 2 \
                                      - new_area / 2
                        new_maximum = (SW['min'] + SW['max']) / 2 \
                                      + new_area / 2
                        # Update min / max for center attractor.
                        SW['min'] = np.copy(new_minimum)
                        SW['max'] = np.copy(new_maximum)
                        SW['area'] = SW['max'] - SW['min']
                else:
                    # Update all items.
                    if len(self.items_selected) == 0:
                        self.np_dist = self.np_dist + factor * self.np_dist
                        self.np_dist = max(self.np_dist, 40)
                        for x, X in self.graph.items():
                            dist = X['pos'] - self.POS
                            X['pos'] += factor * dist
                    else:
                        for x in self.items_selected:
                            X = self.graph[x]
                            dist = X['pos'] - self.POS
                            X['pos'] += factor * dist


            # =================================================================
            # Check LMB click / hold / drag.
            # =================================================================
            self.viz_prof_start['LMB'] = time.time()
            # Check all widgets for button clicked action.
            if LMB_click:
                LMB_click = self.wcollector.LMB_clicked(POS)
            if RMB_click:
                RMB_click = self.wcollector.RMB_clicked(POS)

            # Check if item clicked.
            if LMB_click:
                # Check if item or its subwindow clicked.
                for x,X in self.graph.items():
                    # Item clicked.
                    if X['rect'].collidepoint(POS):
                        if self.is_shift_pressed:
                            if x in self.items_selected:
                                self.items_selected \
                                    = [i for i in self.items_selected if i != x]
                            else:
                                self.items_selected += [x]
                        else:
                            self.items_selected = [x]
                        LMB_click = False
                        break

            if LMB_click:
                for sw_id in reversed(range(len(self.subwin_blit_queue))):
                    SW = self.subwin_blit_queue[sw_id]
                    x = SW['source'][0]
                    if SW['rect close'].collidepoint(POS):
                        self.delete_subwindow(SW)
                        LMB_click = False
                    elif SW['rect tiles'].collidepoint(POS):
                        # Show / hide tiles for subwindow.
                        if SW['tileable']:
                            if SW['tiles']:
                                SW['tiles'] = False
                            else:
                                SW['tiles'] = True
                        LMB_click = False
                    elif SW['rect mode'].collidepoint(POS):
                        selections = self.subwin_mode_exists(SW['dat_shape'])
                        sel_info = ['' for s in selections]
                        ListSelectionWindow(None,
                                            self.wcollector,
                                            pos=self.POS,
                                            selections=selections,
                                            selections_info=sel_info,
                                            source=str(sw_id),
                                            modus='sw mode',
                                            cb_LMB_clicked=self.cb_selection_window_click)
                        LMB_click = False
                    elif SW['rect magic'].collidepoint(POS):
                        # Change magic function.
                        # Create ParamSpecWindow to enter new magic fcn.
                        ParamSpecWindow(parent=None,
                                        wcollector=self.wcollector,
                                        pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                        modus='sw magic',
                                        ptype=str(sw_id),
                                        parameter=[
                                        {
                                            "name": "fcn: ",
                                            "type": "string",
                                            "min": None,
                                            "max": None,
                                            "default": SW['magic']
                                        }],
                                        cb_LMB_clicked=self.cb_parameter_specification_done)
                        LMB_click = False
                    elif SW['rect patterns'].collidepoint(POS):
                        # Change array access pattern.
                        # Create ParamSpecWindow to enter new pattern.
                        default = ''
                        if SW['patterns']:
                            default = SW['patterns'][0]
                        ParamSpecWindow(parent=None,
                                        wcollector=self.wcollector,
                                        pos=np.array([self.POS[0], self.POS[1]], dtype=np.float32),
                                        modus='sw pattern',
                                        ptype=str(sw_id),
                                        parameter=[
                                        {
                                            "name": "pattern: ",
                                            "type": "string",
                                            "min": None,
                                            "max": None,
                                            "default": default
                                        }],
                                        cb_LMB_clicked=self.cb_parameter_specification_done)
                        LMB_click = False
                    elif SW['cm rect'].collidepoint(POS):
                        # colormap clicked: change cm
                        SW['colormap'] \
                            = self.cm[(self.cm.index(SW['colormap']) + 1) \
                                % len(self.cm)]
                        LMB_click = False
                    elif SW['rect tmem+'].collidepoint(POS):
                        # Add 1 to tmem index of sw (go into past).
                        if SW['tmem'] < len(self.param['core']['temporal_memory']) - 1:
                            SW['tmem'] += 1
                        LMB_click = False
                    elif SW['rect tmem-'].collidepoint(POS):
                        # Substract 1 from tmem index of sw (go into future).
                        if SW['tmem'] >= 0:
                            SW['tmem'] -= 1
                        LMB_click = False
                    elif SW['rect'].collidepoint(POS) and x in self.graph:
                        X = self.graph[x]
                        # For the moment only allowed for np.
                        if X['type'] == 'np':
                            # Mere sw clicked.
                            # Get total source shape.
                            shape = self.mn.np_shape[SW['source'][0]]
                            # For feature maps switch to single feature map clicked on.
                            if SW['map'] == -1:
                                # Only proceed for nps with space.
                                if shape[-1] > 1 and shape[-2] > 1 and shape[-3] > 1:
                                    # Get sw coordintes of click.
                                    c_x = POS[0] - X['pos'][0] - SW['pos'][0]
                                    c_y = POS[1] - X['pos'][1] - SW['pos'][1]
                                    # Get feature x/y index of click.
                                    f_x = c_x // (SW['size'][0] / SW['sub shape'][0])
                                    f_y = c_y // (SW['size'][1] / SW['sub shape'][1])
                                    # Finally compute clicked feature map id.
                                    SW['map'] = f_x + f_y * SW['sub shape'][0]
                                    # Check selected map index.
                                    if SW['map'] > shape[1]:
                                        print('WARNING: Visualization try to visualize feature map ' \
                                              + str(SW['map']) + ' of np ' \
                                              + str(SW['source']))
                                        SW['map'] = -1
                                    # Use spatial dimension of state.
                                    SW['shape'] = [shape[2], shape[3]]
                            elif SW['map'] >= 0:
                                SW['map'] = -1
                                # General vector of feature maps.
                                SW['sub shape'][0] = int(np.ceil(np.sqrt(shape[1])))
                                SW['sub shape'][1] \
                                    = int(np.ceil(float(shape[1]) / SW['sub shape'][0]))
                                SW['shape'] \
                                    = [SW['sub shape'][0] * shape[2],
                                       SW['sub shape'][1] * shape[3]]
                            # Reset buffer / array / array float.
                            SW['buffer'] \
                                = pg.Surface(SW['shape'])
                            SW['buffer'].convert()
                            SW['array'] \
                                = np.zeros(SW['shape'], 
                                           dtype=np.int32)
                            SW['array double'] \
                                = np.zeros(SW['shape'], 
                                           dtype=np.double)
                        LMB_click = False
                    if not LMB_click:
                        break
                    
            # Check for settings.
            if LMB_click:
                if self.active_setting is not None:
                    for s,S in self.settings[self.active_setting]['sets'].items():
                        if S['type'] == 'iter':
                            if S['rect'].collidepoint(POS):
                                new_idx = (1 + S['values'].index(S['value'])) \
                                          % len(S['values'])
                                S['value'] = copy.copy(S['values'][new_idx])
                                LMB_click = False

            # Check meta-variables overview.
            if LMB_click:
                for mv,MV in enumerate(self.meta_variables):
                    if MV.rects['close'].collidepoint(POS):
                        # Instruct core to shutdown system client.
                        core_comm = {}
                        core_comm['instruction string'] \
                            = "remove_sys_client " + MV.name
                        core_comm['instruction'] = 1
                        self.core_comm_queue.append(copy.deepcopy(core_comm))
                        # Remove meta-variables subwin.
                        while MV.SW:
                            self.delete_subwindow(MV.SW[0])
                        # Remove meta-variable from list.
                        del self.meta_variables[mv]
                        LMB_click = False
                    elif MV.rects['itemize'].collidepoint(POS):
                        # Switch itemizeable mode if possible.
                        if MV.itemable > 0:
                            MV.itemized = (MV.itemized + 1) % MV.itemable
                        LMB_click = False
                    elif MV.rects['name'].collidepoint(POS):
                        # Select all children.
                        self.items_selected = []
                        for i,I in enumerate(MV.sv):
                            if I[0] not in self.items_selected:
                                self.items_selected.append(I[0])
                        LMB_click = False
                    elif MV.rects['viz'].collidepoint(POS):
                        # Open menue of available 'variables' for this meta-variable.
                        selection = []
                        sel_info = []
                        max_sel = 0
                        for v,V in MV.shm_layout['variables'].items():
                            selection.append(v)
                            sel_info.append(str(V['shape']))
                            max_sel = max(max_sel, len(v))
                        for i in range(len(MV.shm_layout['variables'])):
                            selection[i] = selection[i].ljust(max_sel + 2)
                        pos = self.get_visible_pos(default_pos=self.POS, 
                                                   strlist1=selection, 
                                                   strlist2=sel_info)
                        ListSelectionWindow(None,
                                            self.wcollector,
                                            pos=np.array(pos, 
                                                         dtype=np.float32),
                                            selections=selection,
                                            selections_info=sel_info,
                                            source=MV.name,
                                            modus='meta_var viz_sel var',
                                            cb_LMB_clicked=self.cb_selection_window_click)
                        LMB_click = False
                    if not LMB_click:
                        break

            # If nothing clicked, clear selection.
            if LMB_click:
                self.items_selected = []
                self.active_setting = None
                LMB_click = False
                
            # Catch LMB drag.
            if LMB_hold:
                # Determine dragged object.
                if LMB_drag_type is None and self.POS == LMB_hold_origin:
                    hold_eval = False
                    if not hold_eval:
                        # Item related objects.
                        for x,X in self.graph.items():
                            # Graph item dragged.
                            if X['rect'].collidepoint(LMB_hold_origin):
                                LMB_drag_type = X['type']
                                LMB_drag_inst = x
                                hold_eval = True
                                break
                        # Subwindow drag / scale.
                        for sw,SW in enumerate(self.subwin_blit_queue):
                            if SW['rect drag'].collidepoint(LMB_hold_origin):
                                LMB_drag_type = 'subwin'
                                LMB_drag_inst = str(sw) + ' drag'
                                hold_eval = True
                                break
                            if SW['rect scale'].collidepoint(LMB_hold_origin):
                                LMB_drag_type = 'subwin'
                                LMB_drag_inst = str(sw) + ' scale'
                                hold_eval = True
                                break
                    if not hold_eval:
                        # Settings drag.
                        if self.active_setting is not None:
                            for s,S in self.settings[self.active_setting]['sets'].items():
                                if S['type'] == 'float':
                                    if S['rect'].collidepoint(LMB_hold_origin):
                                        LMB_drag_type = '__setting__'
                                        LMB_drag_inst = self.active_setting + ' ' + s
                                        S['manip orig'] \
                                            = copy.copy(self.settings[self.active_setting]['sets'][s]['value'])
                                        S['manip dv'] = 0
                                        S['manip px'] = 0
                                        hold_eval = True
                                        break
                    # Drag delay.
                    if not hold_eval:
                        if self.delay_rect.collidepoint(LMB_hold_origin):
                            LMB_drag_type = '__delay__'
                            LMB_drag_inst = ''
                            hold_eval = True
                    # If LMB in blank: select mouse rectangle.
                    if not hold_eval:
                        LMB_drag_type = '__seldrag__'
                        LMB_drag_inst = ''
                else:
                    pass

                if LMB_drag_type == '__seldrag__':
                    # Nothing to do. Only interesting on button release.
                    pass
                elif LMB_drag_type in self.graph_i:
                    # If selected item dragged, drag also all other selected.
                    if LMB_drag_inst in self.items_selected:
                        for i in self.items_selected:
                            if LMB_drag_inst != i:
                                self.graph[i]['pos'] += POS - self.graph[LMB_drag_inst]['pos']
                    # Update dragged item.
                    self.graph[LMB_drag_inst]['pos'] = POS
                elif LMB_drag_type == 'subwin':
                    sw_idx = int(LMB_drag_inst.split()[0])
                    SW = self.subwin_blit_queue[sw_idx]
                    mod = LMB_drag_inst.split()[1]
                    # Check if parent is an item.
                    if SW['type'] == 'item':
                        if mod == 'drag':
                            SW['pos'] \
                                = POS - self.graph[SW['source'][0]]['pos']
                        elif mod == 'scale':
                            old_pos = copy.copy(SW['pos'])
                            SW['pos'] \
                                = POS - self.graph[SW['source'][0]]['pos'] \
                                  - np.array([self.sw_size + self.sw_size / 2, 
                                              self.sw_size / 2])
                            SW['size'][0] \
                                = np.clip(SW['size'][0] \
                                  + old_pos[0] \
                                  - SW['pos'][0], 48, 512)
                            SW['size'][1] \
                                = np.clip(SW['size'][1] \
                                  + old_pos[1] \
                                  - SW['pos'][1], 48, 512)
                    else:
                        if mod == 'drag':
                            SW['pos'] = POS
                        elif mod == 'scale':
                            old_pos = copy.copy(SW['pos'])
                            SW['pos'] = POS - np.array([self.sw_size + self.sw_size / 2, 
                                                        self.sw_size / 2])
                            SW['size'][0] = np.clip(SW['size'][0] \
                                            + old_pos[0] - SW['pos'][0], 48, 512)
                            SW['size'][1] = np.clip(SW['size'][1] \
                                            + old_pos[1] - SW['pos'][1], 48, 512)
                elif LMB_drag_type == '__delay__':
                    self.delay = int(np.clip(10 * self.POS[0] - 100, 0, 1000))
                elif LMB_drag_type == '__setting__':
                        # Get 'dragged' setting.
                        mods = LMB_drag_inst.split()
                        S = self.settings[mods[0]]['sets'][mods[1]]
                        # Orig in pixel.
                        dist = np.clip(POS[0] - 64 - 100 / 2,
                                       -100 / 2,
                                       100 / 2)
                        S['manip px'] = int(dist)
                        S['manip dv'] \
                            = abs(float(S['manip orig'])) * np.sign(dist) \
                              * (float(dist) / float(100 / 2))**2
                    
            else:
                # If not dragged, set to default.
                for x,X in self.graph.items():
                    for sw in range(len(X['sw'])):
                        if self.graph[x]['sw'][sw]['manip flag']:
                            self.graph[x]['sw'][sw]['manip px'] = 0
                            self.graph[x]['sw'][sw]['manip dv'] = 0
                if self.active_setting is not None:
                    for s,S in self.settings[self.active_setting]['sets'].items():
                        if S['type'] == 'float':
                            S['manip px'] = 0
                            S['manip dv'] = 0
            self.viz_prof_dur["LMB"][viz_prof_frame] \
                = time.time() - self.viz_prof_start["LMB"]
            # =================================================================


            # =================================================================
            # RMB click / drag / hold.
            # =================================================================
            # Check RMB click.
            if RMB_click:
                # Check if clicked over item.
                for x,X in self.graph.items():
                    if X['rect'].collidepoint(POS):
                        if x in self.items_selected:
                            # If item is selected determine available 
                            # meta-variable types.
                            selections = []
                            sel_info = []
                            for t in self.meta_variable_types:
                                mv_type_exists \
                                    = getattr(importlib.import_module('statestream.meta.system_clients.' + t), 
                                              'type_exists_for')
                                if mv_type_exists(self.net, self.items_selected, self.shm):
                                    selections.append(t)
                                    sel_info.append('')
                            # In case no interfaces are selected add on/off option.
                            on_off = True
                            for i in self.items_selected:
                                if i in self.net['interfaces']:
                                    on_off = False
                                    break
                            if on_off:
                                selections.insert(0, 'on/off')
                                sel_info.append('')
                            modus = 'meta_variable_type'
                            sorted_selections = selections
                            sorted_sel_info = sel_info
                        else:
                            # If not selected, determine item adjusted selections.
                            selections = copy.copy(self.options[X['type']])
                            sel_info = ['' for s in selections]
                            # add 'pause' and 'on/off' option for np with own proc
                            if X['type'] == 'np':
                                selections = selections + ['pause', 'on/off']
                                sel_info = sel_info + ['', '']
                            if X['type'] in ['np', 'sp', 'plast', 'if']:
                                for par in self.shm.dat[x]['parameter']:
                                    selections.append('par: ' + par)
                                    if is_scalar_shape(self.shm.layout[x]['parameter'][par].shape):
                                        if is_int_dtype(self.shm.layout[x]['parameter'][par].dtype):
                                            sel_info.append(' = ' + str(int(copy.copy(self.shm.dat[x]['parameter'][par][0]))))
                                        else:
                                            sel_info.append(' = ' + num2str(copy.copy(self.shm.dat[x]['parameter'][par][0])))
                                    else:
                                        sel_info.append(' ' + str(self.shm.layout[x]['parameter'][par].shape).replace(' ', ''))
                                for var in self.shm.dat[x]['variables']:
                                    selections.append('var: ' + var)
                                    if is_scalar_shape(self.shm.layout[x]['variables'][var].shape):
                                        if is_int_dtype(self.shm.layout[x]['variables'][var].dtype):                 
                                            sel_info.append(' = ' + str(int(copy.copy(self.shm.dat[x]['variables'][var][0]))))
                                        else:
                                            sel_info.append(' = ' + num2str(copy.copy(self.shm.dat[x]['variables'][var][0])))
                                    else:
                                        sel_info.append(' ' + str(self.shm.layout[x]['variables'][var].shape).replace(' ', ''))
                                if X['type'] == 'plast':
                                    for upd_id in self.shm.dat[x]['updates']:
                                        for upd_par in self.shm.dat[x]['updates'][upd_id]:
                                            selections.append('upd: ' + upd_id + ' ' + upd_par)
                                            sel_info.append('')
                            modus = 'item'
                            # Sort entries.
                            if X['type'] == 'plast':
                                sorted_selections = [s1 for (s1, s2) in sorted(zip(selections, sel_info))]
                                sorted_sel_info = [s2 for (s1, s2) in sorted(zip(selections, sel_info))]
                            else:
                                sorted_selections = selections
                                sorted_sel_info = sel_info
                        # Add selection window to widget collector.
                        selwin_pos = self.get_visible_pos(default_pos=RMB_hold_origin,
                                                          font_size='small',
                                                          strlist1=selections,
                                                          strlist2=sel_info)
                        ListSelectionWindow(None,
                                            self.wcollector,
                                            pos=np.array(selwin_pos, 
                                                         dtype=np.float32),
                                            selections=sorted_selections,
                                            selections_info=sorted_sel_info,
                                            source=x,
                                            modus=modus,
                                            cb_LMB_clicked=self.cb_selection_window_click)
                        
                        # Reset RMB.
                        RMB_click = False
                        break

            # Check for RMB click on meta-var name.
            if RMB_click:
                for mv,MV in enumerate(self.meta_variables):
                    if MV.rects['name'].collidepoint(self.POS):
                        # Get current parameters of current meta variable.
                        mv_param = copy.deepcopy(MV.mv_param)
                        t = copy.copy(MV.type)
                        self.mv_all_selected_values = copy.copy(MV.sv)
                        # Replace defaults with current values.
                        for p in range(len(mv_param)):
                            if mv_param[p]['name'] in MV.parameter:
                                mv_param[p]['default'] = MV.parameter[mv_param[p]['name']]
                        # Delete clicked on meta variable.
                        core_comm = {}
                        core_comm['instruction string'] = "remove_sys_client " + MV.name
                        core_comm['instruction'] = 1
                        self.core_comm_queue.append(copy.deepcopy(core_comm))
                        # Remove meta-variables subwin.
                        while MV.SW:
                            self.delete_subwindow(MV.SW[0])
                        del self.meta_variables[mv]
                        # Open selection window for new meta variable.
                        ParamSpecWindow(parent=None,
                                        wcollector=self.wcollector,
                                        pos=np.array([self.vparam['screen_width'] // 2, 
                                                      self.vparam['screen_height'] // 2], 
                                                     dtype=np.float32),
                                        modus='meta_variable',
                                        ptype=t,
                                        parameter=mv_param,
                                        cb_LMB_clicked=self.cb_parameter_specification_done)

                        RMB_click = False
                        break

            # If nothing clicked (RMB).
            if RMB_click:
                RMB_click = False

            # RMB hold actions.
            if RMB_hold:
                if RMB_drag_type is None:
                    if self.mouse_over[0] == 'sw' \
                            and self.subwin_blit_queue[self.mouse_over[1]]['mode'] == 'scatter':
                        RMB_drag_type = '__sw__'
                        RMB_drag_inst = self.mouse_over[1]
                    else:
                        RMB_drag_type = '__all__'
                        RMB_drag_inst = ''
                # Evaluate drag for different RMB drag types.
                if RMB_drag_type == '__all__':
                    # Update all items.
                    for x, X in self.graph.items():
                        X['pos'][0] += 0.1 * (self.POS[0] - RMB_hold_origin[0])
                        X['pos'][1] += 0.1 * (self.POS[1] - RMB_hold_origin[1])
                elif RMB_drag_type == '__sw__':
                    # Update min/max/area of meta-var. sub-window.
                    SW = self.subwin_blit_queue[self.mouse_over[1]]
                    SW['min'][0] -= 0.0005 * SW['area'][0] * (self.POS[0] - RMB_hold_origin[0])
                    SW['min'][1] += 0.0005 * SW['area'][1] * (self.POS[1] - RMB_hold_origin[1])
                    SW['max'][0] -= 0.0005 * SW['area'][0] * (self.POS[0] - RMB_hold_origin[0])
                    SW['max'][1] += 0.0005 * SW['area'][1] * (self.POS[1] - RMB_hold_origin[1])
                    SW['area'] = SW['max'] - SW['min']
            # =================================================================


            # =================================================================
            # Graph overview: Compute and update item item forces / positions.
            # =================================================================
            self.viz_prof_start["item force comput."] \
                = time.time()
            self.items_double_array_0 *= 0.0
            self.items_double_array_1 *= 0.0
            self.items_double_array_2 *= 0.0
            self.items_double_array_3 *= 0.0
            for x in self.np_n2i:
                self.graph[x]['force'] *= 0
                # Generate np arrays for np pos and force.
                self.items_double_array_0[self.np_n2i[x]] \
                    = self.graph[x]['pos'][0]
                self.items_double_array_1[self.np_n2i[x]] \
                    = self.graph[x]['pos'][1]
            # Compute np-np repulsion forces.
            if self.force_np_repulsion:
                cgraphics_np_force(self.items_double_array_0,
                                   self.items_double_array_1,
                                   self.conn_mat,
                                   len(self.np_n2i),
                                   int(self.np_dist * 0.95),
                                   int(self.np_dist * 1.05),
                                   self.items_double_array_2,
                                   self.items_double_array_3)

            # Reset all forces.
            self.items_double_array_0 *= 0.0
            self.items_double_array_1 *= 0.0
            self.items_double_array_4 *= 0.0
            self.items_double_array_5 *= 0.0
            # Compute sp-sp repulsion forces.
            if self.force_sp_repulsion:
                for x, sp_id in self.sp_n2i.items():
                    self.graph[x]['force'] *= 0
                    # Generate np arrays for np pos and force.
                    self.items_double_array_0[sp_id] \
                        = self.graph[x]['pos'][0]
                    self.items_double_array_1[sp_id] \
                        = self.graph[x]['pos'][1]
                cgraphics_np_force(self.items_double_array_0,
                                   self.items_double_array_1,
                                   self.conn_mat_sp,
                                   len(self.sp_n2i),
                                   32,
                                   1,   
                                   self.items_double_array_4,
                                   self.items_double_array_5)

            # Update forces in graph structure with those computed.
            for x, np_id in self.np_n2i.items():
                self.graph[x]['force'][0] += self.items_double_array_2[np_id]
                self.graph[x]['force'][1] += self.items_double_array_3[np_id]
            for x, sp_id in self.sp_n2i.items():
                self.graph[x]['force'][0] += self.items_double_array_4[sp_id]
                self.graph[x]['force'][1] += self.items_double_array_5[sp_id]

            # Update forces for np-sp-np positioning.
            for x0, X0 in self.graph.items():
                if X0['type'] == 'sp':
                    tgt_np = self.net['synapse_pools'][x0]['target']
                    # List of all sources.
                    srcs = [src for src_list in self.net["synapse_pools"][x0]["source"] for src in src_list]
                    if srcs.count(tgt_np) == len(srcs):
                        # If all src equal to target, move to tgt until close.
                        dx = self.graph[srcs[0]]['pos'][0] - X0['pos'][0]
                        dy = self.graph[srcs[0]]['pos'][1] - X0['pos'][1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 40:
                            X0['force'][0] += 0.1 * dx
                            X0['force'][1] += 0.1 * dy
                    else:
                        # Compute mean over all sources unequal to target
                        # this will become the "source" position.
                        src_pos = [0.0, 0.0]
                        cntr = 0
                        for s in srcs:
                            if s != tgt_np:
                                src_pos[0] += self.graph[s]['pos'][0]
                                src_pos[1] += self.graph[s]['pos'][1]
                                cntr += 1
                        # Take mean.
                        src_pos[0] /= cntr
                        src_pos[1] /= cntr
                        diff = np.array([self.graph[tgt_np]['pos'][0] - src_pos[0],
                                         self.graph[tgt_np]['pos'][1] - src_pos[1]])
                        attractor = self.graph[tgt_np]['pos'] - 0.5 * diff
                        # Update force.
                        diff = attractor - X0['pos']
                        X0['force'] += 0.1 * diff
                            
            # Apply force to vel, vel to pos, and friction to vel.
            # And while at it compute min/max pos rectangle.
            min_item_pos = None
            max_item_pos = None
            for x, X in self.graph.items():
                if np.abs(X['force'][0]) < 0.5:
                    X['force'][0] = 0.0 
                if np.abs(X['force'][1]) < 0.5:
                    X['force'][1] = 0.0 
                X['vel'] += X['force']
                X['vel'] = np.clip(X['vel'], -10, 10)
                X['pos'] += X['vel']
                X['vel'] *= self.graph_friction
                if min_item_pos is None:
                    min_item_pos = copy.copy(X['pos'])
                    max_item_pos = copy.copy(X['pos'])
                else:
                    min_item_pos[0] = min(min_item_pos[0], X['pos'][0])
                    min_item_pos[1] = min(min_item_pos[1], X['pos'][1])
                    max_item_pos[0] = max(max_item_pos[0], X['pos'][0])
                    max_item_pos[1] = max(max_item_pos[1], X['pos'][1])
            self.viz_prof_dur["item force comput."][viz_prof_frame] \
                = time.time() - self.viz_prof_start["item force comput."]
            # =================================================================


            # =================================================================
            # Monitors: Compute monitor states.
            # =================================================================
            self.viz_prof_start["monitor comput."] = time.time()
            # Update monitor for new frame.
            if self.new_frame and False:
                if self.settings['monitor']['sets']['m state 0/1']['value'] != 'off':
                    for n in self.monitor_item['mean_state']:
                        self.monitor_item['mean_state'][n] \
                            = np.mean(self.shm.dat[n]['state'], axis=(1,2,3))
                if self.settings['monitor']['sets']['v state 0/1']['value'] != 'off':
                    for n in self.monitor_item['var_state']:
                        self.monitor_item['var_state'][n] \
                            = np.var(self.shm.dat[n]['state'], axis=(1,2,3))
                if self.settings['monitor']['sets']['n state 0/1']['value'] != 'off':
                    for n in self.monitor_item['nan_state']:
                        self.monitor_item['nan_state'][n] \
                            = np.isnan(np.sum(self.shm.dat[n]['state'].flatten()))
                        
            # Check monitor: update bad items and break if escalation
            for m in ['mean_state', 'var_state', 'nan_state']:
                if self.settings['monitor']['sets'][m[0] + ' state 0/1']['value'] in ['on']:
                    for x,X in self.graph.items():
                        if X['type'] == 'np':
                            if m in ['mean_state', 'var_state']:
                                if max(abs(self.monitor_item[m][x])) > self.settings['monitor']['sets'][m]['value']:
                                    self.monitor_bad_items[m].append(x)
                                    # Set system to pause.
                                    self.IPC_PROC['break'].value = 1
                            elif m == 'nan_state':
                                if self.monitor_item['nan_state'][x]:
                                    self.monitor_bad_items[m].append(x)
                                    # Set system to pause.
                                    self.IPC_PROC['break'].value = 1
            self.viz_prof_dur["monitor comput."][viz_prof_frame] \
                = time.time() - self.viz_prof_start["monitor comput."]
            # =================================================================


            # Compute NN FPS.
            if time.time() - NNFPS_old_timer > 2.0:
                tmp_n = int(self.IPC_PROC['now'].value)
                NNFPS = (tmp_n - NNFPS_old_frame) // 2
                NNFPS_old_frame = tmp_n
                NNFPS_old_timer = time.time()


            # Blit background - tabula rasa.
            # All plotting / drawing / blitting should come afterwards.
            self.screen.blit(background, (0, 0))


            # =================================================================
            # Command line.
            # =================================================================
            # Check command line input.
            if self.perform_command:
                # Reset perform flag.
                self.perform_command = False
                # Do some preprocessing on the command line.
                if len(self.command_line) > 0:
                    if self.command_line[-1] == '':
                        del self.command_line[-1]
                # Correct last entry of command line for probably missing !
                if len(self.command_line) > 0:
                    if self.command_line[-1][-1] != '!':
                        self.command_line[-1] += '!'
                # If not each entry ends with an '!' --> invalid command.
                valid_command = True
                for c in range(len(self.command_line)):
                    if self.command_line[c][-1] != '!':
                        valid_command = False
                if len(self.command_line) > 0:
                    if self.command_line[0] not in ['reset!', 'mv!']:
                        valid_command = False
                    if self.command_line[0] == 'mv!':
                        if self.command_line[2] not in ['get!', 
                                                        'class_label!', 
                                                        'conf_mat!', 
                                                        'tmp_acc!', 
                                                        'var_mean!']:
                            valid_command = False
                        if self.command_line[2] not in ['class_label!', 'tmp_acc!'] \
                                and self.command_line[3] not in ['np!', 
                                                                 'sp!', 
                                                                 'plast!', 
                                                                 'if!']:
                            valid_command = False
                else:
                    valid_command = False
                # Dependent on main command.
                if valid_command:
                    if self.command_line[0] == 'reset!':
                        # Generate single string.
                        command_line = ''
                        for c in self.command_line:
                            command_line += c + ' '
                        # Send command line to core.
                        core_string = np.array([ord(c) for c in command_line])
                        self.IPC_PROC['instruction len'].value = len(core_string)
                        self.IPC_PROC['string'][0:len(core_string)] = core_string[:]
                        self.IPC_PROC['instruction'].value = 1
                else:
                    print('\nWARNING: Ignored an invalid command in visualization.')

            # Print command line.
            command_line = '>> '
            if self.is_typing_command:
                for c in self.command_line:
                    if len(c) > 0:
                        if c[-1] == '!':
                            command_line += c[0:-1] + ' '
                        else:
                            command_line += c
                # Get suggestions.
                sugs = suggest_data(self.net, self.mn, self.command_line)
                # Get offset for blit due to removed '!'.
                cc_correct = 16 * len(command_line.split())
                for s in range(len(sugs)):
                    self.screen.blit(self.fonts['small'].render(str(sugs[s]),
                                                                1, 
                                                                self.cc(DEFAULT_COLORS['light'])), 
                                     (240 + len(command_line) * 14 - cc_correct, 60 + s * 20))
                # Add active cursor at end of command.
                if (current_viz_frame // 10) % 2 == 0:
                    command_line += '_'
            # Blit command line.
            self.screen.blit(self.fonts['small'].render(command_line, 
                                                        1, 
                                                        self.cc(DEFAULT_COLORS['light'])), 
                             (240, 40))
            # =================================================================


            # Check if mouse over widget.
            self.wcollector.mouse_over(POS)


            # =================================================================
            # Setting buttons on the left.
            # =================================================================
            # Draw settings buttons.
            cntr = 1
            self.settings['monitor']['rect'] \
                = self.screen.blit(self.button_sprite['monitor'], 
                                   (4, 200 + cntr * 48))
            cntr += 1

            # Check for mouse over settings buttons.
            for s,S in self.settings.items():
                if S['rect'].collidepoint(POS):
                    self.active_setting = s

            # Draw active button settings.
            if self.active_setting is not None:
                for s,S in self.settings[self.active_setting]['sets'].items():
                    if S['type'] == 'iter':
                        S['rect'] = pg.draw.rect(self.screen, 
                                                 self.cc((127,127,127)), 
                                                 [60, S['y'] - 4, 120, 48], 4)
                        self.screen.blit(self.fonts['large'].render(s, 
                                                                    1, 
                                                                    self.cc((255,255,255))), 
                                         (64, S['y']))
                        self.screen.blit(self.fonts['large'].render(str(S['value']), 
                                                                    1, 
                                                                    self.cc((255,255,255))), 
                                         (82, S['y'] + 22))
                    elif S['type'] == 'float':
                        self.screen.blit(self.fonts['large'].render(s, 
                                                                    1, 
                                                                    self.cc((255,255,255))), 
                                         (64, S['y']))
                        S['value'] = copy.copy(S['value'] + S['manip dv'])
                        self.screen.blit(self.fonts['small'].render(num2str(S['value']), 
                                                                    1, 
                                                                    self.cc((255,255,255))), 
                                         (64, S['y'] + 22))
                        # blit manipulator
                        S['rect'] = pg.draw.circle(self.screen, self.cc((255,255,255)), 
                                                   (64 + int(100 / 2 + S['manip px']), int(S['y'] + 52)),
                                                   8, 2)
                        pg.draw.line(self.screen, 
                                     self.cc((255,255,255)), 
                                     [64, S['y'] + 52], 
                                     [164, S['y'] + 52], 
                                     1)
            # =================================================================


            # =================================================================
            # Graph overview: Draw items and their connections.
            # =================================================================
            self.viz_prof_start["draw connections"] = time.time()
            # Draw all inter-item connections.
            items_drawn = False
            if self.mouse_over[0] == 'item':
                if self.graph[self.mouse_over[1]]['type'] == 'np':
                    for x,X in self.graph.items():
                        if X['type'] == 'sp':
                            if self.net['synapse_pools'][x]['target'] == self.mouse_over[1]:
                                conn_col = (255, 200, 200)
                        elif X['type'] == 'np':
                            pass
            if not items_drawn:
                conn_col = (127, 127, 127)
                for x,X in self.graph.items():
                    # Draw all np-sp-np connections.
                    if self.conn_type_viz_flag['sp'] and X['type'] == 'sp':
                        if not self.graph_type_viz_flag['sp'] \
                                and len(self.net['synapse_pools'][x]['source']) == 1 \
                                and len(self.net['synapse_pools'][x]['source'][0]) == 1:
                            source = self.net['synapse_pools'][x]['source'][0][0]
                            target = self.net['synapse_pools'][x]['target']
                            draw_dashed_line(self.screen, 
                                             self.cc(conn_col), 
                                             np.array([self.graph[source]['pos'][0], 
                                                       self.graph[source]['pos'][1]]), 
                                             np.array([self.graph[target]['pos'][0], 
                                                       self.graph[target]['pos'][1]]), 
                                             width=2,
                                             dash=self.dash_size, 
                                             offset=(current_frame % 32) / 16.0,
                                             dash_type=1)
                        else:
                            for c in X['in']:
                                if self.graph[c]['type'] == 'np':
                                    draw_dashed_line(self.screen, 
                                                     self.cc(conn_col), 
                                                     np.array([self.graph[c]['pos'][0], 
                                                               self.graph[c]['pos'][1]]), 
                                                     np.array([X['pos'][0], X['pos'][1]]), 
                                                     width=2, 
                                                     dash=self.dash_size, 
                                                     offset=(current_frame % 32) / 16.0,
                                                     dash_type=1)
                            for c in X['out']:
                                if self.graph[c]['type'] == 'np':
                                    draw_dashed_line(self.screen, 
                                                     self.cc(conn_col), 
                                                     np.array([X['pos'][0], X['pos'][1]]), 
                                                     np.array([self.graph[c]['pos'][0], 
                                                               self.graph[c]['pos'][1]]), 
                                                     width=2, 
                                                     dash=self.dash_size, 
                                                     offset=(current_frame % 32) / 16.0,
                                                     dash_type=1)

            # Draw all for sub-windows, ifs and plasts.
            for x,X in self.graph.items():
                # Draw all subwindow connections.
                if self.conn_type_viz_flag['sw']:
                    for sw,SW in enumerate(X['sw']):
                        pg.draw.line(self.screen, 
                                     self.cc((255,255,255)), 
                                     [X['pos'][0], X['pos'][1]], 
                                     [X['pos'][0] + SW['pos'][0] \
                                        + SW['size'][0] / 2, 
                                     X['pos'][1] + SW['pos'][1] \
                                        + SW['size'][1] / 2], 
                                     1)
                # Draw interface connections.
                if self.conn_type_viz_flag['if'] and X['type'] == 'if':
                    for c in X['in']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([self.graph[c]['pos'][0], 
                                            self.graph[c]['pos'][1]]), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         width=2, 
                                         dash=self.dash_size, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
                    for c in X['out']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         np.array([self.graph[c]['pos'][0], 
                                            self.graph[c]['pos'][1]]), 
                                         width=2, 
                                         dash=self.dash_size,
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
                # Draw all plasticity connections.
                if self.conn_type_viz_flag['plast'] and X['type'] == 'plast':
                    for c in X['in']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([self.graph[c]['pos'][0], 
                                                   self.graph[c]['pos'][1]]), 
                                         np.array([X['pos'][0], 
                                                   X['pos'][1]]), 
                                         width=2, 
                                         dash=self.dash_size, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
                    for c in X['out']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         np.array([self.graph[c]['pos'][0], 
                                                   self.graph[c]['pos'][1]]), 
                                         width=2, 
                                         dash=self.dash_size, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
            self.viz_prof_dur["draw connections"][viz_prof_frame] \
                = time.time() - self.viz_prof_start["draw connections"]
            self.viz_prof_start["draw items"] = time.time()
            for x, X in self.graph.items():
                # Dependent on state set colors of circles.
                if X['state'] not in [0, 6]:
                    X['col'] = self.graph_col[X['type']]
                    X['COL'] = (255, 255, 255)
                else:
                    self.all_up = False
                    X['col'] = self.graph_col_idle[X['type']]
                    X['COL'] = (127, 127, 127)

                # Check type visibility.
                if self.graph_type_viz_flag[X['type']]:
                    # If selected, change border width.
                    if x in self.items_selected:
                        border = 6
                    else:
                        border = 2
                    # If edited, change border color.
                    if x in self.edited_items:
                        border_col = DEFAULT_COLORS['yellow']
                        border = 4
                    else:
                        border_col = DEFAULT_COLORS['light']
                    # Blit item circle filled.
                    self.graph[x]['rect'] = plot_circle(self.screen, 
                                                        int(X['pos'][0]), 
                                                        int(X['pos'][1]), 
                                                        X['rad'] + 2, 
                                                        self.cc(border_col), 
                                                        self.cc(X['col']), 
                                                        border)
                    # Check for pause state.
                    if self.graph[x]['pause'] == 1:
                        pg.draw.line(self.screen, 
                                     self.cc((127,127,127)), 
                                     [X['pos'][0] - 16, X['pos'][1] + 16], 
                                     [X['pos'][0] + 16, X['pos'][1] - 16], 
                                     8)
                    elif self.graph[x]['pause'] == 2:
                        pg.draw.line(self.screen, 
                                     self.cc((0,0,196)), 
                                     [X['pos'][0] - 16, X['pos'][1] + 16], 
                                     [X['pos'][0] + 16, X['pos'][1] - 16], 
                                     8)
            self.viz_prof_dur["draw items"][viz_prof_frame] \
                = time.time() - self.viz_prof_start["draw items"]
            # =================================================================


            # =================================================================
            # Draw selection window.
            # =================================================================
            if LMB_drag_type == '__seldrag__':
                pg.draw.rect(self.screen, 
                             self.cc(DEFAULT_COLORS['light']), 
                             [int(min(self.POS[0], LMB_hold_origin[0])),
                              int(min(self.POS[1], LMB_hold_origin[1])), 
                              int(abs(self.POS[0] - LMB_hold_origin[0])),
                              int(abs(self.POS[1] - LMB_hold_origin[1]))],
                             2)
            # =================================================================


            # =================================================================
            # Profiler and/or names at items.
            # =================================================================
            # Draw profiler information if requested.
            if self.show_profiler in [1, 2]:
                for x in self.shm.proc_id:
                    if self.show_profiler == 2 and x not in self.items_selected:
                        continue
                    X = self.graph[x]
                    prof = [copy.copy(self.IPC_PROC['profiler'][self.shm.proc_id[x][0]][0]),
                            copy.copy(self.IPC_PROC['profiler'][self.shm.proc_id[x][0]][1])]
                    self.screen.blit(self.fonts['small'].render('R:', 
                                                                1, 
                                                                self.cc(self.vparam['text_color'])), 
                                     (X['pos'][0] + X['rad'] + 5, X['pos'][1] + X['rad']))
                    self.screen.blit(self.fonts['small'].render('W:', 
                                                                1, 
                                                                self.cc(self.vparam['text_color'])), 
                                     (X['pos'][0] + X['rad'] + 5, X['pos'][1] + X['rad'] + 20))
                    self.screen.blit(self.fonts['small'].render(str(int(1000*prof[0])), 
                                                                1, 
                                                                self.cc(self.vparam['number_color'])), 
                                     (X['pos'][0] + X['rad'] + 32, X['pos'][1] + X['rad']))
                    self.screen.blit(self.fonts['small'].render(str(int(1000*prof[1])), 
                                                                1, 
                                                                self.cc(self.vparam['number_color'])), 
                                     (X['pos'][0] + X['rad'] + 32, X['pos'][1] + X['rad'] + 20))
            # Draw names if requested.
            if self.show_name in [1, 2]:
                for x,X in self.graph.items():
                    if self.show_name == 2 and x not in self.items_selected:
                        continue
                    self.screen.blit(self.fonts['small'].render(x, 
                                                                1, 
                                                                self.cc(self.vparam['text_color'])), 
                                     (X['pos'][0] + X['rad'] + 5, X['pos'][1] - X['rad'] - 20))
            # =================================================================


            # =================================================================
            # Draw meta-variables and their overview.
            # =================================================================
            # Draw meta variable overview.
            X0 = self.vparam['screen_width'] - 200
            for mv,MV in enumerate(self.meta_variables):
                MV.rects['name'] \
                    = self.screen.blit(self.fonts['small'].render(MV.name, 
                                                                  1, 
                                                                  self.cc(MV.col)), 
                                       (X0, 60 + mv * 48))
                MV.rects['close'] \
                    = self.screen.blit(self.button_sprite['small red exit'], 
                                       (X0, 60 + mv * 48 + 16))
                if MV.itemized == 0:
                    MV.rects['itemize'] \
                        = self.screen.blit(self.button_sprite['small item enable'], 
                                           (X0 + 2 * 16, 60 + mv * 48 + 16))
                elif MV.itemized == 1:
                    MV.rects['itemize'] \
                        = self.screen.blit(self.button_sprite['small item disable'], 
                                           (X0 + 2 * 16, 60 + mv * 48 + 16))
                elif MV.itemized == 2:
                    MV.rects['itemize'] \
                        = self.screen.blit(self.button_sprite['small curve'], 
                                           (X0 + 2 * 16, 60 + mv * 48 + 16))
                MV.rects['viz'] \
                    = self.screen.blit(self.button_sprite['small color'], 
                                       (X0 + 3 * 16, 60 + mv * 48 + 16))

                # Check for mouse over name and if so outline all children.
                if MV.rects['name'].collidepoint(POS):
                    # Draw pretty print.
                    ppl = MV.pprint()
                    for l,L in enumerate(ppl):
                        self.screen.blit(self.fonts['small'].render(L, 
                                                                    1, 
                                                                    self.cc((255,255,255))), 
                                         (10, 400 + l * 16))
                    # Loop over all selected values.
                    for v in range(len(MV.sv)):
                        C = MV.sv[v][0]
                        X = self.graph[C]
                        pg.draw.circle(self.screen, 
                                       self.cc((255,255,255)), 
                                       (int(X['pos'][0]), int(X['pos'][1])), 
                                       24, 
                                       4)
                    # Set over info flag to not blit minimap.
                    self.over_info = True
            # Draw all meta variables.
            for mv,MV in enumerate(self.meta_variables):
                # Try specialized plotting function.
                if MV.itemized > 0:
                    MV.plot(self.shm, 
                            self.screen, 
                            self.graph, 
                            self.ccol)
            # =================================================================


            # =================================================================
            # Subwindows.
            # =================================================================
            self.viz_prof_start["draw subwindows"] = time.time()
            # Draw all subwindows for items.
            for SW in self.subwin_blit_queue:
                s = SW['source']
                # Check if source is shared.
                is_shared = False
                if s[0] in self.net['synapse_pools']:
                    if 'share params' in self.net['synapse_pools'][s[0]] and len(s) == 3:
                        if s[2] in self.net['synapse_pools'][s[0]]['share params']:
                            is_shared = True
                if is_shared:
                    src_item = self.net['synapse_pools'][s[0]]['share params'][s[2]][0]
                    src_param = self.net['synapse_pools'][s[0]]['share params'][s[2]][1]
                    src_dat = np.copy(self.shm.dat[src_item]['parameter'][src_param])
                else:
                    if len(s) == 2:
                        src_dat = np.copy(self.shm.dat[s[0]][s[1]])
                    elif len(s) == 3:
                        src_dat = np.copy(self.shm.dat[s[0]][s[1]][s[2]])
                    elif len(s) == 4:
                        src_dat = np.copy(self.shm.dat[s[0]][s[1]][s[2]][s[3]])
                    elif len(s) == 5:
                        src_dat = np.copy(self.shm.dat[s[0]][s[1]][s[2]][s[3]][s[4]])
                    elif len(s) == 6:
                        src_dat = np.copy(self.shm.dat[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]])

                # Apply magic function.
                if SW['magic'] == '':
                    if SW['patterns']:
                        dat = eval('src_dat' + SW['patterns'][0])
                    else:
                        dat = src_dat
                else:
                    if SW['magic'].find('#0') != -1:
                        m = SW['magic'].replace('#0', 'src_dat' + SW['patterns'][0])
                    else:
                        m = SW['magic'].replace('#', 'src_dat' + SW['patterns'][0])
                    dat = eval(m)

                dat = dat.astype(np.double)
                if SW['type'] == 'meta_var':
                    self.blit_subwindow(SW, dat)
                else:
                    self.blit_subwindow(SW, dat, parent_pos=self.graph[s[0]]['pos'])


                    # Draw only if on screen.
#                    if SW['pos'][0] + X['pos'][0] < self.vparam['screen_width'] \
#                            and SW['pos'][1] + X['pos'][1] < self.vparam['screen_height'] \
#                            and SW['pos'][0] + X['pos'][0] + SW['size'][0] > 0 \
#                            and SW['pos'][1] + X['pos'][1] + SW['size'][1] > 0:


                            # Check for shared variable.
#                            if X['type'] == 'sp':
#                                if 'share params' in self.net['synapse_pools'][x]:
#                                    if sw_ar in self.net['synapse_pools'][x]['share params']:
#                                        source_item = self.net['synapse_pools'][x]['share params'][sw_ar][0]
#                                        source_val = self.net['synapse_pools'][x]['share params'][sw_ar][1]
#                            if SW['tmem'] != -1:


            self.viz_prof_dur["draw subwindows"][viz_prof_frame] \
                = time.time() - self.viz_prof_start["draw subwindows"]
            # =================================================================



            # Check if mouse over item.
            if self.mouse_over[0] == "item":
                x = self.mouse_over[1]
                X = self.graph[x]
                # Show name over item if requested.
                if self.show_name == 0:
                    self.screen.blit(self.fonts['small'].render(x, 
                                                                1, 
                                                                self.cc(self.vparam['text_color'])), 
                                     (X['pos'][0] + X['rad'] + 5, X['pos'][1] - X['rad'] - 20))
                    if X['type'] == 'np':
                        self.screen.blit(self.fonts['small'].render(str(self.net['neuron_pools'][x]['shape']), 
                                                                    1, 
                                                                    self.cc(self.vparam['text_color'])), 
                                         (X['pos'][0] + X['rad'] + 5, X['pos'][1] - X['rad']))
                # Print network dict entry for item.
                if X['type'] in ['np', 'sp', 'plast', 'if']:
                    cntr = 0
                    self.screen.blit(self.fonts['large'].render(x, 
                                                                1, 
                                                                self.cc(self.vparam['text_color'])), 
                                     (20, self.vparam['screen_height'] - 310))
                    for i,I in self.net[self.graph_I[self.graph_i.index(X['type'])]][x].items():
                        cntr += 1
                        # For list of lists we need separate print.
                        if is_list_of_lists(I):
                            self.screen.blit(self.fonts['small'].render(i, 
                                                                        1, 
                                                                        self.cc(self.vparam['text_color'])), 
                                             (100, self.vparam['screen_height'] - 300 + cntr * 16))
                            for ol in range(len(I)):
                                cntr += 1
                                self.screen.blit(self.fonts['small'].render(str(I[ol]), 
                                                                            1, 
                                                                            self.cc(self.vparam['text_color'])), 
                                                 (200, self.vparam['screen_height'] - 300 + cntr * 16))
                        else:
                            self.screen.blit(self.fonts['small'].render(i + ': ' + str(I), 
                                                                        1, 
                                                                        self.cc(self.vparam['text_color'])), 
                                             (100, self.vparam['screen_height'] - 300 + cntr * 16))
                # Print connection for if / plast if over.
                if not self.conn_type_viz_flag['if'] and X['type'] == 'if':
                    for c in X['in']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([self.graph[c]['pos'][0], self.graph[c]['pos'][1]]), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         width=2, 
                                         dash=12, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
                    for c in X['out']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         np.array([self.graph[c]['pos'][0], self.graph[c]['pos'][1]]), 
                                         width=2, 
                                         dash=12, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
                if not self.conn_type_viz_flag['plast'] and X['type'] == 'plast':
                    for c in X['in']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([self.graph[c]['pos'][0], self.graph[c]['pos'][1]]), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         width=2, 
                                         dash=12, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
                    for c in X['out']:
                        draw_dashed_line(self.screen, 
                                         self.cc((127,127,127)), 
                                         np.array([X['pos'][0], X['pos'][1]]), 
                                         np.array([self.graph[c]['pos'][0], self.graph[c]['pos'][1]]), 
                                         width=2, 
                                         dash=12, 
                                         offset=(current_frame % 32) / 16.0,
                                         dash_type=1)
            

            # =================================================================
            # Monitors: Blit monitor information.
            # =================================================================
            for m in ['mean_state', 'var_state', 'nan_state']:
                if self.settings['monitor']['sets'][m[0] + ' state 0/1']['value'] != 'off':
                    if m == 'mean_state':
                        rad = 30
                        state_col = (0, 255, 255)
                    elif m == 'var_state':
                        rad = 40
                        state_col = (255, 0, 255)
                    elif m == 'nan_state':
                        rad = 50
                        state_col = (255, 255, 0)
                    for x,X in self.graph.items():
                        if self.settings['monitor']['sets'][m[0] + ' state 0/1']['value'] in ['soft on', 'soft off']:
                            # blit blended warning cycle with
                            if x in self.monitor_item[m]:
                                factor = np.sqrt(max(abs(self.monitor_item[m][x])) \
                                                 / (abs(self.settings['monitor']['sets'][m]['value']) + 1e-8))
                                col = (0,0,0)
                                if factor > 1:
                                    col = (255, 0, 0)
                                elif factor <= 1 and factor >= 0:
                                    col = (255, 255, 0)
                                pg.draw.rect(self.screen, 
                                             self.cc(col), 
                                             [int(X['pos'][0]) - rad, int(X['pos'][1]) - rad, 10, 10], 
                                             0)
                        # blit warning circle only on hard violation
                        elif self.settings['monitor']['sets'][m[0] + ' state 0/1']['value'] == 'on':
                            for m in self.monitor_bad_items:
                                if x in self.monitor_bad_items[m]:
                                    pg.draw.circle(self.screen, 
                                                   self.cc(state_col), 
                                                   (int(X['pos'][0]), int(X['pos'][1])), 
                                                   rad, 
                                                   10)
            # =================================================================


            # =================================================================
            # If not all up, plot not ready.
            # =================================================================
            if not self.all_up:
                a = np.pi * (self.all_up_current / self.vparam['FPS'])
                x = self.vparam['screen_width'] - 60 + 16 * np.cos(a)
                y = 60 + 16 * np.sin(a)
                plot_circle(self.screen,
                            int(x), 
                            int(y), 
                            16, 
                            fill_col=self.cc((128, 220, 220)))
                x = self.vparam['screen_width'] - 60 + 16 * np.cos(a + np.pi)
                y = 60 + 16 * np.sin(a + np.pi)
                plot_circle(self.screen,
                            int(x), 
                            int(y), 
                            16, 
                            fill_col=self.cc((128, 220, 220)))
                self.all_up_current = (self.all_up_current + 1) % self.vparam['FPS']
            # =================================================================


            # =================================================================
            # Plot debug and viz. profiler information.
            # =================================================================
            if self.debug_flag:
                for i in range(len(self.debug_items)):
                    self.screen.blit(self.fonts['small'].render(str(self.debug_items[i]), 
                                                                1, 
                                                                self.cc((255,255,255))), 
                                     (self.vparam["screen_width"] - 400, 10 + i * 16))
                    try:
                        self.screen.blit(self.fonts['small'].render(str(eval(self.debug_items[i])), 
                                                                    1, 
                                                                    self.cc((127,255,127))), 
                                         (self.vparam["screen_width"] - 200, 10 + i * 16))
                    except:
                        self.screen.blit(self.fonts['small'].render(str(eval("self." + self.debug_items[i])), 
                                                                    1, 
                                                                    self.cc((127,255,127))), 
                                         (self.vparam["screen_width"] - 200, 10 + i * 16))
                Y0 = 10 + 16 * len(self.debug_items) + 40
                cntr = 0
                for p,P in self.viz_prof_dur.items():
                    self.screen.blit(self.fonts['small'].render(p.rjust(18) + ": " + str(int(1e+3 * np.mean(P))), 
                                                                1, 
                                                                self.cc((255,255,255))), 
                                     (self.vparam["screen_width"] - 400, Y0 + cntr * 16))
                    cntr += 1
            # =================================================================


            # =================================================================
            # Draw mini-map if no over-info drawn already.
            # =================================================================
            self.viz_prof_start["mini-map"] = time.time()
            X0 = 10
            Y0 = self.vparam['screen_height'] - self.minimap_size[1] - 10
            if not self.over_info:
                # Black background for minimap.
                pg.draw.rect(self.screen, self.cc((0,0,0)), 
                                          [X0, 
                                           Y0, 
                                           self.minimap_size[0], 
                                           self.minimap_size[1]],
                                          0)
                # Blit minimap border.
                pg.draw.rect(self.screen, self.cc(DEFAULT_COLORS['light']), 
                                          [X0, 
                                           Y0, 
                                           self.minimap_size[0], 
                                           self.minimap_size[1]],
                                          1)
                # Blit small items.
                fx = self.minimap_size[0] / (max_item_pos[0] - min_item_pos[0])
                fy = self.minimap_size[1] / (max_item_pos[1] - min_item_pos[1])
                for x, X in self.graph.items():
                    if X['type'] in ['np', 'if', 'plast', 'sp']:
                        px = fx * (X['pos'][0] - min_item_pos[0])
                        py = fy * (X['pos'][1] - min_item_pos[1])
                        if x in self.items_selected:
                            plot_circle(self.screen,
                                        int(X0 + px), 
                                        int(Y0 + py), 
                                        5, 
                                        fill_col=self.cc(DEFAULT_COLORS['light']))
                        else:
                            plot_circle(self.screen,
                                        int(X0 + px), 
                                        int(Y0 + py), 
                                        3, 
                                        fill_col=self.cc(self.graph_col[X['type']]))
                # Blit cross for current mouse position.
                px = fx * (self.POS[0] - min_item_pos[0])
                py = fy * (self.POS[1] - min_item_pos[1])
                px = min(max(0, px), self.minimap_size[0])
                py = min(max(0, py), self.minimap_size[1])
                pg.draw.line(self.screen, 
                             self.cc(DEFAULT_COLORS['light']),
                             [X0 + px - 10, Y0 + py],
                             [X0 + px + 10, Y0 + py],
                             2)
                pg.draw.line(self.screen, 
                             self.cc(DEFAULT_COLORS['light']),
                             [X0 + px, Y0 + py - 10],
                             [X0 + px, Y0 + py + 10],
                             2)
                # Blit current screen to minimap.
                # Screens left upper corner.
                px = fx * (0 - min_item_pos[0])
                py = fy * (0 - min_item_pos[1])
                # Screens lower right corner
                PX = fx * (self.vparam['screen_width'] - min_item_pos[0])
                PY = fy * (self.vparam['screen_height'] - min_item_pos[1])
                pg.draw.rect(self.screen, self.cc(DEFAULT_COLORS['light']), 
                                          [min(max(X0, X0 + px), X0 + PX - 20), 
                                           min(max(Y0, Y0 + py), Y0 + PY - 20), 
                                           min(self.minimap_size[0], PX - px), 
                                           min(self.minimap_size[1], PY - py)],
                                          1)
            self.viz_prof_dur["mini-map"][viz_prof_frame] \
                = time.time() - self.viz_prof_start["mini-map"]
            # =================================================================


            # End overall frame timer.
            timer_overall[current_viz_frame % 12] -= time.time()

            # Draw black background for better main button visibility.
            pg.draw.rect(self.screen, 
                         self.cc(DEFAULT_COLORS['dark1']), 
                         [0, 0, 200, 150], 0)
            pg.draw.rect(self.screen, 
                         self.cc(DEFAULT_COLORS['dark1']), 
                         [200, 0, self.vparam['screen_width'] - 400, 80], 0)

            # Print overall timing information.
            self.screen.blit(self.fonts['small'].render(' fps nn: ' + str(NNFPS), 
                                                        1, 
                                                        self.cc(self.vparam['text_color'])), 
                             (10, 10))
            # Blit overall timer.
            self.screen.blit(self.fonts['small'].render('viz dur: ' + str(int(1000 * np.mean(-timer_overall))) + ' ms', 
                                                        1, 
                                                        self.cc(self.vparam['text_color'])), 
                             (10, 30))
            # Blit current frame.
            self.screen.blit(self.fonts['small'].render('  frame: ' + str(int(current_frame)), 
                                                        1, 
                                                        self.cc(self.vparam['text_color'])), 
                             (10, 50))
            # Blit delay stuff.
            self.IPC_PROC['delay'].value = self.delay
            self.screen.blit(self.fonts['small'].render('  delay: ' + str(self.delay) + ' ms', 
                                                        1, 
                                                        self.cc(self.vparam['text_color'])), 
                             (10, 70))
            self.delay_rect = pg.draw.circle(self.screen, 
                                             self.cc((255,255,255)), 
                                             (int(20 + float(self.delay) / 10.0), 100), 
                                             8, 
                                             2)
            pg.draw.line(self.screen, self.cc((255,255,255)), [20, 100], [120, 100], 2)

            self.over_info = False


            # =================================================================
            # Draw all widgets.
            # =================================================================
            self.wcollector.draw()
            # =================================================================


            # Draw current preview.
            if self.current_preview_active:
                self.screen.blit(self.current_preview, (0,0))
                self.current_preview_active = False

            # Draw top widget.
            self.wcollector.draw_top()

            # Flip display.
            pg.display.flip()

            # Update core communication queue.
            self.update_core_comm_queue()
            
            # Maintain frames per seconds delay.
            clock.tick(self.vparam['FPS'])
            
            # Evaluate trigger and viz flag.
            if self.IPC_PROC['trigger'].value == 2 or self.IPC_PROC['gui flag'].value != 1:
                self.shutdown()

        # Quit pygame.
        pg.display.quit()






# =============================================================================
# =============================================================================
# =============================================================================

    def shutdown(self):
        """Exiting the visualization.
        """
        self.state_running = False
        # Dump local graphview.
        self.dump_graphview()
        # Shutdown all clients.
        for mv,MV in enumerate(self.meta_variables):
            instruction = "remove_sys_client " + MV.name
            core_string = np.array([ord(c) for c in instruction])
            self.IPC_PROC['instruction len'].value = len(core_string)
            self.IPC_PROC['string'][0:len(core_string)] = core_string[:]
            self.IPC_PROC['instruction'].value = 1
            # Wait until shutdown of client is finished.
            client_shutdown_finished = False
            while not client_shutdown_finished:
                if self.IPC_PROC['instruction'].value != 1:
                    client_shutdown_finished = True
                else:
                    time.sleep(0.005)
        self.IPC_PROC['gui flag'].value = 0

# =============================================================================
# =============================================================================
# =============================================================================

    def blit_subwindow(self, SW, dat, parent_pos=None):
        """Method to draw data in a sub-window.
        """
        border = -1

        if dat is not None:
            if parent_pos is None:
                parent_pos = np.zeros([2,])
            shape = dat.shape

            # Get meta-variable index.
            mv = None
            MV = None
            if SW['type'] == 'meta_var':
                for i,I in enumerate(self.meta_variables):
                    if I.name == SW['source'][0]:
                        mv = i
                        MV = I
                        break

            # Tabula rasa.
            if SW['mode'] in ['maps', 'scatter'] or SW['mode'].startswith('RGB'):
                if SW['mode'] == 'scatter':
                    self.tmp_surf.fill(0)
                else:
                    SW['array'].fill(0)
                border = 2

            # Dependent on sub-window mode compute ("render") main graphics.
            if SW['mode'] == 'maps':
                # Dependent on length show value.
                if len(shape) == 1:
                    if shape[0] == 1:
                        pass
                    else:
                        # [>1], general 1D parameter vector
                        img = dat.flatten()[:]
                        cgraphics_colorcode(img, shape[0], 1, self.CM[SW['colormap']], int(self.ccol))
                        SW['array'] = np.reshape(img, [1, shape[0]]).astype(np.int32)
                elif len(shape) == 2:
                    if shape[0] == 1 and shape[1] == 0:
                        pass
                    else:
                        img = dat[:,:].flatten()[:]
                        cgraphics_colorcode(img, shape[0], shape[1], self.CM[SW['colormap']], int(self.ccol))
                        SW['array'] = img.reshape([shape[0], shape[1]]).astype(np.int)
                elif len(shape) == 3:
                    if shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
                        pass
                    elif shape[0] not in [1, 3] and shape[1] != 1 and shape[2] == 1:
                        # [!1/3, >1, 1], general 1D feature map
                        img = np.swapaxes(dat, 0, 1).flatten()[:]
                        cgraphics_colorcode(img, shape[1], shape[0], self.CM[SW['colormap']], int(self.ccol))
                        SW['array'] = img.reshape([shape[1], shape[0]]).astype(np.int)
                    elif shape[0] != 1 and shape[1] == 1 and shape[2] == 1:
                        # [>1], general 1D parameter vector
                        img = dat[:,0,0].flatten()[:]
                        cgraphics_colorcode(img, shape[0], 1, self.CM[SW['colormap']], int(self.ccol))
                        SW['array'] = np.reshape(img, [1, shape[0]]).astype(np.int32)
                    elif shape[0] == 1 and shape[1] != 1:
                        # [1, >1, *]
                        img = dat[0,:,:].flatten()[:]
                        cgraphics_colorcode(img, shape[1], shape[2], self.CM[SW['colormap']], int(self.ccol))
                        SW['array'] = img.reshape([shape[1], shape[2]]).astype(np.int)
                    elif shape[0] != 1 and shape[1] != 1 and shape[2] != 1:
                        # [>1, >1, >1], general 2D feature map
                        # convert 3D feature map tensor to 2D image.
                        if SW['map'] == -1:
                            rearrange_3D_to_2D(dat, SW['array double'], SW['shape'], SW['sub shape'])
                            img = SW['array double'].flatten()[:]
                            cgraphics_colorcode(img, SW['shape'][0], SW['shape'][1], self.CM[SW['colormap']], int(self.ccol))
                            SW['array'] = img.astype(np.int32).reshape(SW['shape'])
                        elif SW['map'] >= 0 and SW['map'] < dat.shape[0]:
                            # Show single selected feature map.
                            img = dat[int(SW['map']),:,:].flatten()[:]
                            cgraphics_colorcode(img, 
                                                shape[1], 
                                                shape[2], 
                                                self.CM[SW['colormap']], 
                                                int(self.ccol))
                            SW['array'] = img.reshape([shape[1], shape[2]]).astype(np.int)
                elif len(shape) == 4:
                    if shape[0] == 1 and shape[1] == 1 and shape[2] == 1 and shape[3] == 1:
                        pass
                    elif shape[0] == 1 or shape[1] == 1:
                        # Source or target features == 1.
                        if shape[0] == 1:
                            dat = dat[0,:,:,:]
                        else:
                            dat = dat[:,0,:,:]
                        #print("\nSHAPE: " + str(shape) + " SWshape: " + str(SW['shape']) + " SWsubshape: " + str(SW['sub shape']))
                        # [*, *, *], general 2D feature map
                        # convert 3D feature map tensor to 2D image
                        rearrange_3D_to_2D(dat, SW['array double'], SW['shape'], SW['sub shape'])
                        img = SW['array double'].flatten()[:]
                        cgraphics_colorcode(img, 
                                            SW['shape'][0], 
                                            SW['shape'][1], 
                                            self.CM[SW['colormap']], 
                                            int(self.ccol))
                        if shape[0] == 1 and shape[1] == 1:
                            SW['array'] = np.transpose(img.astype(np.int32).reshape(SW['shape']))
                        else:
                            SW['array'] = img.astype(np.int32).reshape(SW['shape'])
                    elif shape[0] != 1 and shape[1] != 1:
                        if shape[2] == 1 and shape[3] == 1:
                            # Fully connected (no-space to no-space).
                            dat = dat[:,:,0,0]
                            img = dat[:,:].flatten()[:]
                            cgraphics_colorcode(img, 
                                                shape[0], 
                                                shape[1], 
                                                self.CM[SW['colormap']], 
                                                int(self.ccol))
                            SW['array'] = img.reshape([shape[0], shape[1]]).astype(np.int)
                        elif shape[2] != 1 and shape[3] != 1:
                            # Full 4D convolution weight matrix.
                            rearrange_4D_to_2D(dat, SW['array double'], SW['shape'])
                            img = SW['array double'].flatten()[:]
                            cgraphics_colorcode(img, 
                                                SW['shape'][0], 
                                                SW['shape'][1], 
                                                self.CM[SW['colormap']], 
                                                int(self.ccol))
                            SW['array'] = img.astype(np.int32).reshape(SW['shape'])
            elif SW['mode'].startswith('RGB'):
                # All RGB image visualizations.
                dim = int(SW['mode'].split()[1])
                # Rescale entries.
                dat = dat - np.min(dat)
                m = np.max(dat)
                if m > 1e-6:
                    dat = dat / m
                if len(shape) == 3 and dim == 0:
                    SW['array'] = (255 * dat[0,:,:]).astype(np.int32) + 256 * (dat[1,:,:] * 255).astype(np.int32) + 256*256 * (dat[2,:,:] * 127).astype(np.int32)
                elif len(shape) == 4:
                    if dim == 0:
                        dat_R = dat[0,:,:,:]
                        dat_G = dat[1,:,:,:]
                        dat_B = dat[2,:,:,:]
                    elif dim == 1:
                        dat_R = dat[:,0,:,:]
                        dat_G = dat[:,1,:,:]
                        dat_B = dat[:,2,:,:]
                    rearrange_3D_to_2D(dat_R, SW['array double'], SW['shape'], SW['sub shape'])
                    SW['array'][:,:] = (SW['array double'][:,:] * 255).astype(np.int32)
                    rearrange_3D_to_2D(dat_G, SW['array double'], SW['shape'], SW['sub shape'])
                    SW['array'][:,:] = SW['array'][:,:] + 256 * (SW['array double'] * 255).astype(np.int32)
                    rearrange_3D_to_2D(dat_B, SW['array double'], SW['shape'], SW['sub shape'])
                    SW['array'][:,:] = SW['array'][:,:] + 256 * 256 * (SW['array double'] * 127).astype(np.int32)
            # Plot a color coded 2D angle (e.g. for optic-flow).
            elif SW['mode'] == 'angle2D':
                self.tmp_surf.fill(0)
                if len(shape) == 3:
                    # Assume dat of shape [2, dim_x, dim_y].
                    img = dat.flatten()[:]
                    cgraphics_vec_to_RGBangle(img, SW['shape'][0], SW['shape'][1], self.CM[SW['colormap']], int(self.ccol))
                    SW['array'] = np.reshape(img[0:SW['shape'][0] * SW['shape'][1]], [SW['shape'][0], SW['shape'][1]]).astype(np.int32)
                elif len(shape) == 4:
                    array_y = np.copy(SW['array double'])
                    rearrange_3D_to_2D(dat[:,0,:,:], SW['array double'], SW['shape'], SW['sub shape'])
                    rearrange_3D_to_2D(dat[:,1,:,:], array_y, SW['shape'], SW['sub shape'])
                    img = np.stack([SW['array double'], array_y]).flatten()[:]
                    cgraphics_vec_to_RGBangle(img, SW['shape'][0], SW['shape'][1], self.CM[SW['colormap']], int(self.ccol))
                    SW['array'] = np.reshape(img[0:SW['shape'][0] * SW['shape'][1]], [SW['shape'][0], SW['shape'][1]]).astype(np.int32)
            # Histogram plot for everything.
            elif SW['mode'] == 'hist':
                self.tmp_surf.fill(0)
                blit_hist(self.tmp_surf, self.fonts['small'], SW['size'], dat.flatten(), self.ccol)
                # Now blit tmp_surf to self.screen.
                self.screen.blit(self.tmp_surf, [parent_pos[0] + SW['pos'][0],
                                                 parent_pos[1] + SW['pos'][1], 
                                                 SW['size'][0], 
                                                 SW['size'][1]],
                                 area=[0, 0, SW['size'][0], SW['size'][1]])

            elif SW['mode'].startswith('plot'):
                dim = int(SW['mode'].split()[1])
                # Get color palette.
                cm = SW.get('colormap', 'magma')
                idx = (self.CM_CC[cm].shape[0] * np.arange(dat.shape[dim])) // dat.shape[dim]
                colors = self.CM_CC[cm][idx,:]
                # Clear temporal surface.
                self.tmp_surf.fill(0)
                # Blit plot histogram to surface.
                blit_pcp(self.tmp_surf, 
                         self.fonts['small'], 
                         SW['size'], 
                         dat, 
                         self.ccol,
                         dim=dim,
                         color=colors)
                # Now blit tmp_surf to self.screen.
                self.screen.blit(self.tmp_surf, [parent_pos[0] + SW['pos'][0],
                                                 parent_pos[1] + SW['pos'][1], 
                                                 SW['size'][0], 
                                                 SW['size'][1]],
                                 area=[0, 0, SW['size'][0], SW['size'][1]])


                # Blit dimensions.
                s = int(np.prod(dat.shape[:]) / dat.shape[dim])
                self.screen.blit(self.fonts['small'].render(str(s) + "  X  " \
                                                            + str(dat.shape[dim]), 1, self.cc(self.vparam['number_color'])), 
                                 (parent_pos[0] + SW['pos'][0], parent_pos[1] + SW['pos'][1] + SW['size'][1] + 2))

                # Blit min / max to plot.
                self.screen.blit(self.fonts['small'].render(num2str(np.max(dat.flatten())), 1, self.cc(self.vparam['number_color'])), (SW['pos'][0] - 80, SW['pos'][1]))
                self.screen.blit(self.fonts['small'].render(num2str(np.min(dat.flatten())), 1, self.cc(self.vparam['number_color'])), (SW['pos'][0] - 80, SW['pos'][1] + SW['size'][1] - 36))



                                # This should be handled via another subwin button for subwins of types 'curves'
                                # ------------------------------------------------------------------------------
#                                img = dat.astype(np.double).flatten()[:]
#                                cgraphics_colorcode(img, SW['shape'][0], SW['shape'][1], self.CM[SW['colormap']], int(self.ccol))
#                                SW['array'] = img.reshape([SW['shape'][0], SW['shape'][1]]).astype(np.int)
                                # Blit array to buffer.
#                                pg.surfarray.blit_array(SW['buffer'],
#                                                        SW['array'])
                                # Scale surface.
#                                scaled_surf = pg.transform.scale(SW['buffer'], 
#                                                                 [int(SW['size'][0]), 
#                                                                  int(SW['size'][1])])
                                # Finally blit to screen.
 #                               SW['rect'] = self.screen.blit(scaled_surf,
 #                                                             (SW['pos'][0],
 #                                                              SW['pos'][1]))
                                # Draw separation lines.
#                                for v in range(SW['shape'][1]):
#                                    pg.draw.line(self.screen, 
#                                                 self.cc(self.vparam['text_color']), 
#                                                 (SW['pos'][0], SW['pos'][1] + int(SW['size'][1] * float(v) / SW['shape'][1])),
#                                                 (SW['pos'][0] + SW['size'][0], SW['pos'][1] + int(SW['size'][1] * float(v) / SW['shape'][1])), 1)
#                                legend = 'left'
#                                SW['cm flag'] = True
#                                border = True

            elif SW['mode'] == "scatter":
                cm = SW.get('colormap', 'hsv')
                idx = (self.CM_CC[cm].shape[0] * np.arange(dat.shape[1])) // dat.shape[1]
                colors = self.CM_CC[cm][idx,:]
                # Determine min/max values.
                if SW['min'] is None or np.prod(SW['area']) == 0:
                    SW['min'] = np.min(dat[:,:,:], axis=(1, 2))
                    SW['max'] = np.max(dat[:,:,:], axis=(1, 2))
                    SW['area'] = SW['max'] - SW['min']
                for i in range(3):
                    dat[i,:,:] -= SW['min'][i]
                    if SW['area'][i] > 1e-6:
                        dat[i,:,:] /= SW['area'][i]
                dat = np.clip(dat, 0.0, 1.0)
                # Blit all values to tmp surface.
                for t in range(dat.shape[2]):
                    for v in range(dat.shape[1]):
                        col = colors[v] * float(t) / dat.shape[2]
                        pg.draw.circle(self.tmp_surf, 
                                       self.cc(col), 
                                       (int(dat[0,v,t] * SW['size'][0]), 
                                        int((1.0 - dat[1,v,t]) * SW['size'][1])), 
                                       8, 0)
                # Now blit tmp_surf to screen.
                SW['rect'] = self.screen.blit(self.tmp_surf, 
                                              [SW['pos'][0],
                                               SW['pos'][1], 
                                               SW['size'][0], 
                                               SW['size'][1]],
                                 area=[0, 0, SW['size'][0], SW['size'][1]])

            # Now blit scaled SW['array'] to screen.
            if SW['mode'] in ['maps', 'angle2D'] or SW['mode'].startswith('RGB'):
                try:
                    pg.surfarray.blit_array(SW['buffer'],
                                            SW['array'])
                except:
                    print("\nERROR: Viz failed to copy array to buffer: " \
                          + str(SW['array'].shape) + " -> " + str(SW['buffer'].get_size()))
                    pg.surfarray.blit_array(SW['buffer'],
                                            SW['array'])
                scaled_surf = pg.transform.scale(SW['buffer'], 
                                                 [int(SW['size'][0]), 
                                                  int(SW['size'][1])])
                SW['rect'] = self.screen.blit(scaled_surf,
                                              (parent_pos[0] + SW['pos'][0],
                                               parent_pos[1] + SW['pos'][1]))

            # Check if mouse is over SW and if so compute some information.
            self.subwin_info = []
            if SW['rect'].collidepoint(self.POS) and SW['mode'] == 'maps':
                # Dependent on data/state shape compute informations.
                if len(shape) == 1:
                    if shape[0] == 1:
                        # Single value [1,].
                        self.subwin_info.append(['scalar index   ', str([0])])
                        self.subwin_info.append(['value          ', '{:2.2e}'.format(dat[0])])
                    else:
                        # Vector [*,].
                        fy = shape[0] / float(SW['rect'].height)
                        tl = np.array(SW['rect'].topleft)
                        cy = int(float(self.POS[1] - tl[1]) * fy)
                        self.subwin_info.append(['vector index   ', str([cy])])
                        self.subwin_info.append(['value          ', '{:2.2e}'.format(dat[cy])])
                elif len(shape) == 3:
                    if shape[2] == 1:
                        if shape[0] != 1 or shape[1] != 1:
                            # 1D feature map [*, * , 1].
                            fx = shape[1] / float(SW['rect'].width)
                            fy = shape[0] / float(SW['rect'].height)
                            tl = np.array(SW['rect'].topleft)
                            cx = int(float(self.POS[0] - tl[0]) * fx)
                            cy = int(float(self.POS[1] - tl[1]) * fy)
                            self.subwin_info.append(['index   ', str([cy, cx, 0])])
                            self.subwin_info.append(['value   ', '{:2.2e}'.format(dat[cy, cx, 0])])
                        else:
                            # Single value [1, 1, 1].
                            self.subwin_info.append(['scalar index   ', str([0, 0, 0])])
                            self.subwin_info.append(['value          ', '{:2.2e}'.format(dat[0, 0, 0])])
                    else:
                        if shape[0] == 1:
                            # Grayscale map [1, *, *].
                            fx = shape[1] / float(SW['rect'].width)
                            fy = shape[2] / float(SW['rect'].height)
                            tl = np.array(SW['rect'].topleft)
                            cx = int(float(self.POS[0] - tl[0]) * fx)
                            cy = int(float(self.POS[1] - tl[1]) * fy)
                            self.subwin_info.append(['index           ', str([0, cx, cy])])
                            self.subwin_info.append(['value           ', '{:2.2e}'.format(dat[0, cx, cy])])
                            self.subwin_info.append(['mean sample act ', '{:2.2e}'.format(np.mean(dat[0,:,:]))])
                            self.subwin_info.append(['var sample act  ', '{:2.2e}'.format(np.var(dat[0,:,:]))])
                        elif shape[0] == 3:
                            # 1 RGB map [3, *, *].
                            fx = shape[1] / float(SW['rect'].width)
                            fy = shape[2] / float(SW['rect'].height)
                            tl = np.array(SW['rect'].topleft)
                            cx = int(float(self.POS[0] - tl[0]) * fx)
                            cy = int(float(self.POS[1] - tl[1]) * fy)
                            self.subwin_info.append(['index           ', str([0, cx, cy])])
                            self.subwin_info.append(['rgb             ', '({:2.2e}, {:2.2e}, {:2.2e})'.format(dat[0, cx, cy], dat[1, cx, cy], dat[2, cx, cy])])
                        else:
                            # General 3D feature map [*, *, *].
                            if SW['map'] == -1:
                                # Get converted neuron coordinates.
                                fx = (shape[1] * SW['sub shape'][0]) / float(SW['rect'].width)
                                fy = (shape[2] * SW['sub shape'][1]) / float(SW['rect'].height)
                                tl = np.array(SW['rect'].topleft)
                                cx = int(float(self.POS[0] - tl[0]) * fx)
                                cy = int(float(self.POS[1] - tl[1]) * fy)
                                # Determine feature map.
                                feat = cx // shape[1] + SW['sub shape'][0] * (cy // shape[2])
                                # Determine local neuron spatial coordinates.
                                nx = cx % shape[1]
                                ny = cy % shape[2]
                                if feat < dat.shape[0]:
                                    self.subwin_info.append(['index         ', str([feat, nx, ny])])
                                    self.subwin_info.append(['value         ', '{:2.2e}'.format(dat[int(SW['map']), nx, ny])])
                                    self.subwin_info.append(['mean feat.map ', '{:2.2e}'.format(np.mean(dat[int(SW['map']),:,:]))])
                                    self.subwin_info.append(['var feat.map  ', '{:2.2e}'.format(np.var(dat[int(SW['map']),:,:]))])
                                    self.subwin_info.append(['mean feat.col ', '{:2.2e}'.format(np.mean(dat[:,nx,ny]))])
                                    self.subwin_info.append(['var feat.col  ', '{:2.2e}'.format(np.var(dat[:,nx,ny]))])
                                else:
                                    # Mouse is out of scope.
                                    self.subwin_info.append(['index         ', 'invalid map'])
                                    self.subwin_info.append(['activation    ', 'NaN'])
                            elif SW['map'] >= 0 and SW['map'] < dat.shape[0]:
                                # Same as for all other 2d things.
                                fx = shape[1] / float(SW['rect'].width)
                                fy = shape[2] / float(SW['rect'].height)
                                tl = np.array(SW['rect'].topleft)
                                cx = int(float(self.POS[0] - tl[0]) * fx)
                                cy = int(float(self.POS[1] - tl[1]) * fy)
                                self.subwin_info.append(['index         ', str([int(SW['map']), cx, cy])])
                                self.subwin_info.append(['value         ', '{:2.2e}'.format(dat[int(SW['map']), cx, cy])])
                                self.subwin_info.append(['mean feat.map ', '{:2.2e}'.format(np.mean(dat[int(SW['map']),:,:]))])
                                self.subwin_info.append(['var feat.map  ', '{:2.2e}'.format(np.var(dat[int(SW['map']),:,:]))])
                                self.subwin_info.append(['mean feat.col ', '{:2.2e}'.format(np.mean(dat[:,cx,cy]))])
                                self.subwin_info.append(['var feat.col  ', '{:2.2e}'.format(np.var(dat[:,cx,cy]))])

            # Check if mouse over controll buttons and change subwin_info.
            if SW['rect magic'].collidepoint(self.POS) \
                or SW['rect patterns'].collidepoint(self.POS):
                self.subwin_info = []
                self.subwin_info.append(['magic function: '.ljust(16), SW['magic']])
                self.subwin_info.append(['access pattern: '.ljust(16), SW['patterns'][0]])
                self.subwin_info.append(['source shape: '.ljust(16), str(SW['src_shape'])])
                self.subwin_info.append(['data shape: '.ljust(16), str(SW['dat_shape'])])

            # Draw subwindow grid / tiles.
            if SW['tileable'] and SW['tiles']:
                if SW['mode'] in ['maps', 'angle2D'] or SW['mode'].startswith('RGB'):
                    # Only if all sub-divisions are shown.
                    if SW['map'] == -1:
                        # Draw horizontal lines.
                        for t in range(SW['sub shape'][0]):
                            tx = int(parent_pos[0] + SW['pos'][0] \
                                     + t * SW['rect'].width / SW['sub shape'][0])
                            pg.draw.line(self.screen, 
                                         self.cc((196,196,196)), 
                                         (tx, parent_pos[1] + SW['pos'][1]), 
                                         (tx, parent_pos[1] + SW['pos'][1] + SW['size'][1]), 
                                         1)
                        # Draw vertical lines.
                        for t in range(SW['sub shape'][1]):
                            ty = int(parent_pos[1] + SW['pos'][1] \
                                     + t * SW['rect'].height / SW['sub shape'][1])
                            pg.draw.line(self.screen, 
                                         self.cc((196,196,196)), 
                                         (parent_pos[0] + SW['pos'][0], ty), 
                                         (parent_pos[0] + SW['pos'][0] + SW['size'][0], ty), 
                                         1)

            # Blit border.
            if border > 0:
                pg.draw.rect(self.screen, 
                     self.cc((127,127,127)), 
                     [parent_pos[0] + SW['pos'][0], parent_pos[1] + SW['pos'][1], \
                      SW['size'][0], SW['size'][1]], 2)

            # Show legends (options: colored, left/right, relative/absolute)
            if SW['legend right']:
                if SW['legend right relative']:
                    for l,L in enumerate(SW['legend right']):
                        if SW['legend right color']:
                            col = colors[l]
                        else:
                            col = self.vparam['text_color']
                        self.screen.blit(self.fonts['small'].render(L, 1, self.cc(col)), 
                                         (SW['pos'][0] + SW['size'][0] + 4, \
                                          SW['pos'][1] + SW['size'][1] * float(l) / len(SW['legend right'])))
                else:
                    for l,L in enumerate(SW['legend right']):
                        if SW['legend right color']:
                            col = colors[l]
                        else:
                            col = self.vparam['text_color']
                        self.screen.blit(self.fonts['small'].render(L, 1, self.cc(col)), 
                                         (SW['pos'][0] + SW['size'][0] + 4, \
                                          SW['pos'][1] + l * self.vparam['font small']))
            if SW['legend left']:
                max_l = 0
                for l,L in enumerate(SW['legend left']):
                    max_l = max(max_l, len(L))
                if SW['legend left relative']:
                    for l,L in enumerate(SW['legend left']):
                        if SW['legend left color']:
                            col = colors[l]
                        else:
                            col = self.vparam['text_color']
                        self.screen.blit(self.fonts['small'].render(L.rjust(max_l), 1, self.cc(col)), 
                                         (int(SW['pos'][0] - max_l * self.font_size['small'] / 1.5), 
                                          int(SW['pos'][1] + SW['size'][1] * float(l) / len(SW['legend left']))))
                else:
                    for l,L in enumerate(SW['legend left']):
                        if SW['legend left color']:
                            col = colors[l]
                        else:
                            col = self.vparam['text_color']
                        self.screen.blit(self.fonts['small'].render(L.rjust(max_l), 1, self.cc(col)), 
                                         (int(SW['pos'][0] - max_l * self.font_size['small'] / 1.5), 
                                          int(SW['pos'][1] + l * self.vparam['font small'])))

            # Draw sub-window controls on top of sw.
            SW['rect drag'] = self.screen.blit(self.button_sprite['small grey drag'], 
                                               (parent_pos[0] + SW['pos'][0], 
                                                parent_pos[1] + SW['pos'][1] - self.sw_size / 2))
            SW['rect scale'] = self.screen.blit(self.button_sprite['small grey scale'], 
                                                (parent_pos[0] + SW['pos'][0] + 1 * self.sw_size, 
                                                 parent_pos[1] + SW['pos'][1] - self.sw_size / 2))
            SW['rect close'] = self.screen.blit(self.button_sprite['small red exit'], 
                                                (parent_pos[0] + SW['pos'][0] + 2 * self.sw_size, 
                                                 parent_pos[1] + SW['pos'][1] - self.sw_size / 2))
            SW['rect mode'] = self.screen.blit(self.button_sprite['small hist'], 
                                               (parent_pos[0] + SW['pos'][0] + 3 * self.sw_size, 
                                                parent_pos[1] + SW['pos'][1] - self.sw_size / 2))

            # Right side controls.
            if SW['tileable']:
                if SW['tiles']:
                    SW['rect tiles'] \
                        = self.screen.blit(self.button_sprite['small tiles off'], 
                                           (parent_pos[0] + SW['pos'][0] + 4 * self.sw_size, 
                                            parent_pos[1] + SW['pos'][1] - self.sw_size / 2))
                else:
                    SW['rect tiles'] \
                        = self.screen.blit(self.button_sprite['small tiles on'], 
                                           (parent_pos[0] + SW['pos'][0] + 4 * self.sw_size, 
                                            parent_pos[1] + SW['pos'][1] - self.sw_size / 2))
            SW['rect magic'] = self.screen.blit(self.button_sprite['small star'], 
                                                (parent_pos[0] + SW['pos'][0] + 5 * self.sw_size, 
                                                 parent_pos[1] + SW['pos'][1] - self.sw_size / 2))
            SW['rect patterns'] = self.screen.blit(self.button_sprite['small bracket'], 
                                                   (parent_pos[0] + SW['pos'][0] + 6 * self.sw_size, 
                                                    parent_pos[1] + SW['pos'][1] - self.sw_size / 2))

            # Draw tmem plus / minus below sw.
#            if SW['mode'] in ['maps', 'hist']:
#                SW['rect tmem-'] = self.screen.blit(self.button_sprite['small right'], (parent_pos[0] + SW['pos'][0] + SW['size'][0] - 16, parent_pos[1] + SW['pos'][1] + SW['size'][1]))
#                SW['rect tmem+'] = self.screen.blit(self.button_sprite['small left'], (parent_pos[0] + SW['pos'][0], parent_pos[1] + SW['pos'][1] + SW['size'][1]))
#                rel_frame = 0
#                if SW['tmem'] >= 0:
#                    factor = 1
#                    for f in range(SW['tmem'] + 1):
#                        factor *= self.param['core']['temporal_memory'][f]
#                        rel_frame += factor
#                self.screen.blit(self.fonts['small'].render(str(-int(rel_frame)), 1, self.cc(self.vparam['number_color'])), (parent_pos[0] + SW['pos'][0] + SW['size'][0] // 2, parent_pos[1] + SW['pos'][1] + SW['size'][1]))
#                self.screen.blit(self.fonts['small'].render(str(SW['tmem'] + 1), 1, self.cc(self.vparam['number_color'])), (parent_pos[0] + SW['pos'][0] + 16, parent_pos[1] + SW['pos'][1] + SW['size'][1]))
            
            # Blit neuron info if any.
            if self.subwin_info:
                self.screen.blit(self.fonts['small'].render('sub-win info', 
                                                            1, 
                                                            self.cc(self.vparam['text_color'])), 
                                 (20, 400))
                for i in range(len(self.subwin_info)):
                    self.screen.blit(self.fonts['small'].render(self.subwin_info[i][0], 1, self.cc(self.vparam['text_color'])), 
                                     (40, 420 + i * 16))
                    self.screen.blit(self.fonts['small'].render(self.subwin_info[i][1], 1, self.cc(self.vparam['number_color'])), 
                                     (200, 420 + i * 16))
                self.subwin_info = []
                self.over_info = True
                
            # Draw colormap on right side of sw.
            if SW['cm flag']:
                SW['cm array'] = self.CM_RAW[SW['colormap']]
                # Blit array to buffer.
                pg.surfarray.blit_array(SW['cm buffer'],
                                        SW['cm array'])
                # Scale surface.
                scaled_surf = pg.transform.scale(SW['cm buffer'], [12, int(SW['size'][1])])
                # Finally blit to screen.
                SW['cm rect'] = self.screen.blit(scaled_surf,
                                                        (parent_pos[0] + SW['pos'][0] + SW['size'][0],
                                                         parent_pos[1] + SW['pos'][1]))
                # Blit min / max to colormap.
                self.screen.blit(self.fonts['small'].render(num2str(np.max(dat.flatten())), 1, self.cc(self.vparam['number_color'])), (parent_pos[0] + SW['pos'][0] + SW['size'][0] + 16, parent_pos[1] + SW['pos'][1]))
                self.screen.blit(self.fonts['small'].render(num2str(np.min(dat.flatten())), 1, self.cc(self.vparam['number_color'])), (parent_pos[0] + SW['pos'][0] + SW['size'][0] + 16, parent_pos[1] + SW['pos'][1] + SW['size'][1] - 16))
        
            if SW['axis x']:
                offset = int(len(SW['axis x']) * self.font_size['small'] / 1.5) // 2
                self.screen.blit(self.fonts['small'].render(SW['axis x'], 1, self.cc(self.vparam['text_color'])), 
                                 (parent_pos[0] + SW['pos'][0] + SW['size'][0] // 2 - offset, 
                                  parent_pos[1] + SW['pos'][1] + SW['size'][1]))
            if SW['axis y']:
                offset = int(len(SW['axis y']) * self.font_size['small'] / 1.5)
                self.screen.blit(self.fonts['small'].render(SW['axis y'], 1, self.cc(self.vparam['text_color'])), 
                                 (parent_pos[0] + SW['pos'][0] - 4 - offset, 
                                  parent_pos[1] + SW['pos'][1] + SW['size'][1] // 2))

            # Finally blit name of sub-win target.
            self.screen.blit(self.fonts['small'].render(str(' '.join(SW['source'])), 1, self.cc(self.vparam['text_color'])), 
                             (parent_pos[0] + SW['pos'][0], 
                              parent_pos[1] + SW['pos'][1] - 42))
            self.screen.blit(self.fonts['small'].render(str(SW['dat_shape']), 1, self.cc(self.vparam['text_color'])), 
                             (parent_pos[0] + SW['pos'][0], 
                              parent_pos[1] + SW['pos'][1] - 26))
