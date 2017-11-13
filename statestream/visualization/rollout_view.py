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
                                     compute_distance_matrix, \
                                     get_the_input, \
                                     clever_sort
from statestream.meta.synapse_pool import sp_get_dict
from statestream.meta.network import S2L

from statestream.ccpp.cgraphics import cgraphics_colorcode

from statestream.utils.rearrange import rearrange_3D_to_2D
from statestream.utils.rearrange import rearrange_4D_to_2D

from statestream.utils.helper import is_scalar_shape
from statestream.utils.helper import is_int_dtype
from statestream.utils.helper import is_float_dtype
from statestream.utils.helper import is_list_of_lists
from statestream.utils.shared_memory import SharedMemory

from statestream.visualization.graphics import num2str, brighter, darker
from statestream.visualization.graphics import blit_plot, \
                                               blit_hist, \
                                               plot_circle

from statestream.visualization.base_widgets import Collector
from statestream.visualization.widgets.list_selection_window import ListSelectionWindow
from statestream.visualization.widgets.list_mselection_window import ListMSelWindow
from statestream.visualization.widgets.parameter_specification_window import ParamSpecWindow

from statestream.utils.defaults import DEFAULT_COLORS, \
                                       DEFAULT_VIZ_PARAMETER




class rollout_view(object):
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
                self.vparam = copy.deepcopy(DEFAULT_VIZ_PARAMETER)
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

        # Set internal state.
        self.state_running = True
        self.debug_flag = False
        self.all_up = False
        self.all_up_current = 0

        self.item_size = int(1.2 * copy.copy(self.vparam['font small']))
        self.name_font = 'small'
        self.name_size = copy.copy(self.vparam['font ' + self.name_font])
        self.header_height = 100
        self.default_border = 1
        self.rollback_border = 4

        # Flag for color correction (blue).
        self.ccol = self.vparam['color_correction']

        self.text_color = copy.copy(self.vparam['text_color'])
        self.np_color = copy.copy(self.vparam['np_color'])
        self.rollback_valid_color = (120,200,200)
        self.rollback_invalid_color = (40,100,250)

        # Determine maximal number of input sps.
        self.max_input_len = 0
        for n in net['neuron_pools']:
            no_inputs = 0
            for s,S in net['synapse_pools'].items():
                if S['target'] == n:
                    no_inputs += 1
            if no_inputs > self.max_input_len:
                self.max_input_len = no_inputs
        
        # Get rollout.
        self.rollout = copy.copy(self.vparam.get('rollout', 6))
        self.memory = copy.copy(self.vparam.get('memory', 2))
        self.dt = copy.copy(self.vparam.get('dt', 0))
        self.screen_width = copy.copy(self.vparam['screen_width'])
        self.screen_height = copy.copy(self.vparam['screen_height'])
        
        self.current_mem = 0

        # Initially sort all NPs.
        self.dm = compute_distance_matrix(self.net)
        self.the_input = get_the_input(self.net, dist_mat=self.dm)
        sorted_nps = clever_sort(self.net, self.the_input, the_input=self.the_input, dist_mat=self.dm)

        # List of dicts representation of network for drawing.
        self.list_rep = []
        self.dict_rep = {}
        overall_Y = 0
        for n in sorted_nps:
            rep = {}
            rep['name'] = copy.copy(n)
            rep['no_src_sps'] = 0
            rep['rect name'] = pg.Rect(0, 0, 2, 2)
            rep['rect item'] = pg.Rect(0, 0, 2, 2)
            rep['src_sps'] = []
            rep['src_nps'] = []
            rep['tgt_nps'] = []
            rep['mode'] = 'item'
            rep['col'] = []
            rep['border'] = []
            rep['size'] = []
            rep['row_height'] = 4 * self.name_size
            rep['Y'] = copy.copy(overall_Y)
            overall_Y += rep['row_height']
            rep['mem'] = []
            rep['surf'] = []
            # The next is a list over all rollouts of this item.
            rep['rect'] = []
            for s,S in net['synapse_pools'].items():
                # S is source of N.
                if S['target'] == n:
                    rep['no_src_sps'] += 1
                    rep['src_sps'].append(copy.copy(s))
                    rep['src_nps'].append([inp for factor in S['source'] for inp in factor])
                # S is target of N.
                all_srcs = [src for srcs in S['source'] for src in srcs]
                for src in all_srcs:
                    if src == n:
                        if src not in rep['tgt_nps']:
                            rep['tgt_nps'].append(S['target'])
            for m in range(self.memory):
                rep['mem'].append(np.zeros(net['neuron_pools'][n]['shape'], dtype=np.float32))
                rep['surf'].append(None)
            for r in range(self.rollout + 1):
                rep['rect'].append(pg.Rect(0, 0, 2, 2))
                rep['col'].append(copy.copy(self.np_color))
                rep['border'].append(1)
                rep['size'].append(copy.copy(self.item_size))
            self.dict_rep[n] = len(self.list_rep)
            self.list_rep.append(copy.deepcopy(rep))

        # Plasticity representation for fast viz.
        self.plasts = {}
        for p,P in self.net['plasticities'].items():
            self.plasts[p] = {}
            self.plasts[p]['rect'] = pg.Rect([0, 0, 2, 2])
        self.plast_selected = None

        # This is the list of elected items.
        self.items_selected = []
        
        # Begin with empty core communication queue.
        self.core_comm_queue = []
       
        # Get home directory.
        self.home_path = os.path.expanduser('~')
        self.this_path = os.path.abspath(__file__)[0:-16]
        if not os.path.isdir(self.home_path + '/.statestream/rolloutview'):
            # Create dictionary.
            os.makedirs(self.home_path + '/.statestream/rolloutview')
        # Set rollout view save file.
        self.rollout_view_file = self.home_path + '/.statestream/rolloutview/' \
                                   + self.net['name'] + '-rolloutview'

        # Get colormap.
        self.colormap = np.array(matplotlib.pyplot.get_cmap('magma')(np.arange(256) / 255))

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

    # Dump rollout view parametrization to local directory.
    def dump_rollout_view(self, id=0):
        '''Method to dump some visualization settings.
        '''
        # Generate rollout view dictionary.
        rollout_view = {}
        rollout_view['rollout'] = self.rollout
        rollout_view['mem'] = self.memory
        rollout_view['dt'] = self.dt
        filename = self.rollout_view_file + "-{:02d}".format(id)
        # Save rollout view.
        with open(filename, 'w') as outfile:
            dump_yaml(rollout_view, outfile)

# =============================================================================

    def load_rollout_view(self, id=0):
        # Try to load and apply rollout view save file.
        filename = self.rollout_view_file + "-{:02d}".format(id)
        if os.path.isfile(filename):
            with open(filename, 'r') as infile:
                rv = load_yaml(infile)
                self.rollout = copy.copy(rv.get('rollout', 6))
                self.memory = copy.copy(rv.get('mem', 2))
                self.dt = copy.copy(rv.get('dt', 0))

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

    def render_item(self, item_idx, mem):
        """Renders the item once for efficient blitting.
        """
        dat = np.copy(self.list_rep[item_idx]['mem'][mem])
        # Rescale entries.
        dat = dat - np.min(dat)
        m = np.max(dat)
        if m > 1e-6:
            dat = dat / m
        shape = dat.shape
        if shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
            pass
        elif shape[0] not in [1, 3] and shape[1] != 1 and shape[2] == 1:
            # [!1/3, >1, 1], general 1D feature map
            img = np.swapaxes(dat, 0, 1).flatten()[:]
            cgraphics_colorcode(img, shape[1], shape[0], self.colormap, int(self.ccol))
            array = img.reshape([shape[1], shape[0]]).astype(np.int32)
        elif shape[0] != 1 and shape[1] == 1 and shape[2] == 1:
            # [>1], general 1D parameter vector
            img = dat[:,0,0].flatten()[:].astype(np.double)
            cgraphics_colorcode(img, shape[0], 1, self.colormap, int(self.ccol))
            array = np.reshape(img, [1, shape[0]]).astype(np.int32)
        elif shape[0] == 1 and shape[1] != 1:
            # [1, >1, *]
            img = dat[0,:,:].flatten()[:].astype(np.double)
            cgraphics_colorcode(img, shape[1], shape[2], self.colormap, int(self.ccol))
            array = img.reshape([shape[1], shape[2]]).astype(np.int32)
        elif shape[0] != 1 and shape[1] != 1 and shape[2] != 1:
            # [>1, >1, >1], general 2D feature map
            # convert 3D feature map tensor to 2D image.
            sub_shape_0 = int(np.ceil(np.sqrt(shape[0])))
            sub_shape_1 = int(np.ceil(float(shape[0]) / sub_shape_0))
            array_shape = [sub_shape_0 * shape[1],
                           sub_shape_1 * shape[2]]
            array_double = np.zeros(array_shape, dtype=np.double)
            rearrange_3D_to_2D(dat, array_double, array_shape, [sub_shape_0, sub_shape_1])
            img = array_double.flatten()[:]
            cgraphics_colorcode(img, array_shape[0], array_shape[1], self.colormap, int(self.ccol))
            array = img.astype(np.int32).reshape(array_shape)
        elif shape[0] == 3 and shape[1] != 1 and shape[2] != 1:
            # All RGB image visualizations.
            array = (255 * dat[0,:,:]).astype(np.int32) + 256 * (dat[1,:,:] * 255).astype(np.int32) + 256*256 * (dat[2,:,:] * 127).astype(np.int32)

        loc_buffer = pg.Surface([array.shape[0], array.shape[1]])
        loc_buffer.convert()
        pg.surfarray.blit_array(loc_buffer, array)
        self.list_rep[item_idx]['surf'][mem] = pg.transform.scale(loc_buffer, 
                                                                  [self.list_rep[item_idx]['row_height'], 
                                                                   self.list_rep[item_idx]['row_height']])

# =============================================================================

    def fade_color(self, src_col, tgt_col, factor):
        """Fades color between two colors.
        """
        return (factor * src_col[0] + (1.0 - factor) * tgt_col[0],
                factor * src_col[1] + (1.0 - factor) * tgt_col[1],
                factor * src_col[2] + (1.0 - factor) * tgt_col[2])

# =============================================================================

    def change_item_color_recback(self, item, rollout):
        """Change item color recursively backwards.
        """
        if (rollout > 0):
            if len(self.list_rep[self.dict_rep[item]]['src_nps']) == 0 and rollout + self.dt > 0:
                self.list_rep[self.dict_rep[item]]['col'][rollout] = copy.copy(self.rollback_invalid_color)
            for src_sp_srcs in self.list_rep[self.dict_rep[item]]['src_nps']:
                for src_np in src_sp_srcs:
                    self.list_rep[self.dict_rep[src_np]]['col'][rollout - 1] = copy.copy(self.rollback_valid_color)
                    self.list_rep[self.dict_rep[src_np]]['border'][rollout - 1] = copy.copy(self.rollback_border)
                    self.list_rep[self.dict_rep[src_np]]['size'][rollout - 1] \
                        = copy.copy(self.list_rep[self.dict_rep[item]]['row_height'] - 6)
                    self.change_item_color_recback(src_np, rollout - 1)

    def change_item_color_recforward(self, item, rollout):
        """Change item color recursively forward.
        """
        if (rollout < self.rollout):
            for tgt_np in self.list_rep[self.dict_rep[item]]['tgt_nps']:
                self.list_rep[self.dict_rep[tgt_np]]['col'][rollout + 1] = copy.copy(self.rollback_valid_color)
                self.list_rep[self.dict_rep[tgt_np]]['border'][rollout + 1] = copy.copy(self.rollback_border)
                self.list_rep[self.dict_rep[tgt_np]]['size'][rollout + 1] \
                    = copy.copy(self.list_rep[self.dict_rep[item]]['row_height'] - 6)
                self.change_item_color_recforward(tgt_np, rollout + 1)

# =============================================================================

    def cb_button_LMB_click_break(self):
        '''System break button clicked, pause system.
        '''
        if self.IPC_PROC['break'].value == 0:
            self.IPC_PROC['break'].value = 1
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

    def cb_button_LMB_click_memory_plus(self):
        """Increase rollout by one.
        """
        if self.memory < 16:
            self.memory += 1
            self.update_memory()
    def cb_button_LMB_click_memory_minus(self):
        """Decrease rollout by one.
        """
        if self.memory > 0:
            self.memory -= 1
            self.update_memory()
    def update_memory(self):
        """Reset and update memory structure.
        """
        self.current_mem = 0
        for N in self.list_rep:
            N['surf'] = []
            N['mem'] = []
            for m in range(self.memory):
                N['surf'].append(pg.Surface([N['row_height'], N['row_height']]))
                N['mem'].append(np.zeros(self.net['neuron_pools'][N['name']]['shape'], dtype=np.float32))

# =============================================================================

    def cb_button_LMB_click_rollout_plus(self):
        """Increase rollout by one.
        """
        if self.rollout < 16:
            self.rollout += 1
            self.update_rollout()
    def cb_button_LMB_click_rollout_minus(self):
        """Decrease rollout by one.
        """
        if self.rollout > 1:
            self.rollout -= 1
            self.update_rollout()
    def update_rollout(self):
        """Update internal structures for new rollout.
        """
        for n,N in enumerate(self.list_rep):
            N['rect'] = []
            N['col'] = []
            N['border'] = []
            N['size'] = []
            for r in range(self.rollout + 1):
                N['rect'].append(pg.Rect(0, 0, 2, 2))
                N['col'].append(copy.copy(self.np_color))
                N['border'].append(self.default_border)
                N['size'].append(self.item_size)
        self.rollout_offsetX = (self.screen_width \
                                - self.max_name_px - 4 * self.item_size \
                                - (self.item_size * (2 + self.max_input_len))) \
                                // self.rollout

# =============================================================================

    def cb_button_LMB_click_dt_plus(self):
        """Increase dt by one.
        """
        self.dt += 1
    def cb_button_LMB_click_dt_minus(self):
        """Decrease dt by one.
        """
        self.dt -= 1

# =============================================================================



    def run(self, IPC_PROC, dummy):
        '''Main running method for visualization.
        '''
        # Reference to self.IPC_PROC.
        self.IPC_PROC = IPC_PROC
        # Get and set viz pid.
        self.IPC_PROC['rvgui pid'].value = os.getpid()
        # Init pygame.
        pg.display.init()
        pg.font.init()
        self.screen = pg.display.set_mode((self.vparam['screen_width'],
                                      self.vparam['screen_height']), pg.SRCALPHA, 32)
        clock = pg.time.Clock()
        background = pg.Surface(self.screen.get_size()).convert()
        background.fill(self.cc(self.vparam['background_color']))
        self.second_bg_col = (60,60,20,0)

        # Add surfaces for rollout view.
        for N in self.list_rep:
            for m in range(self.memory):
                N['surf'][m] = pg.Surface([N['row_height'], N['row_height']])

        pg.mouse.set_visible(1)
        pg.key.set_repeat(1, 100)

        # Get shared memory.
        self.shm = SharedMemory(self.net, 
                                self.param,
                                session_id=\
                                    int(self.IPC_PROC['session_id'].value))

        # current visualization frame
        current_rv_frame = 0
        # current neuronal frame (from ipc)
        current_frame = 0
        # previous neuronal frame
        last_frame = 0
        # new neuronal frame flag
        self.new_frame = False
        
        # Set all fonts.
        self.fonts = {}
        self.font_size = {}
        for f in ['tiny', 'small', 'large', 'huge']:
            self.fonts[f] = pg.font.SysFont('Courier', 
                                            self.vparam['font ' + f])
            self.font_size[f] = self.vparam['font ' + f]

        # Determine maximal item name length.
        self.max_name_px = 0
        self.max_name_len = 0
        for t in ['neuron_pools', 'interfaces']:
            for i in self.net[t]:
                dummy = self.fonts[self.name_font].render(i, 1, self.cc(self.text_color))
                if len(i) > self.max_name_len:
                    self.max_name_len = len(i)
                    self.max_name_px = dummy.get_size()[0]
        self.rollout_offsetX = (self.screen_width \
                                - self.max_name_px - 4 * self.item_size \
                                - (self.item_size * (2 + self.max_input_len))) \
                                // self.rollout
        self.Y_offset = 0

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
            'minus': [6 * 32, 128, 32, 32],
            'plus': [7 * 32, 128, 32, 32],
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
        # Small left / right arrow sprite.
        self.button_sprite['small right'] = pg.transform.scale(self.button_sprite['play'], [24, 24])
        self.button_sprite['small left'] = pg.transform.flip(self.button_sprite['small right'], True, False)

        # Create widget collector.
        self.wcollector = Collector(self.screen, 
                                    self.vparam,
                                    self.button_sprite, 
                                    self.fonts,
                                    self.ccol)

        # Dictionary of buttons.
        self.buttons = {}

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
        # Add plus / minus button for rollout.
        self.buttons['rollout-minus'] = self.wcollector.add_button(None, sprite='small left', 
                pos=np.asarray([550, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_rollout_minus())
        self.buttons['rollout-plus'] = self.wcollector.add_button(None, sprite='small right', 
                pos=np.asarray([550 + 24, Y0], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_rollout_plus())

        self.buttons['memory-minus'] = self.wcollector.add_button(None, sprite='small left', 
                pos=np.asarray([550, Y0 + 24], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_memory_minus())
        self.buttons['memory-plus'] = self.wcollector.add_button(None, sprite='small right', 
                pos=np.asarray([550 + 24, Y0 + 24], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_memory_plus())

        self.buttons['dt-plus'] = self.wcollector.add_button(None, sprite='small left', 
                pos=np.asarray([self.max_name_px // 2 - 40, self.header_height - 3 * self.name_size // 2 - 2], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_dt_plus())
        self.buttons['dt-minus'] = self.wcollector.add_button(None, sprite='small right', 
                pos=np.asarray([self.max_name_px // 2 + 32, self.header_height - 3 * self.name_size // 2 - 2], dtype=np.float32), 
                cb_LMB_clicked=lambda: self.cb_button_LMB_click_dt_minus())

        # Load potential rollout view settings.
        self.load_rollout_view()
        self.update_rollout()
        self.update_memory()

        # Some timers
        timer_overall = np.ones([12])

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
        
        # Temporary draw board.
        self.tmp_surf = pg.Surface((self.vparam['screen_width'],
                                    self.vparam['screen_height']))

        # Enter forever loop.
        while self.state_running:
            # Set back all (items) up flag.
            self.all_up = True

            # Current viz frame.
            current_rv_frame += 1

            # Start viz timer.
            timer_overall[current_rv_frame % 12] = time.time()

            # Get current frame.
            current_frame = int(copy.copy(self.IPC_PROC['now'].value))
            # Determine beginning of new neural frame.
            if current_frame > last_frame:
                # Update last frame.
                last_frame = current_frame
                # Set new_frame flag.
                self.new_frame = True
            else:
                self.new_frame = False

            # =================================================================
            # Get all process states.
            # =================================================================
            for X in self.list_rep:
                X['state'] = self.IPC_PROC['state'][self.shm.proc_id[X['name']][0]].value
            # =================================================================

            # =================================================================
            # Update internal neural state memory.
            # =================================================================
            if self.new_frame and self.memory > 0:
                self.current_mem = (self.current_mem + 1) % self.memory
                for n,N in enumerate(self.list_rep):
                    N['mem'][self.current_mem][:] = self.shm.dat[N['name']]['state'][0,:]
                    self.render_item(n, self.current_mem)
            # =================================================================

            
            # =================================================================
            # Update mouse state.
            # =================================================================
            # Get current mouse position.
            POS = pg.mouse.get_pos()
            self.POS = POS
            # Determine mouse over.
            self.mouse_over = ['bg', None]
            # Determine if over item.
            for X in self.list_rep:
                if X['rect name'].collidepoint(POS):
                    self.mouse_over = ['name', X['name']]
                    break
                for r in range(self.rollout + 1):
                    if X['rect'][r].collidepoint(POS):
                        self.mouse_over = ['item', X['name'], r]
                        break
            for p,P in self.plasts.items():
                if P['rect'].collidepoint(POS):
                    self.mouse_over = ['plast', p]
                    break
            # =================================================================


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
                        elif event.key in [pg.K_w, pg.K_r] \
                                and not self.is_shift_pressed \
                                and not self.is_ctrl_pressed:
                            # Open selection to save / recall current rolloutview.
                            # Determine existing rolloutviews for this model.
                            rolloutviews = os.listdir(self.home_path + '/.statestream/viz/')
                            this_bvs = 0
                            for bv in range(len(rolloutviews)):
                                if rolloutviews[bv].find('-rolloutview-') != -1 \
                                        and rolloutviews[bv].find(self.net['name']) != -1:
                                    this_bvs += 1
                            selections = [" slot " + str(s) + " " for s in range(this_bvs)]
                            sel_info = ['' for s in range(this_bvs)]
                            # For saving also add 'new'.
                            if event.key == pg.K_w:
                                selections.append("new slot")
                                sel_info.append('')
                                modus = "save rolloutview"
                            else:
                                modus = "load rolloutview"
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
                        elif event.key == pg.K_g:
                            self.hotkey['g'] = True
                            self.hotkey_origin = np.array([POS[0], POS[1]])

                    else:
                        self.wcollector.key(event.key, self.is_shift_pressed, event.unicode)
            # =================================================================
                



            # =================================================================
            # Check LMB click / hold / drag.
            # =================================================================
            # Check all widgets for button clicked action.
            if LMB_click:
                LMB_click = self.wcollector.LMB_clicked(POS)
            if RMB_click:
                RMB_click = self.wcollector.RMB_clicked(POS)
            # If nothing clicked, clear selection.
            if LMB_click:
                for N in self.list_rep:
                    if N['rect name'].collidepoint(POS):
                        if N['mode'] == 'item':
                            N['mode'] = 'map'
                        elif N['mode'] == 'map':
                            N['mode'] = 'item'
                        LMB_click = False
                        break
                if self.mouse_over[0] == 'plast':
                    if self.mouse_over[1] == self.plast_selected:
                        self.plast_selected = None
                    else:
                        self.plast_selected = copy.copy(self.mouse_over[1])
                    LMB_click = False
                LMB_click = False

            if RMB_click:
                pass
            # =================================================================



            # =================================================================
            # RMB click / drag / hold.
            # =================================================================
            # RMB hold actions.
            if RMB_hold:
                if RMB_drag_type is None:
                    RMB_drag_type = '__all__'
                    RMB_drag_inst = ''
                # Evaluate drag for different RMB drag types.
                if RMB_drag_type == '__all__':
                    # Update Y shift.
                    self.Y_offset += 0.1 * (self.POS[1] - RMB_hold_origin[1])
            # =================================================================

                            


            # Compute NN FPS.
            if time.time() - NNFPS_old_timer > 2.0:
                tmp_n = int(self.IPC_PROC['now'].value)
                NNFPS = (tmp_n - NNFPS_old_frame) // 2
                NNFPS_old_frame = tmp_n
                NNFPS_old_timer = time.time()



            # =================================================================
            # Tabula rasa.
            # =================================================================
            # Blit background - tabula rasa.
            # All plotting / drawing / blitting should come afterwards.
            self.screen.blit(background, (0, 0))
            # =================================================================



            # Check if mouse over widget.
            self.wcollector.mouse_over(POS)



            # =================================================================
            # Rollout overview: Draw items and their connections.
            # =================================================================
            # Dependent on mouse over change item (here, nps) color / border / size.
            if self.mouse_over[0] == 'bg':
                for N in self.list_rep:
                    for r in range(self.rollout + 1):
                        N['col'][r] = copy.copy(self.np_color)
                        N['border'][r] = copy.copy(self.default_border)
                        N['size'][r] = copy.copy(self.item_size)
            elif self.mouse_over[0] == 'item':
                self.list_rep[self.dict_rep[self.mouse_over[1]]]['col'][self.mouse_over[2]] = (0,0,0)
                self.list_rep[self.dict_rep[self.mouse_over[1]]]['border'][self.mouse_over[2]] = copy.copy(self.rollback_border)
                self.list_rep[self.dict_rep[self.mouse_over[1]]]['size'][self.mouse_over[2]] \
                    = copy.copy(self.list_rep[self.dict_rep[self.mouse_over[1]]]['row_height'])
                self.change_item_color_recback(self.mouse_over[1], self.mouse_over[2])
                self.change_item_color_recforward(self.mouse_over[1], self.mouse_over[2])
            if self.plast_selected is not None:
                if self.plast_selected in self.mn.net_plast_nps:
                    mn_ro = self.mn.net_plast_nps[self.plast_selected]
                    for d in range(len(mn_ro)):
                        if -self.dt + d >= 0 and -self.dt + d <= self.rollout:
                            for n in mn_ro[d]:
                                self.list_rep[self.dict_rep[n]]['col'][-self.dt + d] = (0,0,0)
                                self.list_rep[self.dict_rep[n]]['border'][-self.dt + d] \
                                    = copy.copy(self.rollback_border)
                                self.list_rep[self.dict_rep[n]]['size'][-self.dt + d] \
                                    = copy.copy(self.list_rep[self.dict_rep[n]]['row_height'])
                    if 'target' in self.net['plasticities'][self.plast_selected]:
                        tgt_np = self.net['plasticities'][self.plast_selected]['target']
                        tgt_dt = self.net['plasticities'][self.plast_selected]['target_t']
                        if -self.dt + tgt_dt >= 0 and -self.dt + tgt_dt <= self.rollout:
                            self.list_rep[self.dict_rep[tgt_np]]['col'][-self.dt + tgt_dt] = self.vparam['plast_color']
                    if 'source' in self.net['plasticities'][self.plast_selected]:
                        src_np = self.net['plasticities'][self.plast_selected]['source']
                        src_dt = self.net['plasticities'][self.plast_selected]['source_t']
                        if -self.dt + src_dt >= 0 and -self.dt + src_dt <= self.rollout:
                            self.list_rep[self.dict_rep[src_np]]['col'][-self.dt + src_dt] = self.vparam['plast_color']

            # Row wise light / dark background.
            for n,N in enumerate(self.list_rep):
                Y = self.header_height + N['row_height'] + N['Y'] + self.Y_offset
                if n % 2 == 0:
                    if Y > 0 and Y <= self.screen_height:
                        bg_col = copy.copy(self.second_bg_col)
                        pg.draw.rect(self.screen, self.cc(bg_col), 
                                     [0, Y, self.screen_width, N['row_height']], 0)

            # Draw connections.
            for n,N in enumerate(self.list_rep):
                Y = self.header_height + N['row_height'] + N['Y'] + self.Y_offset
                for src_sp,SRC_SP in enumerate(N['src_sps']):
                    X = self.name_size
                    X += self.max_name_px + self.item_size
                    X += self.rollout_offsetX // 2
                    Y_sp = N['Y'] + self.Y_offset
                    Y_sp = Y_sp + self.header_height + 3 * N['row_height'] // 2
                    Y_sp = np.clip(Y_sp, 
                                   self.header_height + N['row_height'] // 2, 
                                   self.screen_height - N['row_height'])
                    dX = int(src_sp * (self.rollout_offsetX // (2 * len(N['src_sps']))))
                    for r in range(self.rollout):
                        # Plot connection sp -> target np
                        if Y > 0 and Y <= self.screen_height:
                            pg.draw.line(self.screen, 
                                         self.cc((127, 127, 127)), 
                                         (int(X + dX), int(Y_sp)), 
                                         (int(X + self.rollout_offsetX // 2), int(Y + N['row_height'] // 2)), 2)
                        # Plot connectionS src_nps -> sp
                        for src_np in N['src_nps'][src_sp]:
                            src_np_id = self.dict_rep[src_np]
                            src_np_Y = self.header_height + N['row_height'] \
                                       + self.list_rep[src_np_id]['Y'] + self.Y_offset
                            if src_np_Y > 0 and src_np_Y <= self.screen_height:
                                pg.draw.line(self.screen, 
                                             self.cc((127, 127, 127)), 
                                             (int(X + dX), int(Y_sp)), 
                                             (int(X - self.rollout_offsetX // 2), int(src_np_Y + N['row_height'] // 2)), 2)
                        X += self.rollout_offsetX
            # Draw SPs.
            for n,N in enumerate(self.list_rep):
                Y = self.header_height + N['row_height'] + N['Y'] + self.Y_offset
                for src_sp,SRC_SP in enumerate(N['src_sps']):
                    X = self.name_size
                    X += self.max_name_px + self.item_size
                    X += self.rollout_offsetX // 2
                    Y_sp = N['Y'] + self.Y_offset
                    Y_sp = Y_sp + self.header_height + 3 * N['row_height'] // 2
                    Y_sp = np.clip(Y_sp, 
                                   self.header_height + N['row_height'] // 2, 
                                   self.screen_height - N['row_height'])
                    dX = int(src_sp * (self.rollout_offsetX // (2 * len(N['src_sps']))))
                    for r in range(self.rollout):
                        plot_circle(self.screen, 
                                    int(X + dX), int(Y_sp), self.item_size // 4, 
                                    self.cc(self.text_color), 
                                    self.cc(self.np_color), 
                                    2)
                        X += self.rollout_offsetX
            # Draw items.
            for n,N in enumerate(self.list_rep):
                name = N['name'].rjust(self.max_name_len)
                Y = self.header_height + N['row_height'] + N['Y'] + self.Y_offset
                X = self.name_size
                if Y > 0 and Y <= self.screen_height:
                    # Blit item name.
                    txt_col = copy.copy(self.text_color)
                    N['rect name'] = self.screen.blit(self.fonts[self.name_font].render(name, 1, self.cc(txt_col)), 
                                                      (self.name_size, Y + (N['row_height'] - self.name_size) // 2))
                    # Blit rolled-out item.
                    X += self.max_name_px + self.item_size 
                    for r in range(self.rollout + 1):
                        if N['mode'] == 'map' and r + self.dt <= 0 and abs(r + self.dt) < self.memory:
                            tmp_idx = (self.current_mem + r + self.dt) % self.memory
                            N['rect'][r] = self.screen.blit(N['surf'][tmp_idx], (int(X - N['row_height'] // 2), int(Y)))
                        else:
                            N['rect'][r] = plot_circle(self.screen, 
                                                       int(X), int(Y + N['row_height'] // 2), 
                                                       N['size'][r] // 2, 
                                                       self.cc(self.text_color), 
                                                       self.cc(N['col'][r]), 
                                                       N['border'][r])
                        X += self.rollout_offsetX

            # =================================================================


            # =================================================================
            # Draw black background for better main button visibility.
            # =================================================================
            pg.draw.rect(self.screen, 
                         self.cc(DEFAULT_COLORS['dark1']), 
                         [0, 0, self.screen_width, self.header_height], 0)
            pg.draw.line(self.screen, 
                         self.cc(self.vparam['text_color']), 
                         (0, self.header_height - 2 * self.name_size), 
                         (self.screen_width - 1, self.header_height - 2 * self.name_size), 2)
            pg.draw.line(self.screen, 
                         self.cc(self.vparam['text_color']), 
                         (0, self.header_height), 
                         (self.screen_width - 1, self.header_height), 2)
            # =================================================================



            # =================================================================
            # Blit plasticities.
            # =================================================================
            cntr = 0
            for p,P in self.plasts.items():
                border = 1
                if self.plast_selected is not None:
                    if p == self.plast_selected:
                        border = 6
                        self.screen.blit(self.fonts[self.name_font].render(p, 1, self.cc(self.text_color)), 
                                         (self.screen_width // 2, 8 + self.item_size - self.name_size // 2))
                P['rect'] = plot_circle(self.screen, 
                                        int(self.screen_width - 2 * (cntr + 1) * self.item_size), 
                                        8 + self.item_size, 
                                        self.item_size, 
                                        self.cc(self.text_color), 
                                        self.cc(self.vparam['plast_color']), 
                                        border)
                cntr += 1
            if self.mouse_over[0] == 'plast':
                self.screen.blit(self.fonts[self.name_font].render(self.mouse_over[1], 1, self.cc(self.text_color)), 
                                 (self.screen_width // 2, 8 + self.item_size - self.name_size // 2))
            # =================================================================



            # =================================================================
            # Blit temporal offset dt.
            # =================================================================
            X = self.name_size
            X += self.max_name_px + self.item_size 
            self.screen.blit(self.fonts[self.name_font].render('dt', 1, self.cc(self.text_color)), 
                             (self.max_name_px // 2, self.header_height - 3 * self.name_size // 2))
            for r in range(self.rollout + 1):
                if r + self.dt == 0:
                    pg.draw.rect(self.screen, 
                                 self.cc(self.text_color), 
                                 [X - 2 * self.name_size, 
                                  self.header_height - 3 * self.name_size // 2, 
                                  4 * self.name_size, 
                                  self.name_size], 0)
                    self.screen.blit(self.fonts[self.name_font].render('NOW', 1, self.cc(DEFAULT_COLORS['dark1'])), 
                                     (X - self.name_size, self.header_height - 3 * self.name_size // 2))
                else:
                    self.screen.blit(self.fonts[self.name_font].render(str(int(r + self.dt)), 1, self.cc(self.text_color)), 
                                     (X - self.name_size // 4, self.header_height - 3 * self.name_size // 2))
                X += self.rollout_offsetX
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



            # End overall frame timer.
            timer_overall[current_rv_frame % 12] -= time.time()


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
            # Blitt current rollout
            self.screen.blit(self.fonts['small'].render('rollout: ' + str(int(self.rollout)), 
                                                        1, 
                                                        self.cc(self.vparam['text_color'])), 
                             (436, 12))
            # Blitt current memory
            self.screen.blit(self.fonts['small'].render('memory: ' + str(int(self.memory)), 
                                                        1, 
                                                        self.cc(self.vparam['text_color'])), 
                             (436, 12 + 24))



            # =================================================================
            # Draw all widgets.
            # =================================================================
            self.wcollector.draw()
            self.wcollector.draw_top()
            # =================================================================



            # =================================================================
            # Update core communication queue.
            # =================================================================
            self.update_core_comm_queue()
            # =================================================================



            # =================================================================
            # Flip display.
            # =================================================================
            pg.display.flip()
            # Maintain frames per seconds delay.
            clock.tick(self.vparam['FPS'])
            # =================================================================


            
            # =================================================================
            # Evaluate trigger and viz flag.
            # =================================================================
            if self.IPC_PROC['trigger'].value == 2 or self.IPC_PROC['rvgui flag'].value != 1:
                self.shutdown()
            # =================================================================

        # Quit pygame.
        pg.display.quit()






# =============================================================================
# =============================================================================
# =============================================================================

    def shutdown(self):
        """Exiting the visualization.
        """
        self.state_running = False
        # Dump local rollout view.
        self.dump_rollout_view()
        self.IPC_PROC['rvgui flag'].value = 0

# =============================================================================
# =============================================================================
# =============================================================================
