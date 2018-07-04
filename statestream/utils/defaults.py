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



__all__ = [
    'DEFAULT_CORE_PARAMETER',
    'DEFAULT_VIZ_PARAMETER'
]

DEFAULT_CORE_PARAMETER = {
    'THEANO_FLAGS': 'dnn.conv.algo_fwd=guess_once, dnn.conv.algo_bwd_filter=none, dnn.conv.algo_bwd_data=guess_once, floatX=float32, optimizer_including=cudnn, mode=FAST_RUN',
    'save_path': '/tmp/',    
    'temporal_memory': [1],
    'random_seed': 42,
    'max_processes': 2**12,
    'visible_devices': [0, 1],
    'dtype_default': 'np.float32',
    'profiler_window': 16
}

DEFAULT_COLORS = {
    'red': (40, 40, 160),
    'green': (40, 160, 40),
    'blue': (160, 40, 40),
    'magenta': (140, 60, 140),
    'yellow': (60, 140, 140),
    'cyan': (140, 140, 60),
    'orange': (40, 140, 180),
    'dark1': (30, 20, 10),
    'dark2': (55, 5, 60),
    'light': (200, 230, 250)
}

DEFAULT_VIZ_PARAMETER = {
    'FPS': 24,
    'screen_width': 1600,
    'screen_height': 900,
    'color_correction': True,
    'background_color': DEFAULT_COLORS['dark1'],
    'number_color': DEFAULT_COLORS['orange'],
    'text_color': DEFAULT_COLORS['light'],
    'window_color': DEFAULT_COLORS['dark2'],
    'np_color': DEFAULT_COLORS['green'],
    'sp_color': DEFAULT_COLORS['blue'],
    'plast_color': DEFAULT_COLORS['red'],
    'if_color': DEFAULT_COLORS['cyan'],
    'font tiny': 12,
    'font small': 16,
    'font large': 20,
    'font huge': 24,
    'rollout': 6,
    'subwindow_width': 512,
    'subwindow_height': 512
}
