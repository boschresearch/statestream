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
import theano.tensor as T



def warp_transform(conv_input, input_shape, x_offset, y_offset):
    """Transforms features of conv_input using x_offset, y_offset.

    Parameter:
    ----------
    conv_input:
        array of shape [batch_size, channels, width, height]
    input_shape:
        the shape of conv_input: [batch_size, channels, width, height]
    x_offset: 
        array of shape [batch_size, 1, width, height]
    y_offset:
        array of shape [batch_size, 1, width, height]

    Return:
    -------
    output:
        The transformed input of shape [batchsize, channels, width, height]

    References:
    -----------
    [1] Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
        Spatial Transformer Networks.
        https://arxiv.org/abs/1506.02025

    [2] https://github.com/skaae/transformer_network
    """
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Generate grid.
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, width - 1.0, width),
                           np.linspace(0.0, height - 1.0, height))
    # Flatten grid.
    flatgrid_x = np.reshape(grid_x, (1, -1))
    flatgrid_y = np.reshape(grid_y, (1, -1))
    flatgrid_x = np.tile(flatgrid_x, np.stack([batch_size, 1])).flatten()
    flatgrid_y = np.tile(flatgrid_y, np.stack([batch_size, 1])).flatten()
    # Compute coordinates.
    x = flatgrid_x + T.flatten(x_offset) * width
    y = flatgrid_y + T.flatten(y_offset) * height
    # Compute / clip indices.
    x0 = T.cast(np.floor(x), "int32")
    x1 = x0 + 1
    y0 = T.cast(np.floor(y), "int32")
    y1 = y0 + 1
    x0 = T.clip(x0, 0, width - 1)
    x1 = T.clip(x1, 0, width - 1)
    y0 = T.clip(y0, 0, height - 1)
    y1 = T.clip(y1, 0, height - 1)

    base = repeat(np.arange(batch_size) * width * height,
                  int(height * width))
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width

    # Lookup pixels by index.
    map_flat = T.reshape(conv_input.dimshuffle(0,2,3,1), [batch_size * height * width, channels])
    Ia = map_flat[base_y0 + x0]
    Ib = map_flat[base_y1 + x0]
    Ic = map_flat[base_y0 + x1]
    Id = map_flat[base_y1 + x1]

    # Calculate interpolated values.
    x0_f = T.cast(x0, "float32")
    x1_f = T.cast(x1, "float32")
    y0_f = T.cast(y0, "float32")
    y1_f = T.cast(y1, "float32")
    wa = ((x1_f - x) * (y1_f - y)).dimshuffle(0, 'x')
    wb = ((x1_f - x) * (y - y0_f)).dimshuffle(0, 'x')
    wc = ((x - x0_f) * (y1_f - y)).dimshuffle(0, 'x')
    wd = ((x - x0_f) * (y - y0_f)).dimshuffle(0, 'x')
    interpolated = T.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    interpolated = T.reshape(T.cast(interpolated, "float32"), 
                                  np.stack([batch_size, height, width, channels]))
    return interpolated.dimshuffle(0,3,1,2)


def repeat(x, n_repeats):
    rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
    return T.dot(x.reshape((-1, 1)), rep).flatten()
