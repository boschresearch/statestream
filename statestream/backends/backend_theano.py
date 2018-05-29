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


# The following functions / classes are derived from Keras 2.1.2
#   (https://github.com/keras-team/keras)
#       def variable()
#       class Function()
#       def function()
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import theano
from theano import tensor as T

import numpy as np


# The backend.
_BACKEND = "theano"

# The default / fallback dtype.
_DTYPE = "float32"

# All available backend functions.
_FUNCTIONS = ["variable",
			  "scalar",
              "get_value",
              "set_value",
              "update",
			  "zeros",
			  "ones",
			  "randomstream",
              "shape",
              "ndim",
			  "cast",
			  "min",
			  "max",
			  "sum",
			  "prod",
			  "mean",
			  "std",
			  "var",
			  "any",
			  "argmin",
			  "argmax",
			  "clip",
              "abs",
			  "Id",
			  "sqrt",
			  "tanh",
			  "relu",
			  "selu",
              "spiky",
			  "leakyrelu",
			  "softmax",
			  "softplus",
			  "exp",
			  "mexp",
			  "elu",
			  "square",
			  "msquare",
			  "sigmoid",
			  "gaussian",
			  "log",
			  "neglog",
			  "concatenated",
			  "reshape",
			  "repeat",
			  "unbroadcast",
			  "dimshuffle",
              "flatten",
			  "cpu_contiguous",
			  "conv2d",
			  "grad",
			  "minimum",
			  "maximum",
              "floor",
			  "dot",
			  "tensordot",
			  "MSE",
			  "MAE",
			  "hinge",
			  "categorical_crossentropy",
			  "negloglikelihood",
			  "minimize",
			  "maximize",
              "warp_transform",
			  "function",
              "free"]





def variable(value, dtype=None, borrow=None, broadcastable=None, name=None, settable=True):
    if dtype is None:
        dtype = np.float32
    value = np.asarray(value, dtype=dtype)
    if borrow is None and broadcastable is None:
        value = theano.shared(value=value, name=name)
    elif borrow is None and broadcastable is not None:
        value = theano.shared(value=value, broadcastable=broadcastable, name=name)
    elif borrow is not None and broadcastable is not None:
        value = theano.shared(value=value, borrow=borrow, name=name)
    else:
        value = theano.shared(value=value, borrow=borrow, name=name, broadcastable=broadcastable)
    return value

def scalar(value, dtype=np.float32, borrow=None, name=None, settable=True):
    if borrow is None:
        value = theano.shared(theano._asarray(value, dtype=dtype), name=name)
    else:
        value = theano.shared(theano._asarray(value, dtype=dtype), borrow=borrow, name=name)
    return value





def get_value(x, order=None):
    value = x.get_value()
    return value

def set_value(x, value, order=None):
    x.set_value(np.asarray(value, dtype=x.dtype))

def update(x, update):
    return (x, update)





def zeros(shape, dtype=None, name=None):
    if dtype is None:
        dtype = np.float32
    return variable(np.zeros(shape, dtype=dtype), dtype=dtype, name=name)

def ones(shape, dtype=None, name=None):
    if dtype is None:
        dtype = np.float32
    return variable(np.ones(shape, dtype=dtype), dtype=dtype, name=name)





def randomstream(seed, dist_type):
    if dist_type == "normal":
        return theano.tensor.shared_randomstreams.RandomStreams(seed).normal
    elif dist_type == "uniform":
        return theano.tensor.shared_randomstreams.RandomStreams(seed).uniform
    elif dist_type == "binomial":
        return theano.tensor.shared_randomstreams.RandomStreams(seed).binomial
    else:
        raise NameError("Invalid theano random distribution: " + str(dist_type))




def shape(x):
    return x.shape

def ndim(x):
    return x.ndim

def cast(x, dtype):
    return T.cast(x, dtype=dtype)

def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)

def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)

def sum(x, axis=None, keepdims=False):
    return T.sum(x, axis=axis, keepdims=keepdims)

def prod(x, axis=None, keepdims=False):
    return T.prod(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    return T.mean(x, axis=axis, keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return T.std(x, axis=axis, keepdims=keepdims)

def var(x, axis=None, keepdims=False):
    return T.var(x, axis=axis, keepdims=keepdims)

def any(x, axis=None, keepdims=False):
    return T.any(x, axis=axis, keepdims=keepdims)

def argmin(x, axis=-1, keepdims=False):
    return T.argmin(x, axis=axis, keepdims=keepdims)

def argmax(x, axis=-1, keepdims=False):
    return T.argmax(x, axis=axis, keepdims=keepdims)

def clip(x, min_value, max_value):
    return T.clip(x, min_value, max_value)

def abs(x):
    return T.abs_(x)





def Id(x):
    """The identity function: a(x) = x
    """
    return x

def sqrt(x):
    return T.sqrt(T.clip(x, 0.0, np.inf))

def tanh(x):
    """Tangenshyperbolicus: a(x) = tanh(x)
    """
    return T.tanh(x)

def relu(x):
    """ReLU function: a(x) = max(x, 0)
    """
    return T.maximum(x, 0)

def selu(x, llambda=1.0507, alpha=1.6733):
    """SeLU function. See also the elu activation.

    Given default parameters (lambda, alpha) correspond to the 
    fixed-point of layer statistics (mu = 0, sigma = 1).
    """
    return T.switch(T.lt(x,0), alpha * llambda * (T.exp(x) - 1.0), llambda * x)

def spiky(x, threshold=1.0, saturation=2.0):
    """Spiky function.

    Simple spiking activation function assuming a self identity recurrence.
    x < 0:                          0
    0 < x < threshold:              x
    threshold < x < saturation:     x + saturation
    saturation <= x:                0


    """
    _x = T.switch(T.lt(x,0), 0.0 * x, x)
    _y = T.switch(T.gt(_x,saturation), 0.0 * _x, _x)
    return T.switch(T.gt(_y, threshold), _y + saturation, _y)

def leakyrelu(x, leak=0.1):
    """Leaky ReLU function: a(x) = max(x, leak*x)
    """
    return T.maximum(x, leak * x)

def softmax(x):
    """Softmax function: a(x) = softmax(x).
    """
    shape_x = shape(x)
    y = T.nnet.softmax(x.swapaxes(0, 1).flatten(2).swapaxes(0, 1))
#    return T.nnet.softmax(np.swapaxes(y.flatten(2), 0, 1))
    return T.reshape(y.swapaxes(0, 1), [shape_x[1], shape_x[0], shape_x[2], shape_x[3]]).swapaxes(0, 1)
    
def softplus(x):
    """Softplus function: a(x) = log(1 + exp(x))
    """
    return T.log(1 + T.exp(x))

def exp(x):
    """Exponential function: a(x) = exp(x)
    """
    return T.exp(x)

def mexp(x):
    """Negative exponential function: a(x) = -exp(x)
    """
    return -T.exp(x)

def square(x):
    """Square function: a(x) = x*x"""
    return x * x

def msquare(x):
    """Negative square function: a(x) = -x*x
    """
    return -x * x
    
def elu(x, alpha=1.0):
    """eLU function: a(x) = {x>0: x, x<=0: exp(x)-1}
    """
    return T.nnet.elu(x, alpha)
    
def sigmoid(x):
    """Sigmoid function: a(x) = 1 / (1 + exp(-x))
    """
    return T.nnet.sigmoid(x)

def gaussian(x):
    """Gaussian function: a(x) = exp(-x*x)
    """
    return T.exp(T.neg(x**2))

def log(x, eps=1e-6):
    """The natural logarithm.
    """
    return T.log(T.maximum(x, eps))

def neglog(x, eps=1e-6):
    """The negative logarithm.
    """
    return -T.log(T.maximum(x, eps))

def softround(x, p=2):
    """The soft-round (activation) function.

    Rounds |R to 0 and 1.
    """
    x_off = 0.5

    x = x - x_off

    normalizer = np.tanh(p * (1 - x_off)) + (1 - x_off) / p
    y = (T.tanh(p * x) + x / p) / normalizer.astype(np.float32)

    y = (y + 1) * 0.5

    return y






def concatenate(states, axis=-1):
    return T.concatenate(states, axis=axis)

def reshape(x, shape):
    return T.reshape(x, shape)

def repeat(x, reps, axis):
    return T.extra_ops.repeat(x, reps, axis)

def unbroadcast(x, axis):
    return T.unbroadcast(x, axis)

def dimshuffle(x, pattern):
    return x.dimshuffle(*pattern)

def flatten(x):
    return T.flatten(x)

def cpu_contiguous(x):
    return T.extra_ops.cpu_contiguous(x)





def conv2d(x, kernel, border_mode="half", subsample=(1, 1), filter_dilation=(1, 1)):
    return T.nnet.conv2d(x,
                         kernel,
                         border_mode=border_mode,
                         subsample=subsample,
                         filter_dilation=filter_dilation)

def grad(loss, variables):
    return T.grad(loss, variables)


def minimum(x, y):
    return T.minimum(x, y)

def maximum(x, y):
    return T.maximum(x, y)

def floor(x):
    return np.floor(x)
    
def dot(x, y):
	return T.dot(x, y)

def tensordot(x, y, axes):
	return T.tensordot(x, y, axes=axes)

def MSE(x, y):
    """Computation of the mean-square error.
    """
    return (x - y)**2

def MAE(x, y):
    """Computation of the mean-absolute error.
    """
    return T.abs_(x - y)

def hinge(x, y):
    """Computation of the hinge error.
    """
    return T.maximum(1.0 - x * y, 0.0)

def categorical_crossentropy(x, y):
    """Computation of the categorical crossentropy.
    """
    return T.nnet.categorical_crossentropy(T.clip(x, 1e-6, 1.0 - 1e-6), y)

def negloglikelihood(x, y):
    """Computation of the negative log-likelihood.
    """
    return -y * T.log(T.maximum(x, 1e-6))

def minimize(x):
    """Simply minimize x.
    """
    return x

def maximize(x):
    """Simply maximize x.
    """
    return -x





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

    x_tmp = np.arange(batch_size) * width * height
    rep = T.ones((int(height * width),), dtype='int32').dimshuffle('x', 0)
    base = T.dot(x_tmp.reshape((-1, 1)), rep).flatten()

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





class Function(object):
    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.function = theano.function(inputs,
                                        outputs,
                                        updates=updates,
                                        **kwargs)
    def __call__(self, inputs=None):
        if inputs is None:
            return self.function()
        else:
            return self.function(*inputs)



def function(inputs, outputs, updates=[], **kwargs):
    return Function(inputs, outputs, updates=updates, **kwargs)


def free():
    pass
