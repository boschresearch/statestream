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


# The following functions / classes are derived from Keras 2.1.2
#   (https://github.com/keras-team/keras)
#       def variable()
#       class Function()
#       def function()
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import tensorflow as tf

import numpy as np


# The backend.
_BACKEND = "tensorflow"

# The default / fallback dtype.
_DTYPE = "float32"

# The current tensorflow session and used device.
_TF_SESSION = None
_DEVICE = None

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
              "sign",
              "Id",
              "sqrt",
              "tanh",
              "relu",
              "selu",
              "spiky",
              "switchsign",
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
    x = tf.Variable(value, dtype=tf.as_dtype(dtype), name=name)
    x._statestream_settable = settable
    if settable:
        x._assign_placeholder = tf.placeholder(dtype, shape=value.shape)
        x._assign_op = x.assign(x._assign_placeholder)
    return x

def scalar(value, dtype=np.float32, borrow=None, name=None, settable=True):
    if dtype is None:
        dtype = np.float32
    x = tf.Variable(value, dtype=tf.as_dtype(dtype), name=name)
    x._statestream_settable = settable
    if settable:
        x._assign_placeholder = tf.placeholder(dtype, shape=x.get_shape().as_list())
        x._assign_op = x.assign(x._assign_placeholder)
    return x





def get_value(x):
    value = x.eval(session=tf_get_session())
    return value

def set_value(x, value):
    if x._statestream_settable:
        value = np.asarray(value, dtype=x.dtype.base_dtype.name)
        tf_dtype = tf.as_dtype(x.dtype.name.split('_')[0])
        tf_get_session().run(x._assign_op, feed_dict={x._assign_placeholder: value})
    else:
        raise TypeError("Tried to set / assign non-settable tensorflow variable: " + str(x.name))

def update(x, update):
    return tf.assign(x, update)





def zeros(shape, dtype=None, name=None):
    if dtype is None:
        dtype = np.float32
    return tf.zeros(shape=shape, dtype=tf.as_dtype(dtype), name=name)

def ones(shape, dtype=None, name=None):
    if dtype is None:
        dtype = np.float32
    return tf.ones(shape=shape, dtype=tf.as_dtype(dtype), name=name)




class RandomStream(object):
    def __init__(self, seed, dist_type, dtype=None):
        if dtype is None:
            dtype = _DTYPE
        self.dtype = tf.as_dtype(dtype)
        self.seed = seed
        self.dist_type = dist_type
    def __call__(self, shape=(), p=0.5, low=0.0, high=1.0, avg=0.0, std=1.0):
        if self.dist_type == "normal":
            return tf.random_normal(shape, mean=avg, stddev=std, seed=self.seed)
        elif self.dist_type == "uniform":
            return tf.random_uniform(shape, minval=low, maxval=high, seed=self.seed)
        elif self.dist_type == "binomial":
            return tf.where(tf.random_uniform(shape, dtype=self.dtype, seed=self.seed) <= p,
                            tf.ones(shape, dtype=self.dtype),
                            tf.zeros(shape, dtype=self.dtype))

def randomstream(seed, dist_type, dtype=None):
    return RandomStream(seed, dist_type, dtype)




def shape(x):
    return x.get_shape().as_list()

def ndim(x):
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    else:
        return None

def cast(x, dtype):
    return tf.cast(x, dtype)

def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, axis=axis, keep_dims=keepdims)

def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, axis=axis, keep_dims=keepdims)

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)

def prod(x, axis=None, keepdims=False):
    return tf.reduce_prod(x, axis=axis, keep_dims=keepdims)

def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)

def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))

def var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    v = tf.square(x - m)
    return tf.reduce_mean(v, axis=axis, keep_dims=keepdims)

def any(x, axis=None, keepdims=False):
    return tf.reduce_any(x, axis=axis, keep_dims=keepdims)

def argmin(x, axis=-1, keepdims=False):
    return tf.argmin(x, axis=axis)

def argmax(x, axis=-1, keepdims=False):
    return tf.argmax(x, axis=axis)

def clip(x, min_value, max_value):
    return tf.clip_by_value(x, min_value, max_value)

def abs(x):
    return tf.abs(x)

def sign(x, thresh=0.0):
    return tf.sign(x - thresh)



def Id(x):
    """The identity function: a(x) = x
    """
    return x

def sqrt(x):
    return tf.sqrt(clip(x, 0.0, np.inf))

def tanh(x):
    """Tangenshyperbolicus: a(x) = tanh(x)
    """
    return tf.nn.tanh(x)

def relu(x, thresh=0.0):
    """ReLU function: a(x) = max(x, 0)
    """
    if thresh == 0.0:
      return tf.nn.relu(x)
    else:
      return tf.nn.relu(x - thresh)
      
def selu(x, llambda=1.0507, alpha=1.6733):
    """SeLU function. See also the elu activation.

    Given default parameters (lambda, alpha) correspond to the 
    fixed-point of layer statistics (mu = 0, sigma = 1).
    """
#    return T.switch(T.lt(x,0), alpha * llambda * (tf.exp(x) - 1.0), llambda * x)
    return tf.nn.selu(x)

def spiky(x, threshold=1.0, saturation=2.0):
    """Spiky function.

    Simple spiking activation function assuming a self identity recurrence.
    x < 0:                          0
    0 < x < threshold:              x
    threshold < x < saturation:     saturation + eps
    saturation <= x:                0


    """
    _x = tf.where(x < 0, 0.0 * x, x)
    _y = tf.where(_x >= saturation, 0.0 * _x, _x)
    return tf.where(_y > threshold, 0.0 * _y + saturation + 1e-8, _y)

def switchsign(x, threshold=1.0):
    """Switch sign function.

    All activations above the threshold switch their sign.
    """
    return tf.where(x > threshold, x, x)

def leakyrelu(x, leak=0.1):
    """Leaky ReLU function: a(x) = max(x, leak*x)
    """
    return maximum(x, leak * x)

def softmax(x):
    """Softmax function: a(x) = softmax(x).
    """
#    shape = shape(x)
#    y = T.nnet.softmax(x.swapaxes(0, 1).flatten(2).swapaxes(0, 1))
#    return T.nnet.softmax(np.swapaxes(y.flatten(2), 0, 1))
#    return T.reshape(y.swapaxes(0, 1), [shape[1], shape[0], shape[2], shape[3]]).swapaxes(0, 1)
    x = tf.transpose(x, (0, 2, 3, 1))
    return tf.transpose(tf.nn.softmax(x), (0, 3, 1, 2))

def softplus(x):
    """Softplus function: a(x) = log(1 + exp(x))
    """
#    return tf.log(1 + tf.exp(x))
    return tf.nn.softplus(x)

def exp(x):
    """Exponential function: a(x) = exp(x)
    """
    return tf.exp(x)

def mexp(x):
    """Negative exponential function: a(x) = -exp(x)
    """
    return -tf.exp(x)

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
    if alpha == 1.0:
        return tf.nn.elu(x)
    else:
        return tf.where(x > 0, tf.nn.elu(x), alpha * tf.nn.elu(x))
    
def sigmoid(x):
    """Sigmoid function: a(x) = 1 / (1 + exp(-x))
    """
    return tf.nn.sigmoid(x)

def gaussian(x):
    """Gaussian function: a(x) = exp(-x*x)
    """
    return tf.exp(-(x**2))

def log(x, eps=1e-6):
    """The natural logarithm.
    """
    return tf.log(maximum(x, eps))

def neglog(x, eps=1e-6):
    """The negative logarithm.
    """
    return -tf.log(maximum(x, eps))

def softround(x, p=2):
    """The soft-round (activation) function.

    Rounds |R to 0 and 1.
    """
    x_off = 0.5

    x = x - x_off

    normalizer = np.tanh(p * (1 - x_off)) + (1 - x_off) / p
    y = (tf.tanh(p * x) + x / p) / normalizer.astype(np.float32)

    y = (y + 1) * 0.5

    return y




def concatenate(states, axis=-1):
    if axis < 0:
        if ndim(states[0]):
            axis %= ndim(states[0])
        else:
            axis = 0
    return tf.concat(states, axis=axis)

def reshape(x, shape):
    return tf.reshape(x, shape)

def repeat(x, reps, axis):
    x_shape = x.get_shape().as_list()
    if x_shape[axis] is not None:
        split = tf.split(value=x,
                         num_or_size_splits=x_shape[axis],
                         axis=axis)
        x_rep = [s for s in split for _ in range(reps)]
        return tf.concat(x_rep, axis)
    else:
        raise TypeError("Repeat state along None axis not allowed.")

def unbroadcast(x, axis):
    return x

def dimshuffle(x, pattern):
    """The dimshuffle function from theano for tensorflow.
    """
    broadcast_ax = [idx for (idx, dim) in enumerate(pattern) if dim == 'x']
    axes = [i for i in pattern if isinstance(i, int)]
    i = 0
    while i < ndim(x):
        if i not in axes:
            x = tf.squeeze(x, axis=i)
            new_axes = []
            for ax in axes:
                if i < ax:
                    new_axes.append(ax)
                else:
                    new_axes.append(ax - 1)
            axes = new_axes
        else:
            i += 1
    if axes != list(range(ndim(x))):
        x = tf.transpose(x, axes)
    if len(broadcast_ax) > 0:
        for i in sorted(broadcast_ax):
            x = tf.expand_dims(x, axis=i)
    return x

def flatten(x):
    return tf.reshape(x, [-1])

def cpu_contiguous(x):
    return x





def conv2d(x, kernel, border_mode="half", subsample=(1, 1), filter_dilation=(1, 1)):
    if border_mode in ["valid", (0, 0)]:
        border_mode = "VALID"
    else:
        border_mode = "SAME"
    # Reordering kernel dimensions.
    result = tf.nn.convolution(input=tf.transpose(x, (0, 2, 3, 1)),
                               filter=tf.transpose(kernel, (2, 3, 1, 0)),
                               padding=border_mode,
                               strides=subsample,
                               dilation_rate=filter_dilation)
    return tf.transpose(result, (0, 3, 1, 2))

def grad(loss, variables):
    return tf.gradients(loss, variables, colocate_gradients_with_ops=True)


def minimum(x, y):
    return tf.minimum(x, y)

def maximum(x, y):
    return tf.maximum(x, y)

def floor(x):
    return tf.floor(x)

def dot(x, y):
    return tf.matmul(x, y)

def tensordot(x, y, axes):
    return tf.tensordot(x, y, axes=axes)

def MSE(x, y):
    """Computation of the mean-square error.
    """
    return (x - y)**2

def MAE(x, y):
    """Computation of the mean-absolute error.
    """
    return tf.abs(x - y)

def hinge(x, y):
    """Computation of the hinge error.
    """
    return tf.maximum(1.0 - x * y, 0.0)

def categorical_crossentropy(x, y):
    """Computation of the categorical crossentropy.
    """
    x = tf.clip_by_value(x, 1e-6, 1.0 - 1e-6)
    return - tf.reduce_sum(y * tf.log(x), axis=len(x.get_shape()) - 1)

def negloglikelihood(x, y):
    """Computation of the negative log-likelihood.
    """
    return - y * tf.log(tf.maximum(x, 1e-6))

def minimize(x):
    """Simply minimize x.
    """
    return x

def maximize(x):
    """Simply maximize x.
    """
    return - x





def warp_transform(conv_input, input_shape, output_shape, **kwargs):
    """Transforms features of conv_input using x_offset, y_offset.

    Parameter:
    ----------
    conv_input:
        array of shape [batch_size, channels, width, height]
    input_shape:
        the shape of conv_input: [batch_size, channels, width, height]
    output_shape:
        the spatial shape of the transformed output: [width, height]


    **kwargs:
    x_pos, y_pos: [1, output_shape], [1, output_shape]
        Contains coordinates in [0,1] for every output pixel.
    theta: [6, 1, 1]
        A single parameter set for a spatial transformation.

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
    output_height = output_shape[0]
    output_width = output_shape[1]

    # Compute coordinates.
#    x = flatgrid_x + flatten(x_offset) * width
#    y = flatgrid_y + flatten(y_offset) * height

    if "x_pos" in kwargs and "y_pos" in kwargs:
        x = flatten(kwargs["x_pos"]) * width
        y = flatten(kwargs["y_pos"]) * height
    elif "theta" in kwargs:
        # Generate grid.
        grid_x, grid_y = np.meshgrid(np.linspace(0.0, output_width - 1.0, output_width),
                           np.linspace(0.0, output_height - 1.0, output_height))
        # Flatten grid.
        flatgrid_x = np.reshape(grid_x, (1, -1))
        flatgrid_y = np.reshape(grid_y, (1, -1))
        tf_flatgrid_x = tf.constant(np.tile(flatgrid_x, np.stack([batch_size, 1])).flatten(), dtype="float32")
        tf_flatgrid_y = tf.constant(np.tile(flatgrid_y, np.stack([batch_size, 1])).flatten(), dtype="float32")
        tf_flatgrid_1 = tf.constant(np.ones(flatgrid_x.shape()), dtype="float32")
        tf_flatgrid = tf.stack([tf_flatgrid_x, tf_flatgrid_y, tf_flatgrid_1], 0)
        xy = tf.matmul(tf.reshape(kwargs["theta"], (-1, 2, 3)), tf_flatgrid)
        x = tf.reshape(tf.slice(xy, [0, 0, 0], [-1, 1, -1]), [-1])
        y = tf.reshape(tf.slice(xy, [0, 1, 0], [-1, 1, -1]), [-1])
    else:
        raise ValueError

    # Compute / clip indices.
    x0 = cast(floor(x), "int32")
    x1 = x0 + 1
    y0 = cast(floor(y), "int32")
    y1 = y0 + 1
    x0 = clip(x0, 0, width - 1)
    x1 = clip(x1, 0, width - 1)
    y0 = clip(y0, 0, height - 1)
    y1 = clip(y1, 0, height - 1)

    x_tmp = np.arange(batch_size, dtype=np.int32) * width * height
#    rep = dimshuffle(ones((int(height * width),), dtype='int32'), ('x', 0))
    rep = dimshuffle(ones((int(output_height * output_width),), dtype='int32'), ('x', 0))
    base = flatten(dot(reshape(x_tmp, (-1, 1)), rep))

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width

    # Lookup pixels by index.
    map_flat = reshape(dimshuffle(conv_input, (0, 2, 3, 1)), 
                       [batch_size * height * width, channels])
    Ia = tf.gather(map_flat, base_y0 + x0)
    Ib = tf.gather(map_flat, base_y1 + x0)
    Ic = tf.gather(map_flat, base_y0 + x1)
    Id = tf.gather(map_flat, base_y1 + x1)

    # Calculate interpolated values.
    x0_f = cast(x0, "float32")
    x1_f = cast(x1, "float32")
    y0_f = cast(y0, "float32")
    y1_f = cast(y1, "float32")
    wa = dimshuffle(((x1_f - x) * (y1_f - y)), (0, 'x'))
    wb = dimshuffle(((x1_f - x) * (y - y0_f)), (0, 'x'))
    wc = dimshuffle(((x - x0_f) * (y1_f - y)), (0, 'x'))
    wd = dimshuffle(((x - x0_f) * (y - y0_f)), (0, 'x'))
    interpolated = sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    interpolated = reshape(cast(interpolated, "float32"), 
                           np.stack([batch_size, output_height, output_width, channels]))
    # print("\nTRAFOSHAPE: " + str(interpolated.get_shape().as_list()))
    return dimshuffle(interpolated, (0, 3, 2, 1))





class Function(object):
    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for u in updates:
                if isinstance(u, tuple):
                    p, u_p = u
                    updates_ops.append(tf.assign(p, u_p))
                else:
                    updates_ops.append(u)
            self.updates_op = tf.group(*updates_ops)
        self.feed_dict = kwargs.pop("feed_dict", {})
        self.fetches = kwargs.pop("fetches", [])
        if not isinstance(self.fetches, list):
            self.fetches = [self.fetches]
        self.kwargs = kwargs

    def __call__(self, inputs=[]):
        feed_dict = self.feed_dict.copy()
        if inputs:
            for tensor, value in zip(self.inputs, inputs):
                feed_dict[tensor] = value
        tf_fetches = self.outputs + [self.updates_op] + self.fetches
        session = tf_get_session()
        updated = session.run(fetches=tf_fetches, feed_dict=feed_dict, **self.kwargs)
        return updated[:len(self.outputs)]

def function(inputs, outputs, updates=[], **kwargs):
    return Function(inputs, outputs, updates=updates, **kwargs)

def free():
    tf.reset_default_graph()



# ============================================================================
# ============================================================================
# ============================================================================

def tf_get_session():
    global _TF_SESSION
    global _DEVICE
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _TF_SESSION is None:
            config = tf.ConfigProto()
            # Do not allocate the entire memory.
            config.gpu_options.allow_growth = True
            _TF_SESSION = tf.Session(config=config)
        session = _TF_SESSION
    return session

