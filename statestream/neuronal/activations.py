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



import theano.tensor as T



# __all__ = [
#     "Id",
#     "tanh",
#     "relu",
#     "leakyrelu",
#     "softmax",
#     "softplus",
#     "exp",
#     "mexp",
#     "square",
#     "msquare",
#     "elu",
#     "sigmoid",
#     "gaussian"
# ]


# =============================================================================
# Some activation functions.
def Id(x):
    """The identity function: a(x) = x
    """
    return x
    
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

def leakyrelu(x, leak=0.1):
    """Leaky ReLU function: a(x) = max(x, leak*x)
    """
    return T.maximum(x, leak * x)

def softmax(x):
    """Softmax function: a(x) = softmax(x).
    """
    shape = T.shape(x)
    y = T.nnet.softmax(x.swapaxes(0, 1).flatten(2).swapaxes(0, 1))
#    return T.nnet.softmax(np.swapaxes(y.flatten(2), 0, 1))
    return T.reshape(y.swapaxes(0, 1), [shape[1], shape[0], shape[2], shape[3]]).swapaxes(0, 1)
    
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
    
def elu(x):
    """eLU function: a(x) = {x>0: x, x<=0: exp(x)-1}
    """
    return T.switch(T.lt(x,0), T.exp(x) - 1, x)
    
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
    """The natural logarithm.
    """
    return -T.log(T.maximum(x, eps))
# =============================================================================
