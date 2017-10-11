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



import theano
import theano.tensor as T

import importlib
import numpy as np



def err_MSE(x, y):
    """Computation of the mean-square error.
    """
    return (x - y)**2



def err_MAE(x, y):
    """Computation of the mean-absolute error.
    """
    return abs(x - y)



def err_hinge(x, y):
    """Computation of the hinge error.
    """
    return T.maximum(1.0 - x * y, 0.0)



def err_categorical_crossentropy(x, y):
    """Computation of the categorical crossentropy.
    """
    return T.nnet.categorical_crossentropy(T.clip(x, 1e-6, 1.0 - 1e-6), y)



def err_negloglikelihood(x, y):
    """Computation of the negative log-likelihood.
    """
    return -y * T.log(T.maximum(x, 1e-6))



def err_minimize(x):
    """Simply minimize x.
    """
    return x



def err_maximize(x):
    """Simply maximize x.
    """
    return -x

