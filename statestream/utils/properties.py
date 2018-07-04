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
import skimage.measure

from statestream.ccpp.cgraphics import cgraphics_tensor_dist



def array_property(a, prop):
    """Function to compute a property of an array.

    Parameter
    ---------
    a : np.ndarray
        The array for which to compute the property.
    prop : str
        String specifying the property.
    """
    value = None

    if prop == "mean":
        value = np.mean(a)
    elif prop == "var":
        value = np.var(a)
    elif prop == "std":
        value = np.std(a)
    elif prop == "max":
        value = np.max(a)
    elif prop == "min":
        value = np.min(a)
    elif prop == "median":
        value = np.median(a)
    elif prop[0:3] in ["mv-", "vm-"]:
        # mean(var(*, axis)) or var(mean(*, axis))
        if len(prop) == 4:
            axis = int(prop[-1])
        elif len(prop) == 5:
            axis = (int(prop[-2]), int(prop[-1]))
        elif len(prop) == 6:
            axis = (int(prop[-3]), int(prop[-2]), int(prop[-1]))
        if prop[0] == "m":
            value = np.mean(np.var(a, axis=axis))
        else:
            value = np.var(np.mean(a, axis=axis))
    elif prop == "L0":
        value = np.linalg.norm(a, ord=0)
    elif prop == "L1":
        value = np.linalg.norm(a, ord=1)
    elif prop in ["L2", "norm"]:
        value = np.linalg.norm(a)
    elif prop == "Linf":
        value = np.linalg.norm(a, ord=np.inf)
    elif prop == "13-mean":
        arr = np.abs(a)
        value = 1.0 / (float(np.prod(arr.shape)) ** (2.0 / 3.0))
        value *= np.sum(arr)
        value *= np.sum(arr * arr * arr) ** (-1.0 / 3.0)

    return value



def np_feature_metric(x, y, metric, samples):
    """Function to compute a metric between two tensors.

    We assume that x and y are 4D:
        [agents, features, dim_x, dim_y]

    In a first step we scale the larger (in sense of space)
    array down to the size of the smaller one.
    In the second step the metric for all feature
    combinations is computed and returned.

    Parameter
    ---------
    x,y : np.ndarray
        4D arrays.
    metric : str
        String specifying the metric:
            cos:
            L0:
            L1:
            L2:
            dot:

    Return
    ------
    z : np.ndarray
        A 2D array of dimension [feature_x, feature_y]
    """

    metric_val = np.zeros([x.shape[1], y.shape[1]], dtype=np.float32).flatten()

    # Determine larger (in sense of space) tensor and rescale it down.
    if x.shape[2] == y.shape[2] and x.shape[3] == y.shape[3]:
        X = x[0:samples,:,:,:]
        Y = y[0:samples,:,:,:]
    elif x.shape[2] > y.shape[2]:
        X = np.zeros([samples, x.shape[1]] + list(y.shape[2:]), dtype=np.float32)
        factor = (int(x.shape[2] / y.shape[2]), 
                  int(x.shape[3] / y.shape[3]))
        for a in range(samples):
            for f in range(x.shape[1]):
                X[a,f,:,:] = skimage.measure.block_reduce(x[a,f,:,:],
                                                          factor,
                                                          np.max)
        Y = y[0:samples,:,:,:]
    else:
        X = x[0:samples,:,:,:]
        Y = np.zeros([samples, y.shape[1]] + list(x.shape[2:]), dtype=np.float32)
        factor = (int(y.shape[2] / x.shape[2]), 
                  int(y.shape[3] / x.shape[3]))
        for a in range(samples):
            for f in range(y.shape[1]):
                Y[a,f,:,:] = skimage.measure.block_reduce(y[a,f,:,:],
                                                          factor,
                                                          np.max)

    # Convert metric to c-modus.
    if metric in ['inf', 'L-inf', 'Linf']:
        m = -1
    elif metric == 'L0':
        m = 0
    elif metric == 'L1':
        m = 1
    elif metric == 'L2':
        m = 2
    elif metric == 'dot':
        m = 3
    elif metric in ['cos', 'cosine']:
        m = 4

    # Compute metric.
    cgraphics_tensor_dist(X.flatten(),
                          Y.flatten(),
                          metric_val,
                          samples,
                          X.shape[1],
                          Y.shape[1],
                          X.shape[2],
                          X.shape[3],
                          m)

    return np.reshape(metric_val, [x.shape[1], y.shape[1]])

