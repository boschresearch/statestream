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



import copy



class SharedMemoryLayout(object):
    """Class stores meta information about single shared memory array.
    """
    def __init__(self, Dtype, shape, dtype, default, minimum=None, maximum=None, broadcastable=None):
        self.type = Dtype
        self.shape = shape
        self.dtype = dtype
        self.default = default
        self.min = minimum
        self.max = maximum
        if broadcastable is None:
            # Scalar variables are never broadcastable.
            if shape is not ():
                # Set default for broadcastable to False.
                self.broadcastable = tuple([False for i in range(len(shape))])
            else:
                self.broadcastable = None
        else:
            self.broadcastable = tuple(copy.copy(broadcastable))
