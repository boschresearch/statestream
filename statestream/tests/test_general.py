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



from __future__ import print_function

import sys

from time import sleep, gmtime, strftime, time
import os
import copy
import multiprocessing as mp
import numpy as np
import ctypes
import importlib



def test_matplotlib():
    """Test if matplotlib can be imported.
    """
    try:
        import matplotlib.pyplot as plt
    except:
        plt = None
    assert(plt is not None), "Error: Unable to import matplotlib.pyplot."



def test_shared_array():
	"""Do some standard tests for SharedArray.
	"""
	try:
		import SharedArray
	except:
		SharedArray = None
	assert(SharedArray is not None), "Error: Unable to load python module SharedArray."
	shm_name = "statestream.test.0.x"
	try:
		x = SharedArray.create(shm_name, 2**20, dtype=np.float32)
	except:
		x = None
	assert(x is not None), "Error: Unable to instantiate shared memory array."
	deleted = True
	try:
		SharedArray.delete(shm_name)
	except:
		deleted = False
	assert(deleted), "Error: Unable to deleted shared memory object."



def test_ruamel_yaml():
    """Do test if ruamel yaml is available.
    """
    try:
        import ruamel_yaml as yaml
    except:
        try:
            import ruamel.yaml as yaml
        except:
            yaml = None
    assert(yaml is not None), "Error: ruamel yaml python package not found."



def test_statestream_import():
    """Test if all statestream submodules can be imported.
    """
    # from statestream.meta.process import process_state
    # from statestream.meta.process import process_trigger
    pass


def test_python_importlib_reload():
    """Test the python module reload function.
    """
    import statestream.visualization.visualization

    reloaded = False
    try:
        importlib.reload(statestream.visualization.visualization)
        reloaded = True
    except:
        try:
            reload(statestream.visualization.visualization)
            reloaded = True
        except:
            pass
    assert(reloaded), "Error: Unable to reload modules."




def test_python_version_import():
    """Try to import certain functions.
    """

    # Try to import pickle.
    try:
        import pickle as pckl
    except:
        try:
            import cPickle as pckl
        except:
            pckl = None
    assert(pckl is not None), "Error: Unable to import cPickle or pickle."



def test_python_version_string():
    """Test python version depeneent string functions.
    """
    failed = False
    try:
        print("test", end="\r")
    except:
    	failed = True
    assert(not failed), "Error: Unable to correctly execute print function"
