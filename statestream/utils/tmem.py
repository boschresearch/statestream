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



def update_tmem(tmem_dists, tmem_updates, tmem_update, index, frame, time_steps = None):                    
    """Update function for temporal memory (tmem) list.
    
    Parameters
    ----------
    tmem_dists : list of ints
    tmem_updates
    tmem_update
    index
    frame
    time_steps
        
    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.
    """
    if tmem_dists != []:
        if index == 0:
            # Update every tmem_dists[0] frames.
            if frame % tmem_dists[0] == 0:
                tmem_update[0] = True
                tmem_updates[0] = tmem_updates[0] + 1
                if time_steps != None:
                    time_steps[0] = 1
            else:
                tmem_update[0] = False
                if time_steps != None:
                    time_steps[0] = time_steps[0] + 1
            if len(tmem_dists) > 1:
                update_tmem(tmem_dists, tmem_updates, tmem_update, 1, frame, time_steps = time_steps)
        else:
            if tmem_updates[index - 1] >= tmem_dists[index]:
                tmem_update[index] = True
                tmem_updates[index] = tmem_updates[index] + 1
                tmem_updates[index - 1] = 0
                if time_steps != None:
                    time_steps[index] = np.prod(tmem_dists[0:index]) + 1
            else:
                tmem_update[index] = False
                if time_steps != None:
                    time_steps[index] = time_steps[index] + 1
            # Recursively proceed if not at end of tmem_dists.
            if index < len(tmem_dists) - 1:
                update_tmem(tmem_dists, tmem_updates, tmem_update, index + 1, frame, time_steps = time_steps)
                