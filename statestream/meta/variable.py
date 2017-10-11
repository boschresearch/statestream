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



from statestream.utils.pygame_import import pg

import numpy as np
import copy as cp



class MetaVariable(object):
    """Meta-variables holding methods and data for derived properties.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.
    mv_param : dict
        Dictionary containing parameters of this meta-variable. Each meta-variable
        has to provide this structure by a function get_parameter(net, children).
    client_param : dict
        Dictionary containing all client parameters.
    """
    def __init__(self, net, param, mv_param, client_param):
        # Copy information.
        self.net = net
        self.param = cp.deepcopy(param)
        self.mv_param = cp.deepcopy(mv_param)
        self.p = cp.deepcopy(client_param)

        # Get / set name.
        self.name = client_param['name']
        
        # Default for presentation is hidden.
        # *-able: number of options
        # *-ed: initial option
        self.itemable = 0
        self.itemized = 0
        self.blitable = []
        self.blitted = []

        self.minimum = None
        self.maximum = None

        # Begin with empty list of children.
        self.selected_values = cp.copy(client_param['selected_values'])
        
        # Default color is white.
        self.col = (255, 255, 255)
        
        # Default position for all meta variable plots.
        self.pos = np.zeros([2,], dtype=np.float32) + 200
        
        # Default rects for meta variable control.
        self.rects = {
            "name": pg.Rect(0, 0, 2, 2),
            "close": pg.Rect(0, 0, 2, 2),
            "col": pg.Rect(0, 0, 2, 2),
            "hide": pg.Rect(0, 0, 2, 2),
            "itemize": pg.Rect(0, 0, 2, 2)
        }


        
    def get_value(self):
        """Method returns the current value of the meta-variable.
        """
        pass


    def pprint(self):
        """Method returns a pretty printable list of lines for visualization.

        Returns:
        --------
            lines : list
                List of strings for a pretty print of some informationf for this
                meta-variable.
        """
        return []
    
    
    
    def plot(self, shm, screen, viz_brain, ccol):
        """Method for specialized meta variable plotting.
        
            Returning False will cause the visualization to use a fallback.

        Parameters:
        -----------
        shm : SharedMemory class
            This is a class holding references to the entire shared memory of this ST session.
        screen : pygame.Surface
            The pygame surface to draw the meta-variable onto.
        viz_brain : dict
            The brain dictionary containing visualization information about all items
            (see also the statestream.visualization.visualization.brain).
        ccol : boolean
            Flag for color correction.

        Returns:
        --------
            plotted : boolean
                Flag if the meta-variable was actually drawn here. If not a fallback drawing can
                be tried in the visualization.
        """
        return False
        

        

    def update_value(self, shm):
        """Method computes a value update for the meta-variable

        Parameters:
        -----------
        shm : SharedMemory class
            This is a class holding references to the entire shared memory of this ST session.
        """
        # Read all computed results from shared memory.
        pass
