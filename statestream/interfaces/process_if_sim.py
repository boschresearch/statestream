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
import time
import copy
import os

from statestream.ccpp.csim import csim_func
from statestream.interfaces.process_if import ProcessIf
from statestream.meta.neuron_pool import np_state_shape
from statestream.neuronal.neuro_convert import float_2_np
from statestream.utils.helper import is_scalar_shape
from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.utils.pygame_import import pg



def if_interfaces():
    """Returns the specific interfaces as strings of the interface.
    Parameters:
    -----------
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string name of this interface.
    """
    # Specify sub-interfaces.
    return {"out": ["dist_fov",
                    "dist_per",
                    "ret_fov",
                    "ret_per",
                    "acc_x",
                    "acc_y",
                    "acc_a",
                    "haptic",
                    "FOV_fov",
                    "lookat"],
            "in": ["Fx",
                   "Fy",
                   "Fa"]
           }



def if_init(net, name, dat_name, dat_layout, mode=None):
    """Return value for interface parameter / variable.

    Parameters:
    -----------
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    name : str
        The unique string name of this interface.
    dat_name : str
        The unique name for the parameter to initialize.
    dat_layout : SharedMemoryLayout
        The layout (see SharedMemoryLayout) of the parameter to be initialized.
    mode : None, value (float, int), str
        The mode determines how a parameter has to be initialized (e.g. 'xavier' or 0.0)
    """
    # Default return is None.
    dat_value = None

    # Return initialized value.
    return dat_value



def if_shm_layout(name, net, param):
    """Return shared memory layout for sim interface.

    Parameters:
    -----------
    name : str
        The unique interface name.
    net : dict
        The network dictionary containing all nps, sps, plasts, ifs.
    param : dict
        Dictionary of core parameters.

    """
    # Get interface dictionary.
    p = net["interfaces"][name]
    # Begin with empty layout.
    shm_layout = {}
    shm_layout["parameter"] = {}
    shm_layout["variables"] = {}

    # Add parameter.
    # -------------------------------------------------------------------------
    # Add mode parameter.
    shm_layout["parameter"]["mode"] \
        = ShmL("np", (), np.int32, p.get("mode", 0), 1, None)

    # Add variables.
    # -------------------------------------------------------------------------
    # Add all outputs as variables.
    for o in p["out"]:
        tmp_target = o
        # Consider remapping.
        if "remap" in p:
            if o in p["remap"]:
                tmp_target = p["remap"][o]
        # Set layout.
        shm_layout["variables"][o] \
            = ShmL("np", np_state_shape(net, tmp_target), np.float32, 0)

    # Return layout.
    return shm_layout





class ProcessIf_sim(ProcessIf):
    """Interface class providing 2D world simulation environment.

        To use this interface compile the (my_)csim.c/h file.

        Parameters:
        -----------
        name : str
            Unique interface identifier.
        ident : int
            Unique id for the interface's process.
        net : dict
            The network dictionary containing all nps, sps, plasts, ifs.
        param : dict
            Dictionary of core parameters.

        Interface parameters:
        ---------------------

        Inputs:
        -------
        Fx : np.array, shape [agents, 1, 1, 1]
            Scalar force in x-direction (agent front) of agent.
        Fy : np.array, shape [agents, 1, 1, 1]
            Scalar force in y-direction (agent side) of agent.
        Fa : np.array, shape [agents, 1, 1, 1]
            Scalar rotational force of agent.

        TODO:
        F_FOV_fov
        F_lookat


        Outputs:
        --------
        dist_fov : np.array, shape [agents, 1, no_f, 1]
            1D-array representing distance measures along no_f different foveal rays.
        dist_per : np.array, shape [agents, 1, no_p, 1]
            1D-array representing distance measures along no_p different peripheral rays.
        ret_fov : np.array, shape [agents, 3, no_f, 1]
            RGB 1D-array representing an RGB measures along no_f different foveal rays.
        ret_per : np.array, shape [agents, 3, no_p, 1]
            RGB 1D-array representing an RGB measures along no_p different peripheral rays.
        acc_x : np.array, shape [agents, 1, 1, 1]
            Agent's perceived acceleration in x-direction (forward - backward).
        acc_y : np.array, shape [agents, 1, 1, 1]
            Agent's perceived acceleration in y-direction (sideways).
        acc_a : np.array, shape [agents, 1, 1, 1]
            Agent's perceived rotational acceleration.
        haptic : np.array, shape [agents, hap_receptors, 1, 1]
            Agent's haptic (near-range) sensors.

        TODO:
        FOV_fov : np.array, shape [agents, 1, 1, 1]
            Agent's perceived field-of-view of its fovea.
        lookat : np.array, shape [agents, 1, 1, 1]
            Agent's perceived head (= ret + per) orientation.


    """ 
    def __init__(self, name, ident, net, param):
        # Initialize parent ProcessIf class
        ProcessIf.__init__(self, name, ident, net, param)

    def initialize(self):
        """Method to initialize the sim interface class.
        """
        # Get some experimental parameters.
        # ---------------------------------------------------------------------

# =============================================================================
# =============================================================================
        # 2D world interface parameter
# =============================================================================
# =============================================================================
        # World size.
        self.world_w = self.p.get("world_width", 1000.0)
        self.world_h = self.p.get("world_height", 720.0)
        # Origin on screen.
        self.oX = self.p.get("screen_width", 1200.0) - self.world_w - 60
        self.oY = 40        
        # Number of line segments.
        if "line_segments" in self.p:
            self.no_ls = len(self.p["line_segments"])
        else:
            self.no_ls = 24
        # Number of mirrors.
        if "mirrors" in self.p:
            self.no_m = len(self.p["mirrors"])
        else:
            self.no_m = 4
        # Number of agents.
        self.no_a = int(self.net["agents"])
        self.no_c = int(2 * self.no_a)
        # Number of Field-Of-View receptors.
        self.no_f = int(self.p.get("fov_receptors", 128))
        # Number of peripheral receptors.
        self.no_p = int(self.p.get("per_receptors", 128))
        # Number of action array nodes.
        self.no_as = int(self.p.get("action_nodes", 16))
        # Number of touch array nodes.
        self.no_ss = int(self.p.get("hap_receptors", 64))
        # List of all variables.
        self.debug_world_vars = ['world_params',
                            'ls_x1', 'ls_y1', 'ls_x2', 'ls_y2',
                            'ls_R', 'ls_G', 'ls_B',
                            'm_x1', 'm_y1', 'm_x2', 'm_y2',
                            'c_x', 'c_y', 'c_r', 'c_a',
                            'c_dx', 'c_dy', 'c_da',
                            'c_Fx', 'c_Fy', 'c_Fa',
                            'c_R', 'c_G', 'c_B',
                            'a_x', 'a_y', 'a_a', 'a_r',
                            'a_dx', 'a_dy', 'a_da',
                            'a_Fx', 'a_Fy', 'a_Fa',
                            'a_R', 'a_G', 'a_B',
                            'a_lookat', 'a_fF', 'a_pF',
                            'a_motor',
                            'a_ddx', 'a_ddy', 'a_dda',
                            'a_sensor',
                            'f_R', 'f_G', 'f_B', 'f_D',
                            'p_R', 'p_G', 'p_B', 'p_D'
                            ]
        # World parameter.
        self.world_params = np.zeros([4,]).astype(np.float32)
        self.world_params[0] = self.p.get("world_friction", 0.2)
        self.world_params[1] = self.p.get("limit_trans_vel", 2.0)
        self.world_params[2] = self.p.get("limit_rot_vel", 1.0)
        self.world_params[3] = self.p.get("repulsion_dist", 32.0)   
        # Line segments.
        self.ls_x1 = 0.0 * np.random.rand(self.no_ls).astype(np.float32)
        self.ls_y1 = 0.0 * np.random.rand(self.no_ls).astype(np.float32)
        self.ls_x2 = 0.0 * np.random.rand(self.no_ls).astype(np.float32)
        self.ls_y2 = 0.0 * np.random.rand(self.no_ls).astype(np.float32)
        self.ls_R = np.zeros([4 * self.no_ls]).astype(np.float32)
        self.ls_G = np.zeros([4 * self.no_ls]).astype(np.float32)
        self.ls_B = np.zeros([4 * self.no_ls]).astype(np.float32)
        # Initialize textures.
        for ls in range(self.no_ls):
            # min
            self.ls_R[4*ls + 0] = 0.0
            self.ls_G[4*ls + 0] = 0.5
            self.ls_B[4*ls + 0] = 0.5
            # max
            self.ls_R[4*ls + 1] = 0.0
            self.ls_G[4*ls + 1] = 0.8
            self.ls_B[4*ls + 1] = 0.8
            # width min
            self.ls_R[4*ls + 2] = 1.0
            self.ls_G[4*ls + 2] = 20.0
            self.ls_B[4*ls + 2] = 30.0
            # width max
            self.ls_R[4*ls + 3] = 1.0
            self.ls_G[4*ls + 3] = 15.0
            self.ls_B[4*ls + 3] = 22.0
        ###############################################################################
        self.ls_x1[0] = 20
        self.ls_y1[0] = 20
        self.ls_x2[0] = self.world_w - 20
        self.ls_y2[0] = 20
        # -----------------------------------------------------------------------------
        self.ls_x1[1] = 20
        self.ls_y1[1] = 20
        self.ls_x2[1] = 20
        self.ls_y2[1] = self.world_h - 20
        # -----------------------------------------------------------------------------
        self.ls_x1[2] = self.world_w - 20
        self.ls_y1[2] = self.world_h - 20 
        self.ls_x2[2] = self.world_w - 20
        self.ls_y2[2] = 20
        # -----------------------------------------------------------------------------
        self.ls_x1[3] = self.world_w - 20
        self.ls_y1[3] = self.world_h - 20 
        self.ls_x2[3] = 20
        self.ls_y2[3] = self.world_h - 20
        # -----------------------------------------------------------------------------
        # some horizontal lines
        self.ls_x1[4] = 100
        self.ls_y1[4] = self.world_h / 2 
        self.ls_x2[4] = 400
        self.ls_y2[4] = self.world_h / 2
        # -----------------------------------------------------------------------------
        self.ls_x1[5] = self.world_w - 300
        self.ls_y1[5] = self.world_h / 2 
        self.ls_x2[5] = self.world_w - 60
        self.ls_y2[5] = self.world_h / 2
        # -----------------------------------------------------------------------------
        self.ls_x1[6] = 400
        self.ls_y1[6] = self.world_h / 4
        self.ls_x2[6] = 500
        self.ls_y2[6] = self.world_h / 4
        # -----------------------------------------------------------------------------
        # some horizontal lines
        self.ls_x1[7] = 500
        self.ls_y1[7] = self.world_h / 2
        self.ls_x2[7] = 500
        self.ls_y2[7] = self.world_h / 4
        
        #######################################################################
        # mirrors
        self.m_x1 = np.zeros([self.no_m]).astype(np.float32)
        self.m_y1 = np.zeros([self.no_m]).astype(np.float32)
        self.m_x2 = np.zeros([self.no_m]).astype(np.float32)
        self.m_y2 = np.zeros([self.no_m]).astype(np.float32)
        # -----------------------------------------------------------------------------
        self.m_x1[0] = 500
        self.m_y1[0] = self.world_h / 4 
        self.m_x2[0] = 700
        self.m_y2[0] = self.world_h / 4
        # -----------------------------------------------------------------------------
        #######################################################################
        #######################################################################
        # circles
        self.c_x = np.clip(self.world_w * np.random.rand(self.no_c), 
                           40, 
                           self.world_w - 40).astype(np.float32)
        self.c_y = np.clip(self.world_h * np.random.rand(self.no_c), 
                           40, 
                           self.world_h - 40).astype(np.float32)
        self.c_r = 8.0 + 0.0 * np.random.rand(self.no_c).astype(np.float32)
        self.c_a = np.zeros([self.no_c]).astype(np.float32)
        self.c_dx = np.zeros([self.no_c]).astype(np.float32)
        self.c_dy = np.zeros([self.no_c]).astype(np.float32)
        self.c_da = np.zeros([self.no_c]).astype(np.float32)
        self.c_Fx = np.zeros([self.no_c]).astype(np.float32)
        self.c_Fy = np.zeros([self.no_c]).astype(np.float32)
        self.c_Fa = np.zeros([self.no_c]).astype(np.float32)
        self.c_R = np.zeros([4 * self.no_c]).astype(np.float32)
        self.c_G = np.zeros([4 * self.no_c]).astype(np.float32)
        self.c_B = np.zeros([4 * self.no_c]).astype(np.float32)
        # Initialize agents dynamic state.
        self.a_r = 12.0 + 0.0 * np.random.rand(self.no_a).astype(np.float32)
        self.a_x = np.clip(self.world_w * np.random.rand(self.no_a), 
                           100, 
                           self.world_w - 100).astype(np.float32)
        self.a_y = np.clip(self.world_h * np.random.rand(self.no_a), 
                           100, 
                           self.world_h - 100).astype(np.float32)
        self.a_dx = np.zeros([self.no_a]).astype(np.float32)
        self.a_dy = np.zeros([self.no_a]).astype(np.float32)
        self.a_da = np.zeros([self.no_a]).astype(np.float32)
        self.a_Fx = np.zeros([self.no_a]).astype(np.float32)
        self.a_Fy = np.zeros([self.no_a]).astype(np.float32)
        self.a_Fa = np.zeros([self.no_a]).astype(np.float32)
        # Agent's energy and duration.
        self.a_e = np.zeros([self.no_a]).astype(np.float32)
        self.a_d = np.zeros([self.no_a]).astype(np.float32)


        # Set agents positions.
        no_a_x = 8
        no_a_y = int(self.no_a / no_a_x)
        for aX in range(no_a_x):
            for aY in range(no_a_y):
                a_id = aX + aY * no_a_x
                c_id1 = 2 * a_id
                c_id2 = 2 * a_id + 1
                self.a_x[a_id] = (aX + 1) * self.world_w / (no_a_x + 1)
                self.a_y[a_id] = (aY + 1) * self.world_h / (no_a_y + 1)
                self.c_x[c_id1] = (aX + 1) * self.world_w / (no_a_x + 1) + self.c_r[c_id1] + self.a_r[a_id] + 10
                self.c_y[c_id1] = (aY + 1) * self.world_h / (no_a_y + 1)
                self.c_x[c_id2] = (aX + 1) * self.world_w / (no_a_x + 1)
                self.c_y[c_id2] = (aY + 1) * self.world_h / (no_a_y + 1) + self.c_r[c_id2] + self.a_r[a_id] + 10
                
        self.a_R = np.zeros([4 * self.no_a]).astype(np.float32)
        self.a_G = np.zeros([4 * self.no_a]).astype(np.float32)
        self.a_B = np.zeros([4 * self.no_a]).astype(np.float32)
        # Initialize agent's textures.
        for a in range(self.no_a):
            # min
            self.a_R[4*a + 0] = 0.2
            self.a_G[4*a + 0] = 0.1
            self.a_B[4*a + 0] = 0.0
            # max
            self.a_R[4*a + 1] = 0.5
            self.a_G[4*a + 1] = 0.2
            self.a_B[4*a + 1] = 0.0
            # width min
            self.a_R[4*a + 2] = 2.0
            self.a_G[4*a + 2] = 5.0
            self.a_B[4*a + 2] = 0.0
            # width max
            self.a_R[4*a + 3] = 5.0
            self.a_G[4*a + 3] = 10.0
            self.a_B[4*a + 3] = 0.0
        self.a_lookat = 0.2 + 0.0 * np.pi * np.random.rand(self.no_a).astype(np.float32)
        # this is the actual orientation of the agent
        self.a_a = np.ones([self.no_a]).astype(np.float32) * np.pi / 4.0
        # this is FOV / 2
        self.a_fF = 0.2 * np.pi + 0.0 * np.random.rand(self.no_a).astype(np.float32)
        # DO NOT CHANGE a_pF, IT MUST BE FULLY PI
        self.a_pF = np.pi + 0.0 * np.random.rand(self.no_a).astype(np.float32)
        self.a_motor = 0.0 * np.random.rand(self.no_as).astype(np.float32)
        # allocate outputs
        self.a_ddx = np.zeros([self.no_a]).astype(np.float32)
        self.a_ddy = np.zeros([self.no_a]).astype(np.float32)
        self.a_dda = np.zeros([self.no_a]).astype(np.float32)
        self.a_sensor = np.zeros([self.no_ss * self.no_a]).astype(np.float32)
        self.f_R = np.zeros([self.no_a * self.no_f]).astype(np.float32)
        self.f_G = np.zeros([self.no_a * self.no_f]).astype(np.float32)
        self.f_B = np.zeros([self.no_a * self.no_f]).astype(np.float32)
        self.f_D = np.zeros([self.no_a * self.no_f]).astype(np.float32)
        self.p_R = np.zeros([self.no_a * self.no_p]).astype(np.float32)
        self.p_G = np.zeros([self.no_a * self.no_p]).astype(np.float32)
        self.p_B = np.zeros([self.no_a * self.no_p]).astype(np.float32)
        self.p_D = np.zeros([self.no_a * self.no_p]).astype(np.float32)
# =============================================================================
# =============================================================================

        # Some mouse related variables.
        self.LMB_click = False
        self.LMB_clicks = 0
        self.LMB_drag = None
        self.LMB_drag_type = None
        self.LMB_drag_inst = None
        self.LMB_hold = False
        self.LMB_origin = np.zeros([2,])
        self.LMB_last_click = 0

        self.steering = "pos"
        self.steering_rect = pg.Rect([0,0,2,2])
        self.steering_Fx = 0.0
        self.steering_Fy = 0.0
        self.steering_Fa = 0.0

        # Flag for color correction (blue).
        self.ccol = True
        
        # Agent of interest.
        self.aoi = 0

        # Initialize some rects for interactivity.
        self.agent_body_rect = []
        for a in range(self.no_a):
            self.agent_body_rect.append(pg.Rect([0,0,2,2]))
        self.circle_rect = []
        for c in range(self.no_c):
            self.circle_rect.append(pg.Rect([0,0,2,2]))



    def cc(self, col):
        """Method for pygame color correction.
        """
        if self.ccol:
            return (col[0] / 2, col[1], col[2])
        else:
            return (col[2], col[1], col[0])



    def update_screen(self):
        """This method updates the sim interface visualization.
        """
        # Catch events.
        for event in self.current_events:
            if event.type == pg.MOUSEBUTTONUP and event.button == 1:
                self.LMB_hold = False
                self.LMB_drag = None
                self.LMB_drag_type = None
                self.LMB_drag_inst = None
                if self.LMB_hold_origin == self.POS:
                    self.LMB_click = True
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                self.LMB_hold = True
                self.LMB_hold_origin = self.POS
                # Check for double click (2 x clicked in 300 ms).
                if 1000 * (time.time() - self.LMB_last_click) < 300:
                    self.LMB_clicks = 2
                else:
                    self.LMB_clicks = 1
                self.LMB_last_click = time.time()

        # Check for LMB click.
        if self.LMB_click:
            # Check for agent selection, update aoi.
            for a in range(self.no_a):
                if self.agent_body_rect[a].collidepoint(self.POS):
                    self.aoi = a
                    self.LMB_click = False
                    break
        if self.LMB_click:
            # Check for steering mode switch
            if self.steering_rect.collidepoint(self.POS):
                if self.steering == "pos":
                    self.steering = "force"
                else:
                    self.steering = "pos"
                self.LMB_click = False

        # Catch LMB drag.
        if self.LMB_hold:
            # Determine dragged object.
            if self.LMB_drag_type is None and self.POS == self.LMB_hold_origin:
                try:
                    # Check if agent dragged.
                    for a in range(self.no_a):
                        if self.agent_body_rect[a].collidepoint(self.POS):
                            self.LMB_drag_type = "agent"
                            self.LMB_drag_inst = a
                            raise
                    # Check if circle dragged.
                    for c in range(self.no_c):
                        if self.circle_rect[c].collidepoint(self.POS):
                            self.LMB_drag_type = "circle"
                            self.LMB_drag_inst = c
                            raise
                except:
                    pass

            if self.steering == 'pos':
                if self.LMB_drag_type == "agent":
                    # Transform from screen to agent coordinates.
                    self.a_x[self.LMB_drag_inst] = self.POS[0] - self.oX
                    self.a_y[self.LMB_drag_inst] = - self.POS[1] + self.oY + self.world_h
                elif self.LMB_drag_type == "circle":
                    # Drag a circle.
                    self.c_x[self.LMB_drag_inst] = self.POS[0] - self.oX
                    self.c_y[self.LMB_drag_inst] = - self.POS[1] + self.oY + self.world_h
            elif self.steering == 'force':
                if self.LMB_drag_type == "agent":
                    # Transform from screen to agent coordinates.
                    self.steering_Fx = self.POS[0] - self.oX - self.a_x[self.LMB_drag_inst]
                    self.steering_Fy = - self.POS[1] + self.oY + self.world_h - self.a_y[self.LMB_drag_inst]
                    self.steering_Fa = 0.0
                elif self.LMB_drag_type == "circle":
                    # Drag a circle.
                    self.steering_Fx = self.POS[0] - self.oX - self.c_x[self.LMB_drag_inst]
                    self.steering_Fy = - self.POS[1] + self.oY + self.world_h - self.c_y[self.LMB_drag_inst]
                    self.steering_Fa = 0.0



        # Switch between pos. / force steering.
        if self.steering == "pos":
            self.steering_rect = self.screen.blit(self.font.render("steering: pos", 
                                                                   1, 
                                                                   self.cc((255,255,255))), 
                                                  (10, 60))
        if self.steering == "force":
            self.steering_rect = self.screen.blit(self.font.render("steering: force", 
                                                                   1, 
                                                                   self.cc((255,255,255))), 
                                                  (10, 60))

        # Blit agent's dynamic information.
        loc_Y = 100
        loc_dY = 20
        loc_Y_cntr = 0
        self.screen.blit(self.font_small.render("glob. pose", 1, self.cc((255,255,255))), (10, loc_Y + loc_Y_cntr * loc_dY))
        self.screen.blit(self.font_small.render("    ({:1.1e}, {:1.1e}, {:1.1e})".format(self.a_x[self.aoi], 
                                                                                   self.a_y[self.aoi],
                                                                                   self.a_a[self.aoi]), 
                                          1, 
                                          self.cc((255,255,255))), (10, loc_Y + (loc_Y_cntr + 1) * loc_dY))
        loc_Y_cntr += 2
        self.screen.blit(self.font_small.render("glob. velocity", 1, self.cc((255,255,255))), (10, loc_Y + loc_Y_cntr * loc_dY))
        self.screen.blit(self.font_small.render("    ({:1.1e}, {:1.1e}, {:1.1e})".format(self.a_dx[self.aoi], 
                                                                                   self.a_dy[self.aoi],
                                                                                   self.a_da[self.aoi]), 
                                          1, 
                                          self.cc((255,255,255))), (10, loc_Y + (loc_Y_cntr + 1) * loc_dY))
        loc_Y_cntr += 2
        self.screen.blit(self.font_small.render("nn force", 1, self.cc((255,255,255))), (10, loc_Y + loc_Y_cntr * loc_dY))
        self.screen.blit(self.font_small.render("    ({:1.1e}, {:1.1e}, {:1.1e})".format(self.inputs['Fx'][self.aoi,0,0,0], 
                                                                                   self.inputs['Fy'][self.aoi,0,0,0],
                                                                                   self.inputs['Fa'][self.aoi,0,0,0]), 
                                          1, 
                                          self.cc((255,255,255))), (10, loc_Y + (loc_Y_cntr + 1) * loc_dY))
        loc_Y_cntr += 2
        self.screen.blit(self.font_small.render("energy {:1.1e}".format(self.a_e[self.aoi]), 
                                                1, 
                                                self.cc((255,255,255))), (10, loc_Y + loc_Y_cntr * loc_dY))
        self.screen.blit(self.font_small.render("duration {:1.1e}".format(self.a_d[self.aoi]), 
                                                1, 
                                                self.cc((255,255,255))), (10, loc_Y + (loc_Y_cntr + 1) * loc_dY))

        # Blit theoretical world boundaries.
        pg.draw.rect(self.screen, self.cc((153,153,153)), [self.oX, self.oY, self.world_w, self.world_h], 4)

        # Draw all line segments.
        for ls in range(self.no_ls):
            pg.draw.line(self.screen, self.cc((255,0,0)), [self.oX + self.ls_x1[ls], self.oY + self.world_h - self.ls_y1[ls]], 
                                                          [self.oX + self.ls_x2[ls], self.oY + self.world_h - self.ls_y2[ls]], 2)
        # Draw all mirrors.
        for m in range(self.no_m):
            pg.draw.line(self.screen, self.cc((255,255,0)), [self.oX + self.m_x1[m], self.oY + self.world_h - self.m_y1[m]], 
                                                          [self.oX + self.m_x2[m], self.oY + self.world_h  - self.m_y2[m]], 2)
        # Draw aoi-th agents peripherie.
        angle_inc = 2 * self.a_pF[int(self.aoi)] / self.no_p
        for r in range(int(self.no_p / 4)):
            x = self.oX + self.a_x[int(self.aoi)]
            y = self.oY + self.world_h  - self.a_y[int(self.aoi)]
            angle = self.a_a[int(self.aoi)] + self.a_lookat[int(self.aoi)] \
                    - self.a_pF[int(self.aoi)] + 4 * r * angle_inc
            ax = np.cos(angle)
            ay = np.sin(angle)
            d = self.p_D[self.no_p * self.aoi + 4 * r]
            pg.draw.line(self.screen, self.cc((127,127,127)), [int(x), int(y)],
                                                              [int(x + d * ax), int(y - d * ay)], 1)

        # Draw aoi-th agents retina.
        angle_inc = 2 * self.a_fF[int(self.aoi)] / self.no_f
        for r in range(int(self.no_f / 4)):
            x = self.oX + self.a_x[int(self.aoi)]
            y = self.oY + self.world_h  - self.a_y[int(self.aoi)]
            angle = self.a_a[int(self.aoi)] + self.a_lookat[int(self.aoi)] \
                    - self.a_fF[int(self.aoi)] + 4 * r * angle_inc
            ax = np.cos(angle)
            ay = np.sin(angle)
            d = self.f_D[self.no_f * self.aoi + 4 * r]
            pg.draw.line(self.screen, self.cc((127,255,127)), [int(x), int(y)],
                                                              [int(x + d * ax), int(y - d * ay)], 1)

        # Draw all agents.
        for a in range(self.no_a):
            # Fill circle for AOI.
            if a == self.aoi:
                filled = 0
            else:
                filled = 2
            # Agent's body.
            self.agent_body_rect[a] = pg.draw.circle(self.screen, 
                                        self.cc((0,0,255)), 
                                        (int(self.oX + self.a_x[a]), int(self.oY + self.world_h  - self.a_y[a])), 
                                        int(self.a_r[a]), 
                                        filled)
            # Agent's orientation.
            off_x = 2 * np.cos(self.a_a[a]) * self.a_r[a]
            off_y = 2 * np.sin(self.a_a[a]) * self.a_r[a]
            pg.draw.line(self.screen, 
                         self.cc((0,0,255)), 
                         [int(self.oX + self.a_x[a]), int(self.oY + self.world_h  - self.a_y[a])],
                         [int(self.oX + self.a_x[a] + off_x), int(self.oY + self.world_h  - self.a_y[a] - off_y)],
                         2)
            # Agent's lookat.
            off_x = 2 * np.cos(self.a_a[a] + self.a_lookat[a]) * self.a_r[a]
            off_y = 2 * np.sin(self.a_a[a] + self.a_lookat[a]) * self.a_r[a]
            pg.draw.line(self.screen, 
                         self.cc((255,255,255)), 
                         [int(self.oX + self.a_x[a]), int(self.oY + self.world_h  - self.a_y[a])],
                         [int(self.oX + self.a_x[a] + off_x), int(self.oY + self.world_h  - self.a_y[a] - off_y)],
                         2)

        # Draw all circles.
        for c in range(self.no_c):
            self.circle_rect[c] = pg.draw.circle(self.screen, 
                               self.cc((0,255,255)), 
                               (int(self.oX + self.c_x[c]), 
                                int(self.oY  + self.world_h - self.c_y[c])), 
                               int(self.c_r[c]), 
                               2)




    def update_frame_writeout(self):
        # Reset forces onto circles and agents.
        self.c_Fx *= 0
        self.c_Fy *= 0
        self.c_Fa *= 0
        self.a_Fx *= 0
        self.a_Fy *= 0
        self.a_Fa *= 0
        # Set agent's forces and convert them into global (= world) frame.
        self.a_Fa[:] = self.inputs['Fa'][:,0,0,0]
        self.a_Fx[:] = (np.cos(self.a_a) * self.inputs['Fx'][:,0,0,0] \
                     - np.sin(self.a_a) * self.inputs['Fy'][:,0,0,0]).astype(np.float32)
        self.a_Fy[:] = (np.sin(self.a_a) * self.inputs['Fx'][:,0,0,0] \
                     + np.cos(self.a_a) * self.inputs['Fy'][:,0,0,0]).astype(np.float32)


        # Update agent's duration.
        self.a_d = self.a_d \
                   + self.a_e * 0.1 \
                   - (np.abs(self.a_Fx) + np.abs(self.a_Fy) + np.abs(self.a_Fa))
        self.a_d = np.clip(self.a_d, 0, 100)

        # Update agent's energy.
        self.a_e = self.a_e - 0.01 \
                   - (np.abs(self.a_Fx) + np.abs(self.a_Fy) + np.abs(self.a_Fa)) * 0.01
        self.a_e = np.clip(self.a_e, 0, 100)


        # Overwrite forces for dragged item.
        if self.LMB_hold and self.steering == 'force':
            if self.LMB_drag_type == 'agent':
                a = self.LMB_drag_inst
                self.a_Fx[a] = self.steering_Fx
                self.a_Fy[a] = self.steering_Fy
                self.a_Fa[a] = self.steering_Fa
            elif self.LMB_drag_type == 'circle':
                self.c_Fx[self.LMB_drag_inst] = self.steering_Fx
                self.c_Fy[self.LMB_drag_inst] = self.steering_Fy
                self.c_Fa[self.LMB_drag_inst] = self.steering_Fa


        # debug: list all dtypes
#                for p in self.debug_world_vars:
#                    print "dtype of " + p + " : " + str(eval('self.'+p+'.dtype'))
        # compute world2d update
        csim_func(self.world_params,
                  self.ls_x1, self.ls_y1, self.ls_x2, self.ls_y2,
                  self.ls_R, self.ls_G, self.ls_B,
                  self.m_x1, self.m_y1, self.m_x2, self.m_y2,
                  self.c_x, self.c_y, self.c_r, self.c_a,
                  self.c_dx, self.c_dy, self.c_da,
                  self.c_Fx, self.c_Fy, self.c_Fa,
                  self.c_R, self.c_G, self.c_B,
                  self.a_x, self.a_y, self.a_a, self.a_r,
                  self.a_dx, self.a_dy, self.a_da,
                  self.a_Fx, self.a_Fy, self.a_Fa,
                  self.a_R, self.a_G, self.a_B,
                  self.a_lookat, self.a_fF, self.a_pF,
                  self.a_motor,
                  self.no_f, self.no_p,
                  self.a_ddx, self.a_ddy, self.a_dda,
                  self.a_sensor,
                  self.f_R, self.f_G, self.f_B, self.f_D,
                  self.p_R, self.p_G, self.p_B, self.p_D)

    
        # write update for np state to shared memory
        if 'ret_fov' in self.p['out']:
            inter = np.swapaxes(np.stack([self.f_R.reshape([self.no_a, -1]),
                                          self.f_G.reshape([self.no_a, -1]),
                                          self.f_B.reshape([self.no_a, -1])]).astype(np.float32),
                                                0, 1)[:,:,:,np.newaxis]
#                    print "WORLD RET SHAPE: " + str(inter.shape)
#                    print "2DW PER HIST: " + str(np.histogram(inter.flatten(), bins=10))
            self.dat["variables"]['ret_fov'][:,:,:,:] = inter[:,:,:,:] 
        if 'ret_per' in self.p['out']:
            inter = np.swapaxes(np.stack([self.p_R.reshape([self.no_a, -1]),
                                          self.p_G.reshape([self.no_a, -1]),
                                          self.p_B.reshape([self.no_a, -1])]).astype(np.float32),
                                                0, 1)[:,:,:,np.newaxis]
            self.dat["variables"]['ret_per'][:,:,:,:] = inter[:,:,:,:]
        if 'dist_fov' in self.p['out']:
            inter = np.swapaxes(self.f_D.reshape([self.no_a, -1])[np.newaxis,:,:,np.newaxis].astype(np.float32), 0, 1)
            self.dat["variables"]['dist_fov'][:,:,:,:] = inter[:,:,:,:]
        if 'dist_per' in self.p['out']:
            inter = np.swapaxes(self.p_D.reshape([self.no_a, -1])[np.newaxis,:,:,np.newaxis].astype(np.float32), 0, 1)
            self.dat["variables"]['dist_per'][:,:,:,:] = inter[:,:,:,:]
        if 'acc_x' in self.p['out']:
            for a in range(self.net['agents']):
                self.dat["variables"]["acc_x"][a,:,:,:] \
                    = float_2_np(self.p['out_par']['acc_x'], 
                                 self.a_ddx[a], 
                                 self.dat_layout["variables"]["acc_x"].shape[1:])
        if 'acc_y' in self.p['out']:
            for a in range(self.net['agents']):
                self.dat["variables"]["acc_y"][a,:,:,:] \
                    = float_2_np(self.p['out_par']['acc_y'], 
                                 self.a_ddy[a], 
                                 self.dat_layout["variables"]["acc_y"].shape[1:])
        if 'acc_a' in self.p['out']:
            for a in range(self.net['agents']):
                self.dat["variables"]["acc_a"][a,:,:,:] \
                    = float_2_np(self.p['out_par']['acc_a'], 
                                 self.a_dda[a], 
                                 self.dat_layout["variables"]["acc_a"].shape[1:])
        if 'FOV_fov' in self.p['out']:
            # TODO
            pass
        if 'lookat' in self.p['out']:
            for a in range(self.net['agents']):
                self.dat["variables"]["lookat"][a,:,:,:] \
                    = float_2_np(self.p['out_par']['lookat'], 
                                 self.a_lookat[a], 
                                 self.dat_layout["variables"]["lookat"].shape[1:])
        if 'haptic' in self.p['out']:
            inter = self.a_sensor.reshape([self.no_a, -1]).astype(np.float32)
            self.dat["variables"]["haptic"][:,:,0,0] = inter[:,:]
            
            



