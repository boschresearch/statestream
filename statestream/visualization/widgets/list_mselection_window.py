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
import copy

from statestream.visualization.base_widgets import Widget
from statestream.utils.pygame_import import pg, pggfx



# =============================================================================
# =============================================================================
# =============================================================================

class ListMSelWindow(Widget):
    def __init__(self,
                 parent,
                 wcollector,
                 font="small",
                 pos=np.zeros([2,], dtype=np.float32),
                 parameter=[],
                 value=[],
                 dvalue=None,
                 ptype=None,
                 modus=None,
                 cb_LMB_clicked=None):
        """Class for multiple item selection window widget.

            rect: Rectangle in global coordinates.
        """
        # Init parent class.
        Widget.__init__(self, parent=parent, pos=pos)

        # Copy initial data.
        self.wcollector = wcollector
        self.parameter = parameter
        self.value = value
        self.ptype = ptype
        self.modus = modus
        self.cb_LMB_clicked = cb_LMB_clicked
        self.entry_height = wcollector.vparam["font " + font]
        self.type = "ListMSelWindow"

        # Determine maximum entry length over list.
        self.max_len = 0
        for i in self.parameter:
            if self.max_len < len(i):
                self.max_len = len(i)

        # Initialize all values with their default.
        self.current_value = [dvalue for v in range(len(parameter))]

        # Create a frame widget as direct child.
        self.frame = wcollector.add_frame(parent=self,
                                          pos=np.array([0, 0], dtype=np.float32))
        # Go through list of parameters and add widgets.
        self.param_entries = []
        for p in range(len(self.parameter)):
            text = " " + str(self.parameter[p]).ljust(self.max_len + 2) + ": " \
                   + str(self.current_value[p])
            self.param_entries.append(wcollector.add_button(parent=self.frame,
                                     pos=np.array([0, p * self.entry_height], 
                                                  dtype=np.float32),
                                     text=text,
                                     font="small",
                                     cb_LMB_clicked=lambda x=p: self.cb_item_clicked(x)))
        # Add the "Done" button.
        self.done = wcollector.add_button(parent=self.frame,
                                          pos=np.array([0, (1 + len(self.parameter)) \
                                                            * self.entry_height], 
                                                        dtype=np.float32),
                                          text=" done ",
                                          cb_LMB_clicked=self.cb_done_clicked)
                                                      
        # Determine and save size of frame.
        self.frame.get_size(self.wcollector.screen, 
                            self.wcollector.vparam, 
                            self.wcollector.sprites, 
                            self.wcollector.fonts)

        # Rectangle for this widget (same as the frame).
        self.rect = pg.Rect(self.pos[0], 
                            self.pos[1], 
                            self.frame.width, 
                            self.frame.height)
        
        # Finally add this to the collectors widget list.
        wcollector.widgets.append(self)
        wcollector.on_top(self)
        
    def cb_done_clicked(self):
        """This is the widget internal callback for a button press.
        """
        # Destroy the list selection window and all its children.
        self.wcollector.destroy(self)
        # Execute the widgets LMB click callback.
        self.cb_LMB_clicked(modus=self.modus, 
                            ptype=self.ptype, 
                            param=self.parameter,
                            value=self.current_value)

    def cb_item_clicked(self, p):
        """This handles the click on an list item.
        """
        # Count current value up by 1.
        next_idx = (self.value.index(self.current_value[p]) + 1) % len(self.value)
        self.current_value[p] = self.value[next_idx]
        # Create new text.
        text = " " + str(self.parameter[p]).ljust(self.max_len + 2) + ": " \
               + str(self.current_value[p])
        # Update text for drawing.
        self.param_entries[p].text = copy.copy(text)

    def draw(self, screen, vparam, sprites, fonts, ccol):
        """Method to draw this widget onto screen.
        """
        # Blit frame widget (buttons are children of frame and will be drawn there).
        self.frame.draw(screen, vparam, sprites, fonts, ccol)

# =============================================================================
# =============================================================================
# =============================================================================
    