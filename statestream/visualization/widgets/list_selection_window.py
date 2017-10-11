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

from statestream.utils.pygame_import import pg, pggfx
from statestream.visualization.base_widgets import Widget



# =============================================================================
# =============================================================================
# =============================================================================

class ListSelectionWindow(Widget):
    def __init__(self, 
                 parent=None, 
                 wcollector=None,
                 font='small',
                 pos=np.zeros([2,], dtype=np.float32), 
                 selections=[],
                 selections_info=[],
                 source=None,
                 modus=None,
                 cb_LMB_clicked=None,
                 cb_over=None):
        """Class for list selection window widget.

            rect: Rectangle in global coordinates.
        """
        # Init parent class.
        Widget.__init__(self, parent=parent, pos=pos)

        # Copy initial data.
        self.wcollector = wcollector
        self.selections = selections
        self.selections_info = selections_info
        self.source = source
        self.modus = modus
        self.cb_LMB_clicked = cb_LMB_clicked
        self.cb_over = cb_over
        self.button_height = wcollector.vparam['font ' + font]
        self.type = 'ListSelectionWindow'

        # Create a frame widget as direct child.
        self.frame = wcollector.add_frame(parent=self,
                                          pos=np.array([0, 0], 
                                                       dtype=np.float32))
        
        # Create a button for each selection as childs of the frame.
        self.buttons = []
        # Callback for over is optionally.
        for s in range(len(self.selections)):
            self.buttons.append(wcollector.add_button(parent=self.frame, 
                                                      text=self.selections[s],
                                                      text_info=self.selections_info[s],
                                                      pos=np.array([0, s * self.button_height], 
                                                                   dtype=np.float32),
                                                      cb_LMB_clicked=lambda x=s: self.cb_button_clicked(x),
                                                      cb_over=lambda x=s: self.cb_button_over(x)))
                                                      
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
        
    def cb_button_clicked(self, button_id):
        """This is the widget internal callback for a button press.
        """
        # Destroy the list selection window and all its children.
        self.wcollector.destroy(self)
        # Execute the widgets LMB click callback.
        self.cb_LMB_clicked(self.source, 
                            self.modus, 
                            self.selections, 
                            button_id)

    def cb_button_over(self, button_id):
        """This is the widget internal callback for mouse over.
        """
        # Execute the widgets over callback.
        if self.cb_over is not None:
            self.cb_over(self.source, self.modus, self.selections, button_id)
        
    def draw(self, screen, vparam, sprites, fonts, ccol):
        """Method to draw this widget onto screen.
        """
        # Blit frame widget (buttons are children of frame and will be drawn there).
        self.frame.draw(screen, vparam, sprites, fonts, ccol)

# =============================================================================
# =============================================================================
# =============================================================================
    