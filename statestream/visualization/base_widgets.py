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



from statestream.utils.pygame_import import pg, pggfx

import numpy as np
import copy

from statestream.visualization.graphics import cc, num2str



# =============================================================================
# =============================================================================
# =============================================================================

class Collector(object):
    def __init__(self, screen, vparam, sprites, fonts, ccol):
        """Collector class for all widgets.

        Attributes
        ----------
        
        """
        # Set references.
        self.screen = screen
        self.sprites = sprites
        self.fonts = fonts
        self.ccol = ccol
        self.vparam = vparam
        # List of all widgets.
        self.widgets = []
        # List of all buttons.
        self.buttons = []
        # List of all text entries.
        self.textentries = []
        # List of all frames.
        self.frames = []
        # Flag for active textentry widget.
        self.active_textentry = None
        # Very on top widget.
        self.top_widget = None
        
    def destroy(self, widget):
        """Method to destroy a widget and all its children.
        """
        if widget is not None:
            # Recursively remove children.
            while len(widget.children) != 0:
                self.destroy(widget.children[0])
            # Remove this from parent"s child list.
            if widget.parent is not None:
                widget.parent.children.remove(widget)
            # Remove from widget type list.
            if widget.type == "Button":
                self.buttons.remove(widget)
            if widget.type == "Frame":
                self.frames.remove(widget)
            if widget.type == "TextEntry":
                self.textentries.remove(widget)
            # Remove from widgets list.
            try:
                self.widgets.remove(widget)
            except:
                print("Error: Unable to remove widget of type: " \
                      + str(widget.type) + "   children: " \
                      + str(len(widget.children)))
            # Remove top pointer.
            if self.top_widget == widget:
                self.top_widget = None

    def on_top(self, widget):
        """Put widget on top of drawing stack.
        """
        # Destroy current on-top widget.
        if self.top_widget is not None:
            self.destroy(self.top_widget)
        # Put given widget on top.
        self.top_widget = widget

    def link(self, w0, w1):
        """Links the values of same type for two widgets.
        """
        # Assert both widgets have the same value type.
        assert (w0.value_type == w1.value_type), "Warning: Unable to link widgets of different value type."
        # Link to all linked widgets.
        if w0 != w1:
            w0.links.append(w1)
            w1.links.append(w0)
            for w in w0.links:
                if w not in w1.links:
                    w1.links.append(w)
                if w1 not in w.links:
                    w.links.append(w1)
            for w in w1.links:
                if w not in w0.links:
                    w0.links.append(w)
                if w0 not in w.links:
                    w.links.append(w0)
        
# =============================================================================
        
    def add_button(self,
                   parent=None, 
                   text=None, 
                   text_info=None,
                   sprite=None, 
                   value=0,
                   font="small",
                   pos=np.zeros([2,], dtype=np.float32), 
                   cb_LMB_clicked=None,
                   cb_RMB_clicked=None,
                   cb_LMB_drag=None,
                   cb_over=None):
        """Method to add a button widget to the collector.
        """
        b = Button(parent=parent, 
                   text=text, 
                   text_info=text_info,
                   sprite=sprite, 
                   value=value,
                   font=font,
                   pos=pos, 
                   cb_LMB_clicked=cb_LMB_clicked,
                   cb_RMB_clicked=cb_RMB_clicked,
                   cb_LMB_drag=cb_LMB_drag,
                   cb_over=cb_over)
        self.buttons.append(b)
        self.widgets.append(b)
        return b

    def add_textentry(self, 
                      parent=None, 
                      text="", 
                      value_type="string",
                      value=None,
                      value_min=None,
                      value_max=None,
                      font="small",
                      pos=np.zeros([2,], dtype=np.float32), 
                      cb_leave=None):
        """Method to add a textentry widget to the collector.
        """
        t = TextEntry(parent=parent, 
                      text=text, 
                      value_type=value_type,
                      value=value,
                      value_min=value_min,
                      value_max=value_max,
                      font=font,
                      pos=pos,
                      cb_leave=cb_leave)
        self.textentries.append(t)
        self.widgets.append(t)
        return t
        
    def add_frame(self,
                  parent=None,
                  title="",
                  border_width=4,
                  font="small",
                  pos=np.zeros([2,], dtype=np.float32),
                  cb_leave=None):
        """Method to add a frame to the collector.
        """
        f = Frame(parent=parent,
                  title=title,
                  border_width=border_width,
                  font=font,
                  pos=pos,
                  cb_leave=cb_leave)
        self.frames.append(f)
        self.widgets.append(f)
        return f

# =============================================================================

    def LMB_clicked(self, pos):
        """Checks if a widget was clicked and executes callback.
        """
        LMB_click_active = True
        # Check if to close on-top widget (click outside top widget).
        if self.top_widget is not None:
            if not self.top_widget.rect.collidepoint(pos):
                self.destroy(self.top_widget)
        # All buttons.
        for b in self.buttons:
            if b.cb_LMB_clicked is not None:
                if b.rect.collidepoint(pos):
                    # For (multi)button go to next sprite.
                    b.value += 1
                    if b.value > b.value_max:
                        b.value = b.value_min
                    b.cb_LMB_clicked()
                    LMB_click_active = False
                    break
        # All text entries.
        for t in self.textentries:
            if t.rect.collidepoint(pos):
                if self.active_textentry is not None:
                    # The one TE that is already active should be closed.
                    self.active_textentry.is_active = False
                    self.active_textentry.entry2value()
                    self.active_textentry.value2entry()
                    if self.active_textentry.cb_leave is not None:
                        self.active_textentry.cb_leave()
                    self.active_textentry = None
                # Now t becomes the active TE.
                t.is_active = True
                t.entry = ""
                self.active_textentry = t
                # Set back mouse click.
                LMB_click_active = False
                break

        # Return if click was considered.
        return LMB_click_active

    def RMB_clicked(self, pos):
        """Checks if a widget was clicked and executes callback.
        """
        RMB_click_active = True
        # Check if to close on-top widget (click outside top widget).
        if self.top_widget is not None:
            if not self.top_widget.rect.collidepoint(pos):
                self.destroy(self.top_widget)
        # Check if RMB clicked for buttons
        for b in self.buttons:
            if b.cb_RMB_clicked is not None:
                if b.rect.collidepoint(pos):
                    b.cb_RMB_clicked()
                    RMB_click_active = False
                    break
        # Return if click was considered.
        return RMB_click_active
    
    def mouse_over(self, pos):
        """Check if mouse is over widget and execute callback.
        """
        for b in self.buttons:
            if b.cb_over is not None:
                if b.rect.collidepoint(pos):
                    b.cb_over()
        
    def key(self, key, shift, uc):
        """Let callbacks process the key event.
        """
        key_active = True
        # Check if there is an active textentry.
        if self.active_textentry is not None:
            # Check for tabulator in parameter specification widget.
            if uc == u"\u0009" and self.active_textentry.parent is not None:
                if self.active_textentry.parent.type == "Frame" \
                        and self.active_textentry.parent.parent is not None:
                    if self.active_textentry.parent.parent.type == "ParamSpecWindow":
                        # Finish current entry.
                        self.active_textentry.is_active = False
                        self.active_textentry.entry2value()
                        self.active_textentry.value2entry()
                        if self.active_textentry.cb_leave is not None:
                            self.active_textentry.cb_leave()
                        # Get index of text entry.
                        idx = self.active_textentry.parent.parent.param_entries.index(self.active_textentry)
                        # Dependent of last textentry or in the middle.
                        if idx == len(self.active_textentry.parent.parent.param_entries) - 1:
                            # Close parameter selection and return parameter.
                            self.active_textentry.parent.parent.cb_done_clicked()
                        else:
                            # Set next textentry to be the active one.
                            self.active_textentry.parent.parent.param_entries[idx + 1].is_active = True
                            self.active_textentry.parent.parent.param_entries[idx + 1].entry = ""
                            self.active_textentry = self.active_textentry.parent.parent.param_entries[idx + 1]
            # Execute callback, when leaving textentry.
            elif key == pg.K_RETURN:
                self.active_textentry.is_active = False
                self.active_textentry.entry2value()
                self.active_textentry.value2entry()
                if self.active_textentry.cb_leave is not None:
                    self.active_textentry.cb_leave()
                self.active_textentry = None
            elif key == pg.K_ESCAPE:
                self.active_textentry.is_active = False
                self.active_textentry.value2entry()
                self.active_textentry.entry2value()
                self.active_textentry = None
            elif key == pg.K_BACKSPACE:
                if len(self.active_textentry.entry) > 0:
                    self.active_textentry.entry \
                        = self.active_textentry.entry[0:-1]
            else:
                # Add unicode.
                self.active_textentry.entry += str(uc)
            key_active = False
        else:
            # Check if top widget to be ended.
            if self.top_widget is not None:
                if self.top_widget.type == "ParamSpecWindow":
                    if key == pg.K_RETURN:
                        self.top_widget.cb_done_clicked()
                    elif key == pg.K_ESCAPE:
                        self.destroy(self.top_widget)
                        self.top_widget = None
                elif self.top_widget.type == "ListSelectionWindow":
                    if key == pg.K_ESCAPE:
                        self.destroy(self.top_widget)
                        self.top_widget = None

        return key_active
        
    def draw(self):
        """Method recursively draws all attached widgets.
        """
        # Recursively draw all widgets.
        for w in self.widgets:
            if w.parent is None and w != self.top_widget:
                w.draw(self.screen, 
                       self.vparam, 
                       self.sprites, 
                       self.fonts, 
                       self.ccol)

    def draw_top(self):
        """Method to draw only the top widget.
        """
        if self.top_widget is not None:
            self.top_widget.draw(self.screen, 
                                 self.vparam, 
                                 self.sprites, 
                                 self.fonts, 
                                 self.ccol)

# =============================================================================
# =============================================================================
# =============================================================================

class Widget(object):
    def __init__(self, 
                 parent=None,
                 pos=np.zeros([2,], dtype=np.float32)):
        """Base class for all widgets.
        
            parent: pointer to parent widget
            pos:    relative coordinate to parent widget
        """
        # Copy all widget type specific data.
        self.parent = parent
        self.pos = np.copy(pos)
        # Begin with empty child list.
        self.children = []
        # Add self to list of parent"s childs.
        if parent is not None:
            parent.children.append(self)
        
        # Some members to organize internal values.
        self.value = None
        self.value_type = None
        self.value_min = None
        self.value_max = None
        self.links = [self]
        self.type = None

    def global_pos(self):
        """Returns the recursively computed global position of this widget.
        """
        if self.parent is None:
            return self.pos
        else:
            return self.pos + self.parent.global_pos()

# =============================================================================
# =============================================================================
# =============================================================================

class Button(Widget):
    def __init__(self, 
                 parent=None, 
                 value=0,
                 text=None, 
                 text_info=None,
                 sprite=None, 
                 font="small",
                 pos=np.zeros([2,], dtype=np.float32), 
                 cb_LMB_clicked=None,
                 cb_RMB_clicked=None,
                 cb_LMB_drag=None,
                 cb_over=None):
        """Class for button widget.

            rect: Rectangle in global coordinates.
        """
        # Init parent class.
        Widget.__init__(self, parent=parent, pos=pos)
        
        # Copy all widget type specific data.
        self.text = copy.copy(text)
        self.text_info = copy.copy(text_info)
        self.sprite = copy.copy(sprite)
        self.font = copy.copy(font)
        self.cb_LMB_clicked = cb_LMB_clicked
        self.cb_RMB_clicked = cb_RMB_clicked
        self.cb_LMB_drag = cb_LMB_drag
        self.cb_over = cb_over
        self.rect = pg.Rect(0, 0, 2, 2)
        self.value = value
        self.type = "Button"
        
        # Initially begin with 0-th. button.
        self.value_type = "int"
        self.value_min = 0
        # Determine "lenght" of button.
        if self.sprite is not None:
            if isinstance(self.sprite, list):
                self.value_max = len(self.sprite) - 1
            else:
                self.value_max = 0
        if self.text is not None:
            if isinstance(self.text, list):
                self.value_max = len(self.text) - 1
            else:
                self.value_max = 0
                
    def get_size(self, screen, vparam, sprites, fonts):
        """Determine size of the text and entry in pixels.
        """
        if self.sprite is not None:
            # All sprites are assumed to have the same size.
            if self.value_max > 0:
                size = [sprites[self.sprite[int(self.value)]].get_width(),
                        sprites[self.sprite[int(self.value)]].get_height()]
            else:
                size = [sprites[self.sprite].get_width(),
                        sprites[self.sprite].get_height()]
            width = size[0]
            height = size[1]
        else:
            # Render text.
            if self.value_max > 0:
                font = fonts[self.font].render(self.text[int(self.value)], 1, (0,0,0))
                if self.text_info is not None:
                    font_info = fonts[self.font].render(self.text_info[int(self.value)], 
                                                        1, 
                                                        (0,0,0))
            else:
                font = fonts[self.font].render(self.text, 1, (0,0,0))
                if self.text_info is not None:
                    font_info = fonts[self.font].render(self.text_info, 1, (0,0,0))
            if self.text_info is not None:
                width = font.get_width() + font_info.get_width()
            else:
                width = font.get_width()
            height = vparam["font " + self.font]
        # Return size.
        return np.array([width, height], dtype=np.float32)

    def draw(self, screen, vparam, sprites, fonts, ccol):
        """Method to draw this widget onto screen.
        """
        # Determine global position.
        pos = self.global_pos()
        if self.sprite is not None:
            # Blit sprite button.
            if self.value_max > 0:
                self.rect = screen.blit(sprites[self.sprite[int(self.value)]], 
                                        (int(pos[0]), int(pos[1])))
            else:
                self.rect = screen.blit(sprites[self.sprite], 
                                        (int(pos[0]), int(pos[1])))
        else:
            # Render text.
            if self.value_max > 0:
                font = fonts[self.font].render(self.text[int(self.value)], 
                                               1, 
                                               cc(vparam["background_color"], 
                                                ccol))
                if self.text_info is not None:
                    font_info = fonts[self.font].render(self.text_info[int(self.value)], 
                                                        1, 
                                                        cc(vparam["number_color"], ccol))
            else:
                font = fonts[self.font].render(self.text, 
                                               1, 
                                               cc(vparam["background_color"], ccol))
                if self.text_info is not None:
                    font_info = fonts[self.font].render(self.text_info, 
                                                        1, 
                                                        cc(vparam["number_color"], ccol))
            if self.text_info is not None:
                width = font.get_width() + font_info.get_width()
            else:
                width = font.get_width()
            height = vparam["font " + self.font]
            # Draw (text) button background.
            self.rect = pg.draw.rect(screen, 
                                     cc(vparam["text_color"], ccol), 
                                     [int(pos[0]), int(pos[1]), int(width), int(height)], 
                                     0)
            # Draw text.
            screen.blit(font, (int(pos[0]), int(pos[1])))
            if self.text_info is not None:
                screen.blit(font_info, (int(pos[0] + font.get_width()), int(pos[1])))

# =============================================================================
# =============================================================================
# =============================================================================

class TextEntry(Widget):
    def __init__(self, 
                 parent=None, 
                 text="", 
                 value_type="string",
                 value=None,
                 value_min=None,
                 value_max=None,
                 font="small",
                 pos=np.zeros([2,], dtype=np.float32), 
                 cb_leave=None):
        """Class for textbox widget.

            rect: Rectangle in global coordinates.
        """
        # Init parent class.
        Widget.__init__(self, parent=parent, pos=pos)
        # Begin with empty entry.
        self.value = value
        self.value_type = value_type
        self.value_min = value_min
        self.value_max = value_max
        # Copy all widget type specific data.
        self.text = copy.copy(text)
        self.font = copy.copy(font)
        self.cb_leave = cb_leave
        self.rect = pg.Rect(0, 0, 2, 2)
        self.type = "TextEntry"

        # Initially set str entry dependent on value type.
        if self.value_type == "int":
            self.entry = str(int(self.value))
        elif self.value_type == "float":
            self.entry = num2str(float(self.value))
        elif self.value_type == "string":
            self.entry = self.value

        # Initially a text entry is not active.
        self.is_active = False
        
    def value2entry(self):
        """Determine string entry dependent on value and its type.
        """
        if self.value_type == "int":
            self.entry = str(int(self.value))
        elif self.value_type == "float":
            self.entry = num2str(self.value)
        elif self.value_type == "string":
            self.entry = self.value

    def entry2value(self):
        """Convert a string entry to a value dependent on value type.
        """
        if self.value_type == "int":
            try:
                self.value = int(float(self.entry))
            except:
                self.value = self.value_min
        elif self.value_type == "float":
            try:
                self.value = float(self.entry)
            except:
                self.value = self.value_min
        elif self.value_type == "string":
            self.value = self.entry
        
    def get_size(self, screen, vparam, sprites, fonts):
        """Determine size of the text and entry in pixels.
        """
        if self.is_active:
            font = fonts[self.font].render(self.text + " " + self.entry + "|", 1, (0,0,0))
        else:
            font = fonts[self.font].render(self.text + " " + self.entry, 1, (0,0,0))
        return np.array([font.get_width(), font.get_height()], dtype=np.float32)

    def draw(self, screen, vparam, sprites, fonts, ccol):
        """Method to draw this widget onto screen.
        """
        # Determine global position.
        pos = self.global_pos()
        # Determine text color (on background -> text color, in window: bg color).
        if self.parent is None:
            text_col = vparam["text_color"]
        else:
            text_col = vparam["background_color"]
        # Blit text entry.
        if self.is_active:
            self.rect = screen.blit(fonts[self.font].render(self.text + " " + self.entry + "|", 
                                    1, 
                                    cc(text_col, ccol)),
                                    (int(pos[0]), int(pos[1])))
        else:
            self.rect = screen.blit(fonts[self.font].render(self.text + " " + self.entry, 
                                    1, 
                                    cc(text_col, ccol)),
                                    (int(pos[0]), int(pos[1])))

# =============================================================================
# =============================================================================
# =============================================================================

class Frame(Widget):
    def __init__(self, 
                 parent=None, 
                 title="", 
                 border_width=4,
                 font="small",
                 pos=np.zeros([2,], dtype=np.float32), 
                 cb_leave=None):
        """Class for frame widget to organize other widgets.

            rect: Rectangle in global coordinates.
        """
        # Init parent class.
        Widget.__init__(self, parent=parent, pos=pos)
        
        # Frame dimensions in pixels.
        self.width = 0
        self.height = 0
        self.border_width = border_width
        self.type = "Frame"

    def get_size(self, screen, vparam, sprites, fonts):
        """Determine frame size dependent on its children.
        """
        # Get maximal width / height considering all children.
        max_width = 0
        max_height = 0
        for c in self.children:
            size = c.get_size(screen, vparam, sprites, fonts)
            if max_width < size[0] + c.pos[0]:
                max_width = size[0] + c.pos[0]
            if max_height < size[1] + c.pos[1]:
                max_height = size[1] + c.pos[1]
        # Also set size.
        self.width = max_width
        self.height = max_height
        # Return (max) size.
        return np.array([max_width, max_height], dtype=np.float32)
        
    def draw(self, screen, vparam, sprites, fonts, ccol):
        """Method to draw this widget onto screen.
        """
        # Determine global position.
        pos = self.global_pos()
        # Draw some borders.
        pg.draw.rect(screen, cc(vparam["text_color"], ccol), [int(pos[0] - self.border_width), 
                                                              int(pos[1] - self.border_width),
                                                              int(self.width + 2 * self.border_width), 
                                                              int(self.height + 2 * self.border_width)], 2)
        # Draw Frame.
        self.rect = pg.draw.rect(screen, cc(vparam["text_color"], ccol), [int(pos[0]), int(pos[1]), int(self.width), int(self.height)], 0)

        # Draw all children.
        for c in self.children:
            c.draw(screen, vparam, sprites, fonts, ccol)
