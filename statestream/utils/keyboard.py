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



import sys
import termios
import atexit
from select import select



__all__ = [
    'Keyboard'
]

class Keyboard(object):
    """The :class:Keyboard provides real-time keyboard handling.
    """
    def __init__(self):
        # Get file identifier for stdin and get terminal.
        self.fid = sys.stdin.fileno()
        self.terminal = termios.tcgetattr(self.fid)
        # Backup the terminal settings.
        self.backup = termios.tcgetattr(self.fid)
        # Create a new unbuffered terminal.
        self.terminal[3] = (self.terminal[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fid, termios.TCSAFLUSH, self.terminal)
        # Reset to backup terminal at exit.
        atexit.register(self.reset_term)

    def getch(self):
        """Get the last typed character.
        """
        return sys.stdin.read(1)

    def event(self):
        """Detect any keyboard event.
        """
        dr, dummy, dummy = select([sys.stdin], [], [], 0)
        return dr != []

    def reset_term(self):
        """Reset terminal to backup.
        """
        termios.tcsetattr(self.fid, termios.TCSAFLUSH, self.backup)
