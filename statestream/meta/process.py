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



"""Dictionary holding possible process states.

States:
-------
I : initialization state
W : write state
R : read state
WaW : wait after write state
WaR : wait after read state
E : exit state
C : compiling
"""
process_state = {
    "I": 0,
    "W": 1,
    "R": 2,
    "WaW": 3,
    "WaR": 4,
    "E": 5,
    "C": 6
}



"""Dictionary holding possible triggers who initiate process
state changes.

Triggers:
---------
WaW-R : from WaW into R state
WaR-W : from WaR into W state
-E : from any state into E state
"""
process_trigger = {
    "WaW-R": 0,
    "WaR-W": 1,
    "-E": 2,
    "-B": 3
}
