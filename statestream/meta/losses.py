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


all_losses = ["MSE",
              "MAE",
              "hinge",
              "negloglikelihood",
              "categorical_crossentropy",
              "minimize",
              "maximize"]

def has_target(loss):
    """Checks if a given loss has a target (a.k.a. ground truth).
    """
    targets = ["MSE",
               "MAE",
               "hinge",
               "negloglikelihood",
               "categorical_crossentropy"]
    if loss in targets:
        return True
    return False

