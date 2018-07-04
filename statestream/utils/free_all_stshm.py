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



import SharedArray



# Get list of all existing shared memory arrays.
shm_list = SharedArray.list()
shm_list_name = []
for i in range(len(shm_list)):
    shm_list_name.append(str(shm_list[i].name, "utf-8"))

# Delete all statestream associeated shared memory.
shm_cntr = 0
for i in range(len(shm_list)):
    if shm_list_name[i].find("statestream") != -1:
        SharedArray.delete(shm_list_name[i])
        shm_cntr += 1
        
# Print number of deleted arrays.
print("Deleted " + str(shm_cntr) + " shared memory arrays.")
