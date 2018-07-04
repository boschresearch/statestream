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



try:
	import ruamel_yaml as yaml
except:
	try:
		import ruamel.yaml as yaml
	except:
		yaml = None



def load_yaml(file_handle):
    """Wrapper function to load yaml files.
    """
    assert(yaml is not None), "\nError: ruamel yaml python package not found."
    return yaml.load(file_handle, Loader=yaml.RoundTripLoader)



def dump_yaml(data, file_handle):
    """Wrapper function to nicely dump dictionaries as yaml files.
    """
    assert(yaml is not None), "\nError: ruamel yaml python package not found."
    yaml.dump(data, file_handle, Dumper=yaml.RoundTripDumper)
    
