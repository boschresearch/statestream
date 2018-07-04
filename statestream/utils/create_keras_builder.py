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



from __future__ import print_function

import sys

from time import sleep, gmtime, strftime, time
import os
import copy
import numpy as np

import statestream.meta.network as mn
from statestream.meta.neuron_pool import np_needs_rebuild
from statestream.meta.synapse_pool import sp_needs_rebuild
from statestream.meta.plasticity import plast_needs_rebuild

from statestream.utils.yaml_wrapper import load_yaml, dump_yaml
from statestream.utils.helper import is_scalar_shape
from statestream.utils.helper import is_int_dtype
from statestream.utils.helper import is_float_dtype
from statestream.utils.defaults import DEFAULT_CORE_PARAMETER



class CreateKerasBuilder(object):
    def __init__(self):
        # Load network parameters.
        # --------------------------------------------------------------------
        self.in_file = ""
        self.out_file = ""
        # Read graph file.
        if len(sys.argv) != 3:
            print("Error: Expected call: python create_keras_builder.py [st_graph file] [keras model builder file].")
            sys.exit()
        else:
            self.in_file = copy.copy(sys.argv[1])
            self.out_file = copy.copy(sys.argv[2])
            # Check if st_net or st_graph file.
            if len(self.in_file) > 10:
                if self.in_file[-9:] == ".st_graph":
                    with open(self.in_file) as f:
                        self.meta = load_yaml(f)
                else:
                    print("Error: Invalid filename ending. Expected .st_graph.")
                    sys.exit()
            else:
                print("Error: Source filename is too short.")
                sys.exit()
                    
        # Check sanity of meta.
        if not mn.is_sane(self.meta):
            sys.exit()

        # Preprocess and complement meta dictionary.
        self.net = mn.preprocess(self.meta)

        # Check sanity of network a second time.
        if not mn.is_sane(self.net):
            sys.exit()

        # Generate meta network with ids.
        self.mn = mn.meta_network(self.net)




    def build(self):
        """This is the main routine to generate the keras model builder.

        The builder will consist of:
            import header
            builder function

        Both will be build up here as list of strings.
        """

        # Begin with empty
        self.import_header = []
        self.builder_function = []

        # Some initial imports
        self.import_header.append("# AUTOMATICALLY GENERATED FILE")
        self.import_header.append("")
        self.import_header.append("from __future__ import print_function")
        self.import_header.append("from keras.models import Model")
        self.import_header.append("import keras.backend as K")


        # Start builder with function definition.
        self.builder_function.append("")
        self.builder_function.append("def model_builder(*):")
        self.builder_function.append("    '''The model builder function for a keras model.")
        self.builder_function.append("    '''")

        self.builder_function.append("    # Instantiate keras model.")
        self.builder_function.append("    model = Model(input=input_layer, output=prediction)")
        self.builder_function.append("")
        self.builder_function.append("    # TODO: Load pretrained weights.")
        self.builder_function.append("")
        self.builder_function.append("    # Print model summary.")
        self.builder_function.append("    # model.summary(line_length=80)")
        self.builder_function.append("")
        self.builder_function.append("    return model")
        



        # Create output file.
        with open(self.out_file, "w+") as f:
            for l in self.import_header:
                f.write(l + "\n")
            for l in self.builder_function:
                f.write(l + "\n")

       

# Main.            
if __name__ == "__main__":
    inst_create_keras_builder = CreateKerasBuilder()
    inst_create_keras_builder.build()
                    