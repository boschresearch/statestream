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

#!/bin/bash


all: cgraphics chebbian csim
	@rm ./lib/ -rf
	@mkdir ./lib
	@mv *.o ./lib/
	@mv *.so ./lib/

cgraphics: ./src/cgraphics.c
	@gcc -c -fPIC ./src/cgraphics.c -o cgraphics.o
	@gcc -shared -Wl,-soname,libcgraphics.so -o libcgraphics.so cgraphics.o

chebbian: ./src/chebbian.c
	@gcc -c -fPIC ./src/chebbian.c -o chebbian.o
	@gcc -shared -Wl,-soname,libchebbian.so -o libchebbian.so chebbian.o

csim: ./src/csim.c
	@gcc -c -fPIC ./src/csim.c -o csim.o
	@gcc -shared -Wl,-soname,libcsim.so -o libcsim.so csim.o
