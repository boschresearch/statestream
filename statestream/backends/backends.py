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


import copy
import os


global _BACKEND
global _B
_BACKEND = None
_B = None


def import_backend(net, param, item_name):
    global _BACKEND
    global _B

    mps = False

    if _B is None and net is None:
        raise TypeError("Attempt to import non-initialized backend: " + str(item_name))

    if _B is None:
        backend = net.get("backend", "theano")

        # Get all visible gpu devices for this network.
        try:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            visible_devices = [int(d) for d in visible_devices.split(",")]
            #print("\nINFO: For item >>" + item_name + "<<, using CUDA_VISIBLE_DEVICES: " + str(visible_devices))
        except:
            visible_devices = param["core"].get("visible_devices", [])
            #print("\nINFO: For item >>" + item_name + "<<, using visible_devices from stcore.yml: " + str(visible_devices))

        # Get device to run item on.
        device = None
        dev_type = None
        dev_id = None
        for t in ["neuron_pools", "plasticities", "synapse_pools"]:
            if item_name in net[t]:
                device = net[t][item_name].get("device", "cpu")
                backend = net[t][item_name].get("backend", backend)
                break
        if device.startswith("gpu"):
            dev_type = "gpu"
            dev_id = int(device.split(":")[1])
            if dev_id >= len(visible_devices):
                raise TypeError("Attempt to use device " + str(dev_id) + " with visible devices: " + str(visible_devices))
        elif device == "cpu":
            dev_type = "cpu"
        else:
            raise TypeError("Unknown device: " + str(device))

        # Set FLAGS according to backend.
        if backend == "theano":
            # Define session / process dependent compile directory.
            base_compiledir = os.path.expanduser("~") + "/.statestream/compiledirs/" + str(item_name)
            if not os.path.isdir(base_compiledir):
                os.makedirs(base_compiledir)
            # Set device.
            loc_device = "cpu"
            tmp = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if dev_type == "gpu":
                loc_device = "cuda0"
            # Make only the specified device visible.
            os.environ["THEANO_FLAGS"] = param["core"]["THEANO_FLAGS"] \
                                         + ",device=" + loc_device \
                                         + ",base_compiledir=" + base_compiledir
        elif backend == "tensorflow":
            # Set device.
            loc_device = "cpu"
            tmp = os.environ["CUDA_VISIBLE_DEVICES"]
#               # In case of cpu, use invalid visible device to mask all devices.
#                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            raise ImportError("Unknown backend: " + str(backend))


        # Set CUDA_VISIBLE_DEVICES to specified device.
        if dev_type == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices[dev_id])


            # Load MPS only in case CUDA_MPS_LOG / PIPE not already set and hence MPS already loaded.
            if mps and "CUDA_MPS_LOG_DIRECTORY" not in os.environ and "CUDA_MPS_PIPE_DIRECTORY" not in os.environ:
                # Determine CUDA_MPS_LOG / PIPE folders.
                CUDA_MPS_LOG_DIRECTORY = "/tmp/" + os.getlogin() + "/mps_" + str(visible_devices[dev_id]) + "/log"
                CUDA_MPS_PIPE_DIRECTORY = "/tmp/" + os.getlogin() + "/mps_" + str(visible_devices[dev_id]) + "/pipe"
                # Set environment variables.
                os.environ["CUDA_MPS_LOG_DIRECTORY"] = CUDA_MPS_LOG_DIRECTORY
                os.environ["CUDA_MPS_PIPE_DIRECTORY"] = CUDA_MPS_PIPE_DIRECTORY
                # Create folders.
                try:
                    os.mkdir("/tmp/" + os.getlogin())
                except:
                    pass
                try:
                    os.mkdir("/tmp/" + os.getlogin() + "/mps_" + str(visible_devices[dev_id]))
                except:
                    pass
                try:
                    os.mkdir(CUDA_MPS_LOG_DIRECTORY)
                except:
                    pass
                try:
                    os.mkdir(CUDA_MPS_PIPE_DIRECTORY)
                except:
                    pass
                # Start MPS.
                try:
                    os.system("nvidia-cuda-mps-control -d")
                except:
                    pass

            # Delete CUDA_VISIBLE_DEVICES.
            if mps and "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


        # Import backend.
        if backend == "theano":
            import statestream.backends.backend_theano as B
        elif backend == "tensorflow":
            import statestream.backends.backend_tensorflow as B
            B.tf_get_session()


        # Set global backend.
        _B = B
        _BACKEND = backend

        #os.environ["CUDA_VISIBLE_DEVICES"] = tmp


    return _B


