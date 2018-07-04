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



from skimage.io import imread
from skimage.transform import resize
from scipy.misc import imresize
import numpy as np
import copy
import time



def gaussian_kernel_2D(size, mean=(0.0, 0.0), sigma=1.0):
    """Funciton returns a gaussian kernel.
    """
    # Compute grid.
    X = np.arange(-size // 2 + 1 + mean[0], \
                   size // 2 + 1 + mean[1])
    x, y = np.meshgrid(X, X)
    # Compute kernel.
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # Return normalized kernel.
    return kernel / np.sum(kernel)



def is_int_dtype(dtype):
    """Determines if a given dtype is int.
    """
    if dtype in [np.int, np.int32]:
        return True
    return False



def is_float_dtype(dtype):
    """Determines if a given dtype is float.
    """
    if dtype in [np.float, np.double, np.float32]:
        return True
    return False



def is_scalar_shape(shape):
    """Determines if a shape is scalar.
    """
    if shape == ():
        return True
    return False



def is_list_of_lists(x):
    """Determines if a variable is of type list of lists.
    """
    if type(x) is list:
        if len(x) > 0:
            if type(x[0]) is list:
                return True
    return False



class LoadSample(object):
    def __init__(self, proc_id, if_name, net, item_shapes, item_types, item_params, augmentation={}):
        """Helper class for data handling / loading.
        """
        # Get global structure.
        self.net = net
        # Get parent interface name.
        self.if_name = copy.copy(if_name)
        # Unique proc identifier.
        self.id = copy.copy(proc_id)
        # Begin in running state.
        self.state_running = True
        # Get augmentation.
        self.augmentation = copy.copy(augmentation)
        # Get no samples in reservoir.
        self.no_isamples = self.net['interfaces'][self.if_name].get("samples", 2 * self.net['agents'])
        # Get items.
        self.item_types = copy.copy(item_types)
        # Get size and shape of sample.
        self.sample_shape = copy.copy(item_shapes)
        self.sample_size = {}
        for i in self.sample_shape:
            self.sample_size[i] = int(np.prod(self.sample_shape[i]))
        # String variable.
        self.string = np.zeros([256], dtype=np.uint8)

        # Generate structures for efficient label <-> classes transformation.
        self.item_params = copy.copy(item_params)
        for i in self.item_params:
            if self.item_types[i] == "RGB2one-hot":
                # For this type of item, create some structures for
                # computational efficiency.
                self.item_params[i]["classes"] = np.zeros((256 + 256*256 + 256*256*256), dtype=np.int32)
                for c, C in self.item_params[i]["coding"].items():
                    ind = int(C['rgb'][0]) + int(C['rgb'][1])*256 + int(C['rgb'][2])*256*256
                    self.item_params[i]["classes"][ind] = int(C['class'])
                self.item_params[i]["labels"] = []
                for key in sorted(self.item_params[i]["coding"].keys()):
                    self.item_params[i]["labels"].append([self.item_params[i]["coding"][key]['rgb'][0], 
                                                          self.item_params[i]["coding"][key]['rgb'][1], 
                                                          self.item_params[i]["coding"][key]['rgb'][2]])
                self.item_params[i]["labels"] = np.array(self.item_params[i]["labels"], np.uint8)
                xInd = np.zeros([self.sample_shape[i][1], self.sample_shape[i][2]], dtype=np.int)
                yInd = np.zeros([self.sample_shape[i][1], self.sample_shape[i][2]], dtype=np.int)
                for x in range(self.sample_shape[i][1]):
                    for y in range(self.sample_shape[i][2]):
                        xInd[x,y] = x
                        yInd[x,y] = y
                self.item_params[i]["xInd"] = xInd.flatten()
                self.item_params[i]["yInd"] = yInd.flatten()

    # ========================================================================

    def l2c(self, img_label, item):
        """Convert label to classes (RGB to one-hot).
        """
        img_class = img_label[0,:,:].astype(np.int64) \
                    + img_label[1,:,:].astype(np.int64) * 256 \
                    + img_label[2,:,:].astype(np.int64) * 256 * 256
        return self.item_params[item]["classes"][img_class]

    def c2l(self, img_class, item):
        """Convert classes to label (one-hot to RGB).
        """
        return self.item_params[item]["labels"][img_class]

    # ========================================================================

    # main method incl. (;;) loop
    def run(self, DL_IPC_PROC, DL_IPC_DATA):
        # Create numpy representation of shared memory.
        dl_net = {}
        for i in self.sample_shape:
            dl_net[i] = {}
            for s in range(self.no_isamples):
                dl_net[i][s] = np.frombuffer(DL_IPC_DATA[i][s],
                                             dtype=np.float32,
                                             count=self.sample_size[i])
                dl_net[i][s].shape = [self.sample_shape[i][0], 
                                      self.sample_shape[i][1], 
                                      self.sample_shape[i][2]]

        # Initial handshake.
        DL_IPC_PROC['proc_state'][self.id][0] = 0

        # Enter forever.
        while self.state_running:
            if DL_IPC_PROC['proc_state'][self.id][0] == 1:
                # Get target sample.
                target_sample = int(copy.copy(DL_IPC_PROC['target_sample'][self.id][0]))
                flipX = False
                if "flipX" in self.augmentation:
                    if np.random.random() > 0.5:
                        flipX = True
                # Loop over all items.
                for i in self.sample_shape:
                    # Get len of filename.
                    name_len = int(DL_IPC_PROC['src_filename_len'][i][self.id][0])
                    # By default ignore items whit no name (even no overwrite).
                    if name_len > 0:
                        # Get filename.
                        self.string[0:name_len] = DL_IPC_PROC['src_filename'][i][self.id][0:name_len]
                        # Create string from message.
                        file_name = ""
                        for c in range(name_len):
                            file_name += chr(self.string[c])
                        # Load image [y, x, c]
                        img = imread(file_name)
                        # Apply cropping.
                        if "crop" in self.item_params[i]:
                            c = self.item_params[i]['crop']
                            x_start = int(c[0][0] * img.shape[1])
                            x_end = int(c[1][0] * img.shape[1])
                            y_start = int(c[0][1] * img.shape[0])
                            y_end = int(c[1][1] * img.shape[0])
                            img = img[y_start:y_end, x_start:x_end,:]
                        if flipX:
                            img = np.fliplr(img)
                        # Resize image dependent on item type.
                        if img.shape[1] != self.sample_shape[i][1] or img.shape[0] != self.sample_shape[i][2]:
                            target_shape = (self.sample_shape[i][2], self.sample_shape[i][1])
                            if self.item_types[i] == 'RGB':
                                img = resize(img, target_shape, mode='reflect')
                            elif self.item_types[i] in ["RGB2one-hot", 'RGB-nearest']:
                                img = imresize(img, target_shape, interp="nearest")
                        # Dependent on item type define data.
                        dat = np.zeros(self.sample_shape[i], dtype=np.float32)
                        if self.item_types[i] in ['RGB', 'RGB-nearest']:
                            # Dependent on grey or color image, correct dimensions [feat, x, y] and copy pixel data.
                            if len(img.shape) == 3:
                                img = np.swapaxes(img, 0, 2)
                                dat[:,:,:] = img[0:3,:,:]
                            elif len(img.shape) == 2:
                                # assume grey-scale image and correct (y,x) -> (x,y)
                                img = np.swapaxes(img, 0, 1)
                                dat[0,:,:] = img[:,:]
                                dat[1,:,:] = img[:,:]
                                dat[2,:,:] = img[:,:]
                        elif self.item_types[i] in ["RGB2one-hot"]:
                            img = np.swapaxes(img, 0, 2)
                            classes = self.l2c(img[0:3,:,:].astype(np.int), i)
                            dat[classes.flatten(), self.item_params[i]["xInd"], self.item_params[i]["yInd"]] = 1
                        # Store loaded item in IPC_DATA.
                        try:
                            dl_net[i][target_sample][:,:,:] = dat[:,:,:]
                        except:
                            print("\nIT: " + str(len(dl_net[i])) + "  " + str(i) + "  " + str(target_sample))
                    else:
                        dl_net[i][target_sample][:,:,:] = 0

                # Set sample state to fresh.
                DL_IPC_PROC['sample_state'][target_sample][0] = 2
                # Set back own state.
                DL_IPC_PROC['proc_state'][self.id][0] = 0.0
            elif DL_IPC_PROC['proc_state'][self.id][0] == 2:
                # received exit signal
                self.state_running = False
            else:
                time.sleep(0.01)
        
        
        
        
        
        