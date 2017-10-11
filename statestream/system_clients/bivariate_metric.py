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



import numpy as np
import copy

from statestream.utils.properties import np_feature_metric



# =============================================================================

class bivariate_metric(object):
    """Provide visualization of a metric between two tensors.

    For now we only allow metrics between two (maybe same) nps.

    Parameters:
    -----------
    net : dict
        Complete network dictionary containing all nps, sps, plasts, ifs.
    name : string
        Instance name of this system client.
    client_param : dict
        Dictionary containing system clients parameters.
    """
    def __init__(self, name, net, client_param):
        self.type = "bivariate_metric"
        self.p = copy.deepcopy(client_param)
        self.dat = {}
        self.sv = copy.deepcopy(client_param['selected_values'])
        
        self.parameter = {}
        for p,P in enumerate(client_param['params']):
            self.parameter[P['name']] = P['value']

        # Begin with empty statistics.
        if len(self.sv) == 1:
            self.np0 = self.sv[0][0]
            self.np1 = self.sv[0][0]
        else:
            self.np0 = self.sv[0][0]
            self.np1 = self.sv[1][0]
        self.shape = [net["neuron_pools"][self.np0]['shape'][0],
                      net["neuron_pools"][self.np1]['shape'][0]]

        self.stats = np.zeros(self.shape, dtype=np.float32)
        self.window = np.zeros([self.parameter["window"]] + self.shape, dtype=np.float32)
        self.current_frame = 0
        


    def initialize(self, shm):
        """Initialize this client.
        """
        # Generate client side structure to hold necessary network data.
        self.sv_dat = {}
        if len(self.sv) == 1:
            self.sv_dat[self.np0] = np.zeros(shm.dat[self.np0]["state"].shape,
                                             dtype=np.float32)
        else:
            self.sv_dat[self.np0] = np.zeros(shm.dat[self.np0]["state"].shape,
                                             dtype=np.float32)
            self.sv_dat[self.np1] = np.zeros(shm.dat[self.np1]["state"].shape,
                                             dtype=np.float32)
        # Check / set device and backend.
        self.parameter['device'] = self.parameter.get('device', 'cpu')
        self.parameter['backend'] = self.parameter.get('backend', 'c')
        if self.parameter['backend'] == 'theano':
            # Set GPU.
            os.environ["THEANO_FLAGS"] = self.param["core"]["THEANO_FLAGS"] \
                                         + ",device=" + self.parameter['device']
            # Here now theano may be imported and everything dependent on it.
            import theano
            import theano.tensor as T
            # Generate theano variables.
            self.th_sv_dat = {}
            self.th_sv_dat[self.np0] \
                = theano.shared(np.zeros(shm.dat[self.np0]["state"].shape, dtype=theano.config.floatX),
                                borrow=True,
                                name='np0')
            if len(self.sv) > 1:
                self.th_sv_dat[self.np1] \
                    = theano.shared(np.zeros(shm.dat[self.np1]["state"].shape, dtype=theano.config.floatX),
                                    borrow=True,
                                    name='np1')
            self.th_stats = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          borrow=True,
                                          name='stats')
            # Downsampling of (spatial) larger argument.

            # Dependent on metric define theano function.




    def update_frame_readin(self, shm):
        """System client dependent read in.
        """
        # Read all selected values.
        if self.parameter['backend'] == 'theano':
            self.th_sv_dat[self.np0].set_value(shm.dat[self.np0]["state"])
            if len(self.sv) > 1:
                self.th_sv_dat[self.np1].set_value(shm.dat[self.np1]["state"])
        else:
            self.sv_dat[self.np0][:] = shm.dat[self.np0]["state"]
            if len(self.sv) > 1:
                self.sv_dat[self.np1][:] = shm.dat[self.np1]["state"]



    def update_frame_writeout(self):
        """Method to compute activation statistics for all child nps.
        """
        self.current_frame = (self.current_frame + 1) % self.parameter["window"]
        # Compute stats.
        if self.parameter['backend'] == 'theano':
            self.window[self.current_frame % self.parameter["window"],:,:] \
                = self.th_stats.get_value()
        else:
            self.stats = np_feature_metric(self.sv_dat[self.np0],
                                           self.sv_dat[self.np1],
                                           self.parameter['metric'],
                                           self.parameter['samples'])
            self.window[self.current_frame % self.parameter["window"],:,:] = self.stats[:,:]
        self.dat['variables']['stats'][:,:] = np.mean(self.window, axis=0)
        
