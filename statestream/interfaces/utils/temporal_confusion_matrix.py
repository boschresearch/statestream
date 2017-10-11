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

from statestream.meta.neuron_pool import np_state_shape



class TemporalConfusionMatrix(object):
    """Class to provide a temporal confusion matrix for all interfaces.

    Computation of the confusion matrix (and derived measures) has two
    important parameters:
        conf-mat mean over : int
            This parameter determines over how many frames the estimated
            confusion matrix should be computed as a mean.
        conf-mat window : int
            This parameter determines the temporal window for which the
            confusion matrix is to be computed. We get one confusion
            matrix for every time delay dt = 1 .. 'conf-mat window' giving
            a performance measure of the networks output at time 0 with
            respect to the network's inputs at time -dt.
            To work properly, this parameter together with the minimum
            presentation duration of the interface must at least be the minimal
            path length from input to network prediction.

    Parameters
    ----------
    net : dict
        The network dictionary.
    interface : string
        The name of this interface.
    pred_np : string
        The name of neuron-pool holding the network's prediction.
    """
    def __init__(self, net, interface, pred_np):
        p = net["interfaces"][interface]

        self.agents = net["agents"]

        # Determine number of classes, for this we need the real prediction np.
        tmp_target = pred_np
        if "remap" in p:
            if pred_np in p["remap"]:
                tmp_target = p["remap"][pred_np]
        target_shape = np_state_shape(net, tmp_target)
        self.no_classes = target_shape[1]

        # Check if we have a segmentation (space).
        self.segmentation = False
        if target_shape[2] != 1 or target_shape[3] != 1:
            self.segmentation = True

        # Get temporal length of confusion matrix.
        self.history_len = p.get("conf-mat window", 9)
        self.mean_over = p.get("conf-mat mean over", 32)

        # Instantiate confusion matrix and accuracy.
        self.conf_mat = np.zeros([self.history_len, 
                                  self.no_classes, 
                                  self.no_classes], dtype=np.float32)
        self.accuracy = np.zeros([self.history_len, 1, 1], dtype=np.float32)

        # Initially trigger histories and conf.-mat. history to empty.
        # ---------------------------------------------------------------------
        self.trigger_history = []
        if self.segmentation:
            for a in range(self.agents):
                self.trigger_history.append([-1, None, None])
        else:
            for a in range(self.agents):
                self.trigger_history.append([0, 0])
        # Initialize delayed confusion matrices as empty.
        self.conf_mat_hist = np.zeros([self.mean_over,
                                       self.history_len,
                                       self.no_classes,
                                       self.no_classes],
                                      dtype=np.float32)

        # Variable to story current temporal conf.-mat.
        self.tCM = np.zeros([self.history_len, 
                             self.no_classes, 
                             self.no_classes], 
                            dtype=np.float32)



    def update_history(self, 
                       current_prediction, 
                       current_frame, 
                       current_label, 
                       current_elapsed, 
                       current_trigger,
                       mask=None):
        """Update (if needed) the confusion matrix variable.
        """
        # Compute current temporal confusion matrix.
        self.tCM *= 0
        if mask is None:
            mask = np.ones([self.agents])
        if self.segmentation:
            for a in range(self.agents):
                if mask[a] > 0:
                    if self.trigger_history[a][0] < self.history_len \
                            and self.trigger_history[a][0] != -1:
                        pred_idx = np.argmax(current_prediction[a,:,:,:], axis=0).flatten()
                        gt_idx = np.argmax(current_label[a][0], axis=0).flatten()
                        mask_idx = current_label[a][1][0,:,:].flatten()
                        for p_idx, g_idx, m_idx in zip(pred_idx, gt_idx, mask_idx):
                            self.tCM[self.trigger_history[a][0], g_idx, p_idx] += m_idx
        else:
            for a in range(self.agents):
                if mask[a] > 0:
                    if self.trigger_history[a][0] < self.history_len:
                        pred_idx = np.argmax(np.mean(current_prediction[a,:,:,:], axis=(1,2)))
                        self.tCM[self.trigger_history[a][0], self.trigger_history[a][1], pred_idx] += 1.0

        # Update current frame in conf_mat history.
        self.conf_mat_hist[current_frame % self.mean_over,:,:,:] = self.tCM[:,:,:]
        # Compute mean over history.
        self.conf_mat[:,:,:] \
            = np.mean(self.conf_mat_hist, axis=0)
        # Compute accuracy.
        for w in range(self.history_len):
            tmp_sum = np.sum(self.conf_mat[w,:,:])
            if abs(tmp_sum) > 1e-8:
                self.accuracy[w,0,0] \
                    = np.trace(self.conf_mat[w,:,:]) \
                        / tmp_sum
            else:
                self.accuracy[w,0,0] = 0
        # Add current entry to histories.
        if self.segmentation:
            for a in range(self.agents):
                if mask[a] > 0:
                    if current_trigger[a] == 1:
                        # Onset of new stimuli, so update trigger history.
                        self.trigger_history[a] \
                            = [0, 
                               np.copy(current_label[a][0]),
                               np.copy(current_label[a][1])]
                    else:
                        # No new stimulus, so count elapsed +1.
                        self.trigger_history[a][0] = current_elapsed[a]
        else:
            for a in range(self.agents):
                if mask[a] > 0:
                    if current_trigger[a] == 1:
                        # Onset of new stimuli, so update trigger history.
                        self.trigger_history[a] \
                            = [0, 
                               [current_label[a]]]
                    else:
                        # No new stimulus, so count elapsed +1.
                        self.trigger_history[a][0] = current_elapsed[a]


