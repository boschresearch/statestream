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



import os
import sys
import numpy as np
from time import gmtime, strftime, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statestream.utils.shared_memory_layout import SharedMemoryLayout as ShmL
from statestream.utils.core_client import STCClient



class CClient_trainvaltest(STCClient):
    """This is a client to schedule the learning rate.
    """
    def __init__(self, name, net, param, session_id, IPC_PROC):
        # Initialize parent ProcessIf class.
        STCClient.__init__(self, name, net, param, session_id, IPC_PROC)

        self.session_start = time()



    def initialize(self):
        """Method to initialize this core client type.
        """
        # Get some core client parameters.
        # ---------------------------------------------------------------------
        self.target_folder = self.p.get("target_folder", 
                                        self.param['core']['save_path'])
        self.main_if = self.p["main_interface"]
        self.main_plast = self.p["main_plasticity"]
        self.test_after_epochs = self.p.get("test_after_epochs", 10000)
        self.validate_after_epochs = self.p.get("validate_after_epochs", 1000)

        self.current_phase = 'train'
        self.current_frame = 0
        self.finished_epochs = 0
        self.overall_train_frames = 0
        self.initialized = False
        self.finished = False

        # Structure holding performance measures.
        self.loss_train = [[], []]
        self.loss_valid = [[], []]
        self.acc_valid = []
        self.acc_test = None
        self.best_valid_acc = 0.0
        self.best_valid_acc_idx = 0

        # Generate session folder.
        self.session_folder = self.target_folder \
                              + os.sep + self.net['name'] \
                              + strftime("_%a-%d-%b-%Y-%H-%M-%S", gmtime())
        os.makedirs(self.session_folder)

        # Start streaming.
        self.IPC_PROC["break"].value = 0



    def before_readin(self):
        """Method is executed before readin phase.
        """
        if not self.initialized:
            self.mesg.append("CC::" + self.name + ": Initializing ...")
            # Set interface in train/val mode.
            self.shm.dat[self.main_if]['parameter']['mode'][0] = 2
            # For this example, for training, stimulus duration may be one.
            # NOTE: This only works for the given example.
            #   In case of larger, recurrent networks this fails.
            self.stimulus_dur = len(self.net["neuron_pools"]) + 2
            self.shm.dat[self.main_if]['parameter']['min_duration'][0] \
                    = self.stimulus_dur
            self.shm.dat[self.main_if]['parameter']['max_duration'][0] \
                    = self.stimulus_dur + 1
            # Set split to 0 so that entire batch is used for training.
            self.IPC_PROC['plast split'].value = 0
            self.initialized = True

        # While in training phase:
        if self.current_phase == 'train':
            if self.current_frame > self.stimulus_dur:
                which = [self.main_plast, 'variables', 'loss1']
                self.loss_train[0].append(self.shm.get_shm(which))
                self.loss_train[1].append(self.overall_train_frames)
            self.overall_train_frames += 1
            self.current_frame += 1

        # While in validation phase: 
        if self.current_phase == 'valid':
            self.current_frame += 1

        # Switch train -> train / valid / test.
        if self.shm.dat[self.main_if]['variables']['_epoch_trigger_'][0] == 1:
            self.finished_epochs += 1
            self.save_performance('train')
            self.save_model('last')
            self.mesg.append("CC::" \
                             + self.name \
                             + ": Finished training of epoch " \
                             + str(self.finished_epochs))
            if self.finished_epochs >= self.test_after_epochs:
                self.mesg.append("CC::" + self.name + ": Start testing.")
                # Switch off all plasticities.
                for p in self.net['plasticities']:
                    self.IPC_PROC['pause'][self.shm.proc_id[p][0]].value = 1
                # Set interface in testing mode.
                self.shm.dat[self.main_if]['parameter']['mode'][0] = 3
                self.IPC_PROC['plast split'].value = 0
                self.current_phase = 'test'
            elif self.finished_epochs % self.validate_after_epochs == 0:
                self.mesg.append("CC::" \
                                 + self.name \
                                 + ": Start validation after epoch " \
                                 + str(self.finished_epochs))
                self.IPC_PROC['plast split'].value = self.net['agents']
                self.current_phase = 'valid'
            else:
                self.mesg.append("CC::" \
                                 + self.name \
                                 + ": Continue training with epoch " \
                                 + str(self.finished_epochs + 1))

        # Switch valid -> train.
        if self.shm.dat[self.main_if]['variables']['_epoch_trigger_'][1] == 1:
            self.mesg.append("CC::" \
                             + self.name \
                             + ": Finished validation after epoch " \
                             + str(self.finished_epochs))
            self.mesg.append("CC::" \
                             + self.name \
                             + ": Continue training with epoch " \
                             + str(self.finished_epochs + 1))
            self.acc_valid.append(np.copy(self.shm.get_shm([self.main_if, 
                                                            'variables', 
                                                            'acc_valid'])[:,0,0]))
            which = [self.main_plast, 'variables', 'loss0']
            self.loss_valid[0].append(self.shm.get_shm(which))
            self.loss_valid[1].append(self.overall_train_frames)

            # We always want to keep the best model on the validation dataset.
            self.save_model('best')
            self.save_performance('valid')
            self.save_report('ongoing')
            # Switch on all plasticities.
            for p in self.net['plasticities']:
                self.IPC_PROC['pause'][self.shm.proc_id[p][0]].value = 0
            self.IPC_PROC['plast split'].value = 0
            self.current_phase = 'train'

        # Finished with testing, hence end everything.
        if self.shm.dat[self.main_if]['variables']['_epoch_trigger_'][2] == 1:
            self.mesg.append("CC::" + self.name + ": Finished testing. Quit.")
            self.acc_test = np.copy(self.shm.get_shm([self.main_if, 
                                                      'variables', 
                                                      'acc_train'])[:,0,0])
            self.save_performance('test')
            self.finished = True
            self.save_report('finished')
            # TODO: End Gui.


            # Shutdown.
            self.IPC_PROC["gui request"][0] = 1



    def interrupt(self):
        """In case the core shuts down this will be executed.
        """
        if not self.finished:
            self.save_report('interrupted')



    def save_report(self, state):
        """Generates and saves a short session report.
        """
        report = []
        report.append("model name      : " + self.net["name"] + "\n")
        report.append("training epochs : " + str(self.finished_epochs) + "\n")
        report.append("session duration: " \
                      + str(int(time() - self.session_start)) \
                      + " [sec] \n")
        report.append("session status  : " + str(state) + "\n")

        report.append("\n\n")

        report.append("stimulus duration: ".ljust(40) + str(self.stimulus_dur) + " [frames]\n")
        if self.acc_test is not None:
            report.append("accuracy for test: ".ljust(40) \
                          + str((100 * self.acc_test).astype(np.int)) + "\n")
        report.append("best validation accuracy: ".ljust(40) \
                      + str((100 * self.acc_valid[self.best_valid_acc_idx]).astype(np.int))\
                      + "\n")
        report.append("best validation acc. at epoch: ".ljust(40) \
                      + str(self.best_valid_acc_idx) + "\n")

        with open(self.session_folder + os.sep + "report.txt", "w") as f:
            for l in report:
                f.write(l)
            f.close()



    def save_model(self, which):
        """Saves model with highest score on validation dataset.
        """
        if which == 'best':
            if max(self.acc_valid[-1].flatten()) > self.best_valid_acc:
                self.best_valid_score = max(self.acc_valid[-1].flatten())
                self.best_valid_acc_idx = len(self.acc_valid) - 1
                save_filename = self.session_folder + os.sep \
                                + "model_best.st_net"
                core_string = np.array([ord(c) for c in save_filename])
                self.IPC_PROC['instruction len'].value = len(core_string)
                self.IPC_PROC['string'][0:len(core_string)] = core_string[:]
                self.IPC_PROC["save/load"].value = 1
        elif which == 'last':
                save_filename = self.session_folder + os.sep \
                                + "model_last.st_net"
                core_string = np.array([ord(c) for c in save_filename])
                self.IPC_PROC['instruction len'].value = len(core_string)
                self.IPC_PROC['string'][0:len(core_string)] = core_string[:]
                self.IPC_PROC["save/load"].value = 1




    def save_performance(self, which):
        """Saves validation scores.
        """
        if which == 'valid':
            score = np.stack(self.acc_valid, axis=0)
            np.save(self.session_folder + os.sep + "accuracies_validation.npy",
                    score,
                    allow_pickle=False)
            # Update validation performance plot.
            plt.close('all')
            plt.figure()
            plt.gca().set_aspect('equal')
            plt.imshow(score, cmap=plt.cm.Reds, interpolation="none")
            plt.colorbar()
            plt.xlabel('network response offset to stimulus onset')
            plt.ylabel('epoch')
            plt.gca().set_yticks([i for i in range(score.shape[0])])
            plt.gca().set_yticklabels([self.validate_after_epochs * (i + 1) for i in range(score.shape[0])])
            plt.savefig(self.session_folder + os.sep \
                        + "accuracies_validation.png")

            plt.close('all')
            loss_train = np.stack(self.loss_train[0], axis=0)
            batch_train = np.stack(self.loss_train[1], axis=0)
            loss_valid = np.stack(self.loss_valid[0], axis=0)
            batch_valid = np.stack(self.loss_valid[1], axis=0)
            plt.figure()
            plt.plot(batch_train, loss_train, label='training')
            plt.plot(batch_valid, loss_valid, label='validation')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='best')
            plt.savefig(self.session_folder + os.sep \
                        + "losses.png")
        elif which == 'test':
            np.save(self.session_folder + os.sep + "accuracies_test.npy",
                    self.acc_test,
                    allow_pickle=False)

