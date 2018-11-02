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




import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
import pickle
import copy
import os
import sys
from ruamel_yaml import YAML



"""This file contains plotting utilities.

    The information to be visualized is specified via command line parameters.
    There are three modes for visualization:

    1) For a pair of trained rollouts (streaming + sequential) for a model
        specification in ./model_specifications/, the following command will 
        visualize the accuracy over update steps of both rollout patterns:

        python keras_model_plotter.py model_specifications/<model name>.st_graph

    2) In case the streaming and sequential rollouts have been trained for all
        cifar_DSR*.st_graph models in ./model_specifications/, the following
        command shows the difference in accuracy between the two rollout
        patterns at the time of the first response of the sequential rollout:

        python keras_model_plotter.py cifar10 firstresponse

    3) In case the models 'mnist_hybrid' and 'mnist_recskip' have been trained
        for both (streaming + sequential) rollout patterns, the following
        command can be used to show the accuracies over update steps (similar 
        to 1)) for the three rollout patterns (streaming, sequential, hybrid):

        python keras_model_plotter.py mnist hybrid

        Note, that the architectures in the 'mnist_hybrid' and 'mnist_recskip'
        differ only in the definition of streaming edges in the sequential 
        rollout, and hence the totally streaming rollout is the same for both.

"""



# Get available specifications.
available_specs = ["model_specifications" + os.sep + a for a in os.listdir("./model_specifications/")]



def print_help():
    """Function to print help instructions to konsole.
    """
    print("\nTo plot results for a model, the model name has to be specified.")
    print("    Available models: \n")
    for s in available_specs:
        print("        " + s)
    print("To show the comparison of accuracies at first sequential response for CIFAR-10:")
    print("    python keras_model_plotter.py cifar10 firstresponse")
    print("")
    print("To show the comparison of streaming vs. sequential vs. hybrid on MNIST:")
    print("    python keras_model_plotter.py mnist hybrid")
    print("")



# Check input arguments.
plot_mode = 1
if len(sys.argv) > 3:
    print_help()
    sys.exit()
elif len(sys.argv) == 3:
    if sys.argv[1] == "cifar10" and sys.argv[2] == "firstresponse":
        # Show accuracies at first response of sequential rollouts
        # over the 7 cifar10 models.
        plot_mode = 2
    elif sys.argv[1] == "mnist" and sys.argv[2] == "hybrid":
        # Like mode 1 for mnist, but combine recskip and hybrid networks.
        plot_mode = 3
    else:
        print_help()
        sys.exit()
elif len(sys.argv) == 2:
    # Check if a valid specification file was given.
    if sys.argv[1] not in available_specs:
        print_help()
        sys.exit()
elif len(sys.argv) == 1:
    print_help()
    sys.exit()



# Dependent on plot mode load and visualize results.
results = {"streaming": [], "sequential": []}
# ============================================================================
if plot_mode == 1:
# ============================================================================
    # Load model specification.
    model_name = sys.argv[1].split(".")[0].split("/")[1]
    dataset = None
    try:
        yaml = YAML()
        stg = yaml.load(open(sys.argv[1], "r"))
        dataset = stg["interfaces"]["data"]["type"]
    except:
        print("\nError: Unable to load specification from " + str(sys.argv[1]))
        sys.exit()
    # Determine number of available results samples.
    samples = []
    for f in os.listdir('model_trained/'):
        if model_name in f and '.results' in f:
            samples.append(f)
    # Load results for streaming & sequential version.
    for s in samples:
        if "streaming" in s:
            t = "streaming"
        else:
            t = "sequential"
        try:
            results[t].append(pickle.load(open('model_trained/' + s, "rb")))
        except:
            print("\nError: Unable to load saved results for model: " + s)
            sys.exit()
    impl_rollouts = results["streaming"][0]["test_acc"][0].shape[0]
    # shortest-path:
    #     determines first response for STREAMING rollout
    first_response = {}
    for t in ["streaming", "sequential"]:
        first_response[t] = stg["first_response_" + t]
    shortest_path_sequential = stg["shortest_path_sequential"]
    # rollout-factor:
    #     determines first response for SEQUENTIAL rollout and its x-axis scaling
    rollout_factor = stg["rollout_factor"]
    x = {}
    y_all = {}
    y_mean = {}
    y_std = {}
    x["streaming"] = np.arange(first_response["streaming"], 
                               first_response["streaming"] + impl_rollouts)
    x["sequential"] = shortest_path_sequential + rollout_factor * np.arange(0, 
                                                                            impl_rollouts)
    for t in ["streaming", "sequential"]:
        # Collect all samples.
        y_all[t] = np.array([results[t][s]["test_acc"][0] for s in range(len(results[t]))])
        # Determine mean and variance.
        y_mean[t] = np.mean(y_all[t], axis=0)
        if y_all[t].shape[0] > 1:
            y_std[t] = np.var(y_all[t], axis=0)
        else:
            y_std[t] = np.zeros(y_mean[t].shape)
    # Some parameters for plotting.
    markerSizeSmall = 3
    markerSizeBig = 5
    color = {}
    color["streaming"] = 'r'
    color["sequential"] = 'b'
    fontnumbering = FontProperties()
    fontnumbering.set_weight('bold')
    fontnumbering.set_size(12)
    for t in ["streaming", "sequential"]:
        plt.errorbar(x[t], y_mean[t], \
                 yerr=y_std[t], \
                 marker='o', ls='--', \
                 ms=markerSizeBig, c=color[t], \
                 label=t)
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.title(str(model_name))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel('time (update steps)')
    ax.set_ylabel('accuracy')
    #plt.savefig("/home/fvi2si/12_DL/BigPicture/cst_mnist_norot.pdf", dpi=200, bbox_inches="tight")
    plt.show()



# ============================================================================
elif plot_mode == 2:
# ============================================================================
    # Show comparison of accuracies at first sequential response for cifar10.
    # Determine already trained cifar10 models (.results files).
    x_label = []
    all_files = os.listdir("./model_trained/")
    for s in range(6):
        results_file_streaming = "cifar_DSR" + str(s) + "-streaming.results"
        results_file_sequential = "cifar_DSR" + str(s) + "-sequential.results"
        if results_file_streaming in all_files and results_file_sequential in all_files:
            x_label.append("DSR" + str(s))
    # Open statestream specification files.
    stg = {}
    yaml = YAML()
    for f in x_label:
        stgraph_file = "model_specifications/cifar_" + f + ".st_graph"
        stg[f] = yaml.load(open(stgraph_file, "r"))
    # Load results for streaming & sequential version.
    shortest_path_sequential = {}
    first_response = {}
    results = {}
    for s in x_label:
        shortest_path_sequential[s] = stg[s]["shortest_path_sequential"]
        first_response[s] = {}
        results[s] = {}
        for t in ["streaming", "sequential"]:
            first_response[s][t] = stg[s]["first_response_" + t]
            results_file = 'model_trained/cifar_' + str(s) + '-' + str(t) + '.results'
            try:
                results[s][t] = pickle.load(open(results_file, "rb"))
            except:
                print("\nError: Unable to load saved results for model: " + results_file)
                sys.exit()
    impl_rollouts = results[x_label[0]]["streaming"]["test_acc"][0].shape[0]
    # Get accuracies at first sequential response for both sequential and streaming.
    y = {"streaming": [], "sequential": []}
    for s in x_label:
        # Determine frame in streaming rollout for which sequential rollout yields first response.
        x_streaming = list(np.arange(first_response[s]["streaming"], 
                                     first_response[s]["streaming"] + impl_rollouts))
        frame_streaming = x_streaming.index(shortest_path_sequential[s])
        # Determine accuracies.
        y["sequential"].append(results[s]["sequential"]["test_acc"][0][0])
        y["streaming"].append(results[s]["streaming"]["test_acc"][0][frame_streaming])
    # Some parameters for plotting.
    markerSizeSmall = 3
    markerSizeBig = 5
    color = {}
    color["streaming"] = 'r'
    color["sequential"] = 'b'

    fontnumbering = FontProperties()
    fontnumbering.set_weight('bold')
    fontnumbering.set_size(12)

    for t in ["streaming", "sequential"]:
        plt.plot(range(len(x_label)),
                 y[t], \
                 marker='o', ls='--', \
                 ms=markerSizeBig, c=color[t], \
                 label=t)
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.title("CIFAR10 accuracies at first sequential response")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xticks(range(0, len(x_label), 1))
    ax.set_xticklabels(x_label)
    ax.set_xlabel('network architecture')
    ax.set_ylabel('accuracy')
    #plt.savefig("/home/fvi2si/12_DL/BigPicture/cst_mnist_norot.pdf", dpi=200, bbox_inches="tight")
    plt.show()



# ============================================================================
elif plot_mode == 3:
# ============================================================================
    # Load model specification.
    try:
        yaml = YAML()
        stg = {}
        stg["recskip"] = yaml.load(open('model_specifications/mnist_recskip.st_graph', "r"))
        stg["hybrid"] = yaml.load(open('model_specifications/mnist_hybrid.st_graph', "r"))
        dataset = 'mnist'
    except:
        print("\nError: Unable to load specification.")
        sys.exit()
    # Determine number of available results samples.
    samples = []
    for f in os.listdir('model_trained/'):
        if ('mnist_recskip' in f or 'mnist_hybrid' in f) and '.results' in f:
            samples.append(f)
    # Load results for streaming & sequential version.
    results = {}
    for t in ["streaming", "sequential"]:
        results[t] = {}
        for n in ["recskip", "hybrid"]:
            results[t][n] = []
    for s in samples:
        if "streaming" in s:
            t = "streaming"
        else:
            t = "sequential"
        if "recskip" in s:
            n = "recskip"
        else:
            n = "hybrid"
        try:
            results[t][n].append(pickle.load(open('model_trained/' + s, "rb")))
        except:
            print("\nError: Unable to load saved results for model: " + s)
            sys.exit()

    impl_rollouts = results["streaming"]["recskip"][0]["test_acc"][0].shape[0]
    # shortest-path:
    #     determines first response for STREAMING rollout
    first_response = {}
    for t in ["streaming", "sequential"]:
        first_response[t] = {}
        for n in ["recskip", "hybrid"]:
            first_response[t][n] = stg[n]["first_response_" + t]
    shortest_path_sequential = {}
    for n in ["recskip", "hybrid"]:
        shortest_path_sequential[n] = stg[n]["shortest_path_sequential"]
    # rollout-factor:
    #     determines first response for SEQUENTIAL rollout and its x-axis scaling
    rollout_factor = {}
    for n in ["recskip", "hybrid"]:
        rollout_factor[n] = stg[n]["rollout_factor"]
    x = {}
    y_all = {}
    y_mean = {}
    y_std = {}
    for n in ["recskip", "hybrid"]:
        x[n] = {}
        x[n]["streaming"] = np.arange(first_response["streaming"][n], 
                                      first_response["streaming"][n] + impl_rollouts)
        x[n]["sequential"] = shortest_path_sequential[n] + rollout_factor[n] * np.arange(0, 
                                                                                         impl_rollouts)
    for t in ["streaming", "sequential"]:
        y_all[t] = {}
        y_mean[t] = {}
        y_std[t] = {}
        for n in ["recskip", "hybrid"]:
            # Collect all samples.
            y_all[t][n] = np.array([results[t][n][s]["test_acc"][0] for s in range(len(results[t][n]))])
            # Determine mean and variance.
            y_mean[t][n] = np.mean(y_all[t][n], axis=0)
            if y_all[t][n].shape[0] > 1:
                y_std[t][n] = np.var(y_all[t][n], axis=0)
            else:
                y_std[t][n] = np.zeros(y_mean[t][n].shape)
    # Some parameters for plotting.
    markerSizeSmall = 3
    markerSizeBig = 5
    color = {}
    color["streaming"] = 'r'
    color["sequential"] = 'b'

    fontnumbering = FontProperties()
    fontnumbering.set_weight('bold')
    fontnumbering.set_size(12)

    for t in ["streaming", "sequential"]:
        plt.errorbar(x["recskip"][t], y_mean[t]["recskip"], \
                 yerr=y_std[t]["recskip"], \
                 marker='o', ls='--', \
                 ms=markerSizeBig, c=color[t], \
                 label=t)
    plt.errorbar(x["hybrid"]["sequential"], y_mean["sequential"]["hybrid"], \
             yerr=y_std["sequential"]["hybrid"], \
             marker='o', ls=':', \
             ms=markerSizeBig, c="darkviolet", \
             label="hybrid")
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.title("mnist_recskip (streaming, hybrid, sequential)")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel('time (update steps)')
    ax.set_ylabel('accuracy')
    #plt.savefig("/home/fvi2si/12_DL/BigPicture/cst_mnist_norot.pdf", dpi=200, bbox_inches="tight")
    plt.show()



    # Write post-processed results to csv file.
    models = {
        "streaming": ("streaming", "recskip"),
        "sequential": ("sequential", "recskip"),
        "hybrid": ("sequential", "hybrid")
    }
    for m,mi in models.items():
        with open(m + ".csv", "w") as f:
            f.write("x, y, yerr\n")
            for i in range(len(x[mi[1]][mi[0]])):
                f.write(str(x[mi[1]][mi[0]][i]) + ", " + \
                        str(y_mean[mi[0]][mi[1]][i]) + ", " + \
                        str(y_std[mi[0]][mi[1]][i]) + "\n")
    


















if False:

    print("    Loading " + str(len(models)) + " models ...")
    pm = {}
    for model in models:
        pm[model] = {}
        pm[model]["data"] = pickle.load(open('code/' + model, "rb"))
        pm[model]["x"] = None
        pm[model]["y"] = None
        pm[model]["yerr"] = None


    # Fill plotting data structure.
    for model in models:
        # Determine actual rollout and noise.
        this_rollout = 8
        reps = 6
        impl_rollouts = pm[model]["data"]["test_acc"][0][0].shape[0]
        acc_over_rollout = np.zeros([impl_rollouts, reps]) 

        # Consider last epoch.
        for rep in range(reps):
            acc_over_rollout[:,rep] = pm[model]["data"]["test_acc"][rep][-1][:]

        shortest_path = pm[model]["data"]["net"]["meta"]["shortest_path_len"]
        rollout_factor = pm[model]["data"]["net"]["meta"]["rollout_factor"]

        if model.find("streaming") != -1:
            pm[model]["x"] \
                = np.arange(shortest_path, impl_rollouts + shortest_path)
            pm[model]["y"] \
                = np.mean(acc_over_rollout, axis=1)
            pm[model]["yerr"] \
                = np.std(acc_over_rollout, axis=1) / np.sqrt(acc_over_rollout.shape[1])
        elif model.find("sequential") != -1:
            pm[model]["x"] \
                = rollout_factor * np.arange(1, impl_rollouts + 1)
            pm[model]["y"] \
                = np.mean(acc_over_rollout, axis=1)
            pm[model]["yerr"] \
                = np.std(acc_over_rollout, axis=1) / np.sqrt(acc_over_rollout.shape[1])



    for model in models:
        with open('code/' + model + ".csv", "w") as f:
            f.write("x, y, yerr\n")
            for i in range(len(pm[model]["x"])):
                f.write(str(pm[model]["x"][i]) + ", " + str(pm[model]["y"][i]) + ", " + str(pm[model]["yerr"][i]) + "\n")

    # Plot data structure.
    for model in models:
        plt.errorbar(pm[model]["x"],
                     pm[model]["y"],
                     yerr=pm[model]["yerr"])
    plt.ylim((0.0, 1.0))
    plt.title("std: 2.0, rollouts: 8")

    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    #plt.savefig("/home/fvi2si/12_DL/BigPicture/cst_mnist_norot.pdf", dpi=200, bbox_inches="tight")
    plt.show()



    viz_epochs = [19]
    #viz_epochs = [e for e in range(len(accuracy[0]["sequential"]))]

    #viz_std = [1, 2, 3, 4]
    viz_std = [1, 2, 3]



    # Plot accuracies.
    plt.rc("axes", labelsize=14)
    plt.subplot(len(viz_std), columns, 1)
    for n in range(len(viz_std)):
        for e,E in enumerate(viz_epochs):
            plt.subplot(len(viz_std), columns, (e + 1) + n * columns)
            
            acc = accuracy[viz_std[n]]["sequential"][e]
            acc = acc[0:3]
            update_steps = rollout_factor + rollout_factor * np.arange(0, acc.shape[0])

    #        acc = acc[0:3]
    #        update_steps = update_steps[0:3]

            plt.plot(update_steps, acc, 'rx')

            acc = accuracy[viz_std[n]]["streaming"][e]
            # Here the offset "rollout_factor" is wrong, in case we have skips. Then this should be the shortest-path-length.
            update_steps = rollout_factor - 1 + np.arange(0, acc.shape[0])
            plt.plot(update_steps, acc, 'b+')
            plt.ylim((0.0, 1.0))
    #        if n == 0:
    #            plt.title("epoch " + str(viz_epochs[e] + 1))
            if e == 0:
                plt.ylabel("noise " + str(results[viz_std[n]]["sequential"]["noise_std"]))
            if e > 0:
                plt.yticks([])
            if n < len(viz_std) - 1:
                plt.xticks([])
            else:
                plt.xlabel("# inference update steps")
            plt.plot([update_steps[0], update_steps[0]], [0, 1], '-')

        # Plot mixed parameters.
        if param_swap:
            plt.subplot(len(viz_std), columns, len(viz_epochs) + 1 + n * columns)
            acc = accuracy[viz_std[n]]["streaming_2_sequential"][0]
            acc = acc[0:-1]
            update_steps = rollout_factor + rollout_factor * np.arange(0, acc.shape[0])
            plt.plot(update_steps, acc, 'rx')

            acc = accuracy[viz_std[n]]["sequential_2_streaming"][0]
            # Here the offset "rollout_factor" is wrong, in case we have skips. Then this should be the shortest-path-length.
            update_steps = rollout_factor + np.arange(0, acc.shape[0])
            plt.plot(update_steps, acc, 'b+')
            plt.ylim((0.0, 1.0))
            plt.yticks([])
            if n == 0:
                plt.title("param. swap")
            if n < len(viz_std) - 1:
                plt.xticks([])


        # Plot relative number of saved frames.
    #    plt.subplot(len(accuracy), len(viz_epochs) + 2, len(viz_epochs) + 2 + n * (len(viz_epochs) + 2))
    #    plt.plot(0.2 + 0.1 * np.arange(0, len(smallest_frame_ratio[n])), smallest_frame_ratio[n])
    #    plt.plot([0.0, 1.0], [1.0, 1.0], '-')
    #    plt.ylim((-1.0, 2.0))
    #    if n == 0:
    #        plt.title("rel. frame save")
    #    if n < len(accuracy) - 1:
    #        plt.xticks([])

    fig = plt.gcf()
    fig.set_size_inches(4, 8)
    plt.savefig("/home/fvi2si/12_DL/BigPicture/cifar_12.pdf", dpi=200, bbox_inches="tight")
    plt.show()
