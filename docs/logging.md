Logging
=======

([back to documentation](README.md))

At first start up of the statestream core with or without an **st_graph** file argument, some configuration files are created in the folder **~/.statestream/**. Default values for this first setup and also if a configuration is not found in this local file can be found in [statestream/utils/defaults.py](../statestream/utils/defaults.py). Overview over logging files:

* **~/.statestream/shm-%d.log**: Logs a list of all shared memory variables for this (%d) session and their sizes in bytes.
* **~/.statestream/pid.log**: Logs all process ids.
* **~/.statestream/log_manipulation/name.manip_log**: Logs all parameter changes applied to network of name **name**.
* **~/.statestream/viz/name-brainview-%d(.jpg)**: Logs a GUI parametrization (item positions, meta-variables, etc.) and a preview of this parametrization. Those can be created / saved / loaded from the visualization.

In the **~/.statestream/stcore.yml** configuration file, a general save directory can be specified using the **save_path** parameter. This folder will store:

* All models saved from the provided visualization.
* In the sub-directory **graph_tikz**, a latex file will be generated containing the network architecture as well as the network rollout for all plasticities. This file can be compiled e.g. using **pdflatex demo.tex**.

