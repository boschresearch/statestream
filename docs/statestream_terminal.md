Statestream terminal
====================
([back to documentation](README.md))

The statestream terminal provides a text interface for the user to the statestream core process.
Functionalities of the terminal are very limited and can be listed by executing **?** or **help** inside the terminal.

Available commands
------------------

* **bottleneck** [PID (int)] [factor (float)]: This will reset the bottleneck factor of the process PID to the specified factor. Please see [temporal controls](temporal_controls.md) for further details.

* **clean on/off**: Switch display mode for statestream terminal. In clean mode, the screen is cleared. For debugging
purposes it is sometimes useful to set clean off.

* **exit**: Exit everything. This is the clean way to shutdown statestream.

* **?/help**: Print available commands.

* **nps**: Print a list of all neuron-pools.

* **offset** [PID (int)] [offset (int)]: This will set the offset of the process PID to the specified offset. Please see [temporal controls](temporal_controls.md) for further details.

* **period** [PID (int)] [period (int)]: This will set the period of the process PID to the specified period. Please see [temporal controls](temporal_controls.md) for further details.

* **profile core**: Prints some profiling information for the core process.

* **pstate**: Print some information about parallel processes.

* **savegraph**: Saves only the network graph to a **.st_graph** file.

* **savenet**: Saves the current state of the entire network (incl. neuron pool states, parameters, optimizer states).

* **shm. ...**: General (rudimental) terminal tool to investigate shared memory objects (states, parameters, etc.). This should only used with **clean** set to **on**.

* **split** [split (int)]: This resets the split to a value between 0 and the number of agents. By default split is zero. Please see [parallelization documentation](parallelization.md) for further details.

* **sps**: Print a list of all synapse-pools.

* **state**: Print some information about the current network/system state.

* **stream/pause**: Start and pause the network.

* **viz on/off**: Activate / deactivate GUI. GUI specific code changes may be tested during runtime by closing the GUI
  (viz off), changing / saving the code and turning the GUI on again (viz on).
