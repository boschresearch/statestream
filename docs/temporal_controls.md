Temporal controls
=================
([back to documentation](README.md))



Profiling
---------

The duration (in milliseconds) of the read- / write phases can be shown for each item in the visualization (see top
button 8) in the [visualization](visualization.md) tutorial). At the right lower side of each item there should
appear two numbers indicating the read- / write duration of this process. These durations are computed as a mean 
over a time window which can be specified by changing the 'profiler_window' parameter in **~/.statestream/stcore.yml**.
Please see also the **pstate** command for the [statestream terminal](statestream_terminal.md).



Pause / Play for network
------------------------

The entire network can be paused and streamed again either by hitting the pause / stream button in the visualization
(see top button 1) in the [visualization](visualization.md) tutorial) or by typing pause / stream in the statestream
terminal. In pause mode one can propagate a single frame by hitting the one-step button in the visualization (top
button 2)).



Execution period and offset
---------------------------

It is possible to specify a period and offset for each item. Both are specified in frames. The period defines every 
which frame this item is updated / executed and the offset defines the temporal offset / phase relative to the 
period at which the item is executed. 



"Not the bottleneck" option
---------------------------

Each item may have a **bottleneck** factor parameter. If this factor is set, the core process will online adapt the 
**period** of this item's process in a way such that the estimated future duration (read + write) of the process 
matches **factor * bottleneck_duration** where **bottleneck_duration** is the maximal duration over all items
which have no **bottleneck** factor parameter. This is especially usefull to execute plasticities besides 
the network without having them slowing down network execution.
To reset the bottlenet factor for a process / item, use the **bottleneck** command in the statestream terminal,
e.g. **bottleneck 4 0.5** will set the bottleneck factor of the process with PID 4 (see **pstate** statestream
terminal command) to 0.5 and **bottleneck 4 -1** will disable the online period adaptation and set the period
of the process to be constant 1.



Temporal memory
---------------

Temporal memory is an expensive way of storing the recent past of all network information (all states, parameters, etc.) and making it available
for the visualization. This memory is parameterized by a list of integers 'temporal_memory' in **~/.statestream/stcore.yml**.
From left to right these integers indicate the multitude of frames for which the current frame is stored. Here some examples:

* [1, 1, 1, 1]: The last four frames are stored.
* [2, 2, 2]: Every two frames, the last frame is stored (first 2). Then this stored frame is stored again only every second
time the last entry was updated (so every fourth frame). Finally this frame again is only stored every second time the last
entry in temporal memory was updated (last 2). Hence by always storing 'only' three times the network we
get a time window of about 2**3 = 8 frames into the past (with a corse temporal resolution of three supporting points).
* [1, 2, 4, 8]: The last frame is stored (first 1). This last stored frame is then stored every two frames (the 2). This frame is then stored every 4 times the last entry was updated which is in turn stored every 8 updates. This gives a potential window into the past of about 64 frames (with a very course temporal resolution).
* []: No temporal memory will be used. At any time we only have the current state / parameters of the network.

Because the lenght of this list determines the numbers of copies of the entire network that have to be stored and updated,
one should be carefull with specifying long temporal memory for big networks.


Delay
-----

To slow down fast streaming networks, a temporal delay can be added to each frame execution via the visualization (slider in
the top left corner of the visualization).


Please see also the [parallelization](parallelization.md) section.