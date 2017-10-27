Troubleshooting
===============
[back to main](../../README.md)



* In case of any errors it is a good idea to check the .st_graph file:
	* No commata are allowed at end of line.
	* Names of items etc. are consistent.
* In case the core crashes: Please send the error message to one of the developers. This should not happen.
* In case the visualization crashes: The visualization can be restarted over the statestream terminal (viz on). Mayby one first has to turn it off (viz off) before turning it on again. This is also possible after a code change of the visualization.
* If color problems seem to occur in the visualization, change the boolean **color_correction** parameter in **~/.statestream/stviz.yml**. If still color problems occur please send a screenshot to a developer.
* If your session id at the beginning is counting up while always only working with one session, then this may be
caused by un-save statestream shutdowns. One can use the **statestream/utils/free_all_stshm.py** script (this
should not be done while a session is open):

```
python utils/free_all_stshm.py
```

* Sometimes when changing from the statestream terminal into another system window (especially when this
  other window is full screen) the terminal holds. When this happens one can change back into the terminal
  window and click inside. This most of the times wakes the terminal.
  