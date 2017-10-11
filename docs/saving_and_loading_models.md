Saving and loading models
=========================
([back to documentation](README.md))

There are two ways to save / load a network:

* **Via the statestream terminal**: In the statestream terminal you can save only the graph or the entire network to a file with the commands **savegraph** and **savenet** resprectively and the target file name.
The file will be placed in the directory specified in **~/.statestream/stcore.yml** under the parameter **'save_path'**. Extensions .st_graph or .st_net will be added automatically if needed. To load the graph or the network file, start a new statestream session using the saved file. All loaded parameters will come effective after the first frame.

* **Via the visualization**: See the [hotkey](hotkeys.md) section.
