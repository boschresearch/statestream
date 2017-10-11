Editing topology parameters (heavy UC)
======================================
([back to documentation](README.md))

Statestream to some extent provides online changes of topology parameters during runtime. To change a topology parameter select **edit** after a RMB click on an item and a selection window will open with all available / changeable topology parameters for this item. After hitting **done**, changes
will be stored in a separate network graph file inside the visualization and more changes can be made. All these changes are not effective yet. Hitting the **e** key will delete all editing changes. Hitting **SHIFT+e** will instruct the core to re-build all affected network parts.

The purpose of the editing feature is, for example, in order to change an activation function of a single neuron-pool, not to have to shutdown the entire network, edit the .st_graph file and startup everything again, but to re-start only necessary parts of the network.

**Note:** Changing topology parameters for a certain item may also effect other items, especially plasticities which in turn will also be restarted automatically. For each editing, the core will determine ALL (also indirectly) affected items by the editing and will re-start all these items after starting the editing (**SHIFT+e**). 

This feature is still experimental and should only be used under the following conditions:

* Only start editing when the network is out of its initialization phase (= could be set into streaming mode).
* After initializing an editing, wait for it to become effective before starting a new editing.
* Editing does not go well with saving / loading models because the edited model has the same name as the old one.
* For now, editing will re-initialize all network parameters (not theano graph, but e.g. weights).

List of edit-able topology parameters:

* neuron-pool: **act**, **bias_shape**
* synapse-pool: **rf**, **dilation**

**Note:** This feature, so far, is only a template illustrating how the network topology (e.g. removing / adding items) can be changed arbitrarily in the future. However, at the moment just changing the st_graph file and restarting the network is the preferred solution.

