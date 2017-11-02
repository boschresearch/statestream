The Rollout-View
================
([back to documentation](README.md))

Besides the main visualization, a rollout view of the network graph can be shown during runtime to better understand the network's temporal behavior.
The rollout view lists all neuron-pools and enrolles the network for a specifyable depth into the future. No meta-variables or sub-windows are available in rollout view.

Hovering the mouse over a NP will show all past and future NPs that influenced or will be influenced by this neuron-pool. Red colored NPs indicate that this neuron-pools state is invalid because it is from the future (compare also image below).

The rollout-view is still experimental and can be startet via the statestream terminal command **rv on**. 

![rollout view](resources/rollout_view.png)

