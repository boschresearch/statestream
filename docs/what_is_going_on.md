What is going on?
=================
([back to documentation](README.md))

The statestream toolbox provides tools to explore (train, evaluate, 
visualize, ...) streaming (for more details, what is meant with streaming, 
see the [from one frame to the next](from_one_frame_to_the_next.md) 
tutorial) networks. Every network is hereby modularised into four different
types of building blocks, which are briefly described below. In general 
and dependent on its specification, every building block has parameters 
which govern its behavior over time and may also change over time.

* __Neuron pools__ (brief NP): Neuron pools can be thought of as the nodes of the network graph.
  They store the current state information (a.k.a. feature-maps, activations) of the network.
  The state of a neuron pool is always represented as a 4D array.
  Most prominent parameters of neuron pools are biases.
* __Synapse pools__ (brief SP): Synapse pools can be thought of as the edges of the network graph.
  They represent transformations between neuron pools and work in a multi-input single-output way.
  Most prominent parameters of synapse pools are weight matrices.
* __Plasticities__ (brief PLAST): Plasticities are automated procedures by which parameters of building blocks may change
  over time. Statestream provides loss/gradient based plasticities, such as back propagation for autoencoders or
  classification but also gradient-free plasticities, such as hebbian learning. Another way parameters may generally
  change is via GUIs (user manipulation).
  Plasticities themselves may also have parameters, such as learning rates.
* __Interfaces__ (brief IF): Interfaces provide communication between the streaming network and the outside world.
  They may, for example, present images from a dataset, camera, or different source to the network or let the network steer a robot.

All these building blocks have to be specified in an .st_graph file from which a network can be created and explored.
During runtime, a core process governs and syncronizes all network modules.
When the core is started with a valid st_graph or st_net file as argument, 
the entire network is instantiated in a distributed manner. For the different building blocks
of the network (neuron-pools, synapse-pools, plasticities and interfaces) 
separate processes are spawned.
Every building block of the network has its own process. If specified these processes my push expensive computations on a GPU.
Besides these network processes the core and potential GUIs have own processes.

Several cores can be started with separate networks at the same time and are discriminated with a _session_ index.

During runtime the network (states, parameters and more) is stored in shared memory which is accessible by all processes.

Except GUIs, all processes are synchronized by the core process and work in two phases where they read / write 
from / to shared memory. 

From one time-step or neural frame to the next, all neuron pool states are updated once. This can be
thought of as overwriting all states of the network with the results of a 1-step rollout of the entire network. Hence,
the distance information can travel during one frame is heavily dependent on network topology. Considering, for example,
a VGG16 like network topology, it would take the network 16 frames to yield a classification result based on an image.
Using skip connections, the number of necessary frames may then be reduced. For more details, see 
[from one frame to the next](from_one_frame_to_the_next.md).