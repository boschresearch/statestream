Network specification
=====================
([back to documentation](README.md))

The graph of a network is specified in an .st_graph file using the yaml format. Besides explicitly defining instances of the basic network items (neuron-pools, synapse-pools, plasticities and interfaces) the network specification offers more sophisticated mechanisms to specify complex network architectures. On the top level, the following parameters have to / may be set:

* **name** (type: str): Mandatory. This is the name of the network. It is for example used to save item arrangement in the visualization or the entire model for later use.
* **agents** (type: int): Mandatory. This is the number of agents executed in parallel (a.k.a. batchsize).
* **neuron_pools** (type: dict): Mandatory (but may be empty). This contains the listing of all [neuron pools](neuron_pools.md) (except those from modules / imports). Neuron-pools hold the network's current state.
* **synapse_pools** (type: dict): Mandatory (but may be empty). This contains the listing of all [synapse pools](synapse_pools.md) (except those from modules / imports). Synapse-pools represent transformations between neuron-pool states.
* **plasticities** (type: dict): Mandatory (but may be empty). This contains the listing of all [plasticities](plasticities.md) (except those from modules / imports). Plasticities train / learn network parameters.
* **interfaces** (type: dict): Mandatory (but may be empty). This contains the listing of all [interfaces](interfaces.md) (except those from modules / imports). Interfaces provide communication between the network and the external world (e.g. a robot or a dataset).
* **core_clients** (type: dict): Optional. This contains the listing of all [core clients](clients.md). Core clients are broadly applicable, e.g. as scheduler for the training process, logging, online learning rate adaptation.
* **import** (type: list): Optional. This lists all [imports](import_specification.md) from other network specification files.
* **modules** (type: dict): Optional. This dictionary may specify [modules](modules.md) for more efficient network specifications.
* **tag_specs** (type: dict): Optional. This contains the listing of all used [tag specifications](tag_specification.md).
* **globals** (type: dict): Optional. This contains the listing of all [global variables](globals.md) used in the specification.

Statestream reads the network specification and parses the contained dictionary into a new network dictionary where all basic network items are specified explicitly. This explicit network is initialized at the beginning with the explicitly given basic items and modified afterwards through the follwing mechanisms in this given order:

* **import**
* **modules**
* **tag_specs**
* **globals**

If none needed, it is sufficient for nps, sps, plasts and ifs to give an empty dictionary, e.g.:

```
...
neuron_pools: {}
...
```

Every item (np, sp, plast or if) has a 'device' parameter, which is set to 'cpu' by default. Just set it for example to 'cuda0' to shift the computations of this item onto 'cuda0'. Also the old setting 'gpu0' may work for older theano versions.

Prior to instantiation of this structure in multiple processes, some sanity checks are performed using **statestream.meta.network.is_sane()**.

For examples, please see the examples/ folder.
