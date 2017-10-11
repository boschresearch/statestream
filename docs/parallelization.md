Parallelization
===============
([back to documentation](README.md))

Each item is executed in its own process. Dependent on the computations done for a particular item, heavy computation
can be put on a GPU device. At the moment GPU support for these computations is only available for neuron-pools and plasticities using the theano backend. 

For each item, a different **config.base_compiledir** is defined using the item's process identifier. All these compilation directories can be found in **~/.statestream/compiledirs/**. This enables parallel theano compilation across items and accelerates the initial compilation process of the entire network.

Specifying the target device for a particular item can be easily achieved in the st_graph file by setting the device parameter, e.g.:

```
...
neuron_pools:
	...
	conv1_1:
		shape: [128, 64, 64]
		device: cuda0
	...

```

The device parameter is set to 'cpu' by default.

More than one item's computations can be put on a single GPU device, as long as enough GPU RAM is available. For now, statestream
does not test for this.

Please see also the [temporal controls](temporal_controls.md) section.



The split parameter
-------------------

The split parameter is a session wide integer. It splits the agents into two groups from which only the second one is used session wide to compute parameter updates. It may assume values between zero and the number of agents and is set to zero be default, hence plasticities use all agents to compute parameter updates.

