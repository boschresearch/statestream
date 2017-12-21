Device handling
===============
([back to documentation](README.md))


The association which network part should be executed on which hardware device is realized in a two step process. First, a set of visible devices is fetched from the environment variable **CUDA_VISIBLE_DEVICES** or if not specified from the ~/.statestream/stcore.yml configureation file:

```
	...
	visible_devices: [0, 1, 7]
	...
```

Second, every neuron-pool and plasticity network item can define a **device** specification. This specification can either be a GPU or the cpu. If no **device** was specified for an item or the **device** specification was set to __cpu__, this item will be executed only on cpu. The GPU device is the relative device id from the list of visible devices, e.g.:

```
...
neuron_pools:
	conv1:
		...
		device: gpu:2
		...
	conv2:
		...
		device: gpu:0
		...
	conv3:
		# no device specified
		...
...
```

In this case, the neuron-pool __conv1__ will be executed on GPU 7, __conv2__ on GPU 0, and __conv3__ on the cpu.

