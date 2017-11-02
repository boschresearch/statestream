Neuron-pools
============
[back to documentation](README.md)

Neuron-pools (brief NP) store the current state (a.k.a. feature maps or layer activations) of the network. The neuron-pool state is always represented as a 4-dimensional tensor: agents x features x spatial_X x spatial_Y. Each NP has its own process and is processed in parallel to all other network components.

Each frame, every neuron-pool performs the following processing steps in this order:

1) Sum up all post-synaptic inputs.
2) Add noise terms.
3) Apply batch-normalization.
4) Apply layer-normalization.
5) Multiply gain.
4) Apply activation function.
5) Add bias.
6) Apply dropout.
7) Apply zoneout.


Specification parameters
------------------------

* **period** (type: int): Number of neuronal frames every which the neuron-pool is updated.
* **period offset** (type: int): Offset of neuronal frames for neuron-pool update.
* **device** (type: string): Device type the neuron-pool should be executed on (e.g. cpu, gpu0).
* **tags** (type: [str, .., str]): A list of tags for this neuron-pool.
* **shape** (type: [int, int, int]):
* **act** (type: str): An activation function for the output of this NP. Default is set to 'Id'. An arbitrary activation function can be provided, using the dollar sign $ as a placeholder (compare [text example](../example/test_np_activations.st_graph))
* **noise** (type: str): A noise added after the activation of this NP. See the [noise terms](noise_terms.md) for further details.
* **dropout** (type: float): The dropout rate used for this neuron-pool. Must be between zero and one.
* **zoneout** (type: float): The zoneout rate used for this neuron-pool. Must be between zero and one. For now, zoneout only works for self-connected NPs, otherwise this parameter will be ignored.
* **bias_shape** (type: str): Bias parameter used for this neuron pool. The NP will have a parameter **b**. Available settings are:
	* 'full': bias extends across space and feature dimensions
	* 'feature': a bias value for every feature map (default)
	* 'spatial': a bias value for every spatial location
	* 'scalar': a single scalar bias is applied to the entire layer
	* False: no bias is applied (equal to zero bias)
* **gain_shape** (type: str): Same as **bias_shape** only that this parameter is applied multiplicatively. The NP will have a parameter **g**. Default is False.
* **batchnorm_mean** (type: str): String specifying how batch normalization is applied. Possible values are 
	* 'full': mean is computed over the entire layer (and batch)
	* 'feature': mean is computed across all features (and batch) for each pixel separately
	* 'spatial': mean is computed across space (and batch) but separately for each feature map
	* 'scalar': mean is computed across only across the batch.
	* False: no mean subtraction is performed for batch normalization (default)
* **batchnorm_std** (type: str): Same as **batchnorm_mean**, now for the normalization with the standard deviation. Available settings and default are the same.
* **layernorm_mean** (type: str): String specifying how layer normalization is applied. Possible values are 
	* 'full': mean is computed over the entire layer
	* 'feature': mean is computed across all features for each pixel separately
	* 'spatial': mean is computed across space but separately for each feature map
	* False: no mean subtraction is performed for layer normalization (default)
* **layernorm_std** (type: str): Same as **layernorm_mean**, now for the normalization with the standard deviation. Available settings and default are the same.

Please see also the examples folder for more details.
