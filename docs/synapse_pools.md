Synapse-pools
=============
([back to documentation](README.md))

Synapse-pools (brief SP) represent transformations between [neuron-pool](neuron_pools.md) states. They are specified in a multi-input (sources) single-output (target) way. In principle, every defined neuron-pool can be a synapse-pool's source or target, hence allowing arbitrary network topologies (e.g. cycles, self-connections). We call the output of a synapse-pool its post-synaptic activation and this is always a tensor of the same dimension as its target neuron-pool (modulo broadcasting). All synapse-pools targeting a specific neuron-pool are also processed in this neuron-pools's process.

All sources of a synapse-pool are specified in a list of lists way, e.g.:

```
synapse_pools:
	sp1:
		sources:
		- [np1, np2]
		- [np3, np2, np4]
		- [np1]
		target: np1
```

Each sub-list (e.g. [np1, np2] in the example) specifies the inputs to a factor. In this example, sp1 would have three factors. For the example above, we would have the following six weight matrices: W_0_0, W_0_1, W_1_0, W_1_1, W_1_2, W_2_0 and the SP would compute:

post_synaptic = (W_0_0 * np1 + W_0_1 * np2) x (W_1_0 * np3 + W_1_1 * np2 + W_1_2 * np4) x (W_2_0 * np1)

Here '*' denotes a convolution and 'x' the element-wise multiplication. In case the pre-processing projection (ppp) parameter is specified, the convolutions become:

W_i_j * np -> W_i_j * P_i_j * np

The following steps are performed by each synapse-pool in the given order, some may be omitted depending on the specification:

1) Apply pre-activation function to all inputs of the SP.
2) Apply pre-processing-projection to every source. The target feature dimension of the projection is the parameter given in the **ppp** parameter.
3) Apply main convolution / spatial scaling. In case the target NP has a smaller spatial dimension the main convolution will be performed with an according stride parameter. In case the target NP has a larger spatial dimension, the source representation will be scaled up by locally copying the source representation and applying the main convolution afterwards.
4) For each factor, its inputs are summed up.
5) For each factor, its activation is applied.
6) For each factor, its bias is added.
7) All factors are multiplied.
8) Avgout is applied.
9) Maxout is applied.
10) The SP activation is applied.
11) Noise is added.

In the minimal case, only step 3) is performed.

Specification parameters
------------------------

* **period** (type: int): The NP is updated every this many frames.
* **period offset** (type: int): Offset of neuronal frames for neuron-pool update.
* **device** (type: string): Device type the neuron-pool should be executed on (e.g. cpu, gpu0).
* **tags** (type: [str, .., str]): A list of tags for this neuron-pool.

* **source** (type: list(list(str))): A list of lists of source neuron-pools for this synapse-pool. The length of the outer list defines the number of factors of this SP and the entries of the inner lists define the inputs (neuron-pools) for each of the factors.
* **target** (type: str): The target neuron-pool.
* **ACT** (type: str): An activation function for the output (= post-synaptic activation) of this SP. Similar to NP activation, arbitrary (combined, parameterized) functions can be specified. The default is set to 'Id'.
* **act** (type: list(str)): An activation function for each factor. Similar to NP activation, arbitrary (combined, parameterized) functions can be specified. The default is set to 'Id'.
* **pact** (type: list(list(str))): A pre-activation function for each input of the SP. This is applied before all other transformations. Similar to NP activation, arbitrary (combined, parameterized) functions can be specified. By default all pre-activation functions are the identity and are hence ignored.
* **noise** (type: str): Noise added after the activation of this SP. See the [noise terms](noise_terms.md) for further details.
* **dilation** (type: list(list(int))): Dilations for each input.
* **rf** (type: int or list(list(int))): The receptive field size of the SP. Int is allowed only in the case of one input NP. If rf is 0 or not given then this SP is a full (a.k.a. dense) connection.
* **weight_fnc** (type: list(list(str))): A function applied to each weight parameter (e.g. **exp**). The default is **Id** and will be ignored.
* **min/max str** (type: float): Specifies an optional min / max bound for a specific parameter **str**. E.g. __max W_0_0: 1.0__ would implement an upper bound of 1.0 on the parameter W_0_0. By default no bounds are applied.
* **init str** (type: str): String storing an optional initialization for a specified parameter **str**. See also the section "Provided parameter initializations" below. E.g. __init W_0_0: normal__ would initialize the parameter W_0_0 from the standard normal distribution.
* **factor_shapes** (type: list(str)): A shape for each factor (see also [this](shapes.md)).
* **target_shapes** (type: list(list(str))): A shape for the internal target factor (see also [this](shapes.md)).
* **bias_shapes** (type: list(str)): A shape for each factor bias (see also [this](shapes.md)).
* **weight_shapes** (type: list(list(str))): A shape for each weight matrix (see also [this](shapes.md)).
* **avgout**, **maxout** (type: int): Integer specifying a factor for the number of features used for maxout or avgout. The default is 1 for both and has no effect.
* **ppp** (type: list(list(int))): A Pre-Processing Projection dimension for each source neuron-pool. This projection is a convolution with receptive field sizes equal to one, altering (pre-processing, selecting) the source's features. The default is 0 for all sources and ignores this pre-processing step.

If **NOSPRING** is added as a tag for a SP, this SP will not enforce NP attraction forces in the main visualization.


Provided parameter initializations
----------------------------------

By default, all weight parameters are initialized with Xavier initialization [[Xavier et al. 2010]](references.md). In general, one of the following initializations can be specified:

* **xavier**: All weights are drawn independently between [-bound, bound] where bound = sqrt(6 / (fan_in + fan_out)).
* **xavier_float**: Same as **xavier**, but **float** will be used as a factor for the boundaries of the uniform distribution (e.g. **xavier_4.0**).
* **id**, **Id**: Initialize the parameters so that the matrix multiplication will be the identity. This requires that the number of source features equals the number of target features. In case the number of target features is a multiple / factor of the number of source features, the Kronecker-product of **eye** and **ones** will be used, **id** for kron(eye, one) and **Id** for kron(one,eye) (please see also [examples/test_sp_weight_init_id.st_graph](../examples/test_sp_weight_init_id.st_graph). 
* **-id**: Same as **id** but with opposite sign.
* **bilin**: Bilinear interpolation, which are for example used for upsampling.
* **normal**: All weight parameters are independently drawn from the standard normal distribution.
* **normal_float1_float2**: All weight parameters are independently drawn from the normal distribution with mean equal to **float1** and standard deviation equal to **float2** (e.g. **normal_-0.5_2.0**).
* **uniform**: All weight parameters are independently drawn from the standard uniform distribution in [0,1].
* **uniform_float1_float2**: All weight parameters are independently drawn from the uniform distribution in [**float1**, **float2**].

Initializations are implemented in [meta/synapse_pools.py](../statestream/meta/synapse_pool.py).

Please see also the examples folder for more details.


Transformer synapse-pool
------------------------

One special type of synapse-pool exist to enable spatial transformer operations [[Jaderberg et al. 2015]](references.md). To use this feature the synapse-pool has to have the tag **TRANSFORMER** and exactly two sources, first the to be transformed layer and second the transformation, i.e.:

```
neuron_pools:
	layer:
		shape: [128, 64, 32]
	transform:
		shape: [2, 64, 32]
	transformed_layer:
		shape: [128, 64, 32]
synapse_pools:
	transform:
	    source: [[layer], [transform]]
	    target: transformed_layer
	    tags: [TRANSFORMER]
```

The two transformation features represent the _dx_ and _dy_ components of the pixelwise transformation and are percent of image resolution. Please also see the example in [examples/test_interface_transform_cifar10.st_graph](../examples/test_interface_transform_cifar10.st_graph).
