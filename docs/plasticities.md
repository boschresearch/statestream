Plasticities
============
([back to documentation](README.md))

Plasticities change the network's parameters during runtime. Right now three different types of plasticities are implemented and are specified with the **type** parameter for every plasticity:

* __loss__: Plasticities of this type provide loss-based changes of parameters which are the most common / used rules for parameter updates. Please see also the [loss-plasticity documentation](plasticity_loss.md).
* __L_regularizer__: Plasticities of this type provide norm regularizers of network paramters.
* __hebbian__: To this point hebbian plasticity may only be applied to weight matrices between nps without spatial dimensions. There exist four different hebbian plasticities combining anti/hebbian with a hebbian over the temporal derivative over the target feature maps: "hebbian", "anti_hebbian", "dt hebbian", "dt anti_hebbian"

Other important plasticity specification parameters are:

* **parameters** (type: list): This specification parameter lists all the network's parameters which should be updated by this plasticity.
* **ignore_split** (type: bool): This specifies if the plasticity is affected by the session wide split parameter. Please see also the [parallelization documentation](parallelization.md).

Because plasticity items may be very slow compared to other items in the network graph, they pose a potential bottleneck.
One solution to this can be to specify several instances for the same plasticity with differnt temporal delays / offsets.
This way, a plasticity runs several times in parallel. Another way to prevent these types of bottlenecks is to use the **bottleneck** parameter. See also the [temporal control documentation](temporal_controls.png).

The prefered parameter update direction of each plasticity is now passed through an optimizer. Every optimizer has a learning rate parameter **lr** of type float. Right now two different optimizers are implemented:

* __grad_desc__: This optimizer only passes the raw update scaled by a learning rate.
* __adam__: For more details on the adam optimizer refer to [Kingma et al. 2015](references.md). Specification parameter for the adam optimizer are:
	* **momentum** (type: float): Default value is 0.999.
	* **decay** (type: float): Default value is 0.99.
* __rmsprop__: For more details on the rmsprop refer to [RMSprop](references.md). Specification parameter:
	* **rho**: Default value is 0.9.

New optimizers can be added extending [neuronal/optimizers.py](../statestream/neuronal/optimizers.py) and [meta/optimizers.py](../statestream/meta/optimizers.py).



Please see also the examples folder for more details.
