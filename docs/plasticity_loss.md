The Loss plasticity
===================
([back to documentation](README.md))

Plasticity of type loss have the following parameters:

* **loss_function** (type: string): This specifies the loss function used to compute the loss. Please see also [neuronal/losses.py](../statestream/neuronal/losses.py) and [meta/losses.py](../statestream/meta/losses.py) for available loss functions.
* **mask** (type: int): An interface might produce besides labels, a map defining the areas (pixels) which should be masked / weighted during loss computation. This **mask** comes in the form of a separate neuron-pool published by the interface and can be associated with the loss for masking by this parameter. This parameter can especially be used for localization tasks such as semantic segmentation to actively ignore network predictions in certain areas of an image. A value of zero in the mask will mean that prediction at that position will be ignored in loss computation.

Please see also the examples folder for more details.
