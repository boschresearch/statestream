Getting creative
================
[back to documentation](README.md)

This guide aims to give some advice how the toolbox can be used and extended.



Specifying different network architectures
------------------------------------------

The main application of the toolbox is to specify new network architectures via st_graph files, to explore their behavior via the visualization and share them with the community. At this point, we aim to keep the API of the specification files backward compatible which cannot be said for any other code API in the statestream toolbox. Please see also the [network specification](network_specification.md).



Writing new clients
-------------------

To adapt scheduling of training / evaluation procedures, one can write new clients. Please see the [clients specification](clients.md).



Adding new interfaces
---------------------

On the basis of existing interfaces, it is relatively easy to add new interfaces for different applications, hardware, simulations, datasets, and so on. Please see also the [interface section](interfaces.md).



Adding new features
-------------------

* **activation functions** and **losses**: Additional activation functions and losses can be added in the backends.
* **optimizers**: New optimizers can be added extending **neuronal/optimizers.py** and **meta/optimizers.py**.



Advanced extensions
-------------------

The statestream toolbox can be extended in many more advanced ways which require deeper insight in the toolbox:

* own visualization tools by using the representation of the network in shared memory;
* new plasticities for training;
* new meta-variables for the exising visualization tool;



Not advised
-----------

It is not advised (for now) to change the following:

* neuron-pool and synapse-pool modules
* core module




