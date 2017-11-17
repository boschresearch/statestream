Unsharing parameters
====================
([back to documentation](README.md))

By default, all parameters (e.g. weight matrices, biases, etc.) are shared across agents (samples in a batch). But sometimes one is interested in not using parameter sharing across samples in a batch. In statestream, this can be accomplished by using the **unshare** specification in an .st_graph file. This specification is always given in form of a list, containing all parameters which should not be shared across agents / samples.

Right now this feature is only available for dense / convolutional (esp. not 1-D convolutional) layers. Please also see the example [examples/test_sp_weight_unshare.st_graph](../examples/test_sp_weight_unshare.st_graph).

