Clients
=======
([back to documentation](README.md))

Statestream provides two types of clients (_core_ and _system_ clients) to perform automated user defined actions during runtime. The major differences between core and system-clients are:

* Separate processes (from core process): Core clients provide methods which called inside the core process while system clients run in a separate / parallel process.
* Online parameter changes (e.g. learning rate schedule): Core clients may perform online parameter changes during runtime, system processes can not change the network's parameters.
* GPU backend: Core clients are not supposed to use the GPU backend, system clients are.



Core clients
------------

Core clients are the main approach to schedule / organize the network execution process. They can be thought of as a python class providing certain methods which are executed during certain execution phases of the core process. In case no core clients are specified, the network will "just" stream (incl. training etc.). With core clients for example the following features are meant to be realized:

* Scheduling the learning rate or other optimization parameters.
* Set model save-points during training.
* Online logging of measures.
* ...

Core clients do not run in an own process and are executed inside the core process. Please see also the [core client class](../statestream/utils/core_client.py) from which a new core client type can be inherited.

A simple example of a core client is given with the provided learning-rate scheduler: [st_graph file](../examples/test_core_clients.st_graph), [client file](../examples/core_clients/lr_scheduler.py).

A richer example managing an entire training / validation / test session is also provided: [st_graph file](../examples/test_core_client_trainvaltest.st_graph), [client file](../examples/core_clients/trainvaltest.py).



System clients
--------------

System clients are the main tool to realize user defined heavy computations in a synchronized way parallel to the rest of the network. Every system client is executed as a separate process very similar to all network items (e.g. neuron-pools) and respects the read- and write phases. All [meta-variables](meta_variables.md) are realized as system clients. At the moment system clients have no influence on the network behavior for example cannot change neuron-pool states or networt parameters but are a tool to provide computation expensive metrics / measures / ... online during network execution and training.