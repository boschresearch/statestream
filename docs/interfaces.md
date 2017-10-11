Interfaces
==========
([back to documentation](README.md))

The purpose of interfaces is to provide a transparent module that provides an 
input / output interface to the neuronal network. Interface processes are 
synchronisied with the rest of the network. Interfaces are never supposed to be 
differentiable.

As every other item, each interface has a unique name (e.g. for the example 
below: 'mnist_small') and a type. For the type, there has to exist a file 
process_if_*.py (inside statestream/interfaces/) with a class ProcessIf_* 
(where * is a placeholder for the if type). There may be several interfaces 
of the same type in one network (st_graph), with different names. For every 
specific interface, there should be a short description in how necessary 
datasets should be made available for this interface (e.g. if to unpack and 
how to specify the source folder, etc.).

Every interfaces has to specify its inputs and outputs as list of topic names. 
These names (for the mnist example below these are: mnist_pred, mnist_image, 
mnist_label) are specific for the interface type. If a neuron-pool with this 
name exists, the interface IO will be associated with this neuron-pool. To 
make these interface dependent topic names independent from actual neuron-pool 
names, hence allowing arbitrary np names, a name remapping may be specified as 
shown for the example below. Here, for example the mnist interface specific 
topic 'mnist_image' is remaped onto the neuron-pool with name 'retina'.

Each interface writes all outputs variables to shared memory. In the read-phase 
of the next frame the associated neuron-pool then reads this variable to update 
its state.

In general the interface specification may also include other parameters which 
specify the functionality of the interface. For the mnist example below, we see 
that the path to the data mnist dataset as well as some presentation time parameters
are set. These parameters heavily rely on the interface type (e.g. a mere simulation 
if may not rely on any dataset).

Period and period offset parameters can also be specified for interfaces, 
controlling the execution of an interface.

* **period** (type: int): Number of neuronal frames every which the interface 
is updated. Default is 1.
* **period offset** (type: int): Offset of neuronal frames for interface update. Default is 0.
* **mode** (type: int): Interfaces can be run in _batch_, _single_, _epoch_ or _test_ mode 
(mode equal 0, 1, 2, 3 respectively). Default is batch-mode (mode = 0). Note that _batch_ and _single_ modes are implemented for all interfaces, while _epoch_ and _test_ modes are only available for the mnist and cifar10 interfaces yet.
    * _batch_: Here, samples are drawn from the dataset with replacement. The split parameter is respected.
    * _single_: Here, samples in the batch are identical copies of the first sample. This mode can be used to analyse for example network statistics or (epistemic) network uncertainty. The split parameter is of course not respected by the interface.
    * _epoch_: Samples are drawn epoch wise without replacement. The split parameter is respected.
    * _test_: Samples are drawn epoch wise only from the test dataset. The split parameter is not respected.

```
interfaces:
    mnist_small:
        type: mnist
        in: [mnist_pred]
        out: [mnist_image, mnist_label]
        remap:
            mnist_image: retina
            mnist_label: label
            mnist_pred: prediction
        source_file: /opt/dl/data/mnist.pkl.gz
        min_duration: 12
        max_duration: 16
```

Each interface file process_if_*.py must also provide three extra functions:

* **if_interfaces**: Returns the specific sub-interfaces as dictionary of strings 
for the interface.
* **if_init**: Specifies how the extra interface parameters have to be initialized. 
In principle the same as for the other item types.
* **if_shm_layout**: Specify the shared memory layout for all variables / parameters 
of the interface. In principle the same as for the other item types.

Please note that here the mnist interface uses a local copy of the
mnist dataset **mnist.pkl.gz** which can be downloaded e.g. from 
[here](http://deeplearning.net/tutorial/gettingstarted.html)) or with:

```
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
```

And the path to this file must be specified under **source_file** in the .st_graph
network file, such as **path to reposito/examples/mnist_small.st_graph**.

For now, interfaces provide performance measures such as confusion-matrices or
prediction accuracies. For most of them to work, note that the **min_duration**
specifying the minimun frame number for which a specific class sample is shown,
must be larger then the minimum path-length from network input to network output.

Please see also the examples folder for more examples, especially the files 
**test_interfaces_\*.st_graph** for further details.
