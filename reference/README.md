Reproducing experiments
=======================



This document explains how the experimental results of [will be added after publication] can be reproduced.
These scripts are focussed on the architectures and rollouts used in [will be added after publication] and do not support all features of the statestream toolbox.
These scripts do not aim to provide a general statestream - Keras interface but rather to benchmark rollout patterns against another.



Requirements
------------

The same requirements as for the statestream toolbox and additionally [Keras](https://keras.io) are needed.



Dataset preparation
-------------------

Download datasets:

* MNIST [mnist.pkl.gz](http://deeplearning.net/data/mnist)
* CIFAR10 [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar.html)
* GTSRB [GTSRB_Final_Training_Images.zip](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads) 

Unpack GTSRB (once) and CIFAR10 (twice) and change paths / files in [stdatasets.py](stdatasets.py) accordingly.



Training and Evaluation
-----------------------

To train a certain model, call:

```
	python keras_model_trainer.py <model specification file> <mode>
```

e.g.:

```
	python keras_model_trainer.py model_specifications/mnist_recskip.st_graph streaming
```

This will train and evaluate the specified model and will store three files in the **model_trained/** folder:

```
	model_trained/<model name>.h5
	model_trained/<model name>.json
	model_trained/<model name>.results
```

The first two files store the trained Keras model and the third file contains evaluation results. 
The following command will train all provided models (dependent on available hardware this may take some days / weeks):

```
	./train_all
```

We advise to first train the MNIST models and inspect the results for these experiments.



Visualize results
-----------------

To visualize results from a previously trained model (both, streaming and sequential versions are necessary), call:

```
	python keras_model_plotter.py <model specification file>
```

e.g.:

```
	python keras_model_plotter.py model_specifications/mnist_recskip.st_graph
```

The visualization script will open the stored results for this model and its rollouts and visualize them.



Content overview
----------------

```
	keras_model_builder.py
```

This file contains the central class to build a rollout window as Keras model from a statestream specification file (see model_specifications/ folder,
e.g., [mnist_recskip.st_graph](model_specifications/mnist_recskip.st_graph)).
In the specification file only the non-streaming rollout has to be specified explicitly because the streaming rollout is unambiguous
and all synapse-pools (edges) are unrolled streaming (to next frame).
A non-streaming rollout pattern is specified by adding the tag 'stream' to synaptic-pools (edges) which should be unrolled streaming.
All other synapse-pools will be integrated in the rollout window in the sequential manner (inside same frame).


```
	keras_model_plotter.py
```

This script can be called with certain options (see below) to visualize information about trained models.


```
	keras_model_trainer.py
```

This script can be called with certain options (see below) to train models given as statestream model specifications.


```
	README.md
```

This readme.


```
	stdatasets.py
```

Some tools to load and prepare MNIST, CIFAT-10, and GTSRB datasets.


```
	train_val
```

Script to train all provided models in __model_specification/__ folder.







Model specification
-------------------

Besides the standard options / specifications for st_graph model files, additional parameters have to be set in order to compare 
streaming and sequential rollout patterns:

* __rollout_factor__: This is the update-factor from the paper and specifies the number of necessary update-steps to compute the entire first frame. 
For the streaming rollout pattern, this is always one and the given number here is the update-factor for the specified sequential rollout pattern.
* __first_response_streaming__: This is the least number of **frames** for which the streaming rollout pattern yields a response.
* __first_response_sequential__: This is the least number of **frames** for which the specified sequential rollout pattern yields a response.
* __shortest_path_sequential__: This is the least number of **update-steps** for which the specified sequential rollout pattern yields a response.

