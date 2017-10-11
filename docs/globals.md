Global specification variables
==============================
([back to documentation](README.md))

The specification of globals in a network specification file enables the use of variables in the network definition. For example, the spatial resolution of a hidden layer often depends on the resolution of the input image. With the specification of globals it is possible to express this dependency and not having to re-specify the hidden layer's resolution, in case the resolution of the image changes, e.g:

```
...
globals:
	glob_res: 32
	glob_factor: 2
neuron_pools:
	image:
		shape: [3, glob_res, glob_res]
	hidden:
		shape: [64, glob_res // glob_factor, glob_res // glob_factor]
...
```

Note that only direct specifications of basic items (nps, sps, ifs, plasts), list entries (as shown in the example above) and entries of lists of lists are changed. 

Global specification variables may not be used recursively.

While it is not directly possible to combine variables used in [modules](modules.md) together with global variables, an indirect approach is shown in the [test example](../examples/test_globals.st_graph) for global variables. 

Because globals are realized using the **eval()** function after all global variables have been replaced with their value, it is possible to use operations.

Note that names of global specification variables should be chosen carefully, because the order in which globals are replaced with their value is arbitrary. For example the specification of the two global variables **rf** and **glob_rf** may cause unintended results.

Please also compare the [example](../examples/test_globals.st_graph).
