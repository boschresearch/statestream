Tag specification
=================
([back to documentation](README.md))

Often one needs the same type of neuron-pool or synapse-pool several times with equal parameters, e.g. equal dropout rates or activation functions. The tag system provides a feature with which parameter settings can be associated with tags. For this, define a tag specification as shown below in the **st_graph** file. Now all items with this tag will have these parameter settings. If an item this tag already has a definition of such a parameter, the local setting will overwrite the tag specification.

```
...
tag_specs:
	hidden:
		dropout: 0.25
		act: relu
...
```

Please also compare the [example](../examples/test_tag_spec.st_graph).
