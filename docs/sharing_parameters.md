Sharing parameters
==================
([back to documentation](README.md))

It is some times necessary that different parts of the network utilize / share the same parameters, for example having several layers sharing the same weight matrices. This can be achieved using the **share params** parameter in the specification file, e.g.:

```
synapse_pools:
    sp_conv1:
		source: 
		- [image]
		target: conv1
    sp_conv2:
		source: 
		- [conv1]
		target: conv2
		share params:
		    W: [sp_conv1, W]
```

For now only weight matrices **W** of synapse pools can be shared.

Please see also the example [examples/test_parameter_sharing.st_graph](../examples/test_parameter_sharing.st_graph) for further details.
