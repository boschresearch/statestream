Module specification
====================
([back to documentation](README.md))

In some cases network architectures are build of similar sub-network modules, such as residual-blocks or LSTM-blocks. In these cases, it is convenient to once specify a template for such a building block and then re-use the template to construct the composed network in an efficient way. This can be accomplished by using the **modules** parameter in a **.st_graph** file. With this, one or more types of sub-modules can be specified and later re-used in the network specification. 

Using the [import](import_specification.md) specification, once specified modules can be re-used also in other network specification files. Module definitions can also be combined with [tag specifications](tag_specification.md), because all tag specifications will be applied after module generation.

Specifying modules is a two step process: First one has to specify a module template and second one can specify instances of this module type. The instances are specified the same way other item types like neuron-pools are specified, therefore one should for example not define a module with name plasticities:

```
modules:
	module_A:
		...
	module_B:
		...
module_A:
	inst_A_1:
		...
	inst_A_2:
		...
	...
module_B:
	inst_B_1:
		...
	...
```

The definition of a module is always a network itself and may only contain the basic items: neuron-pools, synapse-pools, plasticities and interfaces as well as core clients:

```
modules:
	module_A:
		core_clients:
			cc1:
				...
		neuron_pools:
			np1:
				...
			np2:
				...
			...
		synapse_pools:
			sp1:
				...
		plasticities:
			pl1:
				...
	module_B:
		neuron_pools:
			...
		synapse_pools:
			...
```

For now, it is not possible to build modules in a recursive way, meaning it is not possible to build modules containing modules.

Item names in the implicit (resulting) specification are always composed of the module instance name and the item name in the module specification separated with an underscore:

```
modules:
	module_A:
		neuron_pools:
			np1:
				...
			npx:
				...
module_A:
	inst_1:
		...
	my_inst:
		...
```

will result in the implicit specification:

```
neuron_pools:
	inst_1_np1:
		...
	inst_1_npx:
		...
	my_inst_np1:
		...
	my_inst_npx:
		...
```

The item specification inside a module specification has the same parameters as it would have in the explicit specification but parameters can be given as an explicit value or as a variable. The use of a variable is indicated by an underscore and all used variables must be specified in the later instance specification. In addition, for the specification of the following item's parameter, the local (inside module specification) neuron-pool and synapse-pool names can be used:

* synapse-pool: **Source** and **target** parameter of a synapse-pool specification may be local.
* plasticity: The np / sp identifier in the **parameter** definition of a plasticity may be local. The **source** and **target** parameter of a plasticity may refer to a local neuron-pool.
* interface: The **remap** parameter of an interface may contain local np identifiers.
* core-clients: If a string argument matches a basic item (np, sp, if, plast) inside the module, it is considered a local identifier.

Example for a synapse-pool specification:

```
modules:
	module_A:
		neuron_pools:
			np1:
				shape: [32, 8, 8]
			np2:
				shape: _my_shape
		synapse_pools:
			sp1:
				source: [[_my_src]]
				target: np1
			sp2:
				source: [[np1]]
				target: np2
module_A:
	inst:
		my_shape: [64, 32, 32]
		my_src: some_np
```

This will result in the following implicit specification:

```
neuron_pools:
	inst_np1:
		shape: [32, 8, 8]
	inst_np2:
		shape: [64, 32, 32]
synapse_pools:
	inst_sp1:
		source: [[some_np]]
		target: inst_np1
	inst_sp2:
		source: [[inst_np1]]
		target: inst_np2
```

A more complex example for a small ring network module is given below:

```
modules:
	ring:
		neuron_pools:
			L1:
				shape: _shape_bottom
			L2:
				shape: _shape_top
				dropout: 0.2
			L3:
				shape: _shape_top
				act: relu
				dropout: 0.2
			L4:
				shape: _shape_bottom
				tags: [layer4]
		synapse_pools:
			L1L2:
				source: [[L1]]
				target: L2
				rf: [[_rf]]
			L2L3:
				source: [[L2]]
				target: L3
				rf: [[_rf]]
			L3L4:
				source: [[L3]]
				target: L4
				rf: [[_rf]]
			L4L1:
				source: [[L4]]
				target: L1
				rf: [[_rf]]
ring:
	little:
		shape_bottom: [32, 128, 128]
		shape_top: [64, 64, 64]
		rf: 3
	big:
		shape_bottom: [8, 16, 16]
		shape_top: [32, 64, 64]
		rf: 5
```

This specification will produce the following network specification:

```
neuron_pools:
	little_L1:
		shape: [32, 128, 128]
	little_L2:
		shape: [64, 64, 64]
		dropout: 0.2
	little_L3:
		shape: [64, 64, 64]
		act: relu
		dropout: 0.2
	little_L4:
		shape: [32, 128, 128]
		tags: [layer4]
	big_L1:
		shape: [8, 16, 16]
	big_L2:
		shape: [32, 64, 64]
		dropout: 0.2
	big_L3:
		shape: [32, 64, 64]
		act: relu
		dropout: 0.2
	big_L4:
		shape: [8, 16, 16]
		tags: [layer4]
synapse_pools:
	little_L1L2:
		source: [[little_L1]]
		target: little_L2
		rf: [[3]]
	little_L2L3:
		source: [[little_L2]]
		target: little_L3
		rf: [[3]]
	little_L3L4:
		source: [[little_L3]]
		target: little_L4
		rf: [[3]]
	little_L4L1:
		source: [[little_L4]]
		target: little_L1
		rf: [[3]]
	big_L1L2:
		source: [[big_L1]]
		target: big_L2
		rf: [[5]]
	big_L2L3:
		source: [[big_L2]]
		target: big_L3
		rf: [[5]]
	big_L3L4:
		source: [[big_L3]]
		target: big_L4
		rf: [[5]]
	big_L4L1:
		source: [[big_L4]]
		target: big_L1
		rf: [[5]]
```

Please also compare the [test example](../examples/test_module.st_graph) and [demonstration example](../examples/cifar10_small_comparison.st_graph).
