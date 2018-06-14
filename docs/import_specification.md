Import specification
====================
([back to documentation](README.md))

Sometimes one wants to reuse already specified networks or network parts. To some extent, this functionality is provided by the import specification parameter with which one can include network parts from other network specifications into a new one. In general the import is recursive but will fail for loops.

The import parameter must be specified as a list, where each list entry specifies one import. Each import list entry itself is again a list with two entries, the first specifying the source network specification file and the second specifying what part to import from this specification. For now, only the */examples** directory of the statestream repository is searched for specified sources and the post-fix **.st_graph** is left out.

There exist several ways to import parts from network specification:
* import all item types from another specification: By specifying the item type (neuron_pools, synapse_pools, plasticities, interfaces), all items of this type will be imported (e.g. **- [mnist_small, synapse_pools]**)
* import all items of a specific type and tag: Same as the item byte but a tag can be specified additionally (e.g. **- [mnist_small, neuron_pools.hidden]** would import all neuron-pools with the tag **hidden**)
* import all modules from another specification: The specification **- [spec_file, modules]** will import all modules contained in **spec_file.st_graph**.
* import a single module: The specification **- [spec_file, modules.module]** will import the module **module** from **spec_file.st_graph**.
* import tag specifications: This works the same as for the module specification. All tag specifications or a specific tag specification can be imported.
* import all items of a specific tag: The specification **- [spec_file, tag]** will import all items (all nps, sps, plasts, ifs) from **spec_file.st_graph** with the tag **tag**.
* import all or specific global variable(s): The specification **- [spec_file, globals]** will import all global variables from **spec_file.st_graph**. The specification **- [spec_file, globals.MYGLOB]** will import only the **MYGLOB** global variable from **spec_file.st_graph**.

An example import illustration with some possible specification:

```
import:
    - [mnist_small, neuron_pools]
    - [mnist_small, synapse_pools]
    - [mnist_small, interfaces]
    - [test_module, modules]
    - [mnist_compare_explicit_loss, modules.mnist_net]
```

Please also compare the [example](../examples/test_import.st_graph).
