Shapes
======
([back to documentation](README.md))

To manipulate the influence of specific dimensions for neuron- and synapse-pools computations, several parameters are provided to be specified in an st_graph file.

Not all combinations of shape parameters are tested. Please see the examples folder for some working examples.



Bias shape:
-----------

The parameter **bias_shape** may be given for a neuron-pool and specifies the dimensionaly of the used bias term. For now, different samples still can only have the same bias. Possible values and their explanations are:

* **full**: The bias has the same dimension as the neuron-pool state itself, hence every neuron has its own affine bias. The broadcastable pattern of the bias is:
	* [True, 0, 1, 2]
* **feature**: This is the default value. The bias has the shape of a number-of-features size vector. The broadcastable pattern of the bias is:
	* [True, 0, True, True]
* **spatial**: The bias has only the spatial shape X x Y and will be applied for all feature-maps equally. The broadcastable pattern of the bias is:
	* [True, True, 0, 1]
* **scalar**: The bias is a single scalar, hence all neurons share the same bias term. The broadcastable pattern of the bias is:
	* [True, True, True, True]



Factor shapes:
--------------

For a synapse-pool, a **factor_shapes** parameter can be defined. This always has to be a list of strings, with the same length as the number of factors in the sp's **sources**. For a factor this shape defines the factor's dimensionaly. Possible values and their meaning are similyr to the ones for the bias shape from above:

* **full**: This is the default value. The factor has the same shape as the full target of the synapse-pool. The broadcastable pattern of a factor then is:
	* [0, 1, 2, 3]
* **feature**: The factor is only a vector with size equal to the number of features of sp's target. The broadcastable pattern of a factor then is:
	* [0, 1, True, True]
* **spatial**: The factor contains only spatial and no feature information. The broadcastable pattern of a factor then is:
	* [0, True, 1, 2]
* **scalar**: The factor is a single scalar per sample which is multiplied with the other factors of the sp. The broadcastable pattern of a factor then is:
	* [0, True, True, True]



Bias shapes:
------------

For each factor of a synapse-pool, the dimensionality of its bias term can be specified using the sp parameter **bias_shapes**. This parameter has to be a list in length equal to the number of factors of the synapse-pool. For each factor exactly the same options as for **bias_shape** above are available and have the same effect. The factor bias shapes are limited by the specified factor shapes. 

* factor_shapes[f] = full : all bias shapes are allowed
* factor_shapes[f] = feature : only [feature, scalar] allowed for bias shapes
* factor_shapes[f] = spatial : only [spatial, scalar] allowed for bias shapes
* factor_shapes[f] = scalar : also only scalar is allowed for bias shapes



Target shapes:
--------------

A synapse-pool may contain a parameter **target_shapes** which must be a list of lists of the same size as the sp's sources. This shape parameter specifies the dimensionality of the target of each of the sp's sources. The target shape is limited by its factor shape. Possible combinations are:

* factor_shapes[f] = full : all target shapes are allowed
* factor_shapes[f] = feature : only [feature, scalar] allowed for target shapes
* factor_shapes[f] = spatial : only [spatial, scalar] allowed for target shapes
* factor_shapes[f] = scalar : also only scalar is allowed for target shapes



Weight shapes:
--------------

A synapse-pool may contain a parameter **weight_shapes** which must be a list of lists of the same size as the sp's sources. This shape parameter specifies the layout of the used weight matrix for this specific input. Available weight shape specifications depend on the target_shapes for this input and hence also on the factor_shapes. In general possible specifications and their broadcast pattern are:

* **full**:             [tgt_c, src_c, rf_x, rf_y]
* **feature**:          [tgt_c, src_c, True, True]
* **spatial**:          [True, True, rf_x, rf_y]
* **scalar**:           [True, True, True, True]
* **src_feature**:      [True, src_c, True, True]
* **tgt_feature**:      [tgt_c, True, True, True]
* **src_spatial**:      [True, src_c, rf_x, rf_y]
* **tgt_spatial**:      [tgt_c, True, rf_x, rf_y]

Effects of specifications dependent on target_shapes:

* target_shapes[f][i] = full: All weight shapes are allowed for weight_shapes[f][i].
* target_shapes[f][i] = feature: All weight shapes are allowed.
* target_shapes[f][i] = spatial: All specifications are allowed but will ignore broadcasting over the target channels (e.g. 'scalar' has the same effect as 'tgt_feature').
* target_shapes[f][i] = scalar: All specifications are allowed but will ignore broadcasting over the target channels (e.g. 'scalar' has the same effect as 'tgt_feature').


