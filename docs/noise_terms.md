Noise terms
===========
([back to documentation](README.md))

There are two different noise terms available: 'normal' and 'uniform'. Dependent on the noise type, two separate parameters can be specified:

* **noise** (type: str): Optional. Default is None so that no noise will be used. Available are 'normal' or 'uniform'.
* **noise mean**, **noise std** (type: float): Parameters for the normal distribution. Defaults are 0 and 1.
* **noise min**, **noise max** (type: float): Parameters for the uniform distribution. Defaults are -1 and 1.

For a more detailed example see **examples/test_np_specification.st_graph**.
