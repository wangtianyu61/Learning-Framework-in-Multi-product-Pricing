# Learning-Framework-in-Multi-product-Pricing
This repo shows some additional numerical experiments for the work on Data-driven Pricing Framework. The experiment setup and result documents can be shown in ```ExperimentResult.pdf``` and the codes are in the ```codes``` folder.

For the codes,

```simulation.py``` shows the simulation setup for different demand models.

```main_test1.py``` and ```main_test2.py``` are two main functions for going through all the optimization problem in one given datasets.

```param.py``` shows the param setup for the experiment.

```task_est_opt.py``` shows the task-based learning model. (```est_opt1.py``` and ```est_opt2.py``` are benchmarks for linear and MNL demand models.)

```e2e_network.py``` is the main framework for model-free learning while ```SigmoidNet.py``` and ```ReLuNet.py``` are two concrete examples.
