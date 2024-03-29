# DynamicCML

This code repo contains the implementation of the DCML algorithm presented in the NeurIPS 2023 paper:

***On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm*** by Qi Chen, Changjian Shui, Ligong Han, and Mario Marchand.



## Prerequisites

To run the code, make sure that the requirements are installed.

```
pip install -r requirements.txt
```

The code was written to run in Python 3.6 or in a more recent version.
## Code Structure
```
.
|-- config
|   `-- yamls
|       |-- omniglot
|       `-- synbols               
|-- output
|   `-- models
|       |-- omniglot_maml.model
|       `-- synbols_maml.model
|-- run_guassian.sh
|-- run_osaka.sh
`-- src
    |-- dataloader
    |   |-- osaka_datasets.py
    |   `-- toy_datasets.py
    |-- envdetector
    |   |-- bocd.py
    |   `--env_change_detection.py
    |-- learners
    |   |-- dmogd.py
    |   |-- __init__.py
    |   |-- maml.py
    |   |-- model.py
    |   `-- osaka_model.py
    |-- main
    |   |-- args.py
    |   |-- run_gaussian.py
    |   `-- run_osaka_benchmark.py
    `-- utils
```

## Continual Meta-Learning Framework
![Illustration](./cml.jpeg)

## Experimental Setting
![Illustration](./experiment.jpeg)

### Run Experiment

#### Moving 2D Gaussian
```
    python src/main/run_gaussian.py --sample_num 100 --K 2 --hazard 0.1 --method 1 --seed 168
```
or 
```
   sh run_gaussian.sh
```
#### OSAKA 
1. Create your wandb account, a new project, and a wandb key
2. Run the code w.r.t different algorithms ("DCML_oracle DCML_win CMAML_pre_kwto_acc online_sgd fine_tuning MetaCOG MetaBGD BGD")
For example:

```
sh run_osaka.sh "DCML_oracle" 0.0 0.2 "synbols" wandb_project_name  wandb_key
```

### Reference

### Acknowledgements
The code for testing the OSAKA benchmark was adapted from https://github.com/ServiceNow/osaka/tree/camera_ready.
