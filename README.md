# MultiModel-OceanGlobalEval
The goal of this repository is to have a central codebase in which agreed upon metrics are applied to different global ocean forecast models, in order to have a fair comparison.

## INSTALL REPO

`conda create -n <your_env> python=3.12`

`conda activate <your_env>`

`pip install -r requirements.txt`

## HOW TO USE THIS REPO

This repository works using metrics configuration files located in [config/metrics/](config/metrics/). You can see how to **create your own configuration file** [here](config/README.md).

The repository is then used like so:

> make sure you execute code from inside the repo   `cd MultiModel-OceanGobalEval`

`python main.py metrics=metrics_config_template`

This code will:

- download the reference data specified in `metrics_config_template.yaml`
- pre-process your model according to the model_type specified in `metrics_config_template.yaml`
- compute metrics specified in `metrics_config_template.yaml`

## CREDITS
The initial metrics codebase is comprised of code from the [ocean data challenges gihtub repo](https://github.com/ocean-data-challenges/2023a_SSH_mapping_OSE).