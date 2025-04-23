# Short Term Global Ocean Forecast evaluation
The goal of this repository is to have a central codebase in which agreed upon metrics are applied to different global ocean forecast models, in order to have a fair comparison.

## Current Leaderboard

| Model                      | &mu (0 day) | SSH (3 days) | SSH (5 days) |
|---------------------------|-------------|--------------|--------------|
| GLO12 SSH                 | 0.818       | 0.816        | 0.814        |
| GLO12 SLA                 | 0.912       | 0.906        | 0.902        |
| **DUACS**                 | **0.939**   | **0.939**    | **0.939**    |
| **4DVarNet** | **_0.936_**     | **_0.931_**      | **_0.924_**      |
| U-Net-17M                     | 0.932       | 0.927        | _0.924_      |
| U-Net-70M        | 0.931       | 0.924        | 0.920        |
| XiHE SSH                  | 0.818       | 0.780        | 0.779        |
| XiHE SLA                  | 0.912       | 0.843        | 0.842        |
| GloNet SSH                | 0.821       | 0.825        | 0.823        |
| GloNet SLA                | 0.906       | _0.913_      | _0.911_      |


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
