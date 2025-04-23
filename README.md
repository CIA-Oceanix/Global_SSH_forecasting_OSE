# Short Term Global Ocean Forecast evaluation
The goal of this repository is to have a central codebase in which agreed upon metrics are applied to different global ocean forecast models, in order to have a fair comparison.

## Current Leaderboard

| Model                      | μ-score (0 day) | μ-score (3 days) | μ-score (5 days) |
|---------------------------|-------------|--------------|--------------|
| GLO12 SSH                 | 0.818       | 0.816        | 0.814        |
| GLO12 SLA                 | 0.912       | 0.906        | 0.902        |
| **DUACS**                 | **0.939**   | **0.939**    | **0.939**    |
| **4DVarNet** | **_0.936_**     | **_0.931_**      | **_0.924_**      |
| U-Net-17M                     | 0.932       | 0.927        | 0.924      |
| U-Net-70M        | 0.931       | 0.924        | 0.920        |
| XiHE SSH                  | 0.818       | 0.780        | 0.779        |
| XiHE SLA                  | 0.912       | 0.843        | 0.842        |
| GloNet SSH                | 0.821       | 0.825        | 0.823        |
| GloNet SLA                | 0.906       | 0.913      | 0.911      |



| Model                      | % correct velocity magnitudes (0 day) | % correct velocity magnitudes (3 days) | % correct velocity magnitudes (5 days) |
|---------------------------|-------------|--------------|--------------|
| GLO12 SSH                 | 60.66       | 57.29        | 55.25        |
| GLO12 SLA                 | 72.72       | 72.09       | 68.06       |
| **DUACS**                 | **0.939**   | **0.939**    | **0.939**    |
| 4DVarNet | _72.96_     | _72.53_      | _69.63_      |
| U-Net-17M                     | 72.86       | 70.08       | 67.89      |
| U-Net-70M        | 71.85       | 69.45        | 67.43        |
| XiHE SSH                  | 60.66       | 64.67        | 63.95        |
| XiHE SLA                  | 68.57       | 64.09        | 64.09        |
| GloNet SSH                | 74.96       | 74.98       | 74.60        |
| **GloNet SLA**                | **_75.82_**       | **_75.91_**     | **_75.30_**      |


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
