# Short Term Global Ocean Forecast evaluation
The goal of this repository is to have a central codebase in which agreed upon metrics are applied to different global ocean forecast models, in order to have a fair comparison.

## Current Leaderboard

| Model           | μ-score (0d) | μ-score (3d) | μ-score (5d) | % Correct Mag (0d) | % Correct Mag (3d) | % Correct Mag (5d) |
|----------------|--------------|--------------|--------------|--------------------|--------------------|--------------------|
| GLO12 SSH       | 0.818        | 0.816        | 0.814        | 71.77              | 70.96              | 70.58             |
| GLO12 SLA       | 0.912        | 0.906        | 0.902        | 72.72              | 72.09              | 71.78              |
| **DUACS**       | **0.939**    | **0.939**    | **0.939**    | **76.51**          | **76.29**          | **76.20**          |
| **4DVarNet**    | _**0.936**_  | _**0.931**_  | _**0.924**_  | _72.96_            | _72.53_            | _69.63_            |
| U-Net-17M       | 0.932        | 0.927        | 0.924        | 72.86              | 70.08              | 67.89              |
| U-Net-70M       | 0.931        | 0.924        | 0.920        | 71.85              | 69.45              | 67.43              |
| XiHE SSH        | 0.818        | 0.780        | 0.779        | 71.77              | 64.67              | 63.95              |
| XiHE SLA        | 0.912        | 0.843        | 0.842        | 72.72              | 67.15              | 66.53             |
| GloNet SSH      | 0.821        | 0.825        | 0.823        | 74.96              | 74.98              | 74.60              |
| **GloNet SLA**  | _0.906_  | _0.913_  | _0.911_  | _**75.82**_        | _**75.91**_        | _**75.30**_        |

Latest benchmark : 
![alt text](https://github.com/CIA-Oceanix/Global_SSH_forecasting_OSE/blob/main/benchmark_complete.png?raw=true)


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
