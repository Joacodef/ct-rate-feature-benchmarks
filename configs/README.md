# Configuration (configs/)

This directory contains all configuration files for the project, managed by [Hydra](https://hydra.cc/).

The configuration system is designed to be composable. The main `config.yaml` file acts as the entry point, which then "pulls in" smaller, swappable configuration pieces from the `data/` and `model/` sub-directories.

## How it Works

1.  **Main Config (`config.yaml`)**: This file defines the default experiment setup. It uses a `defaults` list to specify which "Configuration Groups" to load.

2.  **Configuration Groups (Sub-directories)**:
    * **`data/`**: Contains `.yaml` files that define *how* to load data (e.g., which features to use, which columns to read).
    * **`model/`**: Contains `.yaml` files that define a specific model's architecture and hyperparameters.

## Running Experiments

When you run the main training script, Hydra builds the complete configuration by merging the files.

**Default Run:**
```bash
# This command loads the new defaults:
# 1. config.yaml
# 2. data/default_features.yaml
# 3. model/mlp_visual.yaml
python src/classification/train.py
````

**Swapping Configurations:**
The power of this system is in swapping components from the command line. If you create a new data config `data/multimodal_features.yaml` and a new model config `model/mlp_multimodal.yaml`, you can run a new experiment without changing any code:

```bash
# This command loads:
# 1. config.yaml
# 2. data/multimodal_features.yaml  <-- SWAPPED
# 3. model/mlp_multimodal.yaml      <-- SWAPPED
python src/classification/train.py data=multimodal_features model=mlp_multimodal
```

This composable approach allows us to define and benchmark new models or data-handling strategies simply by adding new `.yaml` files.

## Resuming a Run

The base config exposes a `training.resume` group. Set `training.resume.enabled=true` to continue from the most recent `last_state.pt` in the run's checkpoint folder, or provide an explicit `training.resume.state_path=/path/to/last_state.pt` override when starting the job.
