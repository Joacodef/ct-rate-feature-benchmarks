
# 🔧 Configuration (`configs/`)

This directory contains all configuration files for the project, managed by [Hydra](https://hydra.cc/).

The configuration system is designed to be **composable**. The main `config.yaml` file acts as the entry point, which then "pulls in" smaller, swappable configuration pieces from the `data/` and `model/` sub-directories.

## How it Works

1.  **Main Config (`config.yaml`)**: This file defines the default experiment setup. It uses a `defaults` list to specify which "Configuration Groups" to load.

2.  **Configuration Groups (Sub-directories)**:
    * **`data/`**: Contains `.yaml` files that define *how* to load data (e.g., which features to use, which columns to read).
    * **`model/`**: Contains `.yaml` files that define a specific model's architecture and hyperparameters.

## 🚀 Running Experiments

When you run the main training script, Hydra builds the complete configuration by merging the files.

**Default Run:**
```bash
# This command loads:
# 1. config.yaml
# 2. data/default_features.yaml
# 3. model/linear_probe_visual.yaml
python src/ct_rate_benchmarks/train.py
````

**Swapping Configurations:**
The power of this system is in swapping components from the command line. If you create a new model config `model/mlp_visual.yaml`, you can run it without changing any code:

```bash
# This command loads:
# 1. config.yaml
# 2. data/default_features.yaml
# 3. model/mlp_visual.yaml  <-- SWAPPED
python src/ct_rate_benchmarks/train.py model=mlp_visual
```

This composable approach allows us to define and benchmark new models or data-handling strategies simply by adding new `.yaml` files.

