import logging
import os
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# A-priori import of models and dataloaders to register them
# We will create these files in subsequent steps.
# from . import data  # noqa: F401
# from . import models # noqa: F401

# Initialize a logger for this module
log = logging.getLogger(__name__)


def train_model(cfg: DictConfig) -> float:
    """
    Main training and evaluation function.

    This function orchestrates the entire pipeline:
    1. Sets up seeding for reproducibility.
    2. Loads and prepares datasets and dataloaders.
    3. Initializes the model architecture.
    4. Initializes the optimizer and loss function.
    5. Runs the training loop for the specified number of epochs.
    6. Runs the final evaluation on the test set.
    7. Logs metrics and saves artifacts.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        The primary metric score (e.g., test AUROC) for optimization.
    """
    log.info("Starting training pipeline...")
    log.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Set Up Reproducibility ---
    # TODO: Implement a utility function to set seeds for
    # random, numpy, and torch.
    # set_seed(cfg.training.seed)
    log.info(f"Using seed: {cfg.training.seed}")

    # --- 2. Calculate Dynamic Parameters ---
    # Dynamically determine the number of labels from the config list
    num_labels = len(cfg.training.target_labels)
    log.info(f"Resolved {num_labels} target labels from config.")

    # --- 3. Load Data ---
    # TODO: Instantiate the PyTorch Dataset and DataLoaders
    # The dataloader will read 'cfg.paths.feature_manifest' and use
    # the column names from 'cfg.data.columns'.
    # dataloader_train = ...
    # dataloader_val = ...
    log.info(f"Loading data from manifest: {cfg.paths.feature_manifest}")

    # --- 4. Initialize Model ---
    # TODO: Instantiate the model defined in 'cfg.model'
    # The model's output dimension should be set to 'num_labels'.
    # model = hydra.utils.instantiate(
    #     cfg.model,
    #     out_features=num_labels,
    #     _recursive_=False
    # )
    log.info(f"Instantiating model: {cfg.model.model_name}")

    # --- 5. Initialize Optimizer and Loss ---
    # TODO: Instantiate optimizer
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=cfg.training.learning_rate
    # )

    # TODO: Initialize loss function (e.g., BCEWithLogitsLoss for multilabel)
    # criterion = torch.nn.BCEWithLogitsLoss()
    log.warning("Placeholders: Optimizer and Loss Function not yet implemented.")

    # --- 6. Set Up Logging ---
    # TODO: Initialize W&B or TensorBoard logger based on 'cfg.logging'
    log.info(f"Using logger: {cfg.logging.logger_name}")

    # --- 7. Run Training Loop ---
    log.info(f"Starting training for {cfg.training.max_epochs} epochs.")
    for epoch in range(cfg.training.max_epochs):
        log.info(f"Epoch {epoch + 1}/{cfg.training.max_epochs}")
        # TODO: Implement training_step(model, dataloader_train, optimizer, criterion)
        # TODO: Implement validation_step(model, dataloader_val, criterion)
        # TODO: Log metrics
        pass
    
    log.warning("Placeholder: Training loop not yet implemented.")

    # --- 8. Run Final Evaluation ---
    # TODO: Implement final evaluation on the test set
    log.warning("Placeholder: Final evaluation not yet implemented.")
    
    # --- 9. Save Artifacts ---
    # Hydra automatically saves the config and log in 'cfg.paths.output_dir'
    # TODO: Save the final model checkpoint
    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, "final_model.pt")
    # torch.save(model.state_dict(), checkpoint_path)
    log.info(f"Mock-saving final checkpoint to: {checkpoint_path}")

    # Return a mock metric for now
    mock_auroc = 0.5
    return mock_auroc


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.

    Loads the configuration and passes it to the main training function.

    Args:
        cfg: The Hydra configuration object automatically populated.
    """
    try:
        train_model(cfg)
    except Exception as e:
        log.exception(f"An error occurred during training: {e}")
        # Optionally re-raise or exit
        raise


if __name__ == "__main__":
    main()