# src/ct_rate_benchmarks/data/dataset.py

import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

# Initialize a logger for this module
log = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-computed features and labels.

    This dataset reads a manifest file (CSV) to locate features and
    corresponding labels for a classification or retrieval task.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        target_labels: List[str],
        visual_feature_col: Optional[str] = None,
        text_feature_col: Optional[str] = None,
    ):
        """
        Initializes the dataset.

        Args:
            manifest_path: Path to the .csv manifest file.
            data_root: The absolute base path to the directory
                       where feature files are stored.
            target_labels: A list of column names in the manifest
                           that correspond to the target labels.
            visual_feature_col: The column name in the manifest that
                                contains the relative path to the
                                visual feature file (e.g., .pt).
            text_feature_col: The column name in the manifest that
                              contains the relative path to the
                              text feature file (e.g., .pt).
        """
        super().__init__()

        if not os.path.exists(manifest_path):
            log.error(f"Manifest file not found at: {manifest_path}")
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        if not os.path.exists(data_root):
            log.warning(f"Data root directory not found: {data_root}. "
                        "Ensure it is correctly mounted or specified.")

        self.manifest_path = manifest_path
        self.data_root = data_root
        self.target_labels = target_labels
        self.visual_feature_col = visual_feature_col
        self.text_feature_col = text_feature_col

        # Load the manifest into a pandas DataFrame
        try:
            self.manifest = pd.read_csv(self.manifest_path)
        except Exception as e:
            log.error(f"Failed to load manifest CSV: {e}")
            raise

        # Validate that required columns exist
        self._validate_columns()

    def _validate_columns(self):
        """
        Ensures all specified feature and label columns
        exist in the loaded manifest.
        """
        manifest_cols = set(self.manifest.columns)
        
        if self.visual_feature_col and self.visual_feature_col not in manifest_cols:
            raise ValueError(f"Visual feature column '{self.visual_feature_col}' "
                             "not found in manifest.")
                             
        if self.text_feature_col and self.text_feature_col not in manifest_cols:
            raise ValueError(f"Text feature column '{self.text_feature_col}' "
                             "not found in manifest.")

        for label_col in self.target_labels:
            if label_col not in manifest_cols:
                raise ValueError(f"Target label column '{label_col}' "
                                 "not found in manifest.")
        
        log.info("Manifest columns validated successfully.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.manifest)

    def _load_feature(self, relative_path: str) -> torch.Tensor:
        """
        Loads a feature tensor from a file.

        Args:
            relative_path: The relative path to the feature file
                           (e.g., 'features/scan_001.pt').

        Returns:
            The loaded feature as a torch.Tensor.
        """
        # Ensure relative paths are handled correctly
        if relative_path.startswith(os.sep):
            relative_path = relative_path[1:]

        # Create the absolute path
        absolute_path = os.path.join(self.data_root, relative_path)

        if not os.path.exists(absolute_path):
            log.warning(f"Feature file not found: {absolute_path}")
            # Return a dummy tensor if not found? Or raise error?
            # For now, raise an error.
            raise FileNotFoundError(f"Feature file not found: {absolute_path}")
        
        # Load the tensor. Assumes .pt file,
        # but can be adapted for .parquet, .npy, etc.
        try:
            feature_tensor = torch.load(absolute_path, map_location="cpu")
            return feature_tensor
        except Exception as e:
            log.error(f"Failed to load feature at {absolute_path}: {e}")
            raise

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Fetches a single sample from the dataset.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A dictionary containing:
            - 'visual_features': (Optional) The visual feature tensor.
            - 'text_features': (Optional) The text feature tensor.
            - 'labels': The multi-label target tensor.
        """
        # Get the metadata row for the given index
        sample_row = self.manifest.iloc[index]
        
        output_item = {}

        # Load Visual Feature if configured
        if self.visual_feature_col:
            visual_path = sample_row[self.visual_feature_col]
            output_item["visual_features"] = self._load_feature(visual_path)

        # Load Text Feature if configured
        if self.text_feature_col:
            text_path = sample_row[self.text_feature_col]
            output_item["text_features"] = self._load_feature(text_path)

        # Load Labels
        # Extract label values and convert to a float tensor
        # Assumes labels are numeric (0 or 1)
        labels = sample_row[self.target_labels].values.astype(float)
        output_item["labels"] = torch.tensor(labels, dtype=torch.float32)

        return output_item