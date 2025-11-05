# src/ct_rate_benchmarks/data/dataset.py

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

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
        preload: bool = False,
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
            preload: If True, eagerly load all features and labels
                     into memory during initialisation.
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
        self._preload_enabled = bool(preload)
        self._visual_cache: Optional[List[torch.Tensor]] = None
        self._text_cache: Optional[List[torch.Tensor]] = None
        self._label_cache: Optional[List[torch.Tensor]] = None

        # Load the manifest into a pandas DataFrame
        try:
            self.manifest = pd.read_csv(self.manifest_path)
        except Exception as e:
            log.error(f"Failed to load manifest CSV: {e}")
            raise

        # Validate that required columns exist
        self._validate_columns()

        if self._preload_enabled:
            self._preload_to_memory()

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

    def _preload_to_memory(self) -> None:
        """Read all configured features and labels into RAM upfront."""
        total = len(self.manifest)
        log.info(
            "Preloading %d samples from %s into memory.",
            total,
            self.manifest_path,
        )

        if self.visual_feature_col:
            self._visual_cache = [None] * total
        if self.text_feature_col:
            self._text_cache = [None] * total
        self._label_cache = [None] * total

        progress = tqdm(
            range(total),
            desc=f"Preloading {os.path.basename(self.manifest_path)}",
            unit="sample",
            leave=False,
        )

        for idx in progress:
            row = self.manifest.iloc[idx]

            if self.visual_feature_col:
                visual_path = row[self.visual_feature_col]
                self._visual_cache[idx] = self._load_feature(visual_path)

            if self.text_feature_col:
                text_path = row[self.text_feature_col]
                self._text_cache[idx] = self._load_feature(text_path)

            labels = row[self.target_labels].values.astype(float)
            self._label_cache[idx] = torch.tensor(labels, dtype=torch.float32)

        progress.close()
        log.info("Finished preloading dataset into memory.")

    def _load_feature(self, relative_path: str) -> torch.Tensor:
        """Load a feature tensor from a file, supporting .pt, .npy, and .npz.

        The manifest may contain paths with mixed separators or leading
        slashes. We normalize the path and treat paths starting with a
        separator but without a Windows drive letter as relative to
        `data_root`.
        """
        # Normalize and coerce to str
        relative_path = os.path.normpath(str(relative_path))

        # If the path has no drive letter but starts with a separator
        # (e.g. '\\data\...'), strip leading separators so it's
        # treated as relative to data_root.
        drive, _ = os.path.splitdrive(relative_path)
        if drive == "" and (relative_path.startswith(os.sep) or relative_path.startswith('/')):
            relative_path = relative_path.lstrip('\\/')

        # Build candidate absolute path
        if os.path.isabs(relative_path):
            candidate = os.path.normpath(relative_path)
        else:
            candidate = os.path.normpath(os.path.join(self.data_root, relative_path))

        # Define loaders for known extensions
        def _load_pt(path: str) -> torch.Tensor:
            t = torch.load(path, map_location="cpu")
            if isinstance(t, torch.Tensor):
                # squeeze any singleton leading dims and flatten remaining dims to 1D
                t = t.squeeze()
                if t.dim() > 1:
                    t = t.view(-1)
                return t.to(dtype=torch.float32)
            # If it's not a tensor (e.g., dict/state_dict), try to find a tensor
            if isinstance(t, dict):
                # try common keys
                for key in ("features", "visual_features", "state_dict", "model_state"):
                    if key in t and isinstance(t[key], torch.Tensor):
                        tt = t[key].squeeze()
                        if tt.dim() > 1:
                            tt = tt.view(-1)
                        return tt.to(dtype=torch.float32)
            raise ValueError(f"Loaded .pt file at {path} did not contain a tensor")

        def _load_npy(path: str) -> torch.Tensor:
            arr = np.load(path)
            # if np.load returned an NpzFile-like object, use first entry
            if isinstance(arr, np.lib.npyio.NpzFile):
                keys = list(arr.keys())
                if not keys:
                    raise ValueError(f".npy/.npz at {path} contains no arrays")
                arr = arr[keys[0]]
            # apply pooling/flatten heuristics to turn pre-pooling maps into vectors
            vec = _pool_to_vector(arr)
            return torch.tensor(vec, dtype=torch.float32)

        def _load_npz(path: str) -> torch.Tensor:
            npz = np.load(path)
            if isinstance(npz, np.lib.npyio.NpzFile):
                keys = list(npz.keys())
                if not keys:
                    raise ValueError(f".npz at {path} contains no arrays")
                arr = npz[keys[0]]
            else:
                arr = npz
            vec = _pool_to_vector(arr)
            return torch.tensor(vec, dtype=torch.float32)

        loaders = {
            ".pt": _load_pt,
            ".pth": _load_pt,
            ".npy": _load_npy,
            ".npz": _load_npz,
        }

        base, ext = os.path.splitext(candidate)

        def _pool_to_vector(x: Union[np.ndarray, torch.Tensor]):
            """Convert multi-dim feature maps to a 1D vector.

            Heuristics:
            - squeeze singleton dims
            - if 1D, return as-is
            - if 2D, flatten
            - if 3D: try to detect channel-first (C,H,W) or channel-last (H,W,C)
              by comparing sizes; perform global average pooling over spatial dims
              when detection succeeds; otherwise flatten.
            - if 4D and first dim == 1: recurse on x[0]
            """
            # Convert to numpy for easy inspection
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)

            arr = np.squeeze(arr)

            if arr.ndim == 0:
                return arr.reshape(-1)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2:
                return arr.reshape(-1)
            if arr.ndim == 3:
                c, h, w = arr.shape
                # guess channels-first if channel dim is smaller than spatial dims
                if c <= max(512, min(h, w)):
                    return arr.mean(axis=(1, 2))
                # else try channels-last
                c_last = arr.shape[2]
                if c_last <= max(512, min(arr.shape[0], arr.shape[1])):
                    return arr.mean(axis=(0, 1))
                # fallback
                return arr.reshape(-1)
            if arr.ndim == 4:
                if arr.shape[0] == 1:
                    return _pool_to_vector(arr[0])
                return arr.reshape(-1)
            # fallback flatten
            return arr.reshape(-1)

        # 1) If the exact candidate exists, try to load it with a matching loader
        if os.path.exists(candidate):
            loader = loaders.get(ext.lower())
            if loader:
                try:
                    return loader(candidate)
                except Exception as e:
                    log.error(f"Failed to load feature at {candidate}: {e}")
                    raise
            # Fallback: try torch.load for other extensions
            try:
                return torch.load(candidate, map_location="cpu")
            except Exception as e:
                log.error(f"Failed to load feature at {candidate}: {e}")
                raise

        # 2) Try alternate common extensions for the same base
        tried = []
        for alt_ext, loader in loaders.items():
            alt_path = base + alt_ext
            tried.append(alt_path)
            if os.path.exists(alt_path):
                try:
                    return loader(alt_path)
                except Exception as e:
                    log.error(f"Failed to load feature at {alt_path}: {e}")
                    raise

        # 3) As a last resort, scan the directory for a matching basename
        dirname = os.path.dirname(candidate)
        basename = os.path.basename(base)
        if os.path.isdir(dirname):
            for fname in os.listdir(dirname):
                name, e = os.path.splitext(fname)
                if name == basename and e.lower() in loaders:
                    alt_path = os.path.join(dirname, fname)
                    try:
                        return loaders[e.lower()](alt_path)
                    except Exception as exc:
                        log.error(f"Failed to load feature at {alt_path}: {exc}")
                        raise

        log.warning(f"Feature file not found: {candidate}")
        log.debug(f"Tried paths: {tried}")
        raise FileNotFoundError(f"Feature file not found: {candidate}")

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
            if self._visual_cache is not None:
                output_item["visual_features"] = self._visual_cache[index]
            else:
                visual_path = sample_row[self.visual_feature_col]
                output_item["visual_features"] = self._load_feature(visual_path)

        # Load Text Feature if configured
        if self.text_feature_col:
            if self._text_cache is not None:
                output_item["text_features"] = self._text_cache[index]
            else:
                text_path = sample_row[self.text_feature_col]
                output_item["text_features"] = self._load_feature(text_path)

        # Load Labels
        # Extract label values and convert to a float tensor
        # Assumes labels are numeric (0 or 1)
        if self._label_cache is not None:
            output_item["labels"] = self._label_cache[index]
        else:
            labels = sample_row[self.target_labels].values.astype(float)
            output_item["labels"] = torch.tensor(labels, dtype=torch.float32)

        return output_item