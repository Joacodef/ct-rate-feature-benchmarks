# src/classification/data/dataset.py

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
    """PyTorch dataset that serves precomputed feature tensors and labels.

    Input:
        A manifest CSV where each row references feature paths (visual and/or
        text) and label columns.

    Output:
        Samples as dictionaries containing one or more feature tensors plus a
        ``labels`` tensor, suitable for training/evaluation loops.

    Logic:
        1. Read and validate the manifest schema.
        2. Optionally preload all tensors into RAM caches.
        3. On each item request, return cached tensors or lazily load from
           disk using extension-aware loaders.
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
        """Initialize dataset metadata, validate schema, and optionally preload.

        Args:
            manifest_path: Path to the manifest CSV file. Must exist.
            data_root: Base directory used to resolve relative feature paths.
                If missing, a warning is logged and loading may later fail.
            target_labels: Ordered list of manifest column names to use as
                label targets.
            visual_feature_col: Optional manifest column containing visual
                feature paths.
            text_feature_col: Optional manifest column containing text feature
                paths.
            preload: When ``True``, eagerly load configured features and labels
                into in-memory caches during construction.

        Input:
            File-system paths and manifest column names defining how each row
            maps to tensors.

        Returns:
            ``None``.

        Raises:
            FileNotFoundError: If ``manifest_path`` does not exist.
            Exception: Re-raises parsing errors from ``pandas.read_csv``.
            ValueError: If required feature/label columns are missing in the
                manifest (via ``_validate_columns``).

        Logic:
            1. Verify path existence for manifest (error) and data root
               (warning only).
            2. Read the manifest into a DataFrame.
            3. Validate feature and label columns.
            4. Optionally preload all samples to memory caches.
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
        """Validate manifest schema against configured feature/label columns.

        Input:
            Uses in-memory ``self.manifest`` and the configured
            ``visual_feature_col``, ``text_feature_col``, and ``target_labels``.

        Returns:
            ``None``.

        Raises:
            ValueError: If any configured column is absent from manifest.

        Logic:
            1. Build a set of manifest column names.
            2. Validate optional feature columns when configured.
            3. Validate every target label column.
            4. Log success when all checks pass.
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
        """Return the number of rows (samples) in the manifest.

        Input:
            No explicit parameters; reads ``self.manifest``.

        Returns:
            Total sample count as an integer.

        Logic:
            Delegate directly to ``len(self.manifest)``.
        """
        return len(self.manifest)

    def _preload_to_memory(self) -> None:
        """Eagerly load feature and label tensors for all samples.

        Input:
            Entire manifest DataFrame plus configured feature columns.

        Returns:
            ``None``. Populates ``_visual_cache``, ``_text_cache``, and
            ``_label_cache`` with per-sample tensors.

        Logic:
            1. Allocate cache lists for enabled modalities and labels.
            2. Iterate over all rows with a progress bar.
            3. Load features via ``_load_feature`` and convert labels to
               ``float32`` tensors.
            4. Store tensors by index for fast access in ``__getitem__``.
        """
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

    def _load_feature(
        self, relative_path: str
    ) -> torch.Tensor:
        """Load one feature vector tensor from disk with extension fallback.

        Args:
            relative_path: Path from manifest. May be absolute, relative,
                mixed-separator, or separator-prefixed without drive letter.

        Input:
            A path-like value pointing to feature data saved as ``.pt``,
            ``.pth``, ``.npy``, or ``.npz`` (or another torch-loadable format).

        Returns:
            A ``torch.float32`` tensor representing the feature vector.

        Raises:
            FileNotFoundError: If no matching feature file is found after all
                resolution strategies.
            ValueError: If content is invalid for expected loaders.
            Exception: Re-raises loader errors after logging.

        Logic:
            1. Normalize path text and treat separator-prefixed, drive-less
               paths as relative to ``data_root``.
            2. Resolve a candidate absolute path.
            3. If candidate exists, load with extension-specific loader when
               available; otherwise try ``torch.load`` fallback.
            4. If missing, try alternate known extensions sharing same basename.
            5. As final fallback, scan candidate directory for matching basename
               with supported extension.
            6. Log attempts and raise if unresolved.
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
            """Load tensor data from a PyTorch serialized file.

            Args:
                path: Absolute path to ``.pt``/``.pth`` file.

            Input:
                Serialized tensor or dictionary potentially containing tensor
                entries.

            Returns:
                1D ``torch.float32`` tensor (squeezed and flattened when
                needed).

            Raises:
                ValueError: If the file does not directly or indirectly contain
                    a tensor value.

            Logic:
                Prefer direct tensor payload; otherwise search common dictionary
                keys and normalize discovered tensor shape/dtype.
            """
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

        def _to_feature_vector(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
            """Normalize arrays/tensors into a 1D NumPy feature vector.

            Args:
                arr: NumPy array or Torch tensor from a file loader.

            Input:
                Scalar, vector, or higher-dimensional numeric array-like data.

            Returns:
                1D ``np.ndarray``. Scalars become shape ``(1,)`` and
                multi-dimensional arrays are flattened.

            Logic:
                Convert to NumPy, squeeze singleton dimensions, then coerce to
                a consistent 1D representation.
            """
            if isinstance(arr, torch.Tensor):
                np_arr = arr.detach().cpu().numpy()
            else:
                np_arr = np.asarray(arr)

            np_arr = np.squeeze(np_arr)

            if np_arr.ndim == 0:
                return np_arr.reshape(1)
            if np_arr.ndim > 1:
                return np_arr.reshape(-1)
            return np_arr

        def _load_npy(path: str) -> torch.Tensor:
            """Load feature data from ``.npy`` and coerce to float tensor.

            Args:
                path: Absolute path to NumPy binary file.

            Input:
                ``.npy`` file content, or an ``NpzFile``-like object returned
                by ``np.load`` in edge cases.

            Returns:
                1D ``torch.float32`` feature tensor.

            Raises:
                ValueError: If loaded archive-like object contains no arrays.

            Logic:
                Load, extract first available array when archive-like, convert
                to normalized vector via ``_to_feature_vector``.
            """
            arr = np.load(path)
            # if np.load returned an NpzFile-like object, use first entry
            if isinstance(arr, np.lib.npyio.NpzFile):
                keys = list(arr.keys())
                if not keys:
                    raise ValueError(f".npy/.npz at {path} contains no arrays")
                arr = arr[keys[0]]
            vec = _to_feature_vector(arr)
            return torch.tensor(vec, dtype=torch.float32)

        def _load_npz(path: str) -> torch.Tensor:
            """Load feature data from ``.npz`` and coerce to float tensor.

            Args:
                path: Absolute path to NumPy zipped archive.

            Input:
                ``.npz`` file containing one or more arrays.

            Returns:
                1D ``torch.float32`` feature tensor from the first stored
                array.

            Raises:
                ValueError: If the archive contains no arrays.

            Logic:
                Read archive, select first array key, normalize shape with
                ``_to_feature_vector``, then convert to tensor.
            """
            npz = np.load(path)
            if isinstance(npz, np.lib.npyio.NpzFile):
                keys = list(npz.keys())
                if not keys:
                    raise ValueError(f".npz at {path} contains no arrays")
                arr = npz[keys[0]]
            else:
                arr = npz
            vec = _to_feature_vector(arr)
            return torch.tensor(vec, dtype=torch.float32)

        loaders = {
            ".pt": _load_pt,
            ".pth": _load_pt,
            ".npy": _load_npy,
            ".npz": _load_npz,
        }

        base, ext = os.path.splitext(candidate)

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
        """Fetch one sample dictionary for a given manifest row index.

        Args:
            index: Zero-based sample index.

        Input:
            Row metadata from the manifest and optional in-memory caches.

        Returns:
            Dictionary with keys:
            - ``visual_features`` (optional): visual tensor when configured.
            - ``text_features`` (optional): text tensor when configured.
            - ``labels``: multi-label ``torch.float32`` tensor.

        Logic:
            1. Retrieve row metadata for ``index``.
            2. For each enabled modality, use cached tensor if available;
               otherwise load from disk.
            3. Use cached labels if preloaded; otherwise build tensor from
               target label columns.
            4. Return assembled sample dictionary.
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