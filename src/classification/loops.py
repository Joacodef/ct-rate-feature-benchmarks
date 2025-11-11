"""Training and evaluation loops for classification models."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def compute_metrics(
	preds: torch.Tensor,
	targets: torch.Tensor,
	label_names: Optional[List[str]] = None,
	negative_class_name: Optional[str] = None,
) -> Dict[str, Any]:
	"""Compute macro AUROC and optional per-class precision/recall/f1 for multi-label logits.

	Args:
		preds: Raw model logits tensor (N, C).
		targets: Ground-truth binary labels tensor (N, C).
		label_names: Optional list of label names of length C. If provided, a
			`per_class` mapping will be included in the returned metrics.
		negative_class_name: Optional friendly name for the negative class when
			computing binary metrics (defaults to "No pathology").
	"""
	preds_prob = torch.sigmoid(preds).detach().cpu().numpy()
	targets_np = targets.detach().cpu().numpy()

	try:
		auroc = roc_auc_score(targets_np, preds_prob, average="macro")
		if not math.isfinite(auroc):
			raise ValueError("Non-finite AUROC")
	except ValueError as exc:
		log.warning("Could not compute AUROC (likely due to missing labels): %s", exc)
		auroc = 0.0

	out: Dict[str, Any] = {"auroc": auroc}

	# If the caller requested per-class metrics, compute precision/recall/f1.
	if label_names is not None:
		try:
			# Binarize predictions at 0.5 for precision/recall/f1 computation
			preds_bin = (preds_prob >= 0.5).astype(int)
			if preds_bin.ndim == 1:
				preds_bin_2d = preds_bin.reshape(-1, 1)
				targets_2d = targets_np.reshape(-1, 1)
			else:
				preds_bin_2d = preds_bin
				targets_2d = targets_np

			precision, recall, f1, support = precision_recall_fscore_support(
				targets_2d, preds_bin_2d, average=None, zero_division=0
			)

			per_class: Dict[str, Dict[str, Union[float, int]]] = {}
			num_classes = precision.shape[0]

			resolved_names: List[str] = []
			if preds_bin_2d.shape[1] == 1:
				positive_name = str(label_names[0]) if label_names else "positive"
				negative_name = (
					negative_class_name if negative_class_name is not None else "No pathology"
				)
				resolved_names = [negative_name, positive_name]
			else:
				resolved_names = [str(name) for name in label_names]

			if len(resolved_names) < num_classes:
				for idx in range(len(resolved_names), num_classes):
					resolved_names.append(f"class_{idx}")
			elif len(resolved_names) > num_classes:
				resolved_names = resolved_names[:num_classes]

			for i in range(num_classes):
				per_class[resolved_names[i]] = {
					"precision": float(precision[i]),
					"recall": float(recall[i]),
					"f1": float(f1[i]),
					"support": int(support[i]),
				}

			out["per_class"] = per_class
		except Exception as exc:
			log.warning("Failed to compute per-class metrics: %s", exc)

	return out


def train_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
) -> float:
	"""Run one training epoch and return the mean loss."""

	model.train()
	total_loss = 0.0

	train_iter = tqdm(dataloader, desc="Train", unit="batch")
	for batch in train_iter:
		features = batch["visual_features"].to(device)
		labels = batch["labels"].to(device)

		optimizer.zero_grad()
		preds = model(features)
		loss = criterion(preds, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		try:
			train_iter.set_postfix(train_loss=f"{loss.item():.4f}")
		except Exception:
			pass

	return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	label_names: Optional[List[str]] = None,
	negative_class_name: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
	"""Evaluate a model on the provided dataloader."""

	model.eval()
	total_loss = 0.0
	all_preds = []
	all_labels = []

	eval_iter = tqdm(dataloader, desc="Eval", unit="batch")
	for batch in eval_iter:
		features = batch["visual_features"].to(device)
		labels = batch["labels"].to(device)

		preds = model(features)
		loss = criterion(preds, labels)

		total_loss += loss.item()
		all_preds.append(preds)
		all_labels.append(labels)
		try:
			eval_iter.set_postfix(eval_loss=f"{loss.item():.4f}")
		except Exception:
			pass

	avg_loss = total_loss / len(dataloader)
	metrics = compute_metrics(
		torch.cat(all_preds, dim=0),
		torch.cat(all_labels, dim=0),
		label_names=label_names,
		negative_class_name=negative_class_name,
	)
	return avg_loss, metrics