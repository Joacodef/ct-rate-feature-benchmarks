"""Training and evaluation loops for classification models."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
	"""Compute macro AUROC for multi-label classification logits."""
	preds_prob = torch.sigmoid(preds).detach().cpu().numpy()
	targets_np = targets.detach().cpu().numpy()

	try:
		auroc = roc_auc_score(targets_np, preds_prob, average="macro")
	except ValueError as exc:
		log.warning("Could not compute AUROC (likely due to missing labels): %s", exc)
		auroc = 0.0

	return {"auroc": auroc}


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
) -> Tuple[float, Dict[str, float]]:
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
	metrics = compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0))
	return avg_loss, metrics