"""Unit tests for common.train.wandb_logging helpers."""

from common.train import wandb_logging


def test_build_epoch_payload_includes_primary_and_secondary_metrics():
    payload = wandb_logging.build_epoch_payload(
        epoch=3,
        train_loss=0.12,
        val_loss=0.34,
        val_metrics={
            "auprc": 0.71,
            "auroc": 0.81,
            "f1_macro": 0.61,
            "per_class": {
                "LabelA": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 10},
                "LabelB": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 6},
            },
        },
        best_val_auprc=0.72,
        best_val_auroc=0.82,
        best_val_f1_macro=0.62,
        epochs_no_improve=1,
        improved=True,
    )

    assert payload["epoch"] == 3
    assert payload["val/auprc"] == 0.71
    assert payload["val/auroc"] == 0.81
    assert payload["val/f1_macro"] == 0.61
    assert payload["best/val_auprc"] == 0.72
    assert payload["best/val_auroc"] == 0.82
    assert payload["best/val_f1_macro"] == 0.62
    assert payload["val/LabelA/precision"] == 0.8
    assert payload["val/LabelA/support"] == 10.0
    assert payload["val/LabelB/f1"] == 0.44


def test_build_epoch_payload_ignores_non_numeric_per_class_values():
    payload = wandb_logging.build_epoch_payload(
        epoch=1,
        train_loss=0.1,
        val_loss=0.2,
        val_metrics={
            "auprc": 0.5,
            "auroc": 0.6,
            "f1_macro": 0.4,
            "per_class": {
                "LabelA": {"precision": "not-a-float", "note": "skip", "support": 3}
            },
        },
        best_val_auprc=0.5,
        best_val_auroc=0.6,
        best_val_f1_macro=0.4,
        epochs_no_improve=0,
        improved=False,
    )

    assert "val/LabelA/precision" not in payload
    assert "val/LabelA/note" not in payload
    assert payload["val/LabelA/support"] == 3.0


class _DummyWandbModule:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


class _DummyRun:
    def __init__(self):
        self.summary = {}


def test_log_wandb_metrics_and_finalize_summary_fields():
    module = _DummyWandbModule()
    run = _DummyRun()

    wandb_logging.log_wandb_metrics(module, {"a": 1.0}, step=4)
    assert module.logged == [({"a": 1.0}, 4)]

    wandb_logging.finalize_wandb_run(
        wandb_module=module,
        wandb_run=run,
        best_val_auprc=0.77,
        best_val_auroc=0.88,
        best_val_f1_macro=0.66,
        interrupted=False,
    )

    assert run.summary["best_val_auprc"] == 0.77
    assert run.summary["best_val_auroc"] == 0.88
    assert run.summary["best_val_f1_macro"] == 0.66
    assert run.summary["primary_metric"] == "auprc"
    assert run.summary["interrupted"] is False
    assert module.finished is True


def test_finalize_wandb_run_noop_when_missing_handles():
    wandb_logging.finalize_wandb_run(
        wandb_module=None,
        wandb_run=None,
        best_val_auprc=0.0,
        best_val_auroc=0.0,
        best_val_f1_macro=0.0,
        interrupted=True,
    )