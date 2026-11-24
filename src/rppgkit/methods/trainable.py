"""Interface of a trainable method: Lightning-shaped hooks that training.Trainer calls.

A trainable method is a torch Module and a Method: the Trainer fits it on chunks of the
train/val videos, then predict() runs on whole test videos like any other method.
Hooks are named as in PyTorch Lightning so moving to Lightning later is a base-class
swap: training_step / validation_step return dicts that must contain "loss" (every key
is averaged per epoch and logged as train/<key>, val/<key>), configure_optimizers
returns an optimizer or {"optimizer", "lr_scheduler": {"scheduler", "interval"}}.
Kept separate from method.py so classical methods don't import torch.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import nn

from rppgkit.methods.method import Method

SCHEDULER_INTERVALS = ("step", "epoch")


class TrainableMethod(nn.Module, Method):
    """Method trained by `training.Trainer` on fixed-length chunks.

    Attributes:
        chunk_length: Frames per training chunk.
        chunk_stride: Frames between the starts of two training chunks.
        current_epoch: The epoch being trained, from 0; set by the Trainer, as in
            Lightning.
        trainer: The Trainer fitting this method, or None; set before
            `configure_optimizers` is called.
    """

    chunk_length: int
    chunk_stride: int

    def __init__(self) -> None:
        super().__init__()
        self.current_epoch = 0
        self.trainer = None

    @abstractmethod
    def transform(
        self, chunk: dict[str, np.ndarray], train: bool
    ) -> dict[str, torch.Tensor]:
        """Turns one chunk into the tensors of one batch item.

        Runs in the DataLoader workers; per-chunk normalization and augmentation go
        here.

        Args:
            chunk: The needed arrays, each with `chunk_length` rows.
            train: True for training chunks, False for validation chunks;
                augmentations are applied only when True.

        Returns:
            Tensors by name, as `training_step` and `validation_step` receive them
            (batched).
        """

    @abstractmethod
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Computes the loss of one training batch.

        Args:
            batch: Batched tensors from `transform`, on the Trainer's device.
            batch_idx: Index of the batch within the epoch.

        Returns:
            `{"loss": ...}` plus any other scalars; the Trainer backpropagates "loss"
            and logs the epoch mean of every key as train/<key>.
        """

    @abstractmethod
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Computes the loss of one validation batch; runs without gradients.

        Args:
            batch: Batched tensors from `transform`, on the Trainer's device.
            batch_idx: Index of the batch within the epoch.

        Returns:
            `{"loss": ...}` plus any other scalars; the Trainer logs the epoch mean of
            every key as val/<key>, and the lowest val/loss picks `best_val.pt`.
        """

    @abstractmethod
    def configure_optimizers(self):
        """Creates the optimizer and, optionally, the learning-rate scheduler.

        `self.trainer.estimated_stepping_batches` holds the total number of training
        steps, e.g. for OneCycleLR; `build_optimizers` does this from config.

        Returns:
            An optimizer, or `{"optimizer": ..., "lr_scheduler": {"scheduler": ...,
            "interval": "step" | "epoch"}}` as in Lightning.
        """


def build_optimizers(
    parameters: Iterable[nn.Parameter],
    optimizer: Callable[..., torch.optim.Optimizer],
    scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
    scheduler_interval: str,
    scheduler_length_argument: str | None,
    trainer,
):
    """configure_optimizers from config partials (`_partial_: true` in the YAML).

    The scheduler is called as `scheduler(optimizer, **{scheduler_length_argument:
    length})`, where length is the run's number of scheduler steps: optimizer steps
    for interval "step", epochs for "epoch" (e.g. `total_steps` for OneCycleLR,
    `T_max` for CosineAnnealingLR). `scheduler_length_argument: null` passes nothing.

    Args:
        parameters: The parameters to optimize.
        optimizer: Optimizer partial, e.g. `torch.optim.Adam` with lr set.
        scheduler: Learning-rate scheduler partial, or None for a constant rate.
        scheduler_interval: "step" (after every batch) or "epoch".
        scheduler_length_argument: Scheduler keyword that gets the run length, or None.
        trainer: The Trainer (max_epochs, estimated_stepping_batches).

    Returns:
        The optimizer, or Lightning's {"optimizer", "lr_scheduler"} dict.

    Raises:
        ValueError: If scheduler_interval is not "step" or "epoch".
    """
    if scheduler_interval not in SCHEDULER_INTERVALS:
        raise ValueError(f"scheduler_interval must be one of {SCHEDULER_INTERVALS}")
    optimizer = optimizer(parameters)
    if scheduler is None:
        return optimizer
    length = (
        trainer.estimated_stepping_batches
        if scheduler_interval == "step"
        else trainer.max_epochs
    )
    length_kwargs = (
        {scheduler_length_argument: length} if scheduler_length_argument else {}
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler(optimizer, **length_kwargs),
            "interval": scheduler_interval,
        },
    }
