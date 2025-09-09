import random

import pandas as pd
import torch

from src.logger.utils import plot_tensor_to_tb
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch: dict, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)  # outputs = {"logits": logits}
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    # TODO: add attention logging and weights for features
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":
            self.log_attention_weights(mode, **batch)
        else:
            self.log_attention_weights(mode, **batch)

    def log_attention_weights(self, mode, kt_weights, dt_weights, **batch):
        if kt_weights is None or dt_weights is None:
            return
        batch_size = self.evaluation_dataloaders["val"].batch_size
        idx = 0

        user_w = kt_weights[idx].detach().cpu().numpy().reshape(1, -1)
        item_w = dt_weights[idx].detach().cpu().numpy().reshape(1, -1)

        self.writer.add_image(
            f"{mode}/user_attention", plot_tensor_to_tb("user_attention", user_w)
        )
        self.writer.add_image(
            f"{mode}/item_attention", plot_tensor_to_tb("item_attention", item_w)
        )
