from datetime import datetime
from logging import Logger

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """
    Class for experiment tracking via TensorBoard.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,  # TODO: refoctor
        run_id=None,
        run_name=None,
        mode="online",  # TODO: refactor
        log_dir="./runs",
        **kwargs,
    ):
        """
        Args:
            logger (Logger): logger for essages layout
            project_config (dict): exp config
            project_name (str): project name used in tensorboard logs
            entity (str | None): ignored
            run_id (str | None): run id (used in tensorboard logs)
            run_name (str | None): run name (used in tensorboard logs)
            mode (str): ignored
            log_dir (str): dir for logs
        """
        self.logger: Logger = logger
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{self.run_id}"
        self.log_dir = f"{log_dir}/{project_name}/{self.run_name}"
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Sets current step and regime (train, val, test).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            steps_per_sec = (self.step - previous_step) / duration.total_seconds()
            self.add_scalar("steps_per_sec", steps_per_sec)
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """Adds prefix (`train_`, `val_`, `test_`)."""
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def add_scalar(self, scalar_name, scalar):
        # TODO: check
        self.writer.add_scalar(self._object_name(scalar_name), scalar, self.step)

    def add_scalars(self, scalars):
        for scalar_name, scalar in scalars.items():
            self.add_scalar(scalar_name, scalar)

    def add_image(self, image_name, image):
        # TODO: check
        if isinstance(image, torch.Tensor):
            self.writer.add_image(self._object_name(image_name), image, self.step)
        else:
            self.logger.warning("TensorBoard supports only torch.Tensor for images.")

    def add_text(self, text_name, text):
        self.writer.add_text(self._object_name(text_name), text, self.step)

    def add_histogram(self, hist_name, values_for_hist, bins=30):
        if isinstance(values_for_hist, torch.Tensor):
            values_for_hist = values_for_hist.detach().cpu().numpy()

        self.writer.add_histogram(
            self._object_name(hist_name), values_for_hist, self.step, bins=bins
        )

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Logs CSV instead of table, tables not supported in TensorBoard
        """
        csv_path = f"{self.log_dir}/{self._object_name(table_name)}.csv"
        table.to_csv(csv_path)
        self.logger.info(f"Table saved: {csv_path}")

    def close(self):
        self.writer.close()

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
