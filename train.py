import warnings

import hydra
import torch
import json
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders, compute_epoch_len
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="test")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config in the right format.
    """
    set_random_seed(config.trainer.seed)

    # setting tensorboard verbosity to 0 to avoid spamming in console
    logging.getLogger("tensorboard").setLevel(logging.ERROR)
    logging.getLogger("tensorboard").propagate = False

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"\nUsing device: {device}\n")
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    # TODO: need to pass sizes when init
    with open(config.data.user_feature_sizes_path, "r") as f:
        user_feature_sizes = json.load(f)

    with open(config.data.item_feature_sizes_path, "r") as f:
        item_feature_sizes = json.load(f)

    model = instantiate(
        config.model,
        user_feature_sizes=user_feature_sizes,
        item_feature_sizes=item_feature_sizes,
    ).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)

    # build learning rate scheduler
    # epoch_len = compute_epoch_len(dataloaders["train"])
    epoch_len = 21926
    # epoch_len = 6460
    # epoch_len = 333
    logger.info(f"\033[1;34m{'=' * 10} Epoch length: {epoch_len} steps {'=' * 10}\033[0m")
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, steps_per_epoch=epoch_len)

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config, # config is passed here!
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
