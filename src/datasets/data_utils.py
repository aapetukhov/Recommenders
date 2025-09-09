from itertools import repeat

import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        # TODO: initialise collate_fn properly
        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            drop_last=(dataset_partition == "train"),
            collate_fn=collate_fn,
            worker_init_fn=set_worker_seed,
            # shuffle=(dataset_partition == "train"),
            # not specifying shuffle because it's an iterable dataset
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def compute_epoch_len(dataloader):
    if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
        length = 0
        for _ in dataloader:
            length += 1
            if length % 1000 == 0:
                print(f"WAIT FOR NOW. COUNTED (in batches): {length}")
        # length = sum(1 for _ in dataloader)
        print(f"Length of the dataset (in batches): {length}")
        return length

    print(f"Lenght of the dataset (in batches): {len(dataloader)}")
    return len(dataloader)
