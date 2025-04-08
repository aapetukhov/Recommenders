import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, List]]): dict, containing batch-version
            of the tensors or lists.
    """
    # TODO: implement for the message
    # for example so that we can collate the feature "message":
    # [1, 2, 3, 4] -> [1, 2, 3, 4, 0, ..., 0]
    users = torch.stack([sample["user"] for sample in dataset_items])
    items = torch.stack([sample["item"] for sample in dataset_items])
    double_users = torch.stack([sample["double_user"] for sample in dataset_items])
    double_items = torch.stack([sample["double_item"] for sample in dataset_items])
    labels = torch.stack([sample["label"] for sample in dataset_items])

    return {"user": users, "item": items, "double_user": double_users, "double_item": double_items, "label": labels}
