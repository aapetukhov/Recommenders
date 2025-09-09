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

    batch = {
        "user": torch.stack([x["user"] for x in dataset_items]),
        "double_user": torch.stack([x["double_user"] for x in dataset_items]),
        "dt_emb": torch.stack([x["dt_emb"] for x in dataset_items]),
        "item_pos": torch.stack([x["item_pos"] for x in dataset_items]),
        "double_item_pos": torch.stack([x["double_item_pos"] for x in dataset_items]),
        "kt_emb_pos": torch.stack([x["kt_emb_pos"] for x in dataset_items]),
        "item_neg": torch.stack([x["item_neg"] for x in dataset_items]),
        "double_item_neg": torch.stack([x["double_item_neg"] for x in dataset_items]),
        "kt_emb_neg": torch.stack([x["kt_emb_neg"] for x in dataset_items]),
    }
    return batch
