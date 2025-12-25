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
        "item": torch.stack([x["item"] for x in dataset_items]),
        "double_user": torch.stack([x["double_user"] for x in dataset_items]),
        "double_item": torch.stack([x["double_item"] for x in dataset_items]),
        "label": torch.stack([x["label"] for x in dataset_items]),
        "kt_emb": torch.stack([x["kt_emb"] for x in dataset_items]),
        "dt_emb": torch.stack([x["dt_emb"] for x in dataset_items]),
    }
    if "dt_topic_emb" in dataset_items[0]:
        batch["dt_topic_emb"] = torch.stack([x["dt_topic_emb"] for x in dataset_items])
    if "kt_topic_emb" in dataset_items[0]:
        batch["kt_topic_emb"] = torch.stack([x["kt_topic_emb"] for x in dataset_items])
    return batch
