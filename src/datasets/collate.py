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
    # TODO: implement for case with continuous features and for messages
    # for example so that we can collate the feature "message":
    # [1, 2, 3, 4] -> [1, 2, 3, 4, 0, ..., 0]
    return
