from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict[str, dict]]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    kt_features_names = dataset_items[0]["kt_features"].keys()
    dt_features_names = dataset_items[0]["dt_features"].keys()
    kt_features = {}
    dt_features = {}

    for feature_name in kt_features_names:
        kt_features[feature_name] = [
            item["kt_features"][feature_name] for item in dataset_items
        ]
    for feature_name in dt_features_names:
        dt_features[feature_name] = [
            item["dt_features"][feature_name] for item in dataset_items
        ]

    return {
        "kt_features": kt_features,
        "dt_features": dt_features,
    }
