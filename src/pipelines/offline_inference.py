from __future__ import annotations

import gzip
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from annoy import AnnoyIndex
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils.io_utils import ROOT_PATH


logger = logging.getLogger(__name__)


@dataclass
class EntityBundle:
    """
    Container with cached feature tables for a specific entity (dt | kt).
    """

    name: str
    categorical_features: Sequence[str]
    numerical_features: Sequence[str]
    feature_table: Mapping[int, Mapping[str, float]]
    base_embeddings: Mapping[int, Sequence[float]]
    topic_embeddings: Mapping[int, Sequence[float]]
    base_dim: int
    base_default: np.ndarray
    topic_dim: int
    topic_default: Optional[np.ndarray]

    @property
    def ids(self) -> List[int]:
        return list(self.feature_table.keys())


@dataclass
class OfflineInferenceArtifacts:
    """
    Output of the offline inference pipeline.
    """

    user_embeddings: Dict[int, np.ndarray]
    item_embeddings: Dict[int, np.ndarray]
    user_attentions: Optional[Dict[int, np.ndarray]]
    item_attentions: Optional[Dict[int, np.ndarray]]
    recommendations: Dict[int, List[int]]
    metrics: Dict[str, float]


def run_offline_inference(config: DictConfig) -> OfflineInferenceArtifacts:
    """
    Run the full offline inference loop:
      1. load model and feature tables,
      2. export user/item embeddings (optionally attention weights),
      3. build ANN index and retrieve candidates,
      4. compute offline ranking metrics.

    Args:
        config (DictConfig): Hydra config (src/configs/inference.yaml).
    Returns:
        OfflineInferenceArtifacts: in-memory copies of produced artifacts.
    """

    cfg = OmegaConf.to_container(config, resolve=True)

    inference_cfg = cfg["inference"]
    io_cfg = inference_cfg.get("io", {})
    persist_cfg = inference_cfg.get("persist", {})

    device = _resolve_device(inference_cfg.get("device", "auto"))

    logger.info("Loading user/item feature sizes")
    user_feature_sizes = _load_json(config.data.user_feature_sizes_path)
    item_feature_sizes = _load_json(config.data.item_feature_sizes_path)

    logger.info("Instantiating %s", config.model._target_)
    model = instantiate(
        config.model,
        user_feature_sizes=user_feature_sizes,
        item_feature_sizes=item_feature_sizes,
    ).to(device)
    model.eval()

    checkpoint_path = _resolve_path(inference_cfg["checkpoint_path"])
    logger.info("Loading checkpoint from %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])

    logger.info("Loading offline targets from %s", config.data.test_dict_path)
    test_dict = _load_pickle(config.data.test_dict_path)

    dataset_partition = inference_cfg.get("dataset_partition", "val")
    dataset_cfg = OmegaConf.to_container(
        config.datasets[dataset_partition], resolve=True
    )

    user_entity = inference_cfg.get("user_entity", "dt")
    item_entity = inference_cfg.get("item_entity", "kt")
    user_bundle = _load_entity_bundle(dataset_cfg, user_entity)
    item_bundle = _load_entity_bundle(dataset_cfg, item_entity)

    io_paths = {
        key: _resolve_path(path) if path is not None else None
        for key, path in io_cfg.items()
    }
    output_dir = io_paths.get("output_dir")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    reuse_artifacts = bool(inference_cfg.get("reuse_artifacts", True))
    save_attentions = bool(inference_cfg.get("save_attentions", False))
    batch_size = int(inference_cfg.get("batch_size", 2048))

    user_ids = _resolve_entity_ids(
        inference_cfg.get("user_ids_source", "test_dict"),
        bundle=user_bundle,
        test_dict=test_dict,
    )
    item_ids = _resolve_entity_ids(
        inference_cfg.get("item_ids_source", "all"),
        bundle=item_bundle,
        test_dict=test_dict,
    )

    logger.info("Users for evaluation: %d | items in catalog: %d", len(user_ids), len(item_ids))

    user_embeddings = None
    item_embeddings = None
    user_attentions = None
    item_attentions = None

    if reuse_artifacts and io_paths.get("user_embeddings") and io_paths["user_embeddings"].exists():
        logger.info("Loading cached user embeddings from %s", io_paths["user_embeddings"])
        user_embeddings = _load_pickle(io_paths["user_embeddings"])
        if save_attentions and io_paths.get("user_attentions") and io_paths["user_attentions"].exists():
            user_attentions = _load_pickle(io_paths["user_attentions"])
    if reuse_artifacts and io_paths.get("item_embeddings") and io_paths["item_embeddings"].exists():
        logger.info("Loading cached item embeddings from %s", io_paths["item_embeddings"])
        item_embeddings = _load_pickle(io_paths["item_embeddings"])
        if save_attentions and io_paths.get("item_attentions") and io_paths["item_attentions"].exists():
            item_attentions = _load_pickle(io_paths["item_attentions"])

    if user_embeddings is None:
        user_embeddings, user_attentions = _export_embeddings(
            model=model,
            bundle=user_bundle,
            entity_ids=user_ids,
            batch_size=batch_size,
            device=device,
            embed_mode="user",
            store_attentions=save_attentions,
            desc=f"Embedding {user_entity} (user tower)",
        )
        if persist_cfg.get("user_embeddings", True) and io_paths.get("user_embeddings"):
            io_paths["user_embeddings"].parent.mkdir(parents=True, exist_ok=True)
            _save_pickle(io_paths["user_embeddings"], user_embeddings)
        if (
            save_attentions
            and persist_cfg.get("user_attentions", True)
            and user_attentions is not None
            and io_paths.get("user_attentions")
        ):
            io_paths["user_attentions"].parent.mkdir(parents=True, exist_ok=True)
            _save_pickle(io_paths["user_attentions"], user_attentions)

    if item_embeddings is None:
        item_embeddings, item_attentions = _export_embeddings(
            model=model,
            bundle=item_bundle,
            entity_ids=item_ids,
            batch_size=batch_size,
            device=device,
            embed_mode="item",
            store_attentions=save_attentions,
            desc=f"Embedding {item_entity} (item tower)",
        )
        if persist_cfg.get("item_embeddings", True) and io_paths.get("item_embeddings"):
            io_paths["item_embeddings"].parent.mkdir(parents=True, exist_ok=True)
            _save_pickle(io_paths["item_embeddings"], item_embeddings)
        if (
            save_attentions
            and persist_cfg.get("item_attentions", True)
            and item_attentions is not None
            and io_paths.get("item_attentions")
        ):
            io_paths["item_attentions"].parent.mkdir(parents=True, exist_ok=True)
            _save_pickle(io_paths["item_attentions"], item_attentions)

    recommendations = None
    recs_path = io_paths.get("recommendations")
    if reuse_artifacts and recs_path and recs_path.exists():
        logger.info("Loading cached recommendations from %s", recs_path)
        recommendations = _load_pickle(recs_path)

    if recommendations is None:
        logger.info("Building ANN index with %d items", len(item_embeddings))
        ann_cfg = inference_cfg.get("ann", {})
        ann_metric = ann_cfg.get("metric", "angular")
        ann_trees = int(ann_cfg.get("n_trees", 64))
        ann_search_k = ann_cfg.get("search_k", -1)
        index, reverse_mapping = _build_ann_index(
            item_embeddings=item_embeddings,
            metric=ann_metric,
            n_trees=ann_trees,
        )
        logger.info(
            "Querying ANN index for %d users | top_k=%d",
            len(user_embeddings),
            inference_cfg.get("top_k", 100),
        )
        recommendations = _query_ann(
            user_embeddings=user_embeddings,
            index=index,
            reverse_mapping=reverse_mapping,
            top_k=inference_cfg.get("top_k", 100),
            search_k=ann_search_k,
        )
        if persist_cfg.get("recommendations", True) and recs_path:
            recs_path.parent.mkdir(parents=True, exist_ok=True)
            _save_pickle(recs_path, recommendations)

    logger.info("Computing offline metrics for %d targets", len(test_dict))
    metrics = compute_ranking_metrics(
        test_targets=test_dict,
        predictions=recommendations,
        k_values=inference_cfg.get("metrics_k", [10, 50, 100]),
    )
    metrics_path = io_paths.get("metrics")
    if persist_cfg.get("metrics", True) and metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return OfflineInferenceArtifacts(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_attentions=user_attentions,
        item_attentions=item_attentions,
        recommendations=recommendations,
        metrics=metrics,
    )


def _resolve_device(device_option: str) -> torch.device:
    if device_option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_option)


def _load_json(path: str | Path) -> MutableMapping:
    resolved = _resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(f"JSON path is not set: {path}")
    return json.loads(resolved.read_text())


def _load_pickle(path: str | Path):
    resolved = _resolve_path(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rb") as f:
            return pickle.load(f)
    with resolved.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, payload) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _resolve_path(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT_PATH / path).resolve()
    return path


def _load_entity_bundle(dataset_cfg: Mapping, entity: str) -> EntityBundle:
    features = list(dataset_cfg[f"{entity}_features"])
    doubles = list(dataset_cfg[f"{entity}_double_features"])
    feature_table = _load_pickle(dataset_cfg[f"{entity}_feat_path"])
    base_embeddings = _load_pickle(dataset_cfg[f"{entity}_emb_path"])
    topic_path = dataset_cfg.get(f"{entity}_topic_emb_path")
    topic_embeddings = _load_pickle(topic_path) if topic_path else {}

    base_dim = _infer_vector_dim(base_embeddings)
    topic_dim = _infer_vector_dim(topic_embeddings)

    return EntityBundle(
        name=entity,
        categorical_features=features,
        numerical_features=doubles,
        feature_table=feature_table,
        base_embeddings=base_embeddings,
        topic_embeddings=topic_embeddings,
        base_dim=base_dim,
        base_default=np.zeros(base_dim, dtype=np.float32),
        topic_dim=topic_dim,
        topic_default=(
            np.zeros(topic_dim, dtype=np.float32) if topic_dim > 0 else None
        ),
    )


def _infer_vector_dim(table: Mapping) -> int:
    if not table:
        return 0
    first_key = next(iter(table))
    vector = table[first_key]
    return int(len(vector))


def _resolve_entity_ids(
    source: Optional[str],
    bundle: EntityBundle,
    test_dict: Optional[Mapping[int, Sequence[int]]] = None,
) -> List[int]:
    if source is None or str(source).lower() == "all":
        return sorted(bundle.feature_table.keys())
    normalized = str(source).lower()
    if normalized in {"test", "test_dict"}:
        if test_dict is None:
            raise ValueError("test_dict is required to derive user ids")
        return sorted(test_dict.keys())
    path = _resolve_path(source)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Unable to resolve id source: {source}")
    payload = _load_pickle(path)
    if isinstance(payload, dict):
        return sorted(payload.keys())
    return list(payload)


def _export_embeddings(
    model,
    bundle: EntityBundle,
    entity_ids: Sequence[int],
    batch_size: int,
    device: torch.device,
    embed_mode: str,
    store_attentions: bool,
    desc: str,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    """
    Export embeddings for either user or item tower.
    """
    embedder = model.embed_user if embed_mode == "user" else model.embed_item
    embeddings: Dict[int, np.ndarray] = {}
    attentions: Optional[Dict[int, np.ndarray]]
    attentions = {} if store_attentions else None

    cat_feats = bundle.categorical_features
    num_feats = bundle.numerical_features

    entity_ids = list(entity_ids)
    logger.info("%s | batches: %d", desc, (len(entity_ids) + batch_size - 1) // batch_size)
    for start in tqdm(range(0, len(entity_ids), batch_size), desc=desc):
        chunk_ids = entity_ids[start : start + batch_size]
        cat_tensor = torch.tensor(
            [
                [
                    bundle.feature_table.get(idx, {}).get(feature, 0)
                    for feature in cat_feats
                ]
                for idx in chunk_ids
            ],
            dtype=torch.long,
            device=device,
        )
        num_tensor = torch.tensor(
            [
                [
                    bundle.feature_table.get(idx, {}).get(feature, 0.0)
                    for feature in num_feats
                ]
                for idx in chunk_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        base_tensor = torch.tensor(
            [
                bundle.base_embeddings.get(idx, bundle.base_default)
                for idx in chunk_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        extra_kwargs = {f"{bundle.name}_emb": base_tensor}

        if bundle.topic_dim > 0:
            topic_tensor = torch.tensor(
                [
                    bundle.topic_embeddings.get(idx, bundle.topic_default)
                    for idx in chunk_ids
                ],
                dtype=torch.float32,
                device=device,
            )
            extra_kwargs[f"{bundle.name}_topic_emb"] = topic_tensor

        with torch.no_grad():
            tower_embeddings, attention = embedder(
                cat_tensor,
                num_tensor,
                **extra_kwargs,
            )

        tower_embeddings = tower_embeddings.detach().cpu().numpy()
        for i, idx in enumerate(chunk_ids):
            embeddings[idx] = tower_embeddings[i]

        if attentions is not None:
            attn_np = attention.detach().cpu().numpy()
            for i, idx in enumerate(chunk_ids):
                attentions[idx] = attn_np[i]

    return embeddings, attentions


def _build_ann_index(
    item_embeddings: Mapping[int, np.ndarray],
    metric: str,
    n_trees: int,
) -> Tuple[AnnoyIndex, Dict[int, int]]:
    """
    Build Annoy index over item embeddings.
    """
    ids = list(item_embeddings.keys())
    if not ids:
        raise ValueError("No item embeddings to index")
    dim = len(item_embeddings[ids[0]])
    index = AnnoyIndex(dim, metric)
    reverse_mapping = {}
    for ann_idx, item_id in enumerate(ids):
        index.add_item(ann_idx, item_embeddings[item_id])
        reverse_mapping[ann_idx] = item_id
    index.build(n_trees)
    return index, reverse_mapping


def _query_ann(
    user_embeddings: Mapping[int, np.ndarray],
    index: AnnoyIndex,
    reverse_mapping: Mapping[int, int],
    top_k: int,
    search_k: int | None,
) -> Dict[int, List[int]]:
    """
    Query Annoy index for every user embedding.
    """
    results: Dict[int, List[int]] = {}
    effective_top_k = min(top_k, len(reverse_mapping))
    for user_id, vector in tqdm(user_embeddings.items(), desc="ANN search"):
        indices = index.get_nns_by_vector(
            vector,
            effective_top_k,
            search_k if search_k and search_k > 0 else -1,
        )
        results[user_id] = [reverse_mapping[idx] for idx in indices]
    return results


def compute_ranking_metrics(
    test_targets: Mapping[int, Sequence[int]],
    predictions: Mapping[int, Sequence[int]],
    k_values: Sequence[int],
) -> Dict[str, float]:
    """
    Compute MAP@K, Recall@K, Precision@K and NDCG@K.
    """
    metrics: Dict[str, float] = {}
    unique_k = sorted(set(int(k) for k in k_values))
    for k in unique_k:
        stats = _compute_metrics_at_k(test_targets, predictions, k)
        metrics.update(stats)
    return metrics


def _compute_metrics_at_k(
    test_targets: Mapping[int, Sequence[int]],
    predictions: Mapping[int, Sequence[int]],
    k: int,
) -> Dict[str, float]:
    recalls = []
    precisions = []
    ndcgs = []
    apk_scores = []
    for user_id in tqdm(test_targets.keys(), desc=f"metrics@{k}"):
        actual = test_targets[user_id]
        predicted = predictions.get(user_id, [])
        recalls.append(_recall_at_k(actual, predicted, k))
        precisions.append(_precision_at_k(actual, predicted, k))
        ndcgs.append(_ndcg_at_k(actual, predicted, k))
        apk_scores.append(_apk(actual, predicted, k))

    return {
        f"MAP@{k}": sum(apk_scores) / len(apk_scores) if apk_scores else 0.0,
        f"Recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"Precision@{k}": sum(precisions) / len(precisions) if precisions else 0.0,
        f"NDCG@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }


def _apk(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if actual is None:
        return 1.0
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def _recall_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if actual is None:
        return 1.0
    actual_set = set(actual)
    predicted = predicted[:k]
    return len(actual_set & set(predicted)) / len(actual_set) if actual_set else 0.0


def _precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    if predicted is None:
        return 1.0
    predicted = predicted[:k]
    if not predicted:
        return 0.0
    actual_set = set(actual) if actual is not None else set()
    return len(actual_set & set(predicted)) / k


def _ndcg_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    predicted = predicted[:k]
    actual_set = set(actual) if actual is not None else set()
    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in actual_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_set), k)))
    return dcg / idcg if idcg > 0 else 0.0
