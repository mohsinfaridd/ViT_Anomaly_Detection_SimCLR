"""Prototype banks, kNN scoring, and image aggregation.

PRIMARY aggregation: normalized log-mean-exp with beta=4 (``logsumexp_t4``).
SECONDARY ablation: fixed 25% global + 75% local hybrid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from .config import ProtocolAConfig
from .features import patch_coordinates


@torch.no_grad()
def score_global_knn(query_features, reference_features, k, eval_batch_size: int = 32, device=None):
    """Exact notebook ``score_global_knn``."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k = min(int(k), len(reference_features))
    reference = reference_features.to(device)
    scores = []
    for start in range(0, len(query_features), eval_batch_size):
        query = query_features[start:start + eval_batch_size].to(device)
        distances = torch.cdist(query.float(), reference.float(), p=2)
        nearest = torch.topk(distances, k=k, largest=False, dim=1).values
        scores.append(nearest.mean(dim=1).cpu())
    return torch.cat(scores).numpy()


def build_patch_prototypes(
    category: str,
    train_patch_features: torch.Tensor,
    cfg: ProtocolAConfig,
    patch_grid=(14, 14),
    save_path: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """
    MiniBatchKMeans prototype bank (notebook ``build_or_load_patch_prototypes`` core).

    Category-specific memory: up to ``patch_prototype_count`` prototypes.
    """
    n_images, n_patches, dim = train_patch_features.shape
    flat = train_patch_features.reshape(-1, dim).float()
    positions = patch_coordinates(patch_grid).repeat(n_images, 1)

    if len(flat) > cfg.patch_candidate_max_per_category:
        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed + sum(ord(ch) for ch in category))
        indices = torch.randperm(len(flat), generator=g)[: cfg.patch_candidate_max_per_category]
        flat = flat[indices]
        positions = positions[indices]

    n_clusters = min(int(cfg.patch_prototype_count), len(flat))
    if n_clusters < 2:
        prototypes = F.normalize(flat.clone(), dim=1)
        proto_positions = positions.clone()
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=cfg.seed,
            batch_size=min(cfg.patch_kmeans_batch_size, len(flat)),
            max_iter=cfg.patch_kmeans_max_iter,
            n_init=1,
            reassignment_ratio=0.01,
        )
        labels = kmeans.fit_predict(flat.numpy())
        prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        prototypes = F.normalize(prototypes, dim=1)
        proto_positions = torch.zeros(n_clusters, 2, dtype=torch.float32)
        for cluster_id in range(n_clusters):
            members = positions[torch.from_numpy(labels == cluster_id)]
            if len(members):
                proto_positions[cluster_id] = members.mean(dim=0)

    payload = {
        "prototypes": prototypes,
        "positions": proto_positions,
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            **payload,
            "metadata": {
                "layers": tuple(int(i) for i in cfg.patch_layer_indices),
                "projection_dim": int(cfg.patch_projection_dim),
                "prototype_count": int(cfg.patch_prototype_count),
                "candidate_max": int(cfg.patch_candidate_max_per_category),
                "seed": int(cfg.seed),
                "category": category,
            },
        }, save_path)
    return payload


@torch.no_grad()
def score_patch_knn_raw(
    patch_features: torch.Tensor,
    prototype_bank,
    cfg: ProtocolAConfig,
    device=None,
    spatial_penalty: float = 0.0,
):
    """
    Return per-patch feature-space anomaly distances [N,P].

    Notebook signature retains ``spatial_penalty``; V5.1 primary path uses 0.0
    and does not apply a spatial term (parameter unused, preserved for parity).
    """
    _ = spatial_penalty  # retained for notebook API parity; unused in V5.1
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_images, n_patches, _ = patch_features.shape
    prototypes = prototype_bank["prototypes"].to(device)
    outputs = []

    for start in range(0, n_images, cfg.eval_batch_size):
        batch = patch_features[start:start + cfg.eval_batch_size].to(device)
        b = batch.shape[0]
        flat = batch.reshape(-1, batch.shape[-1])
        distances = torch.cdist(flat.float(), prototypes.float(), p=2)
        k = min(int(cfg.patch_knn_k), prototypes.shape[0])
        nearest = torch.topk(distances, k=k, largest=False, dim=1).values.mean(dim=1)
        outputs.append(nearest.reshape(b, n_patches).cpu())

    return torch.cat(outputs, dim=0)


def aggregate_patch_scores(
    patch_scores: torch.Tensor,
    mode: str,
    logsumexp_temperature: float = 4.0,
):
    """
    Convert position-calibrated [N,P] local anomaly scores to one image score.

    PRIMARY mode: ``logsumexp_t4`` (normalized log-mean-exp, beta=4).
    Legacy modes retained for notebook diagnostic parity.
    """
    mode = str(mode)
    patch_scores = patch_scores.float()
    n_patches = patch_scores.shape[1]

    def top_mean(fraction):
        count = max(1, int(math.ceil(n_patches * fraction)))
        return torch.topk(patch_scores, k=count, largest=True, dim=1).values.mean(dim=1)

    if mode == "max":
        out = patch_scores.max(dim=1).values
    elif mode == "top_0p5pct":
        out = top_mean(0.005)
    elif mode == "top_1pct":
        out = top_mean(0.01)
    elif mode == "top_2pct":
        out = top_mean(0.02)
    elif mode == "max_top1_mix":
        out = 0.5 * patch_scores.max(dim=1).values + 0.5 * top_mean(0.01)
    elif mode == "max_top2_mix":
        out = 0.5 * patch_scores.max(dim=1).values + 0.5 * top_mean(0.02)
    elif mode == "logsumexp_t4":
        # PRIMARY: normalized log-mean-exp with beta=4
        temperature = float(logsumexp_temperature)
        out = (
            torch.logsumexp(temperature * patch_scores, dim=1) - math.log(n_patches)
        ) / temperature
    elif mode == "gem_p6":
        out = torch.mean(torch.clamp(patch_scores, min=0.0) ** 6, dim=1) ** (1.0 / 6.0)
    else:
        raise ValueError(f"Unknown patch aggregation mode: {mode}")
    return out.cpu().numpy()


def fixed_hybrid_scores(global_normalized, local_normalized, global_weight: float = 0.25):
    """
    SECONDARY ablation only: fixed hybrid = w * global + (1-w) * local.

    Default w=0.25 (25% global + 75% local). Not the primary detector.
    """
    w = float(global_weight)
    return w * np.asarray(global_normalized, dtype=float) + (1.0 - w) * np.asarray(
        local_normalized, dtype=float
    )


def score_patch_knn(
    patch_features,
    prototype_bank,
    cfg: ProtocolAConfig,
    device=None,
    return_patch_scores: bool = False,
    aggregation: Optional[str] = None,
):
    """Compatibility helper: raw distance aggregation without position calibration."""
    raw = score_patch_knn_raw(
        patch_features, prototype_bank, cfg, device=device, spatial_penalty=0.0
    )
    mode = aggregation or cfg.final_fixed_patch_aggregation
    image_scores = aggregate_patch_scores(raw, mode, cfg.logsumexp_temperature)
    if return_patch_scores:
        return image_scores, raw
    return image_scores
