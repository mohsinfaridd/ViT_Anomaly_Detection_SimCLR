"""Protocol A orchestration API.

Known-category inference: category identity selects prototype memory,
position calibrator, and image-score calibrator.

Test-image flow
---------------
image
  -> frozen ViT
  -> blocks 6/9/12
  -> projection (2304 -> 256)
  -> category prototype kNN

IMAGE DETECTION (primary: local_patch):
  raw patch scores
  -> position calibration
  -> log-mean-exp beta=4
  -> q=.95 image calibration
  -> score > 1
  -> normal/anomaly

LOCALIZATION:
  raw patch scores
  -> 14x14
  -> bilinear 224x224
  -> heatmap

Hybrid (25% global + 75% local) is a SECONDARY ablation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .anomaly_scoring import (
    aggregate_patch_scores,
    build_patch_prototypes,
    fixed_hybrid_scores,
    score_global_knn,
    score_patch_knn_raw,
)
from .calibration import (
    apply_patch_position_calibrator_from_cfg,
    fit_patch_position_calibrator_from_cfg,
    fit_quantile_margin_calibrator_from_cfg,
    position_calibrator_to_json,
    predict_anomaly,
    quantile_margin_transform,
)
from .config import ProtocolAConfig
from .data import prepare_protocol_a_splits
from .features import build_patch_projection_from_config, extract_global_and_patch_features
from .localization import load_eval_mask, patch_scores_to_maps
from .metrics import binary_metrics, compute_aupro
from .model import SimCLRv2ViT, build_model, freeze_model, load_ssl_checkpoint
from .utils import amp_enabled, ensure_dir, get_device, save_json, seed_everything


@dataclass
class CategoryMemory:
    """Category-specific normal reference banks and calibrators."""

    category: str
    global_reference: torch.Tensor
    prototype_bank: Dict[str, torch.Tensor]
    position_calibrator: Dict[str, Any] = field(default_factory=dict)
    local_image_calibrator: Dict[str, float] = field(default_factory=dict)
    global_image_calibrator: Dict[str, float] = field(default_factory=dict)
    hybrid_image_calibrator: Dict[str, float] = field(default_factory=dict)


def prepare_category_memory(
    category: str,
    train_df: pd.DataFrame,
    model: SimCLRv2ViT,
    projection: torch.Tensor,
    cfg: ProtocolAConfig,
    device: torch.device,
    patch_bank_dir: Optional[Path] = None,
    use_amp: bool = False,
) -> CategoryMemory:
    """
    Build category-specific global reference bank and patch prototype memory.

    Requires known category identity.
    """
    if not category:
        raise ValueError("category identity is required for Protocol A memory")

    train_cat = train_df[train_df["category"] == category].reset_index(drop=True)
    if train_cat.empty:
        raise ValueError(f"No training normals for category: {category}")

    train_global_all, train_patches = extract_global_and_patch_features(
        model, train_cat, projection, cfg, device, use_amp=use_amp
    )

    if cfg.reference_bank_max_per_category and len(train_cat) > cfg.reference_bank_max_per_category:
        rng = np.random.default_rng(cfg.seed + sum(ord(ch) for ch in category))
        idx = rng.choice(len(train_cat), size=cfg.reference_bank_max_per_category, replace=False)
        global_ref = train_global_all[idx].contiguous()
    else:
        global_ref = train_global_all

    save_path = None
    if patch_bank_dir is not None:
        patch_bank_dir = ensure_dir(patch_bank_dir)
        save_path = patch_bank_dir / f"{category}_v5_patch_prototypes.pt"

    prototype_bank = build_patch_prototypes(
        category,
        train_patches,
        cfg,
        patch_grid=model.patch_grid,
        save_path=save_path,
    )
    return CategoryMemory(
        category=category,
        global_reference=global_ref,
        prototype_bank=prototype_bank,
    )


def fit_category_calibrators(
    memory: CategoryMemory,
    calib_df: pd.DataFrame,
    model: SimCLRv2ViT,
    projection: torch.Tensor,
    cfg: ProtocolAConfig,
    device: torch.device,
    use_amp: bool = False,
) -> CategoryMemory:
    """
    Fit position + image calibrators on disjoint normal calibration images.

    Category identity selects which memory/calibrators are updated.
    """
    category = memory.category
    calib_cat = calib_df[calib_df["category"] == category].reset_index(drop=True)
    if calib_cat.empty:
        raise ValueError(f"No calibration normals for category: {category}")

    calib_global, calib_patches = extract_global_and_patch_features(
        model, calib_cat, projection, cfg, device, use_amp=use_amp
    )

    global_raw = score_global_knn(
        calib_global, memory.global_reference, cfg.knn_k, cfg.eval_batch_size, device
    )
    memory.global_image_calibrator = fit_quantile_margin_calibrator_from_cfg(global_raw, cfg)

    raw_patch = score_patch_knn_raw(
        calib_patches, memory.prototype_bank, cfg, device=device, spatial_penalty=0.0
    )
    memory.position_calibrator = fit_patch_position_calibrator_from_cfg(raw_patch, cfg)
    patch_z = apply_patch_position_calibrator_from_cfg(raw_patch, memory.position_calibrator, cfg)
    local_raw = aggregate_patch_scores(
        patch_z, cfg.final_fixed_patch_aggregation, cfg.logsumexp_temperature
    )
    memory.local_image_calibrator = fit_quantile_margin_calibrator_from_cfg(local_raw, cfg)

    global_norm = quantile_margin_transform(global_raw, memory.global_image_calibrator)
    local_norm = quantile_margin_transform(local_raw, memory.local_image_calibrator)
    hybrid_raw = fixed_hybrid_scores(
        global_norm, local_norm, global_weight=cfg.fixed_hybrid_global_weight
    )
    memory.hybrid_image_calibrator = fit_quantile_margin_calibrator_from_cfg(hybrid_raw, cfg)
    return memory


def score_category(
    category: str,
    test_df: pd.DataFrame,
    memory: CategoryMemory,
    model: SimCLRv2ViT,
    projection: torch.Tensor,
    cfg: ProtocolAConfig,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """
    Score a known category on test images.

    Returns image-level predictions and RAW localization maps.
    """
    if category != memory.category:
        raise ValueError(
            f"Category mismatch: requested '{category}' but memory is '{memory.category}'"
        )
    if not memory.position_calibrator or not memory.local_image_calibrator:
        raise RuntimeError("Calibrators not fitted; call fit_category_calibrators first")

    cat_df = test_df[test_df["category"] == category].reset_index(drop=True)
    if cat_df.empty:
        raise ValueError(f"No test images for category: {category}")

    test_global, test_patches = extract_global_and_patch_features(
        model, cat_df, projection, cfg, device, use_amp=use_amp
    )

    global_raw = score_global_knn(
        test_global, memory.global_reference, cfg.knn_k, cfg.eval_batch_size, device
    )
    global_norm = quantile_margin_transform(global_raw, memory.global_image_calibrator)

    # IMAGE DETECTION path uses position calibration.
    raw_patch = score_patch_knn_raw(
        test_patches, memory.prototype_bank, cfg, device=device, spatial_penalty=0.0
    )
    patch_z = apply_patch_position_calibrator_from_cfg(raw_patch, memory.position_calibrator, cfg)
    local_raw = aggregate_patch_scores(
        patch_z, cfg.final_fixed_patch_aggregation, cfg.logsumexp_temperature
    )
    local_norm = quantile_margin_transform(local_raw, memory.local_image_calibrator)

    # SECONDARY hybrid ablation.
    hybrid_raw = fixed_hybrid_scores(
        global_norm, local_norm, global_weight=cfg.fixed_hybrid_global_weight
    )
    hybrid_norm = quantile_margin_transform(hybrid_raw, memory.hybrid_image_calibrator)

    local_pred = predict_anomaly(local_norm)
    global_pred = predict_anomaly(global_norm)
    hybrid_pred = predict_anomaly(hybrid_norm)

    rows = []
    for i, row in cat_df.iterrows():
        rows.append({
            "path": row["path"],
            "category": row["category"],
            "defect_type": row["defect_type"],
            "mask_path": row.get("mask_path", ""),
            "binary_label": int(row["binary_label"]),
            "global_score": float(global_raw[i]),
            "local_score": float(local_raw[i]),
            "global_normalized": float(global_norm[i]),
            "local_normalized": float(local_norm[i]),
            "hybrid_normalized": float(hybrid_norm[i]),
            "normalized_score": float(local_norm[i]),
            "global_prediction": int(global_pred[i]),
            "local_prediction": int(local_pred[i]),
            "hybrid_prediction": int(hybrid_pred[i]),
            "primary_prediction": int(local_pred[i]),
            "primary_detector": "local_patch",
            "selected_patch_aggregation": cfg.final_fixed_patch_aggregation,
        })

    # LOCALIZATION uses RAW patch kNN distances (no position calibration).
    localization_maps = patch_scores_to_maps(
        raw_patch, patch_grid=model.patch_grid, image_size=cfg.image_size
    )
    gt_masks = np.stack([load_eval_mask(cat_df.iloc[i], cfg) for i in range(len(cat_df))])

    return {
        "category": category,
        "predictions": pd.DataFrame(rows),
        "raw_patch_scores": raw_patch,
        "localization_maps": localization_maps,
        "gt_masks": gt_masks,
    }


def run_protocol_a(
    cfg: ProtocolAConfig,
    categories: Optional[Sequence[str]] = None,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end Protocol A runner for a single seed/config.

    Does not train SSL. Requires a compatible ``ssl_best_checkpoint.pt``.
    Category identity is required throughout (no category-agnostic inference).
    """
    seed_everything(cfg.seed)
    device = get_device(require_gpu=cfg.require_gpu)
    use_amp = amp_enabled(cfg.use_amp, device)

    splits = prepare_protocol_a_splits(cfg)
    if categories is None:
        categories = splits["categories"]
    categories = list(categories)

    output_root = ensure_dir(cfg.resolve_output_root() / f"seed{cfg.seed}")
    tables_dir = ensure_dir(output_root / "tables")
    patch_bank_dir = ensure_dir(output_root / "patch_banks")

    model = build_model(cfg, device=device)
    ckpt = Path(checkpoint_path) if checkpoint_path else cfg.resolve_checkpoint_path()
    load_ssl_checkpoint(model, ckpt, cfg=cfg, device=device, check_compatibility=True)
    freeze_model(model)

    projection = build_patch_projection_from_config(
        cfg, feature_dim=model.feature_dim * len(cfg.patch_layer_indices)
    )

    memories: Dict[str, CategoryMemory] = {}
    for category in categories:
        mem = prepare_category_memory(
            category=category,
            train_df=splits["ssl_train_df"],
            model=model,
            projection=projection,
            cfg=cfg,
            device=device,
            patch_bank_dir=patch_bank_dir,
            use_amp=use_amp,
        )
        mem = fit_category_calibrators(
            mem,
            splits["detector_calib_df"],
            model,
            projection,
            cfg,
            device,
            use_amp=use_amp,
        )
        memories[category] = mem

    # Persist calibrator metadata (JSON-serializable).
    threshold_payload = {}
    for category, mem in memories.items():
        threshold_payload[category] = {
            "global": float(mem.global_image_calibrator["threshold"]),
            "local": float(mem.local_image_calibrator["threshold"]),
            "hybrid": float(mem.hybrid_image_calibrator["threshold"]),
            "global_image_calibrator": mem.global_image_calibrator,
            "local_image_calibrator": mem.local_image_calibrator,
            "hybrid_image_calibrator": mem.hybrid_image_calibrator,
            "position_calibrator": position_calibrator_to_json(mem.position_calibrator),
            "patch_aggregation": cfg.final_fixed_patch_aggregation,
            "primary_detector": "local_patch",
            "fixed_hybrid_global_weight": float(cfg.fixed_hybrid_global_weight),
        }
    save_json(threshold_payload, tables_dir / "protocolA_category_thresholds.json")

    all_rows: List[dict] = []
    pixel_rows: List[dict] = []
    for category in categories:
        result = score_category(
            category=category,
            test_df=splits["official_test_df"],
            memory=memories[category],
            model=model,
            projection=projection,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
        )
        pred_df = result["predictions"]
        all_rows.extend(pred_df.to_dict(orient="records"))

        maps = result["localization_maps"]
        masks = result["gt_masks"]
        flat_y = masks.reshape(-1).astype(np.uint8)
        flat_s = maps.reshape(-1).astype(np.float32)
        if len(np.unique(flat_y)) == 2:
            pixel_auc = float(roc_auc_score(flat_y, flat_s))
            pixel_ap = float(average_precision_score(flat_y, flat_s))
            pixel_aupro = compute_aupro(
                maps, masks, max_fpr=cfg.pixel_aupro_max_fpr, num_thresholds=cfg.pixel_aupro_thresholds
            )
        else:
            pixel_auc = pixel_ap = pixel_aupro = float("nan")

        pixel_rows.append({
            "category": category,
            "localization_map_mode": cfg.localization_map_mode,
            "pixel_roc_auc": pixel_auc,
            "pixel_average_precision": pixel_ap,
            "pixel_aupro_0.30": pixel_aupro,
            "n_test_images": int(len(pred_df)),
            "n_anomalous_images": int(pred_df["binary_label"].sum()),
        })

    protocolA_pred_df = pd.DataFrame(all_rows)
    protocolA_pred_df.to_csv(tables_dir / "protocolA_official_test_predictions.csv", index=False)

    protocolA_pixel_df = pd.DataFrame(pixel_rows)
    protocolA_pixel_df.to_csv(tables_dir / "protocolA_per_category_pixel_metrics.csv", index=False)

    y_true = protocolA_pred_df["binary_label"].to_numpy(dtype=int)
    primary = binary_metrics(
        y_true,
        protocolA_pred_df["local_prediction"].to_numpy(dtype=int),
        protocolA_pred_df["local_normalized"].to_numpy(dtype=float),
    )
    save_json(primary, tables_dir / "protocolA_primary_overall_metrics.json")

    ablation = []
    for method, pred_col, score_col in [
        ("global_only", "global_prediction", "global_normalized"),
        ("local_patch", "local_prediction", "local_normalized"),
        ("hybrid_fixed_g25", "hybrid_prediction", "hybrid_normalized"),
    ]:
        ablation.append({
            "method": method,
            **binary_metrics(
                y_true,
                protocolA_pred_df[pred_col].to_numpy(dtype=int),
                protocolA_pred_df[score_col].to_numpy(dtype=float),
            ),
        })
    pd.DataFrame(ablation).to_csv(tables_dir / "protocolA_ablation_metrics.csv", index=False)

    return {
        "config": cfg,
        "categories": categories,
        "predictions": protocolA_pred_df,
        "pixel_metrics": protocolA_pixel_df,
        "primary_metrics": primary,
        "output_root": output_root,
        "memories": memories,
    }
