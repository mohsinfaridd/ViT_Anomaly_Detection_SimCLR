"""Industrial image anomaly detection — Protocol A Python package.

Public API re-exports. Importing this package does not train models,
download datasets, load checkpoints, or allocate CUDA memory.
"""

from .anomaly_scoring import (
    aggregate_patch_scores,
    build_patch_prototypes,
    fixed_hybrid_scores,
    score_global_knn,
    score_patch_knn_raw,
)
from .calibration import (
    apply_patch_position_calibrator,
    fit_patch_position_calibrator,
    fit_quantile_margin_calibrator,
    predict_anomaly,
    quantile_margin_transform,
)
from .config import ProtocolAConfig, config_from_seed, load_config
from .features import make_patch_projection, patch_coordinates
from .localization import (
    category_common_visualization_limits,
    patch_scores_to_maps,
    relative_defect_area,
)
from .metrics import binary_metrics, compute_aupro, mean_and_sample_sd
from .model import SimCLRv2ViT, build_model, freeze_model, load_ssl_checkpoint
from .protocol_a import (
    CategoryMemory,
    fit_category_calibrators,
    prepare_category_memory,
    run_protocol_a,
    score_category,
)

__all__ = [
    "ProtocolAConfig",
    "load_config",
    "config_from_seed",
    "SimCLRv2ViT",
    "build_model",
    "load_ssl_checkpoint",
    "freeze_model",
    "make_patch_projection",
    "patch_coordinates",
    "build_patch_prototypes",
    "score_patch_knn_raw",
    "score_global_knn",
    "aggregate_patch_scores",
    "fixed_hybrid_scores",
    "fit_patch_position_calibrator",
    "apply_patch_position_calibrator",
    "fit_quantile_margin_calibrator",
    "quantile_margin_transform",
    "predict_anomaly",
    "patch_scores_to_maps",
    "relative_defect_area",
    "category_common_visualization_limits",
    "binary_metrics",
    "compute_aupro",
    "mean_and_sample_sd",
    "CategoryMemory",
    "prepare_category_memory",
    "fit_category_calibrators",
    "score_category",
    "run_protocol_a",
]
