"""Protocol A configuration derived from the final seed notebooks.

Scientific defaults match Code1/Code2/Code3 (v5p1_FINAL_FIXED_LOGSUMEXP).
The only intended scientific difference across seed configs is ``seed``
(and the matching checkpoint path).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


EXPECTED_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fixed QR projection seed offset used by make_patch_projection().
PATCH_PROJECTION_SEED_OFFSET = 9107

# Detector-validation split RNG offset (notebook: seed + 17011).
DETECTOR_SPLIT_SEED_OFFSET = 17011


def _project_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/config.py -> src/ -> repository root
    return Path(__file__).resolve().parents[1]


@dataclass
class ProtocolAConfig:
    """Portable Protocol A configuration (notebook Config without Colab paths)."""

    # ---------------- Reproducibility ----------------
    seed: int = 42
    publication_seeds: Tuple[int, ...] = (42, 123, 2026)
    version_tag: str = "v5p1_FINAL_FIXED_LOGSUMEXP"
    selected_categories: Union[str, Sequence[str]] = "all"

    # Paths (portable; override via env or YAML)
    project_root: Optional[str] = None
    data_root: Optional[str] = None
    checkpoint_root: Optional[str] = None
    output_root: Optional[str] = None
    checkpoint_path: Optional[str] = None

    ssl_val_fraction: float = 0.20
    detector_tune_fraction_of_ssl_val: float = 0.30

    # ---------------- Images / encoder ----------------
    image_size: int = 224
    eval_resize: int = 256
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = False
    projection_hidden_dim: int = 4096
    projection_dim: int = 256

    # ---------------- Industrial-preserving SimCLR augmentation ----------------
    augmentation_profile: str = "industrial_preserving_v1"
    crop_scale_min: float = 0.60
    horizontal_flip_p: float = 0.50
    color_jitter_strength: float = 0.40
    hue_jitter: float = 0.05
    color_jitter_p: float = 0.70
    grayscale_p: float = 0.10
    gaussian_blur_p: float = 0.30

    # ---------------- SimCLR (checkpoint compatibility metadata) ----------------
    epochs: int = 200
    batch_size: int = 16
    accumulation_steps: int = 2
    num_workers: int = 2
    eval_num_workers: int = 0
    warmup_epochs: int = 10
    encoder_lr: float = 1e-4
    projector_lr: float = 3e-4
    weight_decay_encoder: float = 1e-2
    weight_decay_projector: float = 1e-4
    temperature: float = 0.10
    max_grad_norm: float = 1.0
    use_amp: bool = True
    ssl_early_stopping_patience: int = 30

    # ---------------- Protocol A scoring ----------------
    eval_batch_size: int = 32
    threshold_quantile: float = 0.95
    knn_k: int = 3
    reference_bank_max_per_category: int = 1000
    normalize_encoder_features: bool = True

    # Zero-based indices -> transformer blocks 6, 9, 12
    patch_layer_indices: Tuple[int, ...] = (5, 8, 11)
    patch_projection_dim: int = 256
    patch_projection_seed_offset: int = PATCH_PROJECTION_SEED_OFFSET
    patch_candidate_max_per_category: int = 25000
    patch_prototype_count: int = 1024
    patch_knn_k: int = 3
    patch_kmeans_batch_size: int = 2048
    patch_kmeans_max_iter: int = 50
    patch_spatial_penalty: float = 0.00

    # PRIMARY aggregation: normalized log-mean-exp with beta=4
    final_fixed_patch_aggregation: str = "logsumexp_t4"
    patch_aggregation_candidates: Tuple[str, ...] = ("logsumexp_t4",)
    logsumexp_temperature: float = 4.0

    # PRIMARY detector; hybrid is secondary ablation only
    primary_detector: str = "local_patch"
    fixed_hybrid_global_weight: float = 0.25
    hybrid_global_weight: float = 0.25

    position_calibration_positive_only: bool = False
    position_calibration_clip_z: float = 15.0
    robust_scale_eps: float = 1e-6

    patch_top_fraction: float = 0.01

    localization_map_mode: str = "raw_patch_knn"
    pixel_aupro_max_fpr: float = 0.30
    pixel_aupro_thresholds: int = 100

    bootstrap_iterations: int = 1000

    # Synthetic anomalies (diagnostic audit only; not used for selection)
    synthetic_variants_per_image: int = 4
    synthetic_patch_min_fraction: float = 0.015
    synthetic_patch_max_fraction: float = 0.14
    synthetic_anomaly_types: Tuple[str, ...] = (
        "scratch",
        "stain",
        "irregular_cutpaste",
        "texture_noise",
        "local_contrast",
        "small_blur",
    )
    synthetic_selection_auc_weight: float = 0.70
    synthetic_selection_balanced_accuracy_weight: float = 0.20
    synthetic_selection_recall_weight: float = 0.10

    require_gpu: bool = False
    quick_test: bool = False
    final_eval_only: bool = True

    # ImageNet normalization used by notebooks
    imagenet_mean: Tuple[float, float, float] = IMAGENET_MEAN
    imagenet_std: Tuple[float, float, float] = IMAGENET_STD

    # SSL checkpoint compatibility field list (notebook SSL_COMPAT_FIELDS)
    ssl_compat_fields: Tuple[str, ...] = (
        "seed",
        "ssl_val_fraction",
        "image_size",
        "backbone",
        "pretrained",
        "projection_hidden_dim",
        "projection_dim",
        "augmentation_profile",
        "crop_scale_min",
        "horizontal_flip_p",
        "color_jitter_strength",
        "hue_jitter",
        "color_jitter_p",
        "grayscale_p",
        "gaussian_blur_p",
        "epochs",
        "batch_size",
        "accumulation_steps",
        "warmup_epochs",
        "encoder_lr",
        "projector_lr",
        "weight_decay_encoder",
        "weight_decay_projector",
        "temperature",
    )

    def __post_init__(self) -> None:
        # Normalize sequence fields that YAML may load as lists.
        if isinstance(self.patch_layer_indices, list):
            object.__setattr__(self, "patch_layer_indices", tuple(self.patch_layer_indices))
        if isinstance(self.publication_seeds, list):
            object.__setattr__(self, "publication_seeds", tuple(self.publication_seeds))
        if isinstance(self.patch_aggregation_candidates, list):
            object.__setattr__(
                self, "patch_aggregation_candidates", tuple(self.patch_aggregation_candidates)
            )
        if isinstance(self.synthetic_anomaly_types, list):
            object.__setattr__(
                self, "synthetic_anomaly_types", tuple(self.synthetic_anomaly_types)
            )
        if isinstance(self.ssl_compat_fields, list):
            object.__setattr__(self, "ssl_compat_fields", tuple(self.ssl_compat_fields))
        if isinstance(self.imagenet_mean, list):
            object.__setattr__(self, "imagenet_mean", tuple(self.imagenet_mean))
        if isinstance(self.imagenet_std, list):
            object.__setattr__(self, "imagenet_std", tuple(self.imagenet_std))

    # ---- resolved path helpers ----
    def resolve_project_root(self) -> Path:
        if self.project_root:
            return Path(self.project_root).expanduser().resolve()
        return _project_root()

    def resolve_data_root(self) -> Path:
        if self.data_root:
            return Path(self.data_root).expanduser().resolve()
        env = os.environ.get("DATA_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        return self.resolve_project_root() / "data" / "mvtec"

    def resolve_checkpoint_root(self) -> Path:
        if self.checkpoint_root:
            return Path(self.checkpoint_root).expanduser().resolve()
        env = os.environ.get("CHECKPOINT_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        return self.resolve_project_root() / "checkpoints"

    def resolve_output_root(self) -> Path:
        if self.output_root:
            return Path(self.output_root).expanduser().resolve()
        env = os.environ.get("OUTPUT_ROOT")
        if env:
            return Path(env).expanduser().resolve()
        return self.resolve_project_root() / "outputs"

    def resolve_checkpoint_path(self) -> Path:
        if self.checkpoint_path:
            p = Path(self.checkpoint_path)
            if not p.is_absolute():
                p = self.resolve_project_root() / p
            return p.expanduser().resolve()
        return (
            self.resolve_checkpoint_root()
            / f"seed{self.seed}"
            / "ssl_best_checkpoint.pt"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Union[str, Path]) -> ProtocolAConfig:
    """Load a ProtocolAConfig from a YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required to load config files. Install with: pip install PyYAML")
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    known = {f.name for f in fields(ProtocolAConfig)}
    filtered = {k: v for k, v in payload.items() if k in known}
    return ProtocolAConfig(**filtered)


def config_from_seed(seed: int, **overrides: Any) -> ProtocolAConfig:
    """Build a seed-specific config with portable checkpoint defaults."""
    defaults = {
        42: {"checkpoint_path": "checkpoints/seed42/ssl_best_checkpoint.pt"},
        123: {"checkpoint_path": "checkpoints/seed123/ssl_best_checkpoint.pt"},
        2026: {"checkpoint_path": "checkpoints/seed2026/ssl_best_checkpoint.pt"},
    }
    payload = {"seed": int(seed)}
    payload.update(defaults.get(int(seed), {}))
    payload.update(overrides)
    return ProtocolAConfig(**payload)
