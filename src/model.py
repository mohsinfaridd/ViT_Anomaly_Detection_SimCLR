"""SimCLR + ViT model extracted from the final Protocol A notebooks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ProtocolAConfig
from .utils import normalize_cfg_value


class SimCLRv2ViT(nn.Module):
    """Exact notebook ``SimCLRv2ViT`` implementation."""

    def __init__(
        self,
        backbone: str,
        proj_hidden: int,
        proj_dim: int,
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = int(self.encoder.num_features)
        self.num_prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 1))
        self.patch_grid = tuple(int(x) for x in self.encoder.patch_embed.grid_size)

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def encode_global_and_multilayer_patches(self, x, layer_indices):
        """
        Return the standard global ViT representation plus patch tokens captured
        from selected transformer blocks. Forward hooks avoid depending on a
        timm-version-specific intermediate-feature API.
        """
        layer_indices = tuple(int(i) for i in layer_indices)
        n_blocks = len(self.encoder.blocks)
        if any(i < 0 or i >= n_blocks for i in layer_indices):
            raise ValueError(
                f"Invalid patch layer indices {layer_indices}; encoder has {n_blocks} blocks"
            )

        captured = {}
        handles = []

        def make_hook(index):
            def hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                captured[index] = output

            return hook

        for index in layer_indices:
            handles.append(self.encoder.blocks[index].register_forward_hook(make_hook(index)))

        try:
            final_tokens = self.encoder.forward_features(x)
        finally:
            for handle in handles:
                handle.remove()

        if final_tokens.ndim != 3:
            raise RuntimeError(
                f"Expected ViT token tensor [B,N,C], got {tuple(final_tokens.shape)}"
            )

        global_features = self.encoder.forward_head(final_tokens, pre_logits=True)
        expected_patches = self.patch_grid[0] * self.patch_grid[1]
        patch_layers = []
        for index in layer_indices:
            if index not in captured:
                raise RuntimeError(f"Failed to capture transformer block {index + 1}")
            tokens = captured[index]
            patches = tokens[:, self.num_prefix_tokens :, :]
            if patches.shape[1] != expected_patches:
                raise RuntimeError(
                    f"Block {index + 1}: expected {expected_patches} patches, "
                    f"got {patches.shape[1]}"
                )
            patch_layers.append(patches)

        return global_features, patch_layers

    def encode_global_and_patches(self, x):
        """Backward-compatible final-layer patch extractor used by helper code."""
        global_features, layers = self.encode_global_and_multilayer_patches(
            x, (len(self.encoder.blocks) - 1,)
        )
        return global_features, layers[0]

    def project(self, features):
        return F.normalize(self.projector(features), dim=1)

    def forward(self, x):
        return self.project(self.encode(x))


class NTXentLoss(nn.Module):
    """Exact notebook NT-Xent loss (kept for checkpoint/training compatibility)."""

    def __init__(self, temperature: float = 0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        if z1.shape[0] != z2.shape[0]:
            raise ValueError("Two SimCLR views must have the same batch size.")
        n = z1.shape[0]
        if n < 2:
            raise ValueError("NT-Xent requires at least two samples in a batch.")

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        logits = (z @ z.T) / self.temperature
        logits = logits.float()
        diagonal = torch.eye(2 * n, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        targets = torch.arange(2 * n, device=logits.device)
        targets = (targets + n) % (2 * n)
        return F.cross_entropy(logits, targets)


def build_model(cfg: ProtocolAConfig, device: Optional[torch.device] = None) -> SimCLRv2ViT:
    """Construct SimCLRv2ViT with notebook hyperparameters."""
    model = SimCLRv2ViT(
        backbone=cfg.backbone,
        proj_hidden=cfg.projection_hidden_dim,
        proj_dim=cfg.projection_dim,
        pretrained=cfg.pretrained,
    )
    if device is not None:
        model = model.to(device)
    return model


def ssl_checkpoint_is_compatible(
    checkpoint_config: Dict[str, Any],
    cfg: ProtocolAConfig,
) -> bool:
    """Exact notebook ``_ssl_checkpoint_is_compatible`` logic."""
    current = asdict(cfg)
    for field in cfg.ssl_compat_fields:
        if field not in checkpoint_config:
            return False
        if normalize_cfg_value(checkpoint_config[field]) != normalize_cfg_value(current[field]):
            return False
    return True


def load_ssl_checkpoint(
    model: SimCLRv2ViT,
    checkpoint_path: Union[str, Path],
    cfg: Optional[ProtocolAConfig] = None,
    device: Optional[torch.device] = None,
    *,
    check_compatibility: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load notebook SSL checkpoint structure.

    Expected payload keys include ``model_state`` and usually ``config``.
    Uses ``strict=True`` by default (notebook default for ``load_state_dict``).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SSL checkpoint not found: {checkpoint_path}")

    map_location = device if device is not None else "cpu"
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(
            f"Checkpoint does not match notebook structure (missing model_state): "
            f"{checkpoint_path}"
        )

    if check_compatibility and cfg is not None:
        ckpt_cfg = payload.get("config", {})
        if not ssl_checkpoint_is_compatible(ckpt_cfg, cfg):
            raise ValueError(
                f"SSL checkpoint is not scientifically compatible with config "
                f"seed={cfg.seed}: {checkpoint_path}"
            )

    model.load_state_dict(payload["model_state"], strict=strict)
    return payload


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze all parameters and switch to eval mode (Protocol A inference)."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
