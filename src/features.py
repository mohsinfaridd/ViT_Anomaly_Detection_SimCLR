"""Feature extraction and fixed orthonormal patch projection.

Primary flow (notebook):
  blocks 6/9/12 -> L2 each -> concat 2304-D -> QR projection 256-D -> L2
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import ProtocolAConfig
from .data import make_eval_loader
from .model import SimCLRv2ViT
from .utils import autocast_context


def make_patch_projection(feature_dim: int, out_dim: int, seed: int, seed_offset: int = 9107):
    """Exact notebook ``make_patch_projection`` (Gaussian + QR, seed+9107)."""
    if out_dim > feature_dim:
        raise ValueError("patch_projection_dim cannot exceed concatenated patch feature dimension")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + seed_offset)
    random_matrix = torch.randn(feature_dim, out_dim, generator=g)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q[:, :out_dim].contiguous()


def patch_coordinates(patch_grid: Tuple[int, int] = (14, 14)) -> torch.Tensor:
    """Exact notebook ``_patch_coordinates_cpu`` for a given patch grid."""
    h, w = patch_grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()


@torch.no_grad()
def project_multilayer_patches(
    patch_layers: Sequence[torch.Tensor],
    projection: torch.Tensor,
) -> torch.Tensor:
    """
    L2-normalize each layer, concatenate, project, L2-normalize.

    ``patch_layers``: list of [B, P, C] tensors (blocks 6/9/12).
    Returns [B, P, projection_dim].
    """
    normalized_layers = [F.normalize(layer.float(), dim=2) for layer in patch_layers]
    multi_patches = torch.cat(normalized_layers, dim=2)
    projected = multi_patches @ projection
    projected = F.normalize(projected, dim=2)
    return projected


@torch.no_grad()
def extract_from_loader(
    model: SimCLRv2ViT,
    loader,
    projection: torch.Tensor,
    cfg: ProtocolAConfig,
    device: torch.device,
    description: str = "Extracting features",
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Notebook ``_extract_from_loader`` (global + projected multilayer patches)."""
    model.eval()
    projection = projection.to(device)
    global_batches, patch_batches = [], []

    for images, _indices in tqdm(loader, leave=False, desc=description):
        images = images.to(device, non_blocking=True)
        with autocast_context(use_amp):
            global_feat, patch_layers = model.encode_global_and_multilayer_patches(
                images, cfg.patch_layer_indices
            )

        global_feat = global_feat.float()
        if cfg.normalize_encoder_features:
            global_feat = F.normalize(global_feat, dim=1)

        projected = project_multilayer_patches(patch_layers, projection)
        global_batches.append(global_feat.cpu())
        patch_batches.append(projected.cpu())

    return torch.cat(global_batches, dim=0), torch.cat(patch_batches, dim=0)


@torch.no_grad()
def extract_global_and_patch_features(
    model: SimCLRv2ViT,
    dataframe,
    projection: torch.Tensor,
    cfg: ProtocolAConfig,
    device: torch.device,
    batch_size: Optional[int] = None,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Notebook ``extract_global_and_patch_features``."""
    return extract_from_loader(
        model=model,
        loader=make_eval_loader(dataframe, cfg, device=device, batch_size=batch_size),
        projection=projection,
        cfg=cfg,
        device=device,
        description="Extracting V5 global+multi-layer patch features",
        use_amp=use_amp,
    )


def build_patch_projection_from_config(
    cfg: ProtocolAConfig,
    feature_dim: Optional[int] = None,
) -> torch.Tensor:
    """Build the fixed seed-controlled orthonormal projection from config."""
    if feature_dim is None:
        # ViT-B/16 feature dim 768 × 3 layers
        feature_dim = 768 * len(cfg.patch_layer_indices)
    return make_patch_projection(
        feature_dim=int(feature_dim),
        out_dim=int(cfg.patch_projection_dim),
        seed=int(cfg.seed),
        seed_offset=int(cfg.patch_projection_seed_offset),
    )
