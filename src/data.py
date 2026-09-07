"""MVTec dataset indexing, splits, transforms, and dataloaders.

Extracted from the final Protocol A notebooks. Does not download or bundle
MVTec images; ``DATA_ROOT`` / ``cfg.data_root`` must point at a local copy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from .config import EXPECTED_CATEGORIES, ProtocolAConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    """Exact notebook ``list_images``."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_mvtec_root(start_path: Union[str, Path]) -> Path:
    """Exact notebook ``find_mvtec_root``."""
    start_path = Path(start_path)
    expected = set(EXPECTED_CATEGORIES)
    for root, dirs, _files in os.walk(start_path):
        if len(expected.intersection(dirs)) >= 10:
            return Path(root)
    raise FileNotFoundError(f"Could not locate MVTec category folders under {start_path}")


def discover_categories(dataset_path: Path, selected: Union[str, Sequence[str]] = "all") -> List[str]:
    available = sorted(
        c for c in EXPECTED_CATEGORIES
        if (dataset_path / c / "train" / "good").is_dir()
        and (dataset_path / c / "test").is_dir()
    )
    if selected == "all":
        categories = available
    else:
        requested = sorted(list(selected))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested categories not found: {missing}")
        categories = requested
    if not categories:
        raise RuntimeError("No valid MVTec categories found.")
    return categories


def build_mvtec_index(dataset_path: Path, categories: Sequence[str]) -> pd.DataFrame:
    """Deterministic MVTec index DataFrame (notebook cell 4 records construction)."""
    records = []
    for category in categories:
        category_root = dataset_path / category

        train_good_dir = category_root / "train" / "good"
        for path in list_images(train_good_dir):
            records.append({
                "path": str(path),
                "category": category,
                "defect_type": "good",
                "binary_label": 0,
                "official_split": "train",
                "mask_path": "",
            })

        test_root = category_root / "test"
        for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            binary_label = 0 if defect_type == "good" else 1
            for path in list_images(defect_dir):
                mask_path = ""
                if binary_label == 1:
                    candidate = (
                        category_root / "ground_truth" / defect_type / f"{path.stem}_mask.png"
                    )
                    if candidate.is_file():
                        mask_path = str(candidate)
                records.append({
                    "path": str(path),
                    "category": category,
                    "defect_type": defect_type,
                    "binary_label": binary_label,
                    "official_split": "test",
                    "mask_path": mask_path,
                })

    index_df = pd.DataFrame(records).sort_values(
        ["official_split", "category", "defect_type", "path"]
    ).reset_index(drop=True)
    return index_df


def split_normal_train_by_category(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exact notebook SSL train/validation split."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []

    for _category, group in df.groupby("category", sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])

    train_out = (
        pd.concat(train_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    val_out = (
        pd.concat(val_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return train_out, val_out


def split_detector_validation_by_category(
    df: pd.DataFrame,
    tune_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exact notebook detector tune/calibration split (seed + 17011)."""
    rng = np.random.default_rng(seed + 17011)
    tune_parts, calib_parts = [], []
    for category, group in df.groupby("category", sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            raise RuntimeError(f"Need >=2 validation normals for detector split: {category}")
        n_tune = max(1, int(round(n * tune_fraction)))
        n_tune = min(n_tune, n - 1)
        tune_parts.append(df.loc[idx[:n_tune]])
        calib_parts.append(df.loc[idx[n_tune:]])
    tune_df = (
        pd.concat(tune_parts, ignore_index=True)
        .sample(frac=1, random_state=seed + 1)
        .reset_index(drop=True)
    )
    calib_df = (
        pd.concat(calib_parts, ignore_index=True)
        .sample(frac=1, random_state=seed + 2)
        .reset_index(drop=True)
    )
    return tune_df, calib_df


def build_ssl_transform(cfg: ProtocolAConfig) -> T.Compose:
    """Stochastic industrial-preserving SimCLR augmentation."""
    j = cfg.color_jitter_strength
    mean = list(cfg.imagenet_mean)
    std = list(cfg.imagenet_std)
    return T.Compose([
        T.RandomResizedCrop(cfg.image_size, scale=(cfg.crop_scale_min, 1.00)),
        T.RandomHorizontalFlip(p=cfg.horizontal_flip_p),
        T.RandomApply([
            T.ColorJitter(
                brightness=j,
                contrast=j,
                saturation=j,
                hue=cfg.hue_jitter,
            )
        ], p=cfg.color_jitter_p),
        T.RandomGrayscale(p=cfg.grayscale_p),
        T.RandomApply([
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
        ], p=cfg.gaussian_blur_p),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_eval_geometry(cfg: ProtocolAConfig) -> T.Compose:
    """Deterministic evaluation geometry: Resize(256) -> CenterCrop(224)."""
    return T.Compose([
        T.Resize(cfg.eval_resize),
        T.CenterCrop(cfg.image_size),
    ])


def build_eval_transform(cfg: ProtocolAConfig) -> T.Compose:
    """Deterministic evaluation preprocessing (geometry + ImageNet normalize)."""
    mean = list(cfg.imagenet_mean)
    std = list(cfg.imagenet_std)
    return T.Compose([
        build_eval_geometry(cfg),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_mask_eval_transform(cfg: ProtocolAConfig) -> T.Compose:
    """Nearest-neighbor mask geometry matching evaluation crops."""
    return T.Compose([
        T.Resize(cfg.eval_resize, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(cfg.image_size),
        T.ToTensor(),
    ])


class TwoViewDataset(Dataset):
    """SSL two-view dataset (notebook)."""

    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            view1 = self.transform(image)
            view2 = self.transform(image)
        return view1, view2


class PathImageDataset(Dataset):
    """Deterministic evaluation dataset (notebook)."""

    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, idx


def seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def make_eval_loader(
    df: pd.DataFrame,
    cfg: ProtocolAConfig,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Notebook ``eval_loader_for``."""
    pin = bool(device is not None and device.type == "cuda")
    return DataLoader(
        PathImageDataset(df, build_eval_transform(cfg)),
        batch_size=batch_size or cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.eval_num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )


def prepare_protocol_a_splits(
    cfg: ProtocolAConfig,
    dataset_path: Optional[Path] = None,
) -> dict:
    """
    Build index + leakage-safe Protocol A splits.

    Returns dict with keys:
    index_df, train_good_df, ssl_train_df, ssl_val_df,
    detector_tune_df, detector_calib_df, official_test_df, categories, dataset_path
    """
    dataset_path = Path(dataset_path) if dataset_path is not None else cfg.resolve_data_root()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"MVTec data root not found: {dataset_path}. "
            "Set data_root / DATA_ROOT to the official MVTec AD directory."
        )
    # Allow either the category root or a parent containing it.
    try:
        categories = discover_categories(dataset_path, cfg.selected_categories)
        root = dataset_path
    except RuntimeError:
        root = find_mvtec_root(dataset_path)
        categories = discover_categories(root, cfg.selected_categories)

    index_df = build_mvtec_index(root, categories)
    train_good_df = index_df[index_df["official_split"] == "train"].copy().reset_index(drop=True)
    official_test_df = index_df[index_df["official_split"] == "test"].copy().reset_index(drop=True)
    official_test_df["joint_class"] = (
        official_test_df["category"] + "::" + official_test_df["defect_type"]
    )

    ssl_train_df, ssl_val_df = split_normal_train_by_category(
        train_good_df,
        val_fraction=cfg.ssl_val_fraction,
        seed=cfg.seed,
    )
    detector_tune_df, detector_calib_df = split_detector_validation_by_category(
        ssl_val_df,
        tune_fraction=cfg.detector_tune_fraction_of_ssl_val,
        seed=cfg.seed,
    )

    # Hard leakage checks (notebook).
    assert set(ssl_train_df["path"]).isdisjoint(set(ssl_val_df["path"]))
    assert set(ssl_train_df["path"]).isdisjoint(set(official_test_df["path"]))
    assert set(ssl_val_df["path"]).isdisjoint(set(official_test_df["path"]))
    assert set(detector_tune_df["path"]).isdisjoint(set(detector_calib_df["path"]))
    assert set(detector_tune_df["path"]).union(set(detector_calib_df["path"])) == set(ssl_val_df["path"])

    return {
        "dataset_path": root,
        "categories": categories,
        "index_df": index_df,
        "train_good_df": train_good_df,
        "ssl_train_df": ssl_train_df,
        "ssl_val_df": ssl_val_df,
        "detector_tune_df": detector_tune_df,
        "detector_calib_df": detector_calib_df,
        "official_test_df": official_test_df,
    }
