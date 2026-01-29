# gepc/datasets/images.py
# -*- coding: utf-8 -*-
"""
Image datasets loader for GEPC experiments.

Supported datasets (torchvision):
  - cifar10, cifar100
  - svhn
  - textures (DTD)
  - celeba

Plus a deterministic synthetic toy dataset:
  - gauss_mu<value>  (e.g., gauss_mu0.0, gauss_mu0.5, gauss_mu1.0)

All pipelines output normalized tensors in [-1, 1] via Normalize(mean=0.5, std=0.5).
We keep the historical behavior of doing a first resize to `image_size` and (optionally)
a second resize to `model_image_size` if different (to match backbone input size).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _interp(mode: str):
    """Map string -> torchvision InterpolationMode with a safe fallback."""
    from torchvision.transforms import InterpolationMode as I

    mode = str(mode).lower()
    table = {
        "bilinear": I.BILINEAR,
        "nearest": I.NEAREST,
        "bicubic": I.BICUBIC,
        "box": I.BOX,
        "hamming": I.HAMMING,
        "lanczos": I.LANCZOS,
    }
    if mode == "nearest_exact":
        # NEAREST_EXACT may not exist in older torchvision; fallback to NEAREST.
        return getattr(I, "NEAREST_EXACT", I.NEAREST)

    if mode not in table:
        raise ValueError(f"Unknown interpolation mode: {mode}")
    return table[mode]


def _resize_pipeline(
    image_size: int,
    model_image_size: int | None = None,
    interpolation: str = "bilinear",
):
    """
    Resize/normalize pipeline.

    Args:
      image_size: "data" logical size (e.g. 32, 64).
      model_image_size: backbone input size; if None -> same as image_size.

    Output is always (C, model_image_size, model_image_size), normalized to [-1, 1].
    """
    image_size = int(image_size)
    if model_image_size is None:
        model_image_size = image_size
    model_image_size = int(model_image_size)

    ops = []
    # 1) Resize to data size (explicit)
    ops.append(transforms.Resize((image_size, image_size), interpolation=_interp(interpolation)))

    # 2) If backbone requires a different size, resize again
    if model_image_size != image_size:
        ops.append(transforms.Resize((model_image_size, model_image_size), interpolation=_interp(interpolation)))

    # 3) ToTensor + normalize to [-1, 1]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def _celeba_pipeline(
    image_size: int,
    model_image_size: int | None = None,
    interpolation: str = "bilinear",
    resized32: bool = False,
):
    """
    CelebA pipeline:
      - CenterCrop(140)
      - optionally resize to 32x32 first (for '*_resized' legacy naming)
      - resize to image_size, then optionally to model_image_size
      - normalize to [-1, 1]
    """
    image_size = int(image_size)
    if model_image_size is None:
        model_image_size = image_size
    model_image_size = int(model_image_size)

    ops = [transforms.CenterCrop(140)]
    if resized32:
        ops.append(transforms.Resize((32, 32), interpolation=_interp(interpolation)))

    ops.append(transforms.Resize((image_size, image_size), interpolation=_interp(interpolation)))

    if model_image_size != image_size:
        ops.append(transforms.Resize((model_image_size, model_image_size), interpolation=_interp(interpolation)))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def _apply_limit(ds, limit):
    """Deterministically take the first `limit` samples (stable subset)."""
    if limit is None:
        return ds
    limit = int(min(int(limit), len(ds)))
    return Subset(ds, range(limit))


class GaussianToyDataset(Dataset):
    """
    Deterministic synthetic dataset: x ~ N(mean, std^2 I) in 3xHxW, already in [-1, 1].

    name format: gauss_mu<value>
      - mean: float in pixel space after normalization (so typically within [-1, 1])
      - std: fixed (default 0.25)
      - split: only used to change seed deterministically
    """

    def __init__(self, mean: float, std: float, image_size: int, split: str = "train", length: int = 10000):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.image_size = int(image_size)
        self.length = int(length)

        base_seed = 12345 + int(1000 * self.mean)
        if str(split).startswith("train"):
            seed = base_seed
        elif str(split).startswith("val"):
            seed = base_seed + 1
        else:
            seed = base_seed + 2

        rng = np.random.RandomState(seed)
        C, H, W = 3, self.image_size, self.image_size
        data = rng.normal(loc=self.mean, scale=self.std, size=(self.length, C, H, W)).astype(np.float32)
        data = np.clip(data, -1.0, 1.0)

        self.data = torch.from_numpy(data)
        self.targets = torch.zeros(self.length, dtype=torch.long)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_data(
    name: str,
    data_dir: str,
    batch_size: int,
    image_size: int,
    train: bool = False,
    split: str | None = None,
    limit: int | None = None,
    interpolation: str = "bilinear",
    shuffle: bool = True,
    num_workers: int = 2,
    download: bool = True,
    model_image_size: int | None = None,
):
    """
    Create a DataLoader for supported datasets.

    Args:
      name: cifar10 / cifar100 / svhn / textures / celeba (and legacy *_resized),
            or gauss_mu<val>.
      data_dir: dataset root (e.g. ./data)
      image_size: logical "data" resize
      model_image_size: final resize for backbone input (if different)
      split: train / test / val / extra (SVHN)
    """
    if model_image_size is None:
        model_image_size = image_size

    name = str(name).lower()
    if split is None:
        split = "train" if train else "test"

    # --- CIFAR ---
    if name in ("cifar10", "cifar10_resized"):
        ds = datasets.CIFAR10(
            data_dir,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
            train=(split == "train"),
        )

    elif name in ("cifar100", "cifar100_resized"):
        ds = datasets.CIFAR100(
            data_dir,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
            train=(split == "train"),
        )

    # --- SVHN ---
    elif name in ("svhn", "svhn_resized"):
        svhn_split = split if split in ("train", "test", "extra") else ("train" if train else "test")
        ds = datasets.SVHN(
            data_dir,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
            split=svhn_split,
        )
        
    # --- CelebA ---
    elif name in ("celeba", "celeba_resized"):
        resized32 = name.endswith("_resized")
        celeba_split = "train" if split == "train" else "test"
        ds = datasets.CelebA(
            data_dir,
            download=download,
            transform=_celeba_pipeline(image_size, model_image_size, interpolation, resized32=resized32),
            split=celeba_split,
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    ds = _apply_limit(ds, limit)

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=False,
    )
