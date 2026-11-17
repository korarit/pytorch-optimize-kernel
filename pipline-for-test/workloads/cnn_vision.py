#!/usr/bin/env python3
"""Workload 1: ResNet-style CNN with CPU image augmentation + DataLoader."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from _common import (
    build_base_parser,
    emit_summary,
    load_checkpoint,
    read_peak_memory_mb,
    reset_peak_memory,
    resolve_device,
    save_checkpoint,
    seed_everything,
    sync_device,
)


class SyntheticVisionDataset(Dataset):
    """Deterministic synthetic image dataset with non-trivial CPU augmentation."""

    def __init__(self, num_samples: int, image_size: int = 64, num_classes: int = 10, seed: int = 42) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 3, image_size + 8, image_size + 8, generator=gen)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=gen)
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        # Deterministic per-index crop + flip + normalize on CPU
        offset_y = idx % 8
        offset_x = (idx * 3) % 8
        crop = img[:, offset_y : offset_y + self.image_size, offset_x : offset_x + self.image_size]
        if idx % 2 == 0:
            crop = torch.flip(crop, dims=[2])
        crop = (crop - self.mean) / self.std
        return crop, self.labels[idx]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class TinyResNet(nn.Module):
    def __init__(self, width: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.down = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(ResidualBlock(width * 2), ResidualBlock(width * 2))
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def main() -> int:
    parser = build_base_parser("Workload 1: CNN Vision Training", default_steps=20, default_batch_size=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)

    total_samples = max(args.steps * args.batch_size, args.batch_size * 4)
    dataset = SyntheticVisionDataset(total_samples, image_size=args.image_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyResNet(width=args.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    start_step = 0
    if args.load_ckpt is not None:
        start_step = load_checkpoint(args.load_ckpt, model, optimizer, device=device)

    model.train()
    reset_peak_memory(device)
    sync_device(device)
    t0 = time.perf_counter()

    losses: list[float] = []
    step_times_ms: list[float] = []
    loader_iter = iter(loader)
    for step in range(args.steps):
        step_t0 = time.perf_counter()
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, labels = next(loader_iter)

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        sync_device(device)
        step_times_ms.append((time.perf_counter() - step_t0) * 1000.0)
        losses.append(float(loss.detach().cpu()))

    sync_device(device)
    elapsed = time.perf_counter() - t0
    alloc_mb, resv_mb = read_peak_memory_mb(device)

    if args.save_ckpt is not None:
        save_checkpoint(args.save_ckpt, model, optimizer, step=start_step + args.steps)

    emit_summary(
        "cnn_vision",
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        elapsed_sec=elapsed,
        losses=losses,
        peak_alloc_mb=alloc_mb,
        peak_reserved_mb=resv_mb,
        step_times_ms=step_times_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
