#!/usr/bin/env python3
"""Workload 3: Deep narrow MLP (many small layers -> kernel launch & optimizer overhead bound)."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DeepNarrowMLP(nn.Module):
    def __init__(self, width: int = 256, depth: int = 24, num_classes: int = 16) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def main() -> int:
    parser = build_base_parser("Workload 3: Deep Narrow MLP Training", default_steps=30, default_batch_size=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=24)
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)

    model = DeepNarrowMLP(width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_step = 0
    if args.load_ckpt is not None:
        start_step = load_checkpoint(args.load_ckpt, model, optimizer, device=device)

    data_gen = torch.Generator().manual_seed(args.seed + 2000)
    batches = [
        (
            torch.randn(args.batch_size, args.width, generator=data_gen),
            torch.randint(0, 16, (args.batch_size,), generator=data_gen),
        )
        for _ in range(args.steps)
    ]

    model.train()
    reset_peak_memory(device)
    sync_device(device)
    t0 = time.perf_counter()

    losses: list[float] = []
    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    sync_device(device)
    elapsed = time.perf_counter() - t0
    alloc_mb, resv_mb = read_peak_memory_mb(device)

    if args.save_ckpt is not None:
        save_checkpoint(args.save_ckpt, model, optimizer, step=start_step + args.steps)

    emit_summary(
        "deep_mlp",
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        elapsed_sec=elapsed,
        losses=losses,
        peak_alloc_mb=alloc_mb,
        peak_reserved_mb=resv_mb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
