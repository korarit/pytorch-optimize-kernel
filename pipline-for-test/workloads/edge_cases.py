#!/usr/bin/env python3
"""Workload 5: Edge-case scripts that an optimization wrapper must never break."""

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


class _LambdaDataset(Dataset):
    """Dataset holding a non-picklable lambda transform (breaks multiprocessing workers)."""

    def __init__(self, count: int, dim: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(count, dim, generator=gen)
        self.y = torch.randint(0, 8, (count,), generator=gen)
        self.transform = lambda t: t * 0.5 + 0.1  # intentionally unpicklable

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.x[idx]), self.y[idx]


def _run_preexisting_amp(device: torch.device, steps: int, batch_size: int, seed: int) -> list[float]:
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    gen = torch.Generator().manual_seed(seed + 10)
    losses: list[float] = []
    for _ in range(steps):
        x = torch.randn(batch_size, 128, generator=gen).to(device)
        y = torch.randint(0, 8, (batch_size,), generator=gen).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return losses


def _run_dynamic_shape(device: torch.device, steps: int, batch_size: int, seed: int) -> list[float]:
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 20)
    losses: list[float] = []
    for step in range(steps):
        cur_bs = max(1, batch_size + (step % 5) - 2)
        seq = 8 + (step % 7) * 4
        x = torch.randn(cur_bs, seq, 64, generator=gen).to(device)
        y = torch.randint(0, 8, (cur_bs,), generator=gen).to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x).mean(dim=1)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def _run_unpicklable_dataset(device: torch.device, steps: int, batch_size: int, seed: int) -> list[float]:
    seed_everything(seed)
    ds = _LambdaDataset(max(steps * batch_size, batch_size * 2), dim=64, seed=seed + 30)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    losses: list[float] = []
    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> int:
    parser = build_base_parser("Workload 5: Wrapper Safety Edge Cases", default_steps=10, default_batch_size=8)
    parser.add_argument(
        "--case",
        default="all",
        choices=["all", "preexisting_amp", "dynamic_shape", "unpicklable_dataset"],
        help="Which edge case to run",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    cases = {
        "preexisting_amp": _run_preexisting_amp,
        "dynamic_shape": _run_dynamic_shape,
        "unpicklable_dataset": _run_unpicklable_dataset,
    }
    selected = list(cases.keys()) if args.case == "all" else [args.case]

    # Optional ckpt check on a tiny module if requested
    if args.save_ckpt is not None or args.load_ckpt is not None:
        tiny = nn.Linear(8, 8).to(device)
        if args.load_ckpt is not None:
            load_checkpoint(args.load_ckpt, tiny, device=device)
        if args.save_ckpt is not None:
            save_checkpoint(args.save_ckpt, tiny, step=args.steps)

    reset_peak_memory(device)
    sync_device(device)
    t0 = time.perf_counter()

    all_losses: list[float] = []
    case_finals: dict[str, float] = {}
    for name in selected:
        case_losses = cases[name](device, args.steps, args.batch_size, args.seed)
        all_losses.extend(case_losses)
        case_finals[name] = round(case_losses[-1], 6)

    sync_device(device)
    elapsed = time.perf_counter() - t0
    alloc_mb, resv_mb = read_peak_memory_mb(device)

    emit_summary(
        f"edge_cases_{args.case}",
        device=device,
        steps=args.steps * len(selected),
        batch_size=args.batch_size,
        elapsed_sec=elapsed,
        losses=all_losses,
        peak_alloc_mb=alloc_mb,
        peak_reserved_mb=resv_mb,
        extra={"cases": case_finals},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
