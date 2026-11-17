#!/usr/bin/env python3
"""Workload 4: Inference / Running benchmark (Batch=1 latency and Batch=N throughput)."""

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


class InferenceBackbone(nn.Module):
    """Lightweight Transformer-style encoder used for latency & throughput inference tests."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 4, seq_len: int = 64, num_classes: int = 16) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)


def main() -> int:
    parser = build_base_parser("Workload 4: Inference Runner", default_steps=30, default_batch_size=1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)

    model = InferenceBackbone(d_model=args.d_model, n_layers=args.n_layers, seq_len=args.seq_len).to(device)
    if args.load_ckpt is not None:
        load_checkpoint(args.load_ckpt, model, device=device)

    model.eval()
    data_gen = torch.Generator().manual_seed(args.seed + 3000)
    inputs = [
        torch.randn(args.batch_size, args.seq_len, args.d_model, generator=data_gen)
        for _ in range(args.steps)
    ]

    reset_peak_memory(device)
    sync_device(device)
    t0 = time.perf_counter()

    checksums: list[float] = []
    step_times_ms: list[float] = []
    with torch.no_grad():
        for x in inputs:
            step_t0 = time.perf_counter()
            x = x.to(device)
            logits = model(x)
            sync_device(device)
            step_times_ms.append((time.perf_counter() - step_t0) * 1000.0)
            checksums.append(float(logits.float().pow(2).mean().sqrt().detach().cpu()))

    sync_device(device)
    elapsed = time.perf_counter() - t0
    alloc_mb, resv_mb = read_peak_memory_mb(device)

    if args.save_ckpt is not None:
        save_checkpoint(args.save_ckpt, model, step=args.steps)

    emit_summary(
        f"infer_runner_bs{args.batch_size}",
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        elapsed_sec=elapsed,
        losses=checksums,
        peak_alloc_mb=alloc_mb,
        peak_reserved_mb=resv_mb,
        step_times_ms=step_times_ms,
        extra={"metric_kind": "output_rms"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
