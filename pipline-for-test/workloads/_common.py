"""Pure PyTorch helpers shared by benchmark workloads (zero dependency on `src/`)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def build_base_parser(description: str, *, default_steps: int = 20, default_batch_size: int = 16) -> argparse.ArgumentParser:
    """Create the standard CLI parser shared across all workloads."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--steps", type=int, default=default_steps, help="Number of training/eval steps")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Target device")
    parser.add_argument("--save-ckpt", type=Path, default=None, help="Optional checkpoint output path")
    parser.add_argument("--load-ckpt", type=Path, default=None, help="Optional checkpoint input path to resume from")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    """Resolve `auto` or explicit device string to a torch.device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    """Synchronize CUDA/MPS device for accurate wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def read_peak_memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        alloc = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        resv = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        return round(alloc, 2), round(resv, 2)
    return 0.0, 0.0


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save standard PyTorch checkpoint dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> int:
    """Load standard PyTorch checkpoint dict and return saved step."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("step", 0))


def emit_summary(
    name: str,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    elapsed_sec: float,
    losses: list[float],
    peak_alloc_mb: float,
    peak_reserved_mb: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Print human-readable summary and machine-parseable WORKLOAD_SUMMARY_JSON line."""
    summary: dict[str, Any] = {
        "workload": name,
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "elapsed_sec": round(elapsed_sec, 4),
        "step_ms_avg": round((elapsed_sec * 1000.0) / max(1, steps), 3),
        "throughput_samples_sec": round((steps * batch_size) / max(1e-9, elapsed_sec), 2),
        "peak_allocated_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "losses": [round(float(x), 6) for x in losses],
        "final_loss": round(float(losses[-1]), 6) if losses else None,
    }
    if extra:
        summary.update(extra)
    print(f"[{name}] device={device} steps={steps} elapsed={summary['elapsed_sec']}s final_loss={summary['final_loss']} peak_vram={peak_alloc_mb}MB")
    print(f"WORKLOAD_SUMMARY_JSON: {json.dumps(summary)}")
